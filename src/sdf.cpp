#include "sdf/sdf.hpp"

#include <iostream>
#include <iomanip>
#include <limits>
#include <atomic>
#include <thread>
#include <random>
#include <algorithm>
#include <chrono>
#include "sdf/internal/RTree.h"
#include "sdf/internal/sdf_util.hpp"

namespace sdf {

struct SDF::Impl {
    Impl(Eigen::Ref<const Points> verts, Eigen::Ref<const Triangles> faces, 
         bool robust)
        : verts(verts), faces(faces), robust(robust), kd_tree(verts) {
        face_normal.resize(faces.rows(), face_normal.ColsAtCompileTime);
        face_area.resize(faces.rows());
        adj_faces.resize(verts.rows());
        for (int i = 0; i < faces.rows(); ++i) {
            for (int j = 0; j < faces.ColsAtCompileTime; ++j)
                adj_faces[faces(i, j)].push_back(i);
        }
        update(false);
    }

    void update(bool need_rebuild_kd_tree = true) {
        if (need_rebuild_kd_tree) kd_tree.rebuild();
        if (robust) {
            rtree.RemoveAll();
        }
        aabb.head<3>() = verts.colwise().minCoeff();
        aabb.tail<3>() = verts.colwise().maxCoeff();

        if (robust) {
            // Generate a random rotation matrix using a unit quaternion
            // to use as raycast frame, ref
            // https://en.wikipedia.org/wiki/Rotation_matrix#Uniform_random_rotation_matrices
            auto& rg = get_rng();
            std::normal_distribution<float> gaussian(0.0f, 1.0f);
            Eigen::Quaternionf rand_rot(gaussian(rg), gaussian(rg),
                                        gaussian(rg), gaussian(rg));
            rand_rot.normalize();
            raycast_axes.noalias() = rand_rot.toRotationMatrix();
        }
        for (int i = 0; i < faces.rows(); ++i) {
            const auto va = verts.row(faces(i, 0)), vb = verts.row(faces(i, 1)),
                       vc = verts.row(faces(i, 2));
            if (robust) {
                Eigen::Matrix<float, 1, 3, Eigen::RowMajor> face_aabb_min =
                                                                va *
                                                                raycast_axes,
                                                            face_aabb_max =
                                                                va *
                                                                raycast_axes;
                face_aabb_min = face_aabb_min.cwiseMin(vb * raycast_axes);
                face_aabb_min = face_aabb_min.cwiseMin(vc * raycast_axes);
                face_aabb_max = face_aabb_max.cwiseMax(vb * raycast_axes);
                face_aabb_max = face_aabb_max.cwiseMax(vc * raycast_axes);
                rtree.Insert(face_aabb_min.data(), face_aabb_max.data(), i);
            }

            if (robust)
            {
                face_normal.row(i).noalias() = util::normal<float>(va, vb, vc);
                face_area[i] = face_normal.row(i).norm();
                face_normal.row(i) /= face_area[i];
            }
        }
        total_area = face_area.sum();
        face_area_cum.resize(0);
    }

    Eigen::VectorXi nn(Eigen::Ref<const Points> points,
                        int n_threads = std::thread::hardware_concurrency()) const {
        Eigen::VectorXi result(points.rows());
        maybe_parallel_for(
            [&](int i) {
                size_t index;
                float dist;
                nanoflann::KNNResultSet<float> resultSet(1);
                resultSet.init(&index, &dist);
                kd_tree.index->findNeighbors(resultSet, points.data() + i * 3,
                                             nanoflann::SearchParams(10));
                result[i] = static_cast<int>(index);
            },
            (int)points.rows(),
            n_threads);
        return result;
    }

    Vector calc(Eigen::Ref<const Points> points,
                bool trunc_aabb = false,
                int n_threads = std::thread::hardware_concurrency()) {
        Vector result(points.rows());
        result.setConstant(std::numeric_limits<float>::max());
        point_gradients.resize(points.rows(), face_normal.ColsAtCompileTime);
        point_triangle_normals.resize(points.rows(), face_normal.ColsAtCompileTime);

        const float DIST_EPS = robust ? 0.f : 1e-5f;

        maybe_parallel_for(
            [&](int i) {
                size_t neighb_index;
                float _dist;
                nanoflann::KNNResultSet<float> result_set(1);

                auto point = points.row(i);

                float sign;
                if (robust) sign = _raycast(point);

                float& min_dist = result[i];
                
                if (trunc_aabb) {
                    // Only care about sign being correct, so we can use
                    // AABB
                    for (int t = 0; t < 3; ++t) {
                        if (point[t] < aabb[t] || point[t] > aabb[t + 3]) {
                            // Out of mesh's bounding box
                            min_dist = -std::numeric_limits<float>::max();
                            continue;
                        }
                    }
                }

                result_set.init(&neighb_index, &_dist);
                kd_tree.index->findNeighbors(result_set, point.data(),
                                             nanoflann::SearchParams(10));

                Eigen::Matrix<float, 1, 3, Eigen::RowMajor> min_gradient;
                Eigen::Matrix<float, 1, 3, Eigen::RowMajor> min_triangle_normal;
                for (int faceid : adj_faces[neighb_index]) {
                    const auto face = faces.row(faceid);

                    Eigen::Matrix<float, 1, 3, Eigen::RowMajor> trinormal; trinormal.setZero();
                    const auto trigradient = util::point2trigrad<float>(
                        point, verts.row(face(0)), verts.row(face(1)),
                        verts.row(face(2)), &trinormal, face_area[faceid]);
                    const float tridist = trigradient.squaredNorm();
                    if (tridist < min_dist - DIST_EPS) {
                        min_gradient = trigradient;
                        min_triangle_normal = trinormal;
                        min_dist = tridist;
                    }
                }
                min_dist = std::sqrt(min_dist);
                if (robust) {
                    min_dist *= sign;
                } else if (min_gradient.dot(point - verts.row(neighb_index)) >
                           0) {
                    // Outside, by normal
                    min_dist = -min_dist;
                }

                point_gradients.row(i).noalias() = min_gradient;
                point_triangle_normals.row(i).noalias() = min_triangle_normal;
            },
            (int)points.rows(),
            n_threads);
        return result;
    }

    Eigen::Matrix<bool, Eigen::Dynamic, 1> contains(
        Eigen::Ref<const Points> points,
        int n_threads = std::thread::hardware_concurrency()) {
        if (robust) {
            Eigen::Matrix<bool, Eigen::Dynamic, 1> result(points.rows());
            maybe_parallel_for(
                [&](int i) { result[i] = _raycast(points.row(i)) >= 0.0f; },
                (int)points.rows(),
                n_threads);
            return result;
        } else {
            Vector vals = calc(points, true, n_threads);
            return vals.array() >= 0;
        }
    }

    Points sample_surface(int num_points) const {
        if (face_area.rows() == 0) {
            std::cerr << "ERROR: No faces, can't sample surface.\n";
            return Points();
        }
        auto& rg = get_rng();
        std::uniform_real_distribution<float> uniform(
            0.0f, 1.0f - std::numeric_limits<float>::epsilon());
        float running = 0.f;

        Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> result(
            num_points, 3);

        // Inverse distribution sampling:
        // pick each face with prob proportional to area
        Eigen::VectorXf rand_area(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            rand_area[i] = uniform(rg) * total_area;
        }
        if (face_area_cum.rows() == 0) {
            face_area_cum.resize(face_area.rows());
            face_area_cum[0] = face_area[0];
            for (int i = 1; i < face_area.rows(); ++i)
                face_area_cum[i] = face_area_cum[i - 1] + face_area[i];
        }

        for (int j = 0; j < num_points; ++j) {
            // Triangle i
            int i =
                std::lower_bound(face_area_cum.data(),
                                 face_area_cum.data() + face_area_cum.rows(),
                                 rand_area[j]) -
                face_area_cum.data();
            i = std::max(std::min<int>(i, faces.rows() - 1), 0);

            const auto face = faces.row(i);
            const auto a = verts.row(face[0]), b = verts.row(face[1]),
                       c = verts.row(face[2]);

            // Point j <-- u.a.r. in triangle i
            const Eigen::Matrix<float, 1, 3, Eigen::RowMajor> ab = b - a,
                                                              ac = c - a,
                                                              bc = c - b;
            const Eigen::Matrix<float, 1, 3, Eigen::RowMajor> perp =
                bc.cross(face_normal.row(i)).normalized();
            bool a_dir_of_bc = ab.dot(perp) < 0.0;

            // Random in quadrilateral
            result.row(j).noalias() =
                a + uniform(rg) * ab +
                uniform(rg) * ac;  // Reflect over bc, if we're over it
            const float bp_dot_perp = (result.row(j) - b).dot(perp);
            const bool p_dir_of_bc = bp_dot_perp >= 0.0;
            if (p_dir_of_bc != a_dir_of_bc) {
                result.row(j).noalias() -= bp_dot_perp * perp * 2.f;
            }
        }
        return result;
    }

    // Input vertices
    Eigen::Ref<const Points> verts;
    // Input triangular faces
    Eigen::Ref<const Triangles> faces;
    // Whether to use 'robust' sign computation
    const bool robust;

    // Stores face normals [n_face, 3]
    Points face_normal;
    // Stores gradients for each queried point [n_points, 3]
    Points point_gradients;
    // Stores normal of the triangle retained for each queried point [n_points, 3]
    Points point_triangle_normals;
    // Stores face areas [n_face]
    Vector face_area;
    // Cumulative face areas for sampling [n_face]
    mutable Vector face_area_cum;
    // Total surface area
    float total_area;
    // Stores adjacent faces to a point [n_points, <n_adj_faces>]
    std::vector<std::vector<int>> adj_faces;

    // Store AABB of entire mesh
    // (minx, miny, minz, maxx, maxy, maxz)
    Eigen::Matrix<float, 1, 6, Eigen::RowMajor> aabb;

   private:
    // KD tree for NN search
    nanoflann::KDTreeEigenRefAdaptor<const sdf::Points, 3,
                                     nanoflann::metric_L2_Simple>
        kd_tree;

    // Face R-Tree (aka AABB Tree)
    RTree<int, float, 3> rtree;

    // Random rotation matrix which transforms points from
    // global space to space used for raycasting.
    // This allows us to use the RTree to do raycasting in an arbitrary
    // direction. Only to be used in robust mode
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> raycast_axes;

    // Raycast to check if point is in or on surface of mesh.
    // Returns 1 if in, -1 else.
    // Only to be used in robust mode
    float _raycast(Eigen::Ref<const Eigen::Matrix<float, 1, 3, Eigen::RowMajor>>
                       point_orig) const {
        for (int t = 0; t < 3; ++t) {
            if (point_orig[t] < aabb[t] || point_orig[t] > aabb[t + 3]) {
                // Out of mesh's bounding box
                return -1;
            }
        }
        Eigen::Matrix<float, 1, 3, Eigen::RowMajor> point =
            point_orig * raycast_axes;

        // ax_idx: axis index, either 0(x) or 2(z).
        //         note: y not supported by current code for efficiency
        // ax_inv: if true, raycasts in negative direction along axis, else
        //         positive direction
        // return: 1 if inside, 0 else
        auto raycast = [&](int ax_idx, bool ax_inv) -> int {
            Eigen::Matrix<float, 1, 3, Eigen::RowMajor> aabb_min, aabb_max;
            int contained = 0;
            int ax_offs = ax_idx == 0 ? 1 : 0;

            auto check_face = [&](int faceid) -> bool {
                const auto face = faces.row(faceid);
                Eigen::Matrix<float, 1, 3, Eigen::RowMajor> normal =
                    face_normal.row(faceid) * raycast_axes;
                if ((normal.dot(point - verts.row(face[0]) * raycast_axes) *
                         normal[ax_idx] >
                     0.f) == ax_inv) {
                    const auto bary = util::bary2d<float>(
                        point.segment<2>(ax_offs),
                        (verts.row(face[0]) * raycast_axes).segment<2>(ax_offs),
                        (verts.row(face[1]) * raycast_axes).segment<2>(ax_offs),
                        (verts.row(face[2]) * raycast_axes)
                            .segment<2>(ax_offs));
                    if (bary[0] >= 0.f && bary[1] >= 0.f && bary[2] >= 0.f) {
                        contained ^= 1;
                    }
                }
                return true;
            };
            aabb_min.noalias() = point;
            aabb_max.noalias() = point;
            if (ax_inv)
                aabb_min[ax_idx] = -std::numeric_limits<float>::max();
            else
                aabb_max[ax_idx] = std::numeric_limits<float>::max();
            rtree.Search(aabb_min.data(), aabb_max.data(), check_face);
            return contained;
        };
        int result = raycast(2, false) + raycast(2, true);
        if (result == 1) result += raycast(0, false);  // Tiebreaker
        return result > 1 ? 1.0f : -1.0f;
    }
};

SDF::SDF(Eigen::Ref<const Points> verts, Eigen::Ref<const Triangles> faces,
         bool robust, bool copy)
    : robust(robust), own_data(copy) {
    if (copy) {
        owned_verts = verts;
        owned_faces = faces;
        p_impl = std::make_unique<Impl>(owned_verts, owned_faces, robust);
    } else {
        p_impl = std::make_unique<Impl>(verts, faces, robust);
    }
}

SDF::~SDF() = default;

const std::vector<int>& SDF::adj_faces(int pointid) const {
    return p_impl->adj_faces[pointid];
}

const float SDF::surface_area() const { return p_impl->total_area; }

const Vector& SDF::face_areas() const { return p_impl->face_area; }

const Points& SDF::face_normals() const { return p_impl->face_normal; }

const Points& SDF::point_gradients() const { return p_impl->point_gradients; }

const Points& SDF::point_triangle_normals() const { return p_impl->point_triangle_normals; }

Eigen::Ref<const Eigen::Matrix<float, 6, 1>> SDF::aabb() const {
    return p_impl->aabb.transpose();
}

Eigen::Ref<const Triangles> SDF::faces() const { return p_impl->faces; }
Eigen::Ref<Triangles> SDF::faces_mutable() {
    if (!own_data) {
        std::cerr
            << "ERROR: 'faces' is non mutable, construct with copy=True\n";
    }
    return owned_faces;
}

Eigen::Ref<const Points> SDF::verts() const { return p_impl->verts; }
Eigen::Ref<Points> SDF::verts_mutable() {
    if (!own_data) {
        std::cerr
            << "ERROR: 'verts' is non mutable, construct with copy=True\n";
    }
    return owned_verts;
}

Vector SDF::operator()(Eigen::Ref<const Points> points, bool trunc_aabb, int n_threads) {
    return p_impl->calc(points, trunc_aabb, n_threads);
}

Eigen::VectorXi SDF::nn(Eigen::Ref<const Points> points, int n_threads) const {
    return p_impl->nn(points, n_threads);
}

Eigen::Matrix<bool, Eigen::Dynamic, 1> SDF::contains(
    Eigen::Ref<const Points> points,
    int n_threads) {
    return p_impl->contains(points, n_threads);
}

void SDF::update() { p_impl->update(); }

Points SDF::sample_surface(int num_points) const {
    return p_impl->sample_surface(num_points);
}
}  // namespace sdf
