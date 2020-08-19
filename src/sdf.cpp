#include "sdf/sdf.hpp"

#include <iostream>
#include <iomanip>
#include <limits>
#include <atomic>
#include <thread>
#include "sdf/nanoflann.hpp"
#include "sdf/RTree.h"

namespace {
// Min number of items to allow multithreading
static const int MULTITHREAD_MIN_ITEMS = 50;
void maybe_parallel_for(std::function<void(int&)> loop_content,
                        int loop_max = MULTITHREAD_MIN_ITEMS,
                        int num_threads = std::thread::hardware_concurrency()) {
    std::atomic<int> counter(-1);
    auto worker = [&]() {
        while (true) {
            int i = ++counter;
            if (i >= loop_max) break;
            loop_content(i);
        }
    };
    if (loop_max >= MULTITHREAD_MIN_ITEMS) {
        std::vector<std::thread> threads;
        for (size_t i = 1; i < num_threads; ++i) {
            threads.emplace_back(worker);
        }
        worker();
        for (auto& thd : threads) {
            thd.join();
        }
    } else {
        worker();
    }
}
}  // namespace

namespace nanoflann {
// Static KD-tree adaptor using Eigen::Ref, so we can pass in
// e.g. blocks and Maps without copying
template <class MatrixType, int DIM,
          class Distance = nanoflann::metric_L2_Simple,
          typename IndexType = int>
struct KDTreeEigenRefAdaptor {
    typedef KDTreeEigenRefAdaptor<MatrixType, DIM, Distance> self_t;
    typedef typename MatrixType::Scalar num_t;
    typedef
        typename Distance::template traits<num_t, self_t>::distance_t metric_t;
    typedef KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType> index_t;
    index_t* index;
    explicit KDTreeEigenRefAdaptor(const Eigen::Ref<const MatrixType> mat,
                                   const int leaf_max_size = 10)
        : m_data_matrix(mat), leaf_max_size(leaf_max_size) {
        index = new index_t(
            DIM, *this,
            nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
        index->buildIndex();
    }
    ~KDTreeEigenRefAdaptor() { delete index; }

    // Rebuild the KD tree from scratch. Call if data updated.
    void rebuild() {
        delete index;
        index = new index_t(
            DIM, *this,
            nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
        index->buildIndex();
    }

    Eigen::Ref<const MatrixType> m_data_matrix;
    const int leaf_max_size;
    /// Query for the num_closest closest points to a given point (entered as
    /// query_point[0:dim-1]).
    inline void query(const num_t* query_point, const size_t num_closest,
                      IndexType* out_indices, num_t* out_distances_sq) const {
        nanoflann::KNNResultSet<typename MatrixType::Scalar, IndexType>
            resultSet(num_closest);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
    }
    /// Query for the closest points to a given point (entered as
    /// query_point[0:dim-1]).
    inline IndexType closest(const num_t* query_point) const {
        IndexType out_indices;
        num_t out_distances_sq;
        query(query_point, 1, &out_indices, &out_distances_sq);
        return out_indices;
    }
    const self_t& derived() const { return *this; }
    self_t& derived() { return *this; }
    inline size_t kdtree_get_point_count() const {
        return m_data_matrix.rows();
    }
    /// Returns the distance between the vector "p1[0:size-1]" and the data
    /// point with index "idx_p2" stored in the class:
    inline num_t kdtree_distance(const num_t* p1, const size_t idx_p2,
                                 size_t size) const {
        num_t s = 0;
        for (size_t i = 0; i < size; i++) {
            const num_t d = p1[i] - m_data_matrix.coeff(idx_p2, i);
            s += d * d;
        }
        return s;
    }
    /// Returns the dim'th component of the idx'th point in the class:
    inline num_t kdtree_get_pt(const size_t idx, int dim) const {
        return m_data_matrix.coeff(idx, dim);
    }
    /// Optional bounding-box computation: return false to default to a standard
    /// bbox computation loop.
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }
};
}  // namespace nanoflann

namespace sdf {

struct SDF::Impl {
    Impl(Eigen::Ref<const Points> verts, Eigen::Ref<const Triangles> faces,
         bool robust)
        : verts(verts), faces(faces), robust(robust), kd_tree(verts) {
        face_points.resize(faces.rows() * 3, face_points.ColsAtCompileTime);
        face_normal.resize(faces.rows(), face_normal.ColsAtCompileTime);
        face_area.resize(faces.rows());
        adj_faces.resize(verts.rows());
        for (int i = 0; i < faces.rows(); ++i) {
            for (int j = 0; j < faces.ColsAtCompileTime; ++j)
                adj_faces[faces(i, j)].push_back(i);
        }
        update(false);
    }

    Eigen::VectorXi nn(Eigen::Ref<const Points> points) {
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
            points.rows());
        return result;
    }

    Vector calc(Eigen::Ref<const Points> points, bool trunc_aabb = false) {
        Vector result(points.rows());
        result.setConstant(std::numeric_limits<float>::max());

        const float DIST_EPS = robust ? 0.f : 1e-5f;

        maybe_parallel_for(
            [&](int i) {
                size_t neighb_index;
                float _dist;
                nanoflann::KNNResultSet<float> resultSet(1);

                auto point = points.row(i);

                float sign;
                if (robust) sign = _raycast_z(point);

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

                resultSet.init(&neighb_index, &_dist);
                kd_tree.index->findNeighbors(resultSet, point.data(),
                                             nanoflann::SearchParams(10));

                Eigen::Matrix<float, 1, 3, Eigen::RowMajor> avg_normal;
                if (!robust) avg_normal.setZero();
                for (int faceid : adj_faces[neighb_index]) {
                    auto face_tri = face_points.block<3, 3>(faceid * 3, 0);
                    const auto normal = face_normal.row(faceid);
                    const float tridist = util::dist_point2tri<float>(
                        point, face_tri, normal, face_area[faceid]);
                    if (tridist < min_dist - DIST_EPS) {
                        min_dist = tridist;
                        if (!robust) avg_normal.noalias() = normal;
                    } else if (!robust && tridist < min_dist + DIST_EPS) {
                        avg_normal.noalias() += normal;
                    }
                }
                if (robust) {
                    min_dist *= sign;
                } else if (avg_normal.dot(point - verts.row(neighb_index)) >
                           0) {
                    // Outside, by normal
                    min_dist = -min_dist;
                }
            },
            points.rows());
        return result;
    }

    Eigen::Matrix<bool, Eigen::Dynamic, 1> contains(
        Eigen::Ref<const Points> points) {
        if (robust) {
            Eigen::Matrix<bool, Eigen::Dynamic, 1> result(points.rows());
            maybe_parallel_for(
                [&](int i) { result[i] = _raycast_z(points.row(i)) >= 0.0f; },
                points.rows());
            return result;
        } else {
            Vector vals = calc(points, true);
            return vals.array() >= 0;
        }
    }

    void update(bool need_rebuild_kd_tree = true) {
        if (need_rebuild_kd_tree) kd_tree.rebuild();
        Eigen::Matrix<float, 1, 3, Eigen::RowMajor> aabb_min, aabb_max;
        if (robust) {
            rtree.RemoveAll();
        }
        aabb.head<3>() = verts.colwise().minCoeff();
        aabb.tail<3>() = verts.colwise().maxCoeff();
        for (int i = 0; i < faces.rows(); ++i) {
            const int a = faces(i, 0), b = faces(i, 1), c = faces(i, 2);
            const auto va = verts.row(a), vb = verts.row(b), vc = verts.row(c);
            auto face_tri = face_points.block<3, 3>(i * 3, 0);
            face_tri.row(0).noalias() = va;
            face_tri.row(1).noalias() = vb;
            face_tri.row(2).noalias() = vc;
            if (robust) {
                aabb_min = face_tri.colwise().minCoeff();
                aabb_max = face_tri.colwise().maxCoeff();
                rtree.Insert(aabb_min.data(), aabb_max.data(), i);
            }

            face_normal.row(i).noalias() = util::normal<float>(face_tri);
            face_area[i] = face_normal.row(i).norm();
            face_normal.row(i) /= face_area[i];
        }
    }

    // Input vertices
    Eigen::Ref<const Points> verts;
    // Input triangular faces
    Eigen::Ref<const Triangles> faces;
    // Whether to use 'robust' sign computation
    const bool robust;

    // Stores face points [3xn_face, 3] (3x3 matrix per face, row per point)
    Points face_points;
    // Stores face normals [n_face, 3]
    Points face_normal;
    // Stores face areas [n_face]
    Vector face_area;
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

    // Only to be used in robust mode
    float _raycast_z(
        Eigen::Ref<const Eigen::Matrix<float, 1, 3, Eigen::RowMajor>> point) {
        auto raycast = [&](int ax_idx, int ax_dir) {
            Eigen::Matrix<float, 1, 3, Eigen::RowMajor> aabb_min, aabb_max;
            float ans = -1.f;
            int ax_begin = (ax_idx == 0) ? 1 : 0;
            auto check_face = [&](int faceid) -> bool {
                auto face2d = face_points.block<3, 2>(faceid * 3, ax_begin);
                auto normal = face_normal.row(faceid);
                if (normal.dot(point - face_points.row(faceid * 3)) *
                            normal[ax_idx] * ax_dir <=
                        0.f &&
                    util::point_in_tri_2d<float>(point.segment<2>(ax_begin),
                                                 face2d)) {
                    ans = -ans;
                }
                return true;
            };
            aabb_min.noalias() = point;
            aabb_max.noalias() = point;
            if (ax_dir > 0)
                aabb_max[ax_idx] = std::numeric_limits<float>::max();
            else
                aabb_min[ax_idx] = -std::numeric_limits<float>::max();
            rtree.Search(aabb_min.data(), aabb_max.data(), check_face);
            return ans;
        };
        float ans_1 = raycast(0, 1);
        float ans_2 = raycast(2, -1);
        float ans_3 = raycast(2, 1);
        return (ans_1 + ans_2 + ans_3) > 0.f ? 1.0f : -1.0f;
    }
};

SDF::SDF(Eigen::Ref<const Points> verts, Eigen::Ref<const Triangles> faces,
         bool robust, bool copy)
    : robust(robust), own_data(copy) {
    if (copy) {
        owned_verts = verts;
        owned_faces = faces;
        p_impl = std::make_unique<SDF::Impl>(owned_verts, owned_faces, robust);
    } else {
        p_impl = std::make_unique<SDF::Impl>(verts, faces, robust);
    }
}

SDF::~SDF() = default;

const std::vector<int>& SDF::adj_faces(int pointid) const {
    return p_impl->adj_faces[pointid];
}

const Vector& SDF::face_areas() const { return p_impl->face_area; }

const Points& SDF::face_normals() const { return p_impl->face_normal; }

Eigen::Ref<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> SDF::face_points(
    int faceid) const {
    return p_impl->face_points.block<3, 3>(3 * faceid, 0);
}

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

Vector SDF::operator()(Eigen::Ref<const Points> points, bool trunc_aabb) {
    return p_impl->calc(points, trunc_aabb);
}

Eigen::VectorXi SDF::nn(Eigen::Ref<const Points> points) {
    return p_impl->nn(points);
}

Eigen::Matrix<bool, Eigen::Dynamic, 1> SDF::contains(
    Eigen::Ref<const Points> points) {
    return p_impl->contains(points);
}

void SDF::update() { p_impl->update(); }
}  // namespace sdf
