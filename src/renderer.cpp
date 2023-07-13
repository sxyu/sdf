#include "sdf/sdf.hpp"

#include <iostream>
#include <limits>
#include <thread>
#include "sdf/internal/RTree.h"
#include "sdf/internal/sdf_util.hpp"

namespace {
using RowVec3 = Eigen::Matrix<float, 1, 3>;
using RefRowVec3 = Eigen::Ref<const RowVec3>;
using RowVec3idx = Eigen::Matrix<sdf::Index, 1, 3>;
using RefRowVec3idx = Eigen::Ref<const RowVec3idx>;
using RowVec2 = Eigen::Matrix<float, 1, 2>;
using RefRowVec2 = Eigen::Ref<const RowVec2>;
}  // namespace

namespace sdf {
struct Renderer::Impl {
    Impl(Eigen::Ref<const Points> verts, Eigen::Ref<const Triangles> faces,
         int width, int height, float fx, float fy, float cx, float cy)
        : verts(verts),
          verts_cam(verts.rows(), 2),
          faces(faces),
          width(width),
          height(height),
          kd_tree(verts_cam, false /* do not build */) {
        cam_f << fx, fy;
        cam_c << cx, cy;
        update();
    }

    void update() {
        kdtree_ready = false;
        rtree.RemoveAll();
        // Manually convert coordinates to clip space
        verts_cam.noalias() = verts.leftCols<2>();
        verts_cam.array().rowwise() *= cam_f.array();
        verts_cam.array().colwise() /= verts.array().rightCols<1>();
        verts_cam.array().rowwise() += cam_c.array();

        for (int i = 0; i < faces.rows(); ++i) {
            const auto va_cam = verts_cam.row(faces(i, 0)),
                       vb_cam = verts_cam.row(faces(i, 1)),
                       vc_cam = verts_cam.row(faces(i, 2));
            Eigen::Matrix<float, 1, 2, Eigen::RowMajor> face_aabb_min = va_cam,
                                                        face_aabb_max = va_cam;
            face_aabb_min = face_aabb_min.cwiseMin(vb_cam);
            face_aabb_min = face_aabb_min.cwiseMin(vc_cam);
            face_aabb_max = face_aabb_max.cwiseMax(vb_cam);
            face_aabb_max = face_aabb_max.cwiseMax(vc_cam);
            if (std::isnan(face_aabb_min[0]) || std::isnan(face_aabb_min[1]) ||
                std::isnan(face_aabb_max[0]) || std::isnan(face_aabb_max[1])) {
                // Invalid face?
                continue;
            }

            rtree.Insert(face_aabb_min.data(), face_aabb_max.data(), i);
        }
    }

   private:
    bool _depth_face_handler(float& depth, RefRowVec3 bary,
                             RefRowVec3idx face) {
        const float new_depth = verts(face[0], 2) * bary[0] +
                                verts(face[1], 2) * bary[1] +
                                verts(face[2], 2) * bary[2];
        depth = std::min(depth, new_depth);
        return true;
    }

    bool _mask_face_handler(bool& contained, RefRowVec3 bary,
                            RefRowVec3idx face) {
        contained = true;
        return false;
    }

    bool _vertex_face_handler(int& vertex, RefRowVec3 bary,
                              RefRowVec3idx face) {
        int close_vert = 0;
        for (int i = 1; i < 3; ++i) {
            if (bary[i] > bary[close_vert]) close_vert = i;
        }
        if (vertex == -1 || verts(face[close_vert], 2) < verts(vertex, 2))
            vertex = face[close_vert];
        return true;
    }

    template <typename T>
    using FaceHandler = bool (Impl::*)(T&, RefRowVec3, RefRowVec3idx) const;

   public:
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    render_depth(int n_threads) const {
        return _render_image<float>(
            (FaceHandler<float>)&Impl::_depth_face_handler,
            std::numeric_limits<float>::max(), true /* convert FLT_MAX to 0 */, false, n_threads);
    }

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    render_mask(int n_threads) const {
        return _render_image<bool>((FaceHandler<bool>)&Impl::_mask_face_handler,
                                   false,
                                   false, false,
                                   n_threads);
    }

    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    render_nn(bool fill_outside, int n_threads) const {
        if (fill_outside && !kdtree_ready) {
            kd_tree.rebuild();
            kdtree_ready = true;
        }
        return _render_image<int>((FaceHandler<int>)&Impl::_vertex_face_handler,
                                  -1, false, fill_outside, n_threads);
    }

    Vector calc_depth(Eigen::Ref<const Points2D> points, int n_threads) const {
        return _calc<float>(
            points, (FaceHandler<float>)&Impl::_depth_face_handler,
            std::numeric_limits<float>::max(), true /* convert FLT_MAX to 0 */,
            false, n_threads);
    }

    Eigen::Matrix<bool, Eigen::Dynamic, 1> calc_mask(
        Eigen::Ref<const Points2D> points, int n_threads) const {
        return _calc<bool>(points, (FaceHandler<bool>)&Impl::_mask_face_handler,
                           uint8_t(0), false, false, n_threads);
    }

    Eigen::VectorXi calc_vertex(Eigen::Ref<const Points2D> points,
                                bool fill_outside,
                                int n_threads) const {
        if (fill_outside && !kdtree_ready) {
            kd_tree.rebuild();
            kdtree_ready = true;
        }
        return _calc<int>(points, (FaceHandler<int>)&Impl::_vertex_face_handler,
                          -1, false, fill_outside, n_threads);
    }

    // Input vertices
    Eigen::Ref<const Points> verts;
    // Input triangular faces
    Eigen::Ref<const Triangles> faces;

    // Vertices in clip space (includes x,y only; z=1)
    Points2D verts_cam;

    // Image size
    int width, height;

    // Intrinsics
    RowVec2 cam_f, cam_c;

   private:
    template <class T>
    void _raycast(const RefRowVec2& point, FaceHandler<T> face_handler,
                  T& data) const {
        auto check_face = [&](int faceid) -> bool {
            const auto face = faces.row(faceid);
            const Eigen::Matrix<float, 1, 3, Eigen::RowMajor> bary =
                util::bary2d<float>(point, verts_cam.row(face[0]),
                                    verts_cam.row(face[1]),
                                    verts_cam.row(face[2]));
            if (bary[0] >= 0.f && bary[1] >= 0.f && bary[2] >= 0.f)
                return (this->*face_handler)(data, bary, face);
            return true;
        };
        Eigen::Matrix<float, 1, 2, Eigen::RowMajor> aabb_min, aabb_max;
        aabb_min.noalias() = point;
        aabb_max.noalias() = point;
        rtree.Search(aabb_min.data(), aabb_max.data(), check_face);
    }

    template <class T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    _render_image(FaceHandler<T> face_handler, T init_val,
                  bool max_to_zero = false,
                  bool fill_outside_nn = false,
                  int n_threads = std::thread::hardware_concurrency()) const {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            result(height, width);
        result.setConstant(init_val);
        maybe_parallel_for(
            [&](int i) {
                int r = i / width, c = i % width;
                RowVec2 point;
                point << (float)c, (float)r;
                T& data = result.data()[i];
                _raycast<T>(point, face_handler, data);

                if (max_to_zero && data == std::numeric_limits<float>::max())
                    data = 0.0f;

                if (fill_outside_nn && data < T(0)) {
                    size_t index;
                    float dist;
                    nanoflann::KNNResultSet<float> resultSet(1);
                    resultSet.init(&index, &dist);
                    kd_tree.index->findNeighbors(resultSet, point.data(),
                                                 nanoflann::SearchParams(10));
                    data = static_cast<int>(index);
                }
            },
            width * height,
            n_threads);
        return result;
    }

    template <class T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _calc(
        const Eigen::Ref<const Points2D>& points, FaceHandler<T> face_handler,
        T init_val,
        bool max_to_zero = false,
        bool fill_outside_nn = false,
        int n_threads = std::thread::hardware_concurrency()) const {
        Eigen::Matrix<T, Eigen::Dynamic, 1> result(points.rows());
        result.setConstant(init_val);
        maybe_parallel_for(
            [&](int i) {
                T& data = result.data()[i];
                _raycast<T>(points.row(i), face_handler, data);

                if (max_to_zero && data == std::numeric_limits<float>::max())
                    data = T(0.0f);

                if (fill_outside_nn && data < T(0)) {
                    size_t index;
                    float dist;
                    nanoflann::KNNResultSet<float> resultSet(1);
                    resultSet.init(&index, &dist);
                    kd_tree.index->findNeighbors(resultSet,
                                                 points.data() + i * 2,
                                                 nanoflann::SearchParams(10));
                    data = static_cast<T>(index);
                }
            },
            result.rows(),
            n_threads);
        return result;
    }

    // Face R-Tree (aka AABB Tree)
    RTree<int, float, 2> rtree;

    // KD tree for NN search (optional)
    mutable nanoflann::KDTreeEigenRefAdaptor<const sdf::Points2D, 2,
                                             nanoflann::metric_L2_Simple>
        kd_tree;

    // Whether KD tree is ready to use
    mutable bool kdtree_ready;
};

Renderer::Renderer(Eigen::Ref<const Points> verts,
                   Eigen::Ref<const Triangles> faces, int width, int height,
                   float fx, float fy, float cx, float cy, bool copy)
    : own_data(copy) {
    if (copy) {
        owned_verts = verts;
        owned_faces = faces;
        p_impl = std::make_unique<Impl>(owned_verts, owned_faces, width, height,
                                        fx, fy, cx, cy);
    } else {
        p_impl =
            std::make_unique<Impl>(verts, faces, width, height, fx, fy, cx, cy);
    }
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Renderer::render_depth(int n_threads) const {
    return p_impl->render_depth(n_threads);
}

Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Renderer::render_mask(int n_threads) const {
    return p_impl->render_mask(n_threads);
}

Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Renderer::render_nn(bool fill_outside, int n_threads) const {
    return p_impl->render_nn(fill_outside, n_threads);
}

void Renderer::update() { p_impl->update(); }

Eigen::Ref<const Triangles> Renderer::faces() const { return p_impl->faces; }
Eigen::Ref<Triangles> Renderer::faces_mutable() {
    if (!own_data) {
        std::cerr
            << "ERROR: 'faces' is non mutable, construct with copy=True\n";
    }
    return owned_faces;
}

Eigen::Ref<const Points> Renderer::verts() const { return p_impl->verts; }
Eigen::Ref<Points> Renderer::verts_mutable() {
    if (!own_data) {
        std::cerr
            << "ERROR: 'verts' is non mutable, construct with copy=True\n";
    }
    return owned_verts;
}

Vector Renderer::operator()(Eigen::Ref<const Points2D> points, int n_threads) const {
    return p_impl->calc_depth(points, n_threads);
}

Eigen::Matrix<bool, Eigen::Dynamic, 1> Renderer::contains(
    Eigen::Ref<const Points2D> points, int n_threads) const {
    return p_impl->calc_mask(points, n_threads);
}

Eigen::VectorXi Renderer::nn(Eigen::Ref<const Points2D> points,
                             bool fill_outside,
                             int n_threads) const {
    return p_impl->calc_vertex(points, fill_outside, n_threads);
}

Renderer::~Renderer() = default;

}  // namespace sdf
