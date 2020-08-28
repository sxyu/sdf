#include "sdf/sdf.hpp"

#include <iostream>
#include <limits>
#include <functional>
#include "sdf/internal/RTree.h"
#include "sdf/internal/sdf_util.hpp"

namespace {
using IntMatrix =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ByteMatrix =
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FloatMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}  // namespace

namespace sdf {
struct Renderer::Impl {
    Impl(Eigen::Ref<const Points> verts, Eigen::Ref<const Triangles> faces,
         int width, int height, float fx, float fy, float cx, float cy)
        : verts(verts), faces(faces), width(width), height(height) {
        cam_f << fx, fy;
        cam_c << cx, cy;
        update();
    }

    void update() {
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

    template <class T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> render(
        std::function<bool(T&, Eigen::Ref<const Eigen::Matrix<float, 1, 3>>,
                           Eigen::Ref<const Eigen::Matrix<Index, 1, 3>>)>
            face_handler,
        T init_val) const {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            result(height, width);
        result.setConstant(init_val);
        maybe_parallel_for(
            [&](int i) {
                int r = i / width, c = i % width;
                Eigen::Matrix<float, 1, 2, Eigen::RowMajor> point;
                point << (float)c, (float)r;
                T& data = result.data()[i];
                auto check_face = [&](int faceid) -> bool {
                    const auto face = faces.row(faceid);
                    const Eigen::Matrix<float, 1, 3> bary = util::bary2d<float>(
                        point, verts_cam.row(face[0]), verts_cam.row(face[1]),
                        verts_cam.row(face[2]));
                    if (bary[0] >= 0.f && bary[1] >= 0.f && bary[2] >= 0.f)
                        return face_handler(data, bary, face);
                    return true;
                };
                Eigen::Matrix<float, 1, 2, Eigen::RowMajor> aabb_min, aabb_max;
                aabb_min.noalias() = point;
                aabb_max.noalias() = point;
                rtree.Search(aabb_min.data(), aabb_max.data(), check_face);
                if (data == std::numeric_limits<float>::max()) data = 0.0f;
            },
            width * height);
        return result;
    }

    FloatMatrix render_depth() const {
        return render<float>(
            [&](float& depth, Eigen::Ref<const Eigen::Matrix<float, 1, 3>> bary,
                Eigen::Ref<const Eigen::Matrix<Index, 1, 3>> face) -> bool {
                const float new_depth = verts(face[0], 2) * bary[0] +
                                        verts(face[1], 2) * bary[1] +
                                        verts(face[2], 2) * bary[2];
                depth = std::min(depth, new_depth);
                return true;
            },
            std::numeric_limits<float>::max());
    }

    ByteMatrix render_mask() const {
        return render<uint8_t>(
            [&](uint8_t& contained,
                Eigen::Ref<const Eigen::Matrix<float, 1, 3>> bary,
                Eigen::Ref<const Eigen::Matrix<Index, 1, 3>> face) -> bool {
                contained = 255;
                return false;
            },
            uint8_t(0));
    }

    IntMatrix render_vertex() const {
        return render<int>(
            [&](int& vertex, Eigen::Ref<const Eigen::Matrix<float, 1, 3>> bary,
                Eigen::Ref<const Eigen::Matrix<Index, 1, 3>> face) -> bool {
                for (int i = 0; i < 3; ++i) {
                    if (vertex == -1 || verts(face[i], 2) < verts(vertex, 2))
                        vertex = face[i];
                }
                return true;
            },
            -1);
    }

    // Input vertices
    Eigen::Ref<const Points> verts;
    // Input triangular faces
    Eigen::Ref<const Triangles> faces;

    // Vertices in camera coords
    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> verts_cam;

    // Image size
    int width, height;

    // Intrinsics
    Eigen::Matrix<float, 1, 2, Eigen::RowMajor> cam_f, cam_c;

   private:
    // Face R-Tree (aka AABB Tree)
    RTree<int, float, 2> rtree;
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
Renderer::render_depth() const {
    return p_impl->render_depth();
}

Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Renderer::render_mask() const {
    return p_impl->render_mask();
}

Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Renderer::render_vertex() const {
    return p_impl->render_vertex();
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

Renderer::~Renderer() = default;

}  // namespace sdf
