// sdf: Triangle mesh to signed-distance function (SDF) library
// Copyright Alex Yu 2020
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once
#ifndef SDF_SDF_F15B6437_01FD_4DBE_AB0D_BC1EE8ACC4C4
#define SDF_SDF_F15B6437_01FD_4DBE_AB0D_BC1EE8ACC4C4

#include <Eigen/Core>
#include <cstdint>

#include <memory>
#include <vector>
#ifdef __GNUC__
#include <experimental/propagate_const>
#endif
#include <Eigen/Geometry>

namespace sdf {

using Index = uint32_t;
using Points = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Triangles = Eigen::Matrix<Index, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Matrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

namespace util {

template <class T>
// 3D point to line shortest distance SQUARED
T dist_point2line(
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& p,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& a,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& b) {
    Eigen::Matrix<T, 1, 3> ap = p - a, ab = b - a;
    return (ap - (ap.dot(ab) / ab.squaredNorm()) * ab).squaredNorm();
}

// 3D point to line segment shortest distance SQUARED
template <class T>
T dist_point2lineseg(
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& p,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& a,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& b) {
    Eigen::Matrix<T, 1, 3> ap = p - a, ab = b - a;
    T t = ap.dot(ab) / ab.squaredNorm();
    t = std::max(T(0.0), std::min(T(1.0), t));
    return (ap - t * ab).squaredNorm();
}

template <class T>
const Eigen::Matrix<T, 1, 3> normal(
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& a,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& b,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& c) {
    return (b - a).cross(c - a);
}

template <class T>
// Find barycentric coords of p in triangle (a,b,c) in 3D
// (p does NOT have to be projected to plane beforehand)
// normal, area_abc to be computed using util::normal,
// where normal is normalized vector, area is magnitude
Eigen::Matrix<T, 1, 3> bary(
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& p,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& a,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& b,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& c,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& normal,
    float area_abc) {
    float area_pbc = normal.dot((b - p).cross(c - p));
    float area_pca = normal.dot((c - p).cross(a - p));

    Eigen::Matrix<T, 1, 3> uvw;
    uvw.x() = area_pbc / area_abc;
    uvw.y() = area_pca / area_abc;
    uvw.z() = T(1.0) - uvw.x() - uvw.y();

    return uvw;
}

template <class T>
// 3D point to triangle shortest distance SQUARED
// normal, area_abc to be computed using util::normal,
// where normal is normalized vector, area is magnitude
T dist_point2tri(
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& p,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& a,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& b,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& c,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 3, Eigen::RowMajor>>& normal,
    float area) {
    const Eigen::Matrix<T, 1, 3> uvw = bary<T>(p, a, b, c, normal, area);
    if (uvw[0] < 0) {
        return dist_point2lineseg<T>(p, b, c);
    } else if (uvw[1] < 0) {
        return dist_point2lineseg<T>(p, a, c);
    } else if (uvw[2] < 0) {
        return dist_point2lineseg<T>(p, a, b);
    } else {
        return (uvw[0] * a + uvw[1] * b + uvw[2] * c - p).squaredNorm();
    }
}

template <class T>
// Returns true if 2D point is in 2D triangle.
bool point_in_tri_2d(
    const Eigen::Ref<const Eigen::Matrix<T, 1, 2, Eigen::RowMajor>>& p,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 2, Eigen::RowMajor>>& a,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 2, Eigen::RowMajor>>& b,
    const Eigen::Ref<const Eigen::Matrix<T, 1, 2, Eigen::RowMajor>>& c) {
    Eigen::Matrix<T, 1, 2, Eigen::RowMajor> v0 = b - a, v1 = c - a, v2 = p - a;
    const float invden = 1.f / (v0.x() * v1.y() - v1.x() * v0.y());
    const float v = (v2.x() * v1.y() - v1.x() * v2.y()) * invden;
    const float w = (v0.x() * v2.y() - v2.x() * v0.y()) * invden;
    const float u = 1.0f - v - w;
    return v >= 0.f && w >= 0.f && u >= 0.f;
}

}  // namespace util

// Signed distance function utility for watertight meshes.
//
// Basic usage: SDF sdf(verts, faces); Vector sdf_vals = sdf(query_points);
// Get nearest neighbor (verts) indices: sdf.nn(query_points);
// Check containment (returns bool): sdf.contains(query_points);
struct SDF {
    // Construct SDF instance from triangle mesh with given vertices and faces
    // Note: Mesh is assumed to be watertight and all vertex positions are
    // expected to be free of nans/infs
    //
    // Basic usage: SDF sdf(verts, faces); Vector sdf_vals = sdf(query_points);
    // Get nearest neighbor (verts) indices: sdf.nn(query_points);
    // Check containment (returns bool): sdf.contains(query_points);
    //
    // @param verts mesh vertices. If the contents of this matrix are modified,
    // please call sdf.update() to update the internal representation.
    // Else the results will be incorrect.
    // @param faces mesh faces. The contents of this matrix should not be
    // modified for the lifetime of this instance.
    // @param robust whether to use robust mode. In robust mode,
    // @param copy whether to make a copy of the data instead of referencing it
    // SDF/containment computation is robust to mesh self-intersections and
    // facewinding but is slower.
    SDF(Eigen::Ref<const Points> verts, Eigen::Ref<const Triangles> faces,
        bool robust = true, bool copy = false);
    ~SDF();

    /*** PRIMARY INTERFACE ***/
    // Fast approximate signed-distance function.
    // Points inside the mesh have positive sign and outside have negative sign.
    //
    // Method: computes minimum distance to a triangular face incident to
    // the nearest vertex for each point.
    //
    // @param points input points
    // @param trunc_aabb if true, returns -FLT_MAX for all points outside mesh's
    // bounding box
    // @return approx SDF values at input points
    //
    // WARNING: if robust=false (from constructor), this WILL FAIL if the mesh
    // has self-intersections. In particular, the signs of points inside the
    // mesh may be flipped.
    Vector operator()(Eigen::Ref<const Points> points, bool trunc_aabb = false);

    // Return exact nearest neighbor vertex index for each point (index as in
    // input verts)
    Eigen::VectorXi nn(Eigen::Ref<const Points> points);

    // Return 1 for each point inside/on surface of the mesh and 0 for outside.
    //
    // @param points input points
    // @return indicator of whether each point is in OR on surface of mesh
    //
    // WARNING: if robust=false (from constructor), this WILL FAIL if the mesh
    // has self-intersections.
    Eigen::Matrix<bool, Eigen::Dynamic, 1> contains(
        Eigen::Ref<const Points> points);

    // Call if vertex positions have been updated to rebuild the KD tree
    // and update face normals+areas
    void update();

    /*** MISC UTILITIES ***/
    // Sample 'num_points' points uniformly on surface, output (num_points, 3).
    // Note: this takes O(num_points * log(num_faces)) time.
    Points sample_surface(int num_points) const;

    /*** DATA ACCESSORS ***/
    // Get adjacent faces of point at verts[pointid]
    const std::vector<int>& adj_faces(int pointid) const;

    // Get total surface area of mesh
    const float surface_area() const;

    // Get vector of face areas, shape (num_faces)
    const Vector& face_areas() const;

    // Get matrix of face normals, shape (num_faces, 3).
    // normal of face i (from faces passed to constructor) is in row i
    const Points& face_normals() const;

    // Get AABB of entire mesh, shape (6).
    // (minx, miny, minz, maxx, maxy, maxz)
    Eigen::Ref<const Eigen::Matrix<float, 6, 1>> aabb() const;

    // Get faces
    Eigen::Ref<const Triangles> faces() const;
    Eigen::Ref<Triangles> faces_mutable();

    // Get verts
    Eigen::Ref<const Points> verts() const;
    Eigen::Ref<Points> verts_mutable();

    // Whether SDF is in robust mode
    const bool robust;

    // Whether we own data
    const bool own_data;

   private:
    // Optional owned data
    Points owned_verts;
    Triangles owned_faces;

    struct Impl;
#ifdef __GNUC__
    std::experimental::propagate_const<std::unique_ptr<Impl>> p_impl;
#else
    std::unique_ptr<Impl> p_impl;
#endif
};

}  // namespace sdf

#endif  // ifndef SDF_SDF_F15B6437_01FD_4DBE_AB0D_BC1EE8ACC4C4
