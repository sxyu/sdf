// Python sdf bindings
#include <sdf/sdf.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <thread>

namespace py = pybind11;
using namespace sdf;

PYBIND11_MODULE(pysdf, m) {
    m.doc() =
        R"pbdoc(SDF: Convert triangle mesh to continuous signed distance function)pbdoc";
    py::class_<SDF>(m, "SDF")
        .def(py::init<Eigen::Ref<const Points>, Eigen::Ref<const Triangles>,
                      bool, bool>(),
             py::arg("verts"), py::arg("faces"), py::arg("robust") = true,
             py::arg("copy") = true)
        .def("__call__", &SDF::operator(),
             "Compute SDF on points (positive iff inside)", py::arg("points"),
             py::arg("trunc_aabb") = false)
        .def("calc", &SDF::operator(),
             "Compute SDF on points (positive iff inside), alias of __call__",
             py::arg("points"), py::arg("trunc_aabb") = false)
        .def("contains", &SDF::contains, "Test if points are in mesh",
             py::arg("points"))
        .def("nn", &SDF::nn, "Find nearest neighbor indices", py::arg("points"))
        .def("update", &SDF::update,
             "Update the SDF to reflect any changes in verts")
        .def("sample_surface", &SDF::sample_surface,
             "Sample num_points uniformly random points on mesh surface. "
             "Internally, picks each triangle wp prop to area and then random "
             "point on triangle.",
             py::arg("num_points"))
        .def_property_readonly("surface_area", &SDF::surface_area,
                               "Get surface area of mesh")
        .def_property_readonly("face_areas", &SDF::face_areas,
                               "ADVANCED: Get vector of face areas (n_faces)")
        .def_property_readonly(
            "face_normals", &SDF::face_normals,
            "ADVANCED: Get matrix of face normals (n_faces, 3)")
        .def_property_readonly("aabb", &SDF::aabb,
                               "ADVANCED: Get AABB of entire mesh.")
        .def_property_readonly("faces", &SDF::faces,
                               "Mesh faces passed to SDF constructor")
        .def_property_readonly("verts", &SDF::verts,
                               "Mesh vertices passed to SDF constructor")
        .def_property(
            "faces_mutable", &SDF::faces_mutable,
            [](SDF& sdf, Eigen::Ref<const Triangles> val) {
                sdf.faces_mutable() = val;
            },
            "Mesh faces passed to SDF constructor")
        .def_property(
            "verts_mutable", &SDF::verts_mutable,
            [](SDF& sdf, Eigen::Ref<const Points> val) {
                sdf.verts_mutable() = val;
            },
            "Mesh vertices passed to SDF constructor")
        .def_property(
            "vertices_mutable", &SDF::verts_mutable,
            [](SDF& sdf, Eigen::Ref<const Points> val) {
                sdf.verts_mutable() = val;
            },
            "Mesh vertices passed to SDF constructor")
        .def_property_readonly("vertices", &SDF::verts,
                               "Mesh vertices passed to SDF constructor (alias "
                               "for verts for trimesh compatibility)")
        .def_readonly("robust", &SDF::robust, "Whether SDF is in robust mode")
        .def_readonly(
            "own_data", &SDF::own_data,
            "Whether SDF owns verts/faces data (constructed with copy=True), "
            "so you can change them with verts_mutable/faces_mutable")
        .def_property_readonly_static(
            "num_threads",
            [](const py::object& _) {
                return std::thread::hardware_concurrency();
            },
            "Max number of threads")
        .def("__repr__", [](const SDF& sdf) {
            return "<sdf.SDF(verts.shape=(" +
                   std::to_string(sdf.verts().rows()) + ", 3), faces.shape=(" +
                   std::to_string(sdf.faces().rows()) +
                   ", 3), robust=" + (sdf.robust ? "True" : "False") + ")>";
        });
    py::module m_util = m.def_submodule("util");
    using RefConstRowVec3f =
        const Eigen::Ref<const Eigen::Matrix<float, 1, 3, Eigen::RowMajor>>;
    m_util
        .def(
            "bary",
            [](RefConstRowVec3f& p, RefConstRowVec3f& a, RefConstRowVec3f& b,
               RefConstRowVec3f& c) {
                auto normal = util::normal<float>(a, b, c);
                float area = normal.norm();
                normal /= area;
                return util::bary<float>(p, a, b, c, normal, area);
            },
            "3D point to barycentric")
        .def("normal", &util::normal<float>,
             "Triangle normal (each row is a point)")
        .def("dist_point2line", &util::dist_point2line<float>,
             "Compute 3d point-line squared distance")
        .def("dist_point2lineseg", &util::dist_point2lineseg<float>,
             "Compute 3d point-line segment squared distance")
        .def(
            "dist_point2tri",
            [](RefConstRowVec3f& p, RefConstRowVec3f& a, RefConstRowVec3f& b,
               RefConstRowVec3f& c) {
                auto normal = util::normal<float>(a, b, c);
                float area = normal.norm();
                normal /= area;
                return util::dist_point2tri<float>(p, a, b, c, normal, area);
            },
            "Compute 3d point-triangle squared distance")
        .def("point_in_tri_2d", &util::point_in_tri_2d<float>,
             "Test if point is in triangle (2d)");
}
