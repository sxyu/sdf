// Python sdf bindings
#include <sdf/sdf.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <iostream>

namespace py = pybind11;
using namespace sdf;

PYBIND11_MODULE(sdf, m) {
    m.doc() =
        R"pbdoc(SDF: Convert triangle mesh to continuous signed distance function)pbdoc";
    py::class_<SDF>(m, "SDF")
        .def(py::init<Eigen::Ref<const Points>, Eigen::Ref<const Triangles>,
                      bool>(),
             py::arg("verts"), py::arg("faces"), py::arg("robust") = true)
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
        .def("face_areas", &SDF::face_areas,
             "ADVANCED: Get vector of face areas (n_faces)")
        .def("face_normals", &SDF::face_normals,
             "ADVANCED: Get matrix of face normals (n_faces, 3)")
        .def("face_normals", &SDF::face_points,
             "ADVANCED: Get matrix points for a face (3,3). Each row is a "
             "point.")
        .def("aabb", &SDF::aabb, "ADVANCED: Get AABB of entire mesh.")
        .def("faces", &SDF::faces, "Mesh faces passed to SDF constructor")
        .def("verts", &SDF::verts, "Mesh verticess passed to SDF constructor")
        .def_property_readonly("robust",
                               [](const SDF& sdf) { return sdf.robust; })
        .def("__repr__", [](const SDF& sdf) {
            return "<sdf.SDF: " + std::to_string(sdf.verts().rows()) +
                   " verts, " + std::to_string(sdf.faces().rows()) + " faces" +
                   (sdf.robust ? ", robust" : "") + ">";
        });
    py::module m_util = m.def_submodule("util");
    m_util
        .def(
            "bary",
            [](const Eigen::Ref<
                   const Eigen::Matrix<float, 1, 3, Eigen::RowMajor>>& p,
               const Eigen::Ref<
                   const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>& tri) {
                auto normal = util::normal<float>(tri);
                float area = normal.norm();
                normal /= area;
                return util::bary<float>(p, tri, normal, area);
            },
            "3D point to barycentric")
        .def("normal", &util::normal<float>,
             "Triangle normal (each row is a point)")
        .def("dist_point2line", &util::dist_point2line<float>,
             "Compute 3d point-line squard distance")
        .def("dist_point2lineseg", &util::dist_point2lineseg<float>,
             "Compute 3d point-line segment squared distance")
        .def(
            "dist_point2tri",
            [](const Eigen::Ref<
                   const Eigen::Matrix<float, 1, 3, Eigen::RowMajor>>& p,
               const Eigen::Ref<
                   const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>& tri) {
                auto normal = util::normal<float>(tri);
                float area = normal.norm();
                normal /= area;
                return util::dist_point2tri<float>(p, tri, normal, area);
            },
            "Compute 3d point-triangle squared distance")
        .def("point_in_tri_2d", &util::point_in_tri_2d<float>,
             "Test if point is in triangle (2d)");
}
