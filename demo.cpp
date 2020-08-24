// USAGE: ./sdf-demo BASIC_OBJ_FILE (with only triangular faces supported1
// Try the OBJ files in sample-obj included in the project.
#include <sdf/sdf.hpp>
#include <meshview/meshview.hpp>
#include <meshview/meshview_imgui.hpp>

#include <iostream>
#include <random>
#include <Eigen/Geometry>

#include <chrono>
// Timing macro
#define _BEGIN_PROFILE auto start = std::chrono::high_resolution_clock::now()
#define _PROFILE(x)                                                            \
    do {                                                                       \
        double _delta = std::chrono::duration<double, std::milli>(             \
                            std::chrono::high_resolution_clock::now() - start) \
                            .count();                                          \
        printf("%s: %f ms = %f fps\n", #x, _delta, 1e3f / _delta);             \
        start = std::chrono::high_resolution_clock::now();                     \
    } while (false)

using namespace sdf;

namespace {

Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rodrigues(
    Eigen::Ref<const Eigen::Vector3f> axis_angle) {
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotation;
    const float norm = axis_angle.norm();
    if (norm < 1e-5f) {
        rotation.setIdentity();
    } else {
        rotation.noalias() =
            Eigen::AngleAxisf(norm, axis_angle / norm).toRotationMatrix();
    }
    return rotation;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Expect 1 argument: path to sample-obj/x.obj (or other "
                     "simple OBJ file)\n";
        return 0;
    }
    // Create meshview viewer
    meshview::Viewer viewer;
    viewer.wireframe = true;

    // Load obj
    meshview::Mesh& obj_mesh = viewer.add_mesh(argv[1]);
    if (obj_mesh.verts_pos().rows() == 0) {
        std::cerr << "Failed to load " << argv[1] << "\n";
        return 1;
    }
    Points mesh_verts_initial = obj_mesh.verts_pos();

    // Create SDF instance from loaded mesh (robust mode)
    sdf::SDF sdf(obj_mesh.verts_pos(), obj_mesh.faces);
    std::cout << sdf.verts().rows() << " verts\n";

    // Cross section visualization parameters
    float csection_z = 0.0f;
    Eigen::Vector3f csection_axisangle, model_axisangle;
    csection_axisangle.setZero();
    model_axisangle.setZero();

    // Generate flat point cloud, for visualizing a cross-section of the SDF
    const int FLAT_CLOUD_DIM = 400;
    const float FLAT_CLOUD_RADIUS =
        (sdf.aabb().tail<3>() - sdf.aabb().head<3>()).maxCoeff();
    const float FLAT_CLOUD_STEP = FLAT_CLOUD_RADIUS * 2 / FLAT_CLOUD_DIM;

    Points _pts_flat(FLAT_CLOUD_DIM * FLAT_CLOUD_DIM, 3);
    for (int i = 0; i < FLAT_CLOUD_DIM; ++i) {
        float y = -FLAT_CLOUD_RADIUS + FLAT_CLOUD_STEP * i;
        for (int j = 0; j < FLAT_CLOUD_DIM; ++j) {
            float x = -FLAT_CLOUD_RADIUS + FLAT_CLOUD_STEP * j;
            _pts_flat.row(i * FLAT_CLOUD_DIM + j) << x, y, 0.f;
        }
    }

    // Add planar cross section point cloud
    auto& flat_cloud = viewer.add_point_cloud(_pts_flat, 0.f, 1.f, 0.f);

    // Make a backup of verts to allow transformations
    Points csection_verts_initial = flat_cloud.verts_pos();
    auto csection_verts = flat_cloud.verts_pos();

    const float MAX_DISTANCE_FUNC = 0.09f;
    bool updated = false;

    // Color by containment only
    bool contains_only = false;

    std::vector<meshview::Mesh*> spheres;
    // This part visualizes the surface sampling
    // sdf::Points rand_pts = sdf.sample_surface(50);
    // for (int i = 0; i < rand_pts.rows(); ++i) {
    //     spheres.push_back(&viewer
    //                            .add_sphere(Eigen::Vector3f(0.f, 0.f, 0.f),
    //                                        0.02f,
    //                                        Eigen::Vector3f(1.f, 0.0f, 0.f))
    //                            .set_translation(rand_pts.row(i).transpose()));
    // }

    // Update the cross section point cloud
    auto update = [&](bool model_rot_updated = false) {
        const float norm = csection_axisangle.norm();
        if (model_rot_updated) {
            // Rotate mesh
            obj_mesh.verts_pos().noalias() =
                mesh_verts_initial * rodrigues(model_axisangle).transpose();
            sdf.update();

            // This part updates the surface sampling
            // rand_pts = sdf.sample_surface(spheres.size());
            // for (int i = 0; i < rand_pts.rows(); ++i) {
            //     spheres[i]->set_translation(rand_pts.row(i).transpose());
            // }
        }

        csection_verts.noalias() = csection_verts_initial;
        csection_verts.array().rightCols<1>() += csection_z;
        csection_verts *= rodrigues(csection_axisangle).transpose();
        _BEGIN_PROFILE;
        Eigen::VectorXf verts_sdf = sdf(csection_verts);
        _PROFILE(compute SDF);
        for (size_t i = 0; i < csection_verts.rows(); ++i) {
            float t = contains_only ? 1.0f
                                    : (1.f - std::min(std::abs(verts_sdf[i]),
                                                      MAX_DISTANCE_FUNC) *
                                                 (1.f / MAX_DISTANCE_FUNC));

            auto rgb = flat_cloud.verts_rgb().row(i);
            rgb[0] = (verts_sdf[i] < 0) ? 0.0f : 1.0f;
            rgb[1] = t;
            rgb[2] = t * 0.5;
        }

        // Update the mesh on-the-fly (send mesh to GPU)
        updated = true;
    };
    update();

    viewer.on_key = [&](int key, meshview::input::Action action,
                        int mods) -> bool {
        if (action != meshview::input::Action::release) {
            if (key == 'J' || key == 'K') {
                csection_z += (key == 'J') ? 0.01 : -0.01;
                update();
                flat_cloud.update();
            }

            if (action == meshview::input::Action::press) {
                if (key == 'M') {
                    obj_mesh.enabled = !obj_mesh.enabled;
                }
            }
        }
        return true;
    };

    viewer.on_open = []() { ImGui::GetIO().IniFilename = nullptr; };
    viewer.on_gui = [&]() {
        updated = false;
        // * GUI code
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(500, 360), ImGuiCond_Once);
        ImGui::Begin("Model and Cross Section", NULL);
        ImGui::TextUnformatted("Reset: ");
        ImGui::SameLine();
        if (ImGui::Button("Axis-angle##ResetCrossSectionAA")) {
            csection_axisangle.setZero();
            update();
        }
        ImGui::SameLine();
        if (ImGui::Button("Z##ResetCrossSectionZ")) {
            csection_z = 0.f;
            update();
        }

        if (ImGui::SliderFloat("cross sec z##slideflatz", &csection_z, -1.0f,
                               1.0f)) {
            update();
        }
        ImGui::TextUnformatted("Tip: press j,k to adjust cross section z");

        if (ImGui::SliderFloat3("cross sec rot##slideflatrot",
                                csection_axisangle.data(), -3.14f, 3.14f)) {
            update();
        }

        if (ImGui::SliderFloat3("model rot##slidemodelrot",
                                model_axisangle.data(), -3.14f, 3.14f)) {
            update(true);
        }

        ImGui::Checkbox("Show mesh", &obj_mesh.enabled);
        ImGui::Checkbox("Wireframe mesh", &viewer.wireframe);
        if (ImGui::Checkbox("Containment only", &contains_only)) {
            update();
        }

        ImGui::End();  // Model and Cross Section
        // Return true if updated to indicate mesh data has been changed
        // the viewer will update the GPU buffers automatically
        return updated;
    };
    viewer.show();
    return 0;
}
