// USAGE: ./sdf-demo BASIC_OBJ_FILE. Try the teapot.obj included in the project.
#include <sdf/sdf.hpp>
#include <meshview/meshview.hpp>
#include <meshview/meshview_imgui.hpp>

#include <iostream>
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Expect 1 argument: path to ./teapot.obj (or other "
                     "simple OBJ file)\n";
        return 0;
    }
    // Use mesh to load OBJ but don't actually show it
    meshview::Mesh dummy_mesh(argv[1]);
    if (dummy_mesh.verts_pos().rows() == 0) {
        std::cerr << "Failed to load " << argv[1] << "\n";
        return 1;
    }

    // Create meshview viewer
    meshview::Viewer viewer;

    // Create SDF instance from loaded mesh (robust mode)
    sdf::SDF sdf(dummy_mesh.verts_pos(), dummy_mesh.faces);

    // Cross section visualization parameters
    float csection_z = 0.0f;
    Eigen::Vector3f csection_axisangle;
    csection_axisangle.setZero();

    // Generate flat point cloud, for visualizing a cross-section of the SDF
    const int FLAT_CLOUD_DIM = 400;
    const float FLAT_CLOUD_RADIUS_X = 1.0, FLAT_CLOUD_RADIUS_Y = 1.4;
    const float FLAT_CLOUD_STEP_X = FLAT_CLOUD_RADIUS_X * 2 / FLAT_CLOUD_DIM;
    const float FLAT_CLOUD_STEP_Y = FLAT_CLOUD_RADIUS_Y * 2 / FLAT_CLOUD_DIM;
    Points _pts_flat(FLAT_CLOUD_DIM * FLAT_CLOUD_DIM, 3);
    for (int i = 0; i < FLAT_CLOUD_DIM; ++i) {
        float y = -FLAT_CLOUD_RADIUS_Y + FLAT_CLOUD_STEP_Y * i;
        for (int j = 0; j < FLAT_CLOUD_DIM; ++j) {
            float x = -FLAT_CLOUD_RADIUS_X + FLAT_CLOUD_STEP_X * j;
            _pts_flat.row(i * FLAT_CLOUD_DIM + j) << x, y, 0.f;
        }
    }

    // Add planar cross section point cloud
    auto& flat_cloud = viewer.add_point_cloud(_pts_flat, 0.0, 1.0, 0.0);

    // Add mesh as point cloud
    auto& mesh_cloud =
        viewer.add_point_cloud(dummy_mesh.verts_pos(), 1.0, 1.0, 1.0);

    // Make a backup of verts to allow transformations
    Points verts_initial = flat_cloud.verts_pos();
    auto verts = flat_cloud.verts_pos();

    const float MAX_DISTANCE_FUNC = 0.0075f;
    bool updated = false;

    // Update the cross section point cloud
    auto update = [&]() {
        const float norm = csection_axisangle.norm();
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotation;
        if (norm < 1e-5f) {
            rotation.setIdentity();
        } else {
            rotation.noalias() =
                Eigen::AngleAxisf(norm, csection_axisangle / norm)
                    .toRotationMatrix();
        }
        verts.noalias() = verts_initial;
        verts.rightCols<1>().setConstant(csection_z);
        verts *= rotation.transpose();
        _BEGIN_PROFILE;
        Eigen::VectorXf verts_sdf = sdf(verts);
        _PROFILE(compute SDF);
        for (size_t i = 0; i < verts.rows(); ++i) {
            float t =
                1.f - std::min(std::abs(verts_sdf[i]), MAX_DISTANCE_FUNC) *
                          (1.f / MAX_DISTANCE_FUNC);

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
        if (ImGui::SliderFloat3("cross sec rot##slideflatz",
                                csection_axisangle.data(), -3.14f, 3.14f)) {
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
