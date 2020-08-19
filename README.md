# Triangle mesh to signed-distance function (SDF)

Given a triangle mesh and a set of points, this library supports:
1. Computing SDF of mesh at each point: `sdf(points)`
2. Computing whether each point is inside mesh: `sdf.contains(points)`
3. Computing nearest neighbor vertex index for each point: `sdf.nn(points)`
All operations are CPU-only and parallelized.

<img src="https://github.com/sxyu/sdf/blob/master/readme-img/human.gif"
    width="400">

Robustness under self-intersections:
<img src="https://github.com/sxyu/sdf/blob/master/readme-img/smpl.png"
    width="400">

Reasonable result for non-watertight mesh with multiple parts:
<img src="https://github.com/sxyu/sdf/blob/master/readme-img/teapot.gif"
    width="400">

Reasonable result for voxel grid
<img src="https://github.com/sxyu/sdf/blob/master/readme-img/voxel.png"
    width="400">

## Usage
```cpp
sdf::SDF sdf(verts, faces); // verts (n, 3) float, faces (m, 3) float

// SDF. points (k, 3) float; return (k) float
Eigen::Vector3f sdf_at_points = sdf(points);

// Containment test, equal (but maybe faster) than sdf >= 0.
// points (k, 3) float; return (k) bool
Eigen::Matrix<bool, -1, 1> contains_points = sdf.contains(points);

// Miscellaneous: nearest neighbor. points (k, 3) float; return (k) int
Eigen::VectorXi nn_verts_idxs = sdf.nn(points);
```
Note SDF is > 0 inside and < 0 outside mesh.

### Robust mode
By default 'robust' mode is used. `sdf::SDF sdf(verts, faces, false)` to disable.
The SDF computation will be slightly faster but may be *incorrect* if the mesh has self-intersections or incorrect winding (not CCW) on some faces.

#### Warning
Robust mode uses raytracing. Currently the ray tracing has the same limitation as embree,
that is when ray exactly hits an edge the intersection gets double counted, inverting the sign
of the distance function. 
This is unlikely for random points but can frequently occur especially if points and mesh vertices are both taken from a grid.
In practice, we trace 3 rays and take the majority sign to decrease the likelihood of this occurring.

### Python
To install Python binding, use `pip install .`
You may need to first install pybind11 from https://github.com/pybind/pybind11.
Usage example:
```python
from sdf import SDF
import trimesh
o = trimesh.load('some.obj')
f = SDF(o.vertices, o.faces);
origin_sdf = f([0, 0, 0])
origin_contained = f.contains([0, 0, 0])
origin_nn = f.nn([0, 0, 0])
other_sdf = f([[0, 0, 0],[1,1,1],[0.1,0.2,0.2]])
```
To modify the vertices/faces, you can use
`f.vertices_mutable` and `f.faces_mutable`.

#### Copying
By default, `SDF(verts, faces)` will copy the vertices/faces to ensure memory safety,
especially since the arguments may be automatically converted.
Use `SDF(verts, faces, copy=False)` to prevent this, if you are sure verts/faces are
of types np.float32/np.uint32 respectively and will not be destroyed before the SDF instance.

## Dependencies
- Eigen 3
- Vendored (no installation needed):
    - nanoflann
    - nushoin/RTree
- Optional:
    - meshview https://github.com/sxyu/meshview for demo

## Build + Install
`mkdir build && cmake .. && make -j4 && sudo make install`

## Demo
A demo program can optionally be build if meshview is installed.
To use it, run `./sdf-demo BASIC_OBJ_FILE`. Try the teapot.obj included in the project.
df::SDF sdf(verts, faces)

## Use in CMake project
- `find_package(sdf)`
- After defining a target: `target_link_libraries(your_target sdf::sdf)`

## Benchmark

All benchmarks are ran on a 6-core CPU (Intel i7 8th generation, high-performance laptop)

| Model vertices | trimesh contains eval/s (numpy) | trimesh contains eval/s (pyembree) | our SDF evals / s (robust) | our SDF evals / s (non-robust)  |
| -------------   | ------------- | ------------- | ------------- | ------------- |
| 3241            | 29,555| 237,855 | 5,077,725   | 8,187,117 |
| 49246           | 6,835 | 62,058  | 2,971,137   | 4,407,045 |
| 179282          | 1,301 | 20,157  | 1,672,859   | 1,987,869 |

## License
Apache 2.0

This library relies on the nanoflann (BSD) and RTree (MIT) libraries.
