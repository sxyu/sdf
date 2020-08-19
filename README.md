# Triangle mesh to signed-distance function (SDF)

Given a triangle mesh and a set of points, this library supports:
1. Computing SDF of mesh at each point: `sdf(points)`
2. Computing whether each point is inside mesh: `sdf.contains(points)`
3. Computing nearest neighbor vertex index for each point: `sdf.nn(points)`
All operations are CPU-only and parallelized.

![Screenshot-teapot](https://github.com/sxyu/sdf/blob/master/readme-img/teapot.gif?raw=true)
![Screenshot-human](https://github.com/sxyu/sdf/blob/master/readme-img/human.gif?raw=true)

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

| Model vertices  | SDF evals / s (robust) | SDF evals / s (non-robust) |
| ------------- | ------------- | ------------- |
| 3241            | 6172554.4     | 8187116.6
| 6890            | 6026525.4     | 8227800.2
| 49246           | 3786387.2     | 4407045.0
| 179282          | 1884640.5     | 1987869.3

## License
Apache 2.0

This library relies on the nanoflann (BSD) and RTree (MIT) libraries.
