# Triangle mesh to signed-distance function (SDF)

Given a triangle mesh and a set of points, this library supports:
1. Computing SDF of mesh at each point: `sdf(points)`
2. Computing whether each point is inside mesh: `sdf.contains(points)`
3. Computing nearest neighbor vertex index for each point: `sdf.nn(points)`
All operations are CPU-only and parallelized.

## Quickstart

**Install python binding**: `pip install pysdf`
Usage example:
```python
from pysdf import SDF

# Load some mesh (don't necessarily need trimesh)
import trimesh
o = trimesh.load('some.obj')
f = SDF(o.vertices, o.faces); # (num_vertices, 3) and (num_faces, 3)

# Compute some SDF values (negative outside);
# takes a (num_points, 3) array, converts automatically
origin_sdf = f([0, 0, 0])
sdf_multi_point = f([[0, 0, 0],[1,1,1],[0.1,0.2,0.2]])

# Contains check
origin_contained = f.contains([0, 0, 0])

# Misc: nearest neighbor
origin_nn = f.nn([0, 0, 0])

# Misc: uniform surface point sampling
random_surface_points = f.sample_surface(10000)

# Misc: surface area
the_surface_area = f.surface_area
```
To modify the vertices/faces, you can use
`f.vertices_mutable` and `f.faces_mutable`.

## Screenshots

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

## C++ Library Usage
```cpp
sdf::SDF sdf(verts, faces); // verts (n, 3) float, faces (m, 3) float

// SDF. points (k, 3) float; return (k) float
Eigen::Vector3f sdf_at_points = sdf(points);

// Containment test, equal (but maybe faster) than sdf >= 0.
// points (k, 3) float; return (k) bool
Eigen::Matrix<bool, -1, 1> contains_points = sdf.contains(points);

// Miscellaneous: nearest neighbor. points (k, 3) float; return (k) int
Eigen::VectorXi nn_verts_idxs = sdf.nn(points);

// Miscellaneous: uniformly random points on surface (generates 10000 in this case)
Eigen::Matrix<float, -1, 3> random_surface_points = sdf.sample_surface(10000);

// Surface area
float surface_area = sdf.surface_area
```
Note SDF is > 0 inside and < 0 outside mesh.

## Use in CMake project
- `find_package(sdf)`
- After defining a target: `target_link_libraries(your_target sdf::sdf)`

### Robust mode
By default 'robust' mode is used. `sdf::SDF sdf(verts, faces, false)` to disable.
The SDF computation will be slightly faster but may be *incorrect* if the mesh has self-intersections or incorrect winding (not CCW) on some faces.

## Python
See the quickstart section for a usage example.

### Copying
By default, `SDF(verts, faces)` will copy the vertices/faces to ensure memory safety,
especially since the arguments may be automatically converted.
Use `SDF(verts, faces, copy=False)` to prevent this, if you are sure verts/faces are
of types np.float32/np.uint32 respectively and will not be destroyed before the SDF instance.

## Warning about reliability
In robust mode (default) we use raytracing (parity count) to check containment.
Currently the ray tracing has the same limitation as embree,
that is when ray exactly hits an edge the intersection gets double counted, inverting the sign
of the distance function. 
This is theoretically unlikely for random points but can occur either due to floating point error or if points and mesh vertices are both taken from a grid.
In practice, we (1) randomize the ray tracing direction and (2) trace 3 rays along different axes and take majority to decrease the likelihood of this occurring.

In non-robust mode we use nearest surface normal to check containment.
 The contains check (and SDF sign) will be wrong under self-intersection or if normals are incorrectly oriented.

## Dependencies
- Eigen 3 (Python: automatically downloaded if needed)
- Vendored (no installation needed):
    - nanoflann
    - nushoin/RTree
- Optional:
    - meshview https://github.com/sxyu/meshview for demo

## Build + Install
`mkdir build && cmake .. && make -j4 && sudo make install`

## Demo
A demo program can optionally be built, if meshview is installed.
To use it, run `./sdf-demo BASIC_OBJ_FILE`. Try the `sample-obj/*.obj` included in the project.

## Benchmark

All benchmarks are ran on a 6-core CPU (Intel i7 8th generation, high-performance laptop)

| Model vertices | trimesh contains eval/s (numpy) | trimesh contains eval/s (pyembree) | our SDF evals / s (robust) | our SDF evals / s (non-robust)  |
| -------------   | ------------- | ------------- | ------------- | ------------- |
| 3241            | 29,555| 237,855 | 5,077,725   | 8,187,117 |
| 49246           | 6,835 | 62,058  | 2,971,137   | 4,407,045 |
| 179282          | 1,301 | 20,157  | 1,672,859   | 1,987,869 |

## License
BSD 2-clause

This library relies on the Eigen (MPL2) nanoflann (BSD) and RTree (MIT) libraries.
