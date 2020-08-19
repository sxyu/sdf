include(CMakeFindDependencyMacro)

find_dependency(Threads)
find_dependency(Eigen3)

include("${CMAKE_CURRENT_LIST_DIR}/sdfTargets.cmake")
