cmake_minimum_required(VERSION 3.3)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (POLICY CMP0069)
    cmake_policy(SET CMP0069 NEW)
endif (POLICY CMP0069)
if (POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif (POLICY CMP0072)

project( sdf )

set( CMAKE_CXX_STACK_SIZE "10000000" )
set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

set( INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" )
set( SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src" )

option( SDF_USE_SYSTEM_EIGEN
    "Use system Eigen. If false, please set EIGEN_INCLUDE_DIR manually" ON )
option( SDF_BUILD_DEMO "Build the demo program, if meshview available" ON )
option( SDF_BUILD_PYTHON "Build the python bindings, if pybind11 available" OFF )
option( SDF_BUILD_INSTALL "Build the install target" ON )

if ( CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang") )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations -O3 -funroll-loops -g" )
elseif( MSVC )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT /GLT /Ox")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT /GLT")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ( )

file(GLOB SDF_SOURCES ${SRC_DIR}/*.cpp)
set(SDF_PUBLIC_HEADERS ${INCLUDE_DIR}/sdf/sdf.hpp)

set( DEPENDENCIES )

find_package(Threads REQUIRED)

if (SDF_USE_SYSTEM_EIGEN)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    message(STATUS "Using system Eigen")
    set(DEPENDENCIES ${DEPENDENCIES} Eigen3::Eigen)
else()
    # Set EIGEN_INCLUDE_DIR manually
    include_directories(${EIGEN_INCLUDE_DIR})
endif()

# Optional sxyu/meshview
find_package( meshview )
if (NOT meshview_FOUND)
    message( WARNING "meshview not found, demo is disabled. Please install from https://github.com/sxyu/meshview if you want to run the demo (totally optional)" )
endif()

# Optional pybind11
find_package( pybind11 )
if (NOT pybind11_FOUND)
    message( WARNING "pybind11 not found, Python binding is disabled. Install from https://github.com/pybind/pybind11" )
endif()

add_library( sdf STATIC ${SDF_SOURCES} )

target_include_directories(
    sdf PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries( sdf PUBLIC
    ${DEPENDENCIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
)

if (${meshview_FOUND} AND ${SDF_BUILD_DEMO})
    add_executable(sdf-demo demo.cpp)
    target_link_libraries( sdf-demo sdf meshview::meshview )
endif ()

if (${pybind11_FOUND} AND ${SDF_BUILD_PYTHON})
    pybind11_add_module(pysdf SHARED ${SDF_SOURCES} pybind.cpp)
    set_target_properties(pysdf PROPERTIES OUTPUT_NAME pysdf)
    target_include_directories(
        pysdf PUBLIC
        $<BUILD_INTERFACE:${INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(pysdf PRIVATE ${DEPENDENCIES})
endif ()

if (SDF_BUILD_INSTALL)
    set_target_properties( sdf PROPERTIES
        PUBLIC_HEADER "${SDF_PUBLIC_HEADERS}"
    )
    include(GNUInstallDirs)
    install(TARGETS sdf 
            EXPORT sdfTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sdf
    )
    export(TARGETS sdf
           NAMESPACE sdf::
           FILE "${CMAKE_CURRENT_BINARY_DIR}/sdfConfig.cmake")
    install(EXPORT
                sdfTargets
                DESTINATION "${CMAKE_INSTALL_DATADIR}/sdf/cmake"
                NAMESPACE sdf::
            )
    install(FILES "${SRC_DIR}/cmake/sdfConfig.cmake"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/sdf/cmake")
endif()
