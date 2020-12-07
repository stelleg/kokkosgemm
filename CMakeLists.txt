
find_package(Kokkos REQUIRED)

# This is a tutorial, not a test, so we don't ask CTest to run it.
add_executable(gemm gemm.cpp)
target_link_libraries(gemm Kokkos::kokkos)

