#include <functional>
#include <thread>
#include <random>

namespace sdf {

// Min number of items to allow multithreading
const int MULTITHREAD_MIN_ITEMS = 50;

// Parallel for
void maybe_parallel_for(std::function<void(int&)> loop_content,
                        int loop_max = MULTITHREAD_MIN_ITEMS,
                        int num_threads = std::thread::hardware_concurrency());

// Get a seeded mersenne twister 19937
std::mt19937& get_rng();

}  // namespace sdf
