#include "sdf/internal/sdf_util.hpp"
#include <vector>
#include <atomic>

namespace sdf {

void maybe_parallel_for(std::function<void(int&)> loop_content, int loop_max,
                        int num_threads) {
    std::atomic<int> counter(-1);
    auto worker = [&]() {
        while (true) {
            int i = ++counter;
            if (i >= loop_max) break;
            loop_content(i);
        }
    };
    if (loop_max >= MULTITHREAD_MIN_ITEMS) {
        std::vector<std::thread> threads;
        for (size_t i = 1; i < num_threads; ++i) {
            threads.emplace_back(worker);
        }
        worker();
        for (auto& thd : threads) {
            thd.join();
        }
    } else {
        worker();
    }
}

// Get a seeded mersenne twister 19937
std::mt19937& get_rng() {
    // Safer seeding with time (random_device can be not availble)
    thread_local std::mt19937 rg{
        std::random_device{}() ^
        static_cast<unsigned int>(std::chrono::high_resolution_clock::now()
                                      .time_since_epoch()
                                      .count())};
    return rg;
}

}  // namespace sdf
