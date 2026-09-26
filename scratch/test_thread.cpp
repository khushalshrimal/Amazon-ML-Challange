#include <iostream>
#include <thread>
#include <vector>
#include <atomic>

int main() {
    std::atomic<int> counter(0);
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&counter]() {
            for (int j = 0; j < 1000; ++j) counter++;
        });
    }
    for (auto& t : threads) t.join();
    std::cout << "Thread test counter: " << counter << std::endl;
    return 0;
}
