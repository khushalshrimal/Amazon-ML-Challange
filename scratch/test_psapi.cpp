#include <iostream>
#include <windows.h>
#include <psapi.h>

int main() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        double ram_gb = (double)pmc.WorkingSetSize / (1024.0 * 1024.0 * 1024.0);
        std::cout << "RAM usage: " << ram_gb << " GB" << std::endl;
    }
    return 0;
}
