#include <iostream>
#include <windows.h>
#include <vector>

DWORD WINAPI ThreadFunc(LPVOID lpParam) {
    int* val = (int*)lpParam;
    *val = 42;
    return 0;
}

int main() {
    int val = 0;
    HANDLE hThread = CreateThread(NULL, 0, ThreadFunc, &val, 0, NULL);
    WaitForSingleObject(hThread, INFINITE);
    CloseHandle(hThread);
    std::cout << "Win32 thread test result: " << val << std::endl;
    return 0;
}
