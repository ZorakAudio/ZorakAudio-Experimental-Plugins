#include "JitPlatform.h"
#include <stdexcept>
#if JUCE_WINDOWS
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace jit_platform {
using namespace juce;
namespace {
[[noreturn]] void fail(const String& message) { throw std::runtime_error(message.toStdString()); }
}
File findRuntime(const void* address) {
#if JUCE_WINDOWS
    HMODULE module = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                      reinterpret_cast<LPCWSTR>(address), &module);
    wchar_t path[32768] {};
    if (!GetModuleFileNameW(module, path, 32768)) fail("Cannot locate the JIT plugin module");
    File folder = File(String(path)).getParentDirectory();
#else
    Dl_info info {};
    if (!dladdr(address, &info) || !info.dli_fname) fail("Cannot locate the JIT plugin module");
    File folder = File(String::fromUTF8(info.dli_fname)).getParentDirectory();
#endif
    auto sibling = folder.getChildFile("JITEditor.runtime");
    if (sibling.isDirectory()) return sibling;
    return folder.getParentDirectory().getChildFile("Resources/JITEditor.runtime");
}
void* symbol(Module module, const char* name) {
#if JUCE_WINDOWS
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name));
#else
    return dlsym(module, name);
#endif
}
Module loadLLVM(const File& runtime) {
#if JUCE_WINDOWS
    auto file = runtime.getChildFile("python/Lib/site-packages/llvmlite/binding/llvmlite.dll");
    auto module = LoadLibraryExW(file.getFullPathName().toWideCharPointer(), nullptr,
                                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!module) fail("Cannot load bundled LLVM DLL: " + String(static_cast<int>(GetLastError())));
    HMODULE pinned = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                      reinterpret_cast<LPCWSTR>(GetProcAddress(module, "LLVMPY_CreateLLJITCompiler")), &pinned);
    return module;
#else
    auto file = runtime.getChildFile("python/site-packages/llvmlite/binding/libllvmlite.so");
    auto module = dlopen(file.getFullPathName().toRawUTF8(), RTLD_NOW | RTLD_LOCAL);
    if (!module) fail("Cannot load bundled LLVM: " + String::fromUTF8(dlerror()));
    // Keep one loader reference for the host process, matching Windows pinning.
    // Other instances and retired native code may still depend on this library.
    return module;
#endif
}
String runHelper(const File& runtime, const File& job, const std::function<bool()>& cancelled) {
#if JUCE_WINDOWS
    auto python = runtime.getChildFile("python/python.exe");
#else
    auto python = runtime.getChildFile("python/bin/python3");
#endif
    auto script = runtime.getChildFile("compiler/compiler_worker.py");
    if (!python.existsAsFile() || !script.existsAsFile()) fail("Bundled compiler is missing beside this plugin");
#if JUCE_WINDOWS
    String command = "\"" + python.getFullPathName() + "\" -I \"" + script.getFullPathName() + "\" \"" + job.getFullPathName() + "\"";
    HANDLE task = CreateJobObjectW(nullptr, nullptr);
    if (!task) fail("Cannot create compiler job");
    struct JobGuard { HANDLE handle; ~JobGuard() { CloseHandle(handle); } } jobGuard {task};
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limit {};
    limit.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(task, JobObjectExtendedLimitInformation, &limit, sizeof(limit))) fail("Cannot set compiler job limits");
    STARTUPINFOW startup {}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION child {};
    std::wstring writable(command.toWideCharPointer());
    if (!CreateProcessW(python.getFullPathName().toWideCharPointer(), writable.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, runtime.getFullPathName().toWideCharPointer(), &startup, &child))
        fail("Cannot launch bundled Python: " + String(static_cast<int>(GetLastError())));
    struct ProcessGuard { PROCESS_INFORMATION value; ~ProcessGuard() { CloseHandle(value.hThread); CloseHandle(value.hProcess); } } processGuard {child};
    if (!AssignProcessToJobObject(task, child.hProcess)) { TerminateProcess(child.hProcess, 1); fail("Cannot contain compiler helper in its job"); }
    ResumeThread(child.hThread);
    while (WaitForSingleObject(child.hProcess, 25) == WAIT_TIMEOUT)
        if (cancelled()) fail("Compilation cancelled by a newer request");
    DWORD code = 1;
    GetExitCodeProcess(child.hProcess, &code);
#else
    auto launcher = runtime.getChildFile("compiler/za_compiler_launcher");
    if (!launcher.existsAsFile()) fail("Bundled compiler process supervisor is missing");
    std::vector<std::string> arguments {launcher.getFullPathName().toStdString(), std::to_string(getpid()),
        python.getFullPathName().toStdString(), "-I", script.getFullPathName().toStdString(), job.getFullPathName().toStdString()};
    std::vector<char*> argv;
    for (auto& arg : arguments) argv.push_back(arg.data());
    argv.push_back(nullptr);
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    struct AttributeGuard { posix_spawnattr_t* value; ~AttributeGuard() { posix_spawnattr_destroy(value); } } attributeGuard {&attributes};
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attributes, 0);
    pid_t child = 0;
    int error = posix_spawn(&child, argv[0], nullptr, &attributes, argv.data(), environ);
    if (error) fail("Cannot launch bundled compiler: " + String::fromUTF8(strerror(error)));
    struct ProcessGuard {
        pid_t pid; bool reaped = false;
        ~ProcessGuard() { kill(-pid, SIGKILL); if (!reaped) { int status; while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {} } }
    } processGuard {child};
    int status = 0;
    for (;;) {
        auto result = waitpid(child, &status, WNOHANG);
        if (result == child) { processGuard.reaped = true; break; }
        if (result < 0 && errno != EINTR) fail("Cannot wait for compiler helper");
        if (cancelled()) fail("Compilation cancelled by a newer request");
        Thread::sleep(25);
    }
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
    if (code != 0) {
        auto error = job.getChildFile("error.txt").loadFileAsString();
        fail(error.isNotEmpty() ? error : "Compiler helper failed (exit " + String(static_cast<int64>(code)) + ")");
    }
    return job.getChildFile("program.ll").loadFileAsString();
}
}
