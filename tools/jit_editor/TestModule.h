#pragma once
#include <string>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
inline std::string testArgument(int index, char**) {
    int count=0;auto** arguments=CommandLineToArgvW(GetCommandLineW(),&count);
    std::string result;
    if(arguments && index<count){
        const int size=WideCharToMultiByte(CP_UTF8,0,arguments[index],-1,nullptr,0,nullptr,nullptr);
        if(size>0){result.resize(size);WideCharToMultiByte(CP_UTF8,0,arguments[index],-1,result.data(),size,nullptr,nullptr);result.pop_back();}
    }
    if(arguments)LocalFree(arguments);
    return result;
}
inline void* loadTestModule(const char* path) {
    const int size=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,nullptr,0);
    if(!size)return nullptr;
    std::wstring wide(size,L'\0');MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,wide.data(),size);
    return LoadLibraryW(wide.c_str());
}
inline void* testSymbol(void* module, const char* name) { return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name)); }
inline void closeTestModule(void* module) { FreeLibrary(static_cast<HMODULE>(module)); }
inline void pumpTestMessages() { MSG m; while(PeekMessageW(&m,nullptr,0,0,PM_REMOVE)){TranslateMessage(&m);DispatchMessageW(&m);} }
#else
#include <dlfcn.h>
inline std::string testArgument(int index,char** argv) { return argv[index]; }
inline void* loadTestModule(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
inline void* testSymbol(void* module, const char* name) { return dlsym(module, name); }
inline void closeTestModule(void* module) { dlclose(module); }
inline void pumpTestMessages() {}
#endif
