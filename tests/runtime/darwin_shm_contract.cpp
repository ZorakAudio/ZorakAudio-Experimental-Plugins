// Linux-only syscall contract shim. This is not a macOS execution test.
// It rejects the two operations Darwin rejects while exercising the real
// mapping/IPC code, real shared memory and real cross-process file locks.
#if !defined(__linux__) || !defined(ZA_JSFX_TEST_DARWIN_SHM)
#error Darwin shared-memory contract shim requires the explicit Linux test mode
#endif
#include <cerrno>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <sys/stat.h>
#include <unistd.h>

extern "C" int __real_shm_open(const char*, int, mode_t);
extern "C" int __real_flock(int, int);
extern "C" int __real_close(int);
extern "C" int __real_ftruncate(int, off_t);
namespace {
std::mutex descriptorsMutex;
std::unordered_set<int> shmDescriptors;
}

extern "C" int __wrap_shm_open(const char* name, int flags, mode_t mode)
{
    if (std::strlen(name) > 31)
    {
        errno = ENAMETOOLONG;
        return -1;
    }
    const int fd = __real_shm_open(name, flags, mode);
    if (fd >= 0)
    {
        std::lock_guard<std::mutex> guard(descriptorsMutex);
        shmDescriptors.insert(fd);
    }
    return fd;
}

extern "C" int __wrap_flock(int fd, int operation)
{
    {
        std::lock_guard<std::mutex> guard(descriptorsMutex);
        if (shmDescriptors.contains(fd))
        {
            errno = ENOTSUP; // XNU flock requires a vnode; POSIX shm is not one.
            return -1;
        }
    }
    return __real_flock(fd, operation);
}

extern "C" int __wrap_close(int fd)
{
    // Hold across close so another thread cannot reopen this descriptor number
    // and then have its new classification erased by the old close.
    std::lock_guard<std::mutex> guard(descriptorsMutex);
    const int result = __real_close(fd);
    if (result == 0)
        shmDescriptors.erase(fd);
    return result;
}

extern "C" int __wrap_ftruncate(int fd, off_t length)
{
    bool isShm;
    {
        std::lock_guard<std::mutex> guard(descriptorsMutex);
        isShm = shmDescriptors.contains(fd);
    }
    if (!isShm)
        return __real_ftruncate(fd, length);
    struct stat info {};
    if (::fstat(fd, &info) != 0)
        return -1;
    if (info.st_size != 0)
    {
        errno = EINVAL; // Darwin permits a nonzero sizing operation only once.
        return -1;
    }
    const auto page = ::sysconf(_SC_PAGESIZE);
    return __real_ftruncate(fd, ((length + page - 1) / page) * page);
}
