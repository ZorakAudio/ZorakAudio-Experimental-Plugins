#include "DspJsfxSharedMemory.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <utility>
#include <cerrno>
#include <chrono>
#include <limits>
#include <thread>

#if JUCE_WINDOWS || defined(_WIN32)
 #include <windows.h>
#else
 #include <fcntl.h>
 #include <sys/mman.h>
 #include <sys/file.h>
 #include <sys/stat.h>
 #include <unistd.h>
#endif

namespace za::jsfx
{

namespace
{
static std::string sanitizeStem(const std::string& stem)
{
    std::string out;
    out.reserve(stem.size() + 16);
    for (const char ch : stem)
    {
        const unsigned char uch = static_cast<unsigned char> (ch);
        if (std::isalnum(uch) != 0)
            out.push_back(static_cast<char> (std::tolower(uch)));
        else
            out.push_back('_');
    }
    if (out.empty())
        out = "default";
    return out;
}

#if defined(__APPLE__) || defined(ZA_JSFX_TEST_DARWIN_SHM)
static std::string compactDarwinName(const std::string& fullName)
{
    // XNU accepts at most 31 bytes, INCLUDING the initial slash. Hash the
    // complete name rather than truncating domain/namespace suffixes. A
    // separate prefix avoids aliasing an ordinary short sanitized stem.
    if (fullName.size() <= 31)
        return fullName;
    std::uint64_t hash = 14695981039346656037ull;
    for (const unsigned char ch : fullName)
    {
        hash ^= ch;
        hash *= 1099511628211ull;
    }
    char compact[32] {};
    std::snprintf(compact, sizeof(compact), "/za_jsfxh_%016llx",
                  static_cast<unsigned long long>(hash));
    return compact;
}
#endif
} // namespace

std::string makeSharedMemoryObjectName(const std::string& stem)
{
   #if JUCE_WINDOWS || defined(_WIN32)
    return std::string("Local\\za_jsfx_") + sanitizeStem(stem);
   #else
    auto name = std::string("/za_jsfx_") + sanitizeStem(stem);
    #if defined(__APPLE__) || defined(ZA_JSFX_TEST_DARWIN_SHM)
    name = compactDarwinName(name);
    #endif
    return name;
   #endif
}

DspJsfxSharedMemorySegment::~DspJsfxSharedMemorySegment()
{
    close();
}

DspJsfxSharedMemorySegment::DspJsfxSharedMemorySegment(DspJsfxSharedMemorySegment&& other) noexcept
{
    *this = std::move(other);
}

DspJsfxSharedMemorySegment& DspJsfxSharedMemorySegment::operator=(DspJsfxSharedMemorySegment&& other) noexcept
{
    if (this == &other)
        return *this;

    close();
    base_ = other.base_;
    sizeBytes_ = other.sizeBytes_;
    objectName_ = std::move(other.objectName_);
    initializationLocked_ = other.initializationLocked_;
    other.initializationLocked_ = false;
   #if JUCE_WINDOWS || defined(_WIN32)
    initializationMutex_ = other.initializationMutex_;
    other.initializationMutex_ = nullptr;
    handle_ = other.handle_;
    other.handle_ = nullptr;
   #else
    fd_ = other.fd_;
    other.fd_ = -1;
    #if defined(__APPLE__) || defined(ZA_JSFX_TEST_DARWIN_SHM)
    initializationLockFd_ = other.initializationLockFd_;
    other.initializationLockFd_ = -1;
    #endif
   #endif
    other.base_ = nullptr;
    other.sizeBytes_ = 0;
    return *this;
}

bool DspJsfxSharedMemorySegment::openOrCreate(const std::string& objectName, std::size_t requestedBytes, bool* created)
{
    close();
    if (created != nullptr)
        *created = false;
    if (requestedBytes == 0)
        return false;
    objectName_ = makeSharedMemoryObjectName(objectName);

   #if JUCE_WINDOWS || defined(_WIN32)
    // Cover mapping creation AND the caller's header initialization. A later
    // attachment cannot observe or reinitialize a half-built layout.
    const auto mutexName = objectName_ + "_init";
    initializationMutex_ = ::CreateMutexA(nullptr, FALSE, mutexName.c_str());
    if (initializationMutex_ == nullptr)
        return false;
    const DWORD wait = ::WaitForSingleObject(static_cast<HANDLE>(initializationMutex_), 1000);
    if (wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED)
    {
        close();
        return false;
    }
    initializationLocked_ = true;
    const auto bytes = static_cast<std::uint64_t>(requestedBytes);
    handle_ = ::CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                  static_cast<DWORD>(bytes >> 32),
                                  static_cast<DWORD>(bytes & 0xffffffffu), objectName_.c_str());
    const bool localCreated = (::GetLastError() != ERROR_ALREADY_EXISTS);
    if (handle_ == nullptr)
    {
        close();
        return false;
    }
    // Zero length maps the complete established section, not the new request's
    // smaller view. Validate the actual mapped region before accepting a header.
    base_ = ::MapViewOfFile(static_cast<HANDLE>(handle_), FILE_MAP_ALL_ACCESS, 0, 0, 0);
    MEMORY_BASIC_INFORMATION info {};
    if (base_ == nullptr || ::VirtualQuery(base_, &info, sizeof(info)) != sizeof(info)
        || info.RegionSize < requestedBytes)
    {
        close();
        return false;
    }
    sizeBytes_ = info.RegionSize;
    if (created != nullptr)
        *created = localCreated;
    return true;
   #else
    if (requestedBytes > static_cast<std::uintmax_t>(std::numeric_limits<off_t>::max()))
        return false;
    #if defined(__APPLE__) || defined(ZA_JSFX_TEST_DARWIN_SHM)
    // Darwin's shm descriptors are not vnodes and reject flock. Lock a
    // private regular file before opening/sizing shared memory instead. Keep
    // the file in place: unlinking it can split simultaneous openers across
    // different inodes. OS descriptor cleanup also releases a crashed owner's
    // lock, unlike a named semaphore whose decremented count could be stranded.
    const auto lockPath = std::string("/tmp/za_jsfx_init_")
                        + std::to_string(static_cast<unsigned long long>(::geteuid()))
                        + "_" + objectName_.substr(1) + ".lock";
    initializationLockFd_ = ::open(lockPath.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (initializationLockFd_ < 0)
        return false;
    struct stat lockInfo {};
    if (::fstat(initializationLockFd_, &lockInfo) != 0
        || !S_ISREG(lockInfo.st_mode) || lockInfo.st_uid != ::geteuid())
    {
        close();
        return false;
    }
    const int lockFd = initializationLockFd_;
    #else
    fd_ = ::shm_open(objectName_.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd_ < 0)
        return false;
    const int lockFd = fd_;
    #endif

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (::flock(lockFd, LOCK_EX | LOCK_NB) != 0)
    {
        if ((errno != EWOULDBLOCK && errno != EAGAIN && errno != EINTR)
            || std::chrono::steady_clock::now() >= deadline)
        {
            close();
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    initializationLocked_ = true;
    #if defined(__APPLE__) || defined(ZA_JSFX_TEST_DARWIN_SHM)
    fd_ = ::shm_open(objectName_.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd_ < 0)
    {
        close();
        return false;
    }
    #endif
    struct stat st {};
    if (::fstat(fd_, &st) != 0 || st.st_size < 0)
    {
        close();
        return false;
    }
    // The lock winner, not the shm_open winner, owns initial sizing. Thus an
    // opener that runs before the creator's ftruncate cannot map unbacked bytes.
    const bool localCreated = (st.st_size == 0);
    if (localCreated)
    {
        if (::ftruncate(fd_, static_cast<off_t>(requestedBytes)) != 0)
        {
            close();
            return false;
        }
        st.st_size = static_cast<off_t>(requestedBytes);
    }
    if (static_cast<std::uintmax_t>(st.st_size) < requestedBytes
        || static_cast<std::uintmax_t>(st.st_size) > std::numeric_limits<std::size_t>::max())
    {
        close();
        return false;
    }
    sizeBytes_ = static_cast<std::size_t>(st.st_size);
    base_ = ::mmap(nullptr, sizeBytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (base_ == MAP_FAILED)
    {
        base_ = nullptr;
        close();
        return false;
    }
    if (created != nullptr)
        *created = localCreated;
    return true;
   #endif
}

void DspJsfxSharedMemorySegment::finishInitialization() noexcept
{
   #if JUCE_WINDOWS || defined(_WIN32)
    if (initializationLocked_ && initializationMutex_ != nullptr)
        ::ReleaseMutex(static_cast<HANDLE>(initializationMutex_));
    initializationLocked_ = false;
    if (initializationMutex_ != nullptr)
        ::CloseHandle(static_cast<HANDLE>(initializationMutex_));
    initializationMutex_ = nullptr;
   #else
    #if defined(__APPLE__) || defined(ZA_JSFX_TEST_DARWIN_SHM)
    const int lockFd = initializationLockFd_;
    #else
    const int lockFd = fd_;
    #endif
    if (initializationLocked_ && lockFd >= 0)
        ::flock(lockFd, LOCK_UN);
    initializationLocked_ = false;
    #if defined(__APPLE__) || defined(ZA_JSFX_TEST_DARWIN_SHM)
    if (initializationLockFd_ >= 0)
        ::close(initializationLockFd_);
    initializationLockFd_ = -1;
    #endif
   #endif
}

void DspJsfxSharedMemorySegment::close() noexcept
{
    finishInitialization();
   #if JUCE_WINDOWS || defined(_WIN32)
    if (base_ != nullptr)
        ::UnmapViewOfFile(base_);
    base_ = nullptr;
    if (handle_ != nullptr)
        ::CloseHandle(static_cast<HANDLE> (handle_));
    handle_ = nullptr;
   #else
    if (base_ != nullptr && sizeBytes_ > 0)
        ::munmap(base_, sizeBytes_);
    base_ = nullptr;
    if (fd_ >= 0)
        ::close(fd_);
    fd_ = -1;
   #endif
    sizeBytes_ = 0;
    objectName_.clear();
}

} // namespace za::jsfx
