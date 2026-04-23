#include "DspJsfxSharedMemory.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <utility>

#if JUCE_WINDOWS || defined(_WIN32)
 #include <windows.h>
#else
 #include <fcntl.h>
 #include <sys/mman.h>
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
} // namespace

std::string makeSharedMemoryObjectName(const std::string& stem)
{
   #if JUCE_WINDOWS || defined(_WIN32)
    return std::string("Local\\za_jsfx_") + sanitizeStem(stem);
   #else
    return std::string("/za_jsfx_") + sanitizeStem(stem);
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
   #if JUCE_WINDOWS || defined(_WIN32)
    handle_ = other.handle_;
    other.handle_ = nullptr;
   #else
    fd_ = other.fd_;
    other.fd_ = -1;
   #endif
    other.base_ = nullptr;
    other.sizeBytes_ = 0;
    return *this;
}

bool DspJsfxSharedMemorySegment::openOrCreate(const std::string& objectName, std::size_t requestedBytes, bool* created)
{
    close();

    if (requestedBytes == 0)
        return false;

    objectName_ = makeSharedMemoryObjectName(objectName);

   #if JUCE_WINDOWS || defined(_WIN32)
    HANDLE mapping = ::CreateFileMappingA(INVALID_HANDLE_VALUE,
                                          nullptr,
                                          PAGE_READWRITE,
                                          static_cast<DWORD> ((requestedBytes >> 32) & 0xffffffffu),
                                          static_cast<DWORD> (requestedBytes & 0xffffffffu),
                                          objectName_.c_str());
    if (mapping == nullptr)
        return false;

    if (created != nullptr)
        *created = (::GetLastError() != ERROR_ALREADY_EXISTS);

    void* view = ::MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, requestedBytes);
    if (view == nullptr)
    {
        ::CloseHandle(mapping);
        return false;
    }

    handle_ = mapping;
    base_ = view;
    sizeBytes_ = requestedBytes;
    return true;
   #else
    bool localCreated = false;
    int fd = ::shm_open(objectName_.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd >= 0)
    {
        localCreated = true;
        if (::ftruncate(fd, static_cast<off_t> (requestedBytes)) != 0)
        {
            ::close(fd);
            ::shm_unlink(objectName_.c_str());
            return false;
        }
    }
    else
    {
        fd = ::shm_open(objectName_.c_str(), O_RDWR, 0600);
        if (fd < 0)
            return false;
    }

    struct stat st {};
    if (::fstat(fd, &st) != 0)
    {
        ::close(fd);
        if (localCreated)
            ::shm_unlink(objectName_.c_str());
        return false;
    }

    const std::size_t mapBytes = static_cast<std::size_t> (std::max<off_t> (st.st_size, static_cast<off_t> (requestedBytes)));
    if (mapBytes < requestedBytes)
    {
        ::close(fd);
        return false;
    }

    void* view = ::mmap(nullptr, mapBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED)
    {
        ::close(fd);
        return false;
    }

    if (created != nullptr)
        *created = localCreated;

    fd_ = fd;
    base_ = view;
    sizeBytes_ = mapBytes;
    return true;
   #endif
}

void DspJsfxSharedMemorySegment::close() noexcept
{
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
