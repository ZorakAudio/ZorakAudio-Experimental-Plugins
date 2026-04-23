#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace za::jsfx
{

class DspJsfxSharedMemorySegment
{
public:
    DspJsfxSharedMemorySegment() = default;
    ~DspJsfxSharedMemorySegment();

    DspJsfxSharedMemorySegment(const DspJsfxSharedMemorySegment&) = delete;
    DspJsfxSharedMemorySegment& operator=(const DspJsfxSharedMemorySegment&) = delete;

    DspJsfxSharedMemorySegment(DspJsfxSharedMemorySegment&& other) noexcept;
    DspJsfxSharedMemorySegment& operator=(DspJsfxSharedMemorySegment&& other) noexcept;

    bool openOrCreate(const std::string& objectName, std::size_t requestedBytes, bool* created = nullptr);
    void close() noexcept;

    void* data() noexcept { return base_; }
    const void* data() const noexcept { return base_; }
    std::size_t size() const noexcept { return sizeBytes_; }
    bool isOpen() const noexcept { return base_ != nullptr && sizeBytes_ > 0; }

private:
    void* base_ = nullptr;
    std::size_t sizeBytes_ = 0;
    std::string objectName_;

   #if JUCE_WINDOWS || defined(_WIN32)
    void* handle_ = nullptr;
   #else
    int fd_ = -1;
   #endif
};

std::string makeSharedMemoryObjectName(const std::string& stem);

} // namespace za::jsfx
