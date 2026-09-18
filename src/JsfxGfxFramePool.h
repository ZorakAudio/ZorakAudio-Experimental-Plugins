// SPDX-License-Identifier: Zlib
// Worker-owned software surfaces, with short publication locking only.
// JUCE must be included before this header.
#ifndef ZA_JSFX_GFX_FRAME_POOL_H
#define ZA_JSFX_GFX_FRAME_POOL_H
#include <array>
#include <cstring>
#include <mutex>

namespace jsfx_gfx
{
class FramePool
{
public:
    // Single worker only. No waiting for the message thread and no writes to a
    // published/leased bitmap. JUCE Image copies are shared references, NOT COW.
    bool acquire(juce::Image& target, int width, int height)
    {
        target = {};
        if (width <= 0 || height <= 0) return false;
        juce::Image* available = nullptr;
        {
            const std::lock_guard<std::mutex> lock(mutex);
            for (auto& slot : slots)
                if (slot.isNull() || slot.getReferenceCount() == 1)
                {
                    available = &slot;
                    break;
                }
        }
        if (!available) return false; // All backs leased: retain input and retry.
        // Non-front, exclusively owned slot; allocation is outside the lock.
        if (available->isNull() || available->getWidth() != width || available->getHeight() != height)
            *available = juce::Image(juce::Image::ARGB, width, height, true, juce::SoftwareImageType());
        target = *available;
        return target.isValid();
    }

    void publish(const juce::Image& image)
    {
        const std::lock_guard<std::mutex> lock(mutex);
        published = image; // O(1) handle exchange. The pixels are immutable to readers.
    }

    juce::Image front() const
    {
        const std::lock_guard<std::mutex> lock(mutex);
        return published;
    }

private:
    std::array<juce::Image, 3> slots {};
    mutable std::mutex mutex;
    juce::Image published;
};

// History is necessary for gfx_clear=-1/partial frames, but not for full clears.
// Copy premultiplied pixels verbatim: drawing with source-over would corrupt the
// history of translucent frames. All pool images have the same ARGB format.
static inline bool copyFramebufferHistory(juce::Image& target, const juce::Image& source)
{
    if (!target.isValid() || !source.isValid() || target == source
        || target.getWidth() != source.getWidth() || target.getHeight() != source.getHeight()
        || target.getFormat() != source.getFormat()) return false;
    const juce::Image::BitmapData from(source, juce::Image::BitmapData::readOnly);
    juce::Image::BitmapData to(target, juce::Image::BitmapData::writeOnly);
    if (from.pixelStride != to.pixelStride) return false;
    const size_t bytes = (size_t)source.getWidth() * (size_t)from.pixelStride;
    for (int y = 0; y < source.getHeight(); ++y)
        std::memcpy(to.getLinePointer(y), from.getLinePointer(y), bytes);
    return true;
}
}
#endif
