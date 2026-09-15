#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <juce_audio_formats/juce_audio_formats.h>

namespace za::jsfx
{
// Shared by both decode paths. The baseline referenced this header but did not
// contain it. Validate allocation metadata, and bound uncompressed RIFF audio
// against physically present data rather than trusting a damaged chunk length.
inline std::int64_t chooseSafeDecodedFrameCount(const juce::File& file,
                                                const juce::AudioFormatReader& reader,
                                                bool& skipUnsafeMalformedFile)
{
    skipUnsafeMalformedFile = false;
    const auto reject = [&]() -> std::int64_t { skipUnsafeMalformedFile = true; return 0; };
    const auto frames = reader.lengthInSamples;
    if (frames < 0 || reader.numChannels == 0 || reader.numChannels > 64
        || !std::isfinite(reader.sampleRate) || reader.sampleRate <= 0
        || static_cast<std::uint64_t>(frames) > std::numeric_limits<std::size_t>::max() / (sizeof(double) * reader.numChannels))
        return reject();
    auto input = file.createInputStream();
    if (!input || input->getTotalLength() < 12)
        return reject();
    const auto size = input->getTotalLength();
    const auto riff = input->readInt();
    input->readInt();
    const auto wave = input->readInt();
    if ((riff != 0x46464952 && riff != 0x34364652) || wave != 0x45564157)
        return frames; // Compressed/non-RIFF formats use their JUCE reader metadata.

    bool pcm = false, haveFormat = false;
    std::uint64_t alignment = 0;
    std::int64_t dataBytes = -1;
    for (int chunks = 0; chunks < 4096 && input->getPosition() <= size - 8; ++chunks)
    {
        const auto id = input->readInt();
        const auto declared = static_cast<std::uint32_t>(input->readInt());
        const auto begin = input->getPosition();
        const auto available = std::min<std::int64_t>(declared, size - begin);
        if (id == 0x20746d66 && available >= 16) // fmt
        {
            const auto tag = static_cast<std::uint16_t>(input->readShort());
            const auto channels = static_cast<std::uint16_t>(input->readShort());
            input->readInt(); input->readInt();
            alignment = static_cast<std::uint16_t>(input->readShort());
            const auto bits = static_cast<std::uint16_t>(input->readShort());
            auto format = tag;
            if (tag == 0xfffe && available >= 40)
            {
                input->setPosition(begin + 24);
                format = static_cast<std::uint16_t>(input->readShort());
            }
            pcm = format == 1 || format == 3;
            haveFormat = true;
            if (pcm && (channels == 0 || bits == 0 || bits > 64
                        || alignment < static_cast<std::uint64_t>(channels) * ((bits + 7) / 8)))
                return reject();
        }
        if (id == 0x61746164 && dataBytes < 0) // first data chunk
            dataBytes = available;
        if (haveFormat && dataBytes >= 0)
            break;
        const auto next = begin + static_cast<std::int64_t>(declared) + (declared & 1u);
        if (next > size || next <= begin || !input->setPosition(next))
            break;
    }
    if (pcm && dataBytes >= 0 && alignment > 0)
        return std::min<std::int64_t>(frames, dataBytes / static_cast<std::int64_t>(alignment));
    return frames;
}
}
