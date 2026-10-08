// SPDX-License-Identifier: Zlib
#pragma once
#include <algorithm>
#include <limits>
#include <vector>
namespace za::jsfx {
// Production bus routing and resampling. The processor is a host port; code
// ownership and compiler mode do not participate in audio channel mapping.
class AudioHostRuntime {
public:
    struct Topology {int channels=0,inputs=0,outputs=0;};
    std::vector<const float*> inPtrs;
    std::vector<float*> outPtrs;
    std::vector<const float*> hostInPtrs;
    std::vector<float*> hostOutPtrs;

    // Scratch buffers for sidechain / missing channels.
    // - zeroIn provides a read-only silent input channel.
    // - scratchOut captures writes for channels that don't map to a host output.
    std::vector<float> zeroIn;
    juce::AudioBuffer<float> scratchOut;
    juce::AudioBuffer<float> oversampledInput;
    juce::AudioBuffer<float> oversampledOutput;
    static void upsampleLinearBlock (const float* src, float* dst, int hostSamples, int factor) noexcept
    {
        if (dst == nullptr || hostSamples <= 0 || factor <= 1)
            return;

        if (src == nullptr)
        {
            std::fill (dst, dst + (size_t) hostSamples * (size_t) factor, 0.0f);
            return;
        }

        for (int i = 0; i < hostSamples; ++i)
        {
            const float a = src[i];
            const float b = (i + 1 < hostSamples) ? src[i + 1] : a;
            float* out = dst + (size_t) i * (size_t) factor;

            for (int sub = 0; sub < factor; ++sub)
            {
                const float t = (float) sub / (float) factor;
                out[sub] = a + (b - a) * t;
            }
        }
    }

    void prepareOversampledAudioPointers (int numCh, int hostSamples, int engineSamples, int factor)
    {
        if (numCh <= 0 || hostSamples <= 0 || engineSamples <= 0 || factor <= 1)
            return;

        if (oversampledInput.getNumChannels() != numCh || oversampledInput.getNumSamples() != engineSamples)
            oversampledInput.setSize (numCh, engineSamples, false, false, true);
        if (oversampledOutput.getNumChannels() != numCh || oversampledOutput.getNumSamples() != engineSamples)
            oversampledOutput.setSize (numCh, engineSamples, false, false, true);

        for (int ch = 0; ch < numCh; ++ch)
        {
            const float* src = (ch < (int) hostInPtrs.size()) ? hostInPtrs[(size_t) ch] : nullptr;
            upsampleLinearBlock (src, oversampledInput.getWritePointer (ch), hostSamples, factor);
        }

        oversampledOutput.clear();

        inPtrs.resize ((size_t) numCh);
        outPtrs.resize ((size_t) numCh);
        for (int ch = 0; ch < numCh; ++ch)
        {
            inPtrs[(size_t) ch] = oversampledInput.getReadPointer (ch);
            outPtrs[(size_t) ch] = oversampledOutput.getWritePointer (ch);
        }
    }

    void downsampleOversampledOutputToHost (int numOutputChannels, int hostSamples, int engineSamples, int factor) noexcept
    {
        if (numOutputChannels <= 0 || hostSamples <= 0 || engineSamples <= 0 || factor <= 1)
            return;

        const int channels = juce::jmin (numOutputChannels,
                                         juce::jmin (oversampledOutput.getNumChannels(),
                                                     (int) hostOutPtrs.size()));

        for (int ch = 0; ch < channels; ++ch)
        {
            float* dst = hostOutPtrs[(size_t) ch];
            const float* src = oversampledOutput.getReadPointer (ch);
            if (dst == nullptr || src == nullptr)
                continue;

            for (int i = 0; i < hostSamples; ++i)
            {
                const int base = i * factor;
                if (base >= engineSamples)
                {
                    dst[i] = 0.0f;
                    continue;
                }

                const int n = juce::jmin (factor, engineSamples - base);
                double sum = 0.0;
                for (int sub = 0; sub < n; ++sub)
                    sum += (double) src[base + sub];

                dst[i] = (float) (sum / (double) juce::jmax (1, n));
            }
        }
    }


    Topology prepareHostAudio(juce::AudioProcessor& processor,juce::AudioBuffer<float>& buffer,int declaredInputs,int declaredOutputs,int factor) {
        const int numSamples=buffer.getNumSamples();
        const int processSamples=safeBlockSize(numSamples,factor);
        if(declaredInputs==0 && declaredOutputs==0) {
            inPtrs.clear();outPtrs.clear();hostInPtrs.clear();hostOutPtrs.clear();return {};
        }

        juce::AudioBuffer<float> mainIn;
        juce::AudioBuffer<float> mainOut;
        juce::AudioBuffer<float> scIn;

        if (processor.getBusCount (true) > 0)
        {
            if (auto* bus = processor.getBus (true, 0); bus != nullptr && bus->isEnabled())
                mainIn = processor.getBusBuffer (buffer, true, 0);
        }

        if (processor.getBusCount (false) > 0)
        {
            if (auto* bus = processor.getBus (false, 0); bus != nullptr && bus->isEnabled())
                mainOut = processor.getBusBuffer (buffer, false, 0);
        }

        const int mainInCh  = juce::jmin (mainIn.getNumChannels(), 64);
        const int mainOutCh = juce::jmin (mainOut.getNumChannels(), 64);

        int scCh = 0;
        if (processor.getBusCount (true) > 1)
        {
            if (auto* bus = processor.getBus (true, 1); bus != nullptr && bus->isEnabled())
            {
                scIn = processor.getBusBuffer (buffer, true, 1);
                scCh = juce::jmin (scIn.getNumChannels(), 64 - mainInCh);
            }
        }

        const int totalInCh  = juce::jmin (mainInCh + scCh, 64);
        const int totalOutCh = juce::jmin (mainOutCh, 64);
        
        

        const int requiredCh = juce::jlimit (0, 64, juce::jmax (declaredInputs, declaredOutputs));
        const int numCh = juce::jmin (64, juce::jmax (requiredCh, juce::jmax (totalInCh, totalOutCh)));

        if ((int) zeroIn.size() != numSamples)
            zeroIn.assign ((size_t) numSamples, 0.0f);
        const float* const zeroPtr = zeroIn.empty() ? nullptr : zeroIn.data();

        const int scratchCh = juce::jmax (0, numCh - totalOutCh);
        if (scratchCh > 0)
        {
            if (scratchOut.getNumChannels() != scratchCh || scratchOut.getNumSamples() != numSamples)
            {
                scratchOut.setSize (scratchCh, numSamples, false, false, true);
                scratchOut.clear();
            }
        }
        else if (scratchOut.getNumChannels() != 0)
        {
            scratchOut.setSize (0, 0);
        }

        inPtrs.resize ((size_t) numCh);
        outPtrs.resize ((size_t) numCh);

        int dst = 0;
        for (int ch = 0; ch < mainInCh && dst < numCh; ++ch, ++dst)
            inPtrs[(size_t) dst] = mainIn.getReadPointer (ch);
        for (int ch = 0; ch < scCh && dst < numCh; ++ch, ++dst)
            inPtrs[(size_t) dst] = scIn.getReadPointer (ch);
        for (; dst < numCh; ++dst)
            inPtrs[(size_t) dst] = zeroPtr;

        for (int ch = 0; ch < totalOutCh && ch < numCh; ++ch)
            outPtrs[(size_t) ch] = mainOut.getWritePointer (ch);
        for (int ch = totalOutCh; ch < numCh; ++ch)
            outPtrs[(size_t) ch] = scratchOut.getWritePointer (ch - totalOutCh);

        hostInPtrs = inPtrs;
        hostOutPtrs = outPtrs;

        if (factor > 1 && numCh > 0 && numSamples > 0)
            prepareOversampledAudioPointers (numCh, numSamples, processSamples, factor);

        return {numCh,totalInCh,totalOutCh};
    }
    static int safeBlockSize(int samples,int factor) noexcept {
        samples=std::max(0,samples);factor=std::clamp(factor,1,8);
        return samples>std::numeric_limits<int>::max()/factor?std::numeric_limits<int>::max():samples*factor;
    }
    static int factorFromChoice(int choice) noexcept {return 1<<std::clamp(choice,0,3);}
};
}
