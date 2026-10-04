// SPDX-License-Identifier: Zlib
// Actual production processor; no editor, same bank/MIDI workload in each mode.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <juce_audio_utils/juce_audio_utils.h>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter();
extern "C" void za_sample_gfx_load_bank(juce::AudioProcessor *,
                                        const char *const *, int);
extern "C" bool za_sample_gfx_legacy();

int main(int argc, char **argv) try {
  if (argc != 4)
    throw std::runtime_error("Pass three test-bank WAV paths");
  juce::ScopedJuceInitialiser_GUI gui;
  std::unique_ptr<juce::AudioProcessor> processor(createPluginFilter());
  processor->setRateAndBufferSizeDetails(48000, 512);
  processor->prepareToPlay(48000, 512);
  za_sample_gfx_load_bank(processor.get(), argv + 1, 3);
  juce::AudioBuffer<float> buffer(2, 512);
  juce::MidiBuffer midi;
  double peak = 0;
  auto process = [&](int block) {
    buffer.clear();
    midi.clear();
    if (block % 64 == 0) {
      midi.addEvent(juce::MidiMessage::allNotesOff(1), 0);
      midi.addEvent(juce::MidiMessage::noteOn(1, 60, 0.8f), 1);
    }
    processor->processBlock(buffer, midi);
    for (int c = 0; c < 2; ++c)
      for (int s = 0; s < 512; ++s) {
        const double value = buffer.getSample(c, s);
        if (!std::isfinite(value))
          throw std::runtime_error("Nonfinite audio");
        peak = std::max(peak, std::abs(value));
      }
  };
  // Allow asynchronous sample loading, then warm DSP/cache before timing.
  for (int i = 0; i < 512; ++i) {
    process(i);
    if (i % 8 == 0)
      juce::MessageManager::getInstance()->runDispatchLoopUntil(5);
  }
  std::vector<double> times;
  constexpr int blocks = 2000;
  times.reserve(blocks);
  for (int i = 0; i < blocks; ++i) {
    buffer.clear();
    midi.clear();
    if (i % 64 == 0) {
      midi.addEvent(juce::MidiMessage::allNotesOff(1), 0);
      midi.addEvent(juce::MidiMessage::noteOn(1, 60, 0.8f), 1);
    }
    const auto start = std::chrono::steady_clock::now();
    processor->processBlock(buffer, midi);
    times.push_back(std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start)
                        .count());
    for (int c = 0; c < 2; ++c)
      for (int s = 0; s < 512; ++s) {
        const double value = buffer.getSample(c, s);
        if (!std::isfinite(value))
          throw std::runtime_error("Nonfinite audio");
        peak = std::max(peak, std::abs(value));
      }
  }
  if (peak < .001)
    throw std::runtime_error("Benchmark bank did not produce audio");
  double mean = 0;
  for (double value : times)
    mean += value / blocks;
  std::sort(times.begin(), times.end());
  std::cout << "{\"legacy\":" << (za_sample_gfx_legacy() ? "true" : "false")
            << ",\"blocks\":" << blocks
            << ",\"block_size\":512,\"mean_us\":" << mean
            << ",\"median_us\":" << times[blocks / 2]
            << ",\"p95_us\":" << times[blocks * 95 / 100]
            << ",\"p99_us\":" << times[blocks * 99 / 100]
            << ",\"max_us\":" << times.back() << ",\"audio_peak\":" << peak
            << "}\n";
  processor->releaseResources();
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
