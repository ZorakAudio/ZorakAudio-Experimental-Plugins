#include <iostream>
#include <juce_audio_utils/juce_audio_utils.h>
#include <stdexcept>
#include <thread>
extern juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter();
extern "C" bool za_native_gfx_snapshot(juce::AudioProcessor *, const char *,
                                       double *, int *, size_t *);
static void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
int main() {
  try {
    juce::ScopedJuceInitialiser_GUI gui;
    for (int instance = 0; instance < 3; ++instance) {
      std::unique_ptr<juce::AudioProcessor> processor(createPluginFilter());
      for (int reset = 0; reset < 2; ++reset) {
        processor->prepareToPlay(48000, 128);
        juce::AudioBuffer<float> audio(
            std::max(1, processor->getTotalNumOutputChannels()), 128);
        juce::MidiBuffer midi;
        bool ready = false;
        for (int attempt = 0; attempt < 500; ++attempt) {
          audio.clear();
          processor->processBlock(audio, midi);
          if (audio.getSample(0, 0) > 0.99f) {
            ready = true;
            break;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        require(ready,
                "Background reduction did not reach processor audio output");
        std::unique_ptr<juce::AudioProcessorEditor> editor(
            processor->createEditorIfNeeded());
        require(editor != nullptr, "Editor creation failed");
        editor->setTopLeftPosition(-10000,-10000);
        editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);
        editor->setVisible(true);
        double result = 0;
        int spans = 0;
        size_t capacity = 0;
        for (int attempt = 0; attempt < 50 && result != 12; ++attempt) {
          juce::MessageManager::getInstance()->runDispatchLoopUntil(20);
          za_native_gfx_snapshot(processor.get(), "ui_result", &result, &spans,
                                 &capacity);
        }
        require(result == 12, "A task submitted from @gfx did not finish");
        editor->createComponentSnapshot(editor->getLocalBounds());
        processor->editorBeingDeleted(editor.get());
        editor.reset(); // Workers must remain owned by the processor.
        audio.clear();
        processor->processBlock(audio, midi);
        require(audio.getSample(0, 0) > 0.99f,
                "Editor closure changed published result");
        processor->releaseResources();
      }
    }
    std::cout << "Task processor and editor lifecycle passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
