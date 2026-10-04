// SPDX-License-Identifier: Zlib
// Actual processor/editor, simulated audio callback and parameter interaction.
#include <atomic>
#include <iostream>
#include <juce_audio_utils/juce_audio_utils.h>
#include <stdexcept>
#include <thread>
extern juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter();
extern "C" bool za_native_gfx_snapshot(juce::AudioProcessor *, const char *, double *, int *, size_t *);
extern "C" uint64_t za_native_gfx_frames();
static void require(bool value, const char *error)
{
    if (!value)
        throw std::runtime_error(error);
}
static void pump(int ms)
{
    juce::MessageManager::getInstance()->runDispatchLoopUntil(ms);
}
static void save(juce::Component &editor, const juce::File &directory, const char *name)
{
    auto image = editor.createComponentSnapshot(editor.getLocalBounds());
    auto out = directory.getChildFile(name).createOutputStream();
    require(out != nullptr, "Cannot save editor snapshot");
    require(out->setPosition(0), "Cannot rewind screenshot output");
    out->truncate();
    juce::PNGImageFormat png;
    require(png.writeImageToStream(image, *out), "PNG write failed");
}
static juce::Slider *firstSlider(juce::Component &root)
{
    if (auto *slider = dynamic_cast<juce::Slider *>(&root))
        return slider;
    for (auto *child : root.getChildren())
        if (auto *slider = firstSlider(*child))
            return slider;
    return nullptr;
}
static juce::Component *graphicsCanvas(juce::Component &root)
{
    if (root.getName() == "Native JSFX graphics")
        return &root;
    for (auto *child : root.getChildren())
        if (auto *canvas = graphicsCanvas(*child))
            return canvas;
    return nullptr;
}
static void waitForText(juce::Component &editor)
{
    auto *canvas = graphicsCanvas(editor);
    require(canvas != nullptr, "Missing native graphics component");
    for (int attempt = 0; attempt < 20; ++attempt)
    {
        pump(50);
        const auto image = canvas->createComponentSnapshot(canvas->getLocalBounds());
        int pixels = 0;
        for (int y = 16; y < 30 && y < image.getHeight(); ++y)
            for (int x = 20; x < image.getWidth() - 60; ++x)
            {
                const auto c = image.getPixelAt(x, y);
                pixels += c.getRed() > 180 && c.getGreen() > 180 && c.getBlue() > 180;
            }
        if (pixels > 100)
            return;
    }
    throw std::runtime_error("Native editor title text did not become visible");
}
int main(int argc, char **argv)
try
{
    juce::ScopedJuceInitialiser_GUI gui;
    require(argc == 2, "Pass screenshot output directory");
    juce::File directory(argv[1]);
    directory.createDirectory();
    std::unique_ptr<juce::AudioProcessor> processor(createPluginFilter());
    processor->prepareToPlay(48000, 512);
    auto *parameter = dynamic_cast<juce::RangedAudioParameter *>(processor->getParameters()[0]);
    require(parameter != nullptr, "Missing threshold parameter");
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> blocks{0};
    std::atomic<double> amplitude{0.1};
    std::thread audio(
        [&]
        {
            juce::AudioBuffer<float> buffer(2, 512);
            juce::MidiBuffer midi;
            double phase = 0;
            while (!stop.load())
            {
                for (int i = 0; i < 512; ++i)
                {
                    const float sample = (float)(amplitude.load() * std::sin(phase));
                    phase += 2 * juce::MathConstants<double>::pi * 440 / 48000;
                    buffer.setSample(0, i, sample);
                    buffer.setSample(1, i, sample);
                }
                midi.clear();
                processor->processBlock(buffer, midi);
                blocks.fetch_add(1);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    struct Join
    {
        std::atomic<bool> &stop;
        std::thread &audio;
        ~Join()
        {
            stop = true;
            if (audio.joinable())
                audio.join();
        }
    } join{stop, audio};
    for (int lifetime = 0; lifetime < 3; ++lifetime)
    {
        std::unique_ptr<juce::AudioProcessorEditor> editor(processor->createEditor());
        require(editor != nullptr, "No plugin editor");
        editor->addToDesktop(juce::ComponentPeer::windowHasTitleBar);
        editor->setVisible(true);
        const auto before = za_native_gfx_frames();
        pump(250);
        require(za_native_gfx_frames() > before, "Native graphics worker did not execute");
        waitForText(*editor);
        if (lifetime == 0)
        {
            std::cout << "Threshold parameter: " << parameter->getName(100) << "\n";
            save(*editor, directory, "easyexpander-open.png");
            auto *slider = firstSlider(*editor);
            require(slider != nullptr, "No JUCE parameter control");
            // Same synchronous listener/attachment route used by the slider's mouse interaction.
            const bool normalised = slider->getMinimum() == 0 && slider->getMaximum() == 1;
            slider->setValue(normalised ? parameter->convertTo0to1(-20) : -20, juce::sendNotificationSync);
            pump(180);
            require(std::abs(parameter->convertFrom0to1(parameter->getValue()) + 20) < 0.2,
                    "Parameter control did not update host value");
            double value = 0;
            int spans = -1;
            size_t capacity = 1;
            require(za_native_gfx_snapshot(processor.get(), "thresh_db", &value, &spans, &capacity),
                    "No scalar snapshot");
            require(std::abs(value + 20) < 0.2, "UI threshold did not reach DSP publication");
            require(spans == 0 && capacity == 0, "Native editor still mirrors a DSP heap");
            parameter->beginChangeGesture();
            parameter->setValueNotifyingHost(parameter->convertTo0to1(-35));
            parameter->endChangeGesture();
            amplitude = 0.0001;
            pump(400);
            require(za_native_gfx_snapshot(processor.get(), "thresh_db", &value, &spans, &capacity) &&
                        std::abs(value + 35) < 0.2,
                    "Host automation did not reach graphics publication");
            save(*editor, directory, "easyexpander-expanding.png");
            editor->setSize(editor->getWidth() + 260, editor->getHeight() + 120);
            pump(160);
            save(*editor, directory, "easyexpander-resized.png");
        }
        editor->setVisible(false);
        editor->removeFromDesktop();
        editor.reset();
        pump(70);
        const auto closed = za_native_gfx_frames();
        pump(100);
        require(closed == za_native_gfx_frames(), "Graphics worker survived editor destruction");
    }
    // Hosts re-prepare existing instances. Do so only after joining callbacks.
    stop = true;
    audio.join();
    processor->releaseResources();
    processor->prepareToPlay(48000, 512);
    {
        std::unique_ptr<juce::AudioProcessorEditor> idle(processor->createEditor());
        idle->addToDesktop(juce::ComponentPeer::windowHasTitleBar);
        idle->setVisible(true);
        const auto before = za_native_gfx_frames();
        pump(180);
        require(za_native_gfx_frames() > before, "Native graphics did not render without audio callbacks");
        waitForText(*idle);
        require(std::abs(parameter->convertFrom0to1(parameter->getValue()) + 35) < 0.2,
                "Re-prepare lost parameter state");
        save(*idle, directory, "easyexpander-idle-reprepared.png");
        idle->removeFromDesktop();
    }
    processor->releaseResources();
    std::cout << "PASS actual JUCE native editor: 4 lifetimes; control + host automation; resize; re-prepare "
                 "+ idle rendering; zero heap snapshot storage; audio_blocks="
              << blocks.load() << " native_frames=" << za_native_gfx_frames() << "\n";
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << "\n";
    return 1;
}
