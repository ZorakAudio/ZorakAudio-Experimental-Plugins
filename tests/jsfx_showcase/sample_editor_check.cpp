// SPDX-License-Identifier: Zlib
// Actual Sample editor: baseline plus native interaction/publication
// qualification.
#include <atomic>
#include <iostream>
#include <juce_audio_utils/juce_audio_utils.h>
#include <stdexcept>
#include <thread>
extern juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter();
extern "C" bool za_sample_gfx_snapshot(juce::AudioProcessor *, const char *,
                                       double *, size_t *, size_t *, int64_t *);
extern "C" uint64_t za_sample_gfx_frames();
extern "C" bool za_sample_gfx_native();
extern "C" bool za_sample_gfx_legacy();
extern "C" double za_sample_gfx_ui(juce::AudioProcessor *, const char *);
extern "C" bool za_sample_gfx_faulted(juce::AudioProcessor *);
extern "C" void za_sample_gfx_copy_stats(juce::AudioProcessor *, uint64_t *,
                                         double *, double *);
struct Gestures final : juce::AudioProcessorParameter::Listener {
    std::atomic<int> begins{0}, ends{0};
    void parameterValueChanged(int, float) override {}
    void parameterGestureChanged(int, bool starting) override {
        if (starting)
            ++begins;
        else
            ++ends;
    }
};
extern "C" void za_sample_gfx_load_bank(juce::AudioProcessor *,
                                        const char *const *, int);
static void require(bool ok, const char *error) {
    if (!ok)
        throw std::runtime_error(error);
}
static void pump(int ms) {
    juce::MessageManager::getInstance()->runDispatchLoopUntil(ms);
}
static juce::Component *canvas(juce::Component &root) {
    if (root.getName() == "EEL JSFX graphics" ||
        root.getName() == "Native JSFX graphics")
        return &root;
    for (auto *child : root.getChildren())
        if (auto *result = canvas(*child))
            return result;
    return nullptr;
}
static void save(juce::Component &editor, const juce::File &directory,
                 const char *name) {
    auto out = directory.getChildFile(name).createOutputStream();
    require(out != nullptr && out->setPosition(0), "Cannot save screenshot");
    out->truncate();
    juce::PNGImageFormat png;
    require(png.writeImageToStream(
                editor.createComponentSnapshot(editor.getLocalBounds()), *out),
            "PNG write failed");
}
static void waitForFrame(juce::Component &editor) {
    auto *view = canvas(editor);
    require(view != nullptr, "Missing existing EEL graphics canvas");
    const auto before = za_sample_gfx_frames();
    for (int attempt = 0; attempt < 60; ++attempt) {
        pump(50);
        if (za_sample_gfx_frames() <= before)
            continue;
        const auto image =
            view->createComponentSnapshot(view->getLocalBounds());
        int textPixels = 0;
        for (int y = 20; y < 60 && y < image.getHeight(); ++y)
            for (int x = 90; x < 420 && x < image.getWidth(); ++x) {
                const auto c = image.getPixelAt(x, y);
                textPixels +=
                    c.getRed() > 180 && c.getGreen() > 180 && c.getBlue() > 180;
            }
        if (textPixels > 100)
            return;
    }
    throw std::runtime_error("Sample title did not render");
}
static juce::MouseEvent event(juce::Component &view, juce::Point<float> p,
                              int flags) {
    const auto now = juce::Time::getCurrentTime();
    return {juce::Desktop::getInstance().getMainMouseSource(),
            p,
            juce::ModifierKeys(flags),
            1,
            0,
            0,
            0,
            0,
            &view,
            &view,
            now,
            p,
            now,
            1,
            false};
}
static void click(juce::Component &view, juce::Point<float> p) {
    view.mouseDown(event(view, p, juce::ModifierKeys::leftButtonModifier));
    pump(150);
    view.mouseUp(event(view, p, juce::ModifierKeys::leftButtonModifier));
    pump(180);
}
static void makeBank(const juce::File &directory,
                     std::vector<std::string> &paths) {
    for (int item = 0; item < 3; ++item) {
        const auto file =
            directory.getChildFile("test-bank-" + juce::String(item) + ".wav");
        auto out = file.createOutputStream();
        require(out != nullptr && out->setPosition(0),
                "Cannot create test sample");
        out->truncate();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(
            wav.createWriterFor(out.get(), 48000, 2, 24, {}, 0));
        require(writer != nullptr, "Cannot encode test sample");
        out.release();
        const int length = 24000 + item * 12000;
        juce::AudioBuffer<float> buffer(2, length);
        for (int i = 0; i < length; ++i) {
            const double t = (double)i / 48000;
            const double envelope =
                std::min(1.0, t / 0.003) * std::exp(-t * (3 + item));
            const double phase = 2 * juce::MathConstants<double>::pi *
                                 (220 * (item + 1) * t + 50 * t * t);
            buffer.setSample(0, i, (float)(0.25 * envelope * std::sin(phase)));
            buffer.setSample(1, i,
                             (float)(0.23 * envelope * std::sin(phase + 0.03)));
        }
        require(writer->writeFromAudioSampleBuffer(buffer, 0, length),
                "Cannot write test bank");
        paths.push_back(file.getFullPathName().toStdString());
    }
}
int main(int argc, char **argv) try {
    juce::ScopedJuceInitialiser_GUI gui;
    require(argc == 2, "Pass screenshot output directory");
    juce::File directory(argv[1]);
    directory.createDirectory();
    std::vector<std::string> bank;
    makeBank(directory, bank);
    std::unique_ptr<juce::AudioProcessor> processor(createPluginFilter());
    processor->prepareToPlay(48000, 512);
    za_sample_gfx_load_bank(processor.get(), nullptr, 0);
    std::atomic<bool> stop{false}, badAudio{false};
    std::atomic<int> noteRequest{0}, hubRequest{0};
    std::atomic<uint64_t> blocks{0};
    std::atomic<float> peak{0};
    const auto runAudio = [&] {
        juce::AudioBuffer<float> buffer(2, 512);
        juce::MidiBuffer midi;
        while (!stop.load()) {
            buffer.clear();
            midi.clear();
            const int request = noteRequest.exchange(0);
            if (request > 0)
                midi.addEvent(
                    juce::MidiMessage::noteOn(1, 60, (juce::uint8)100), 17);
            if (request < 0)
                midi.addEvent(juce::MidiMessage::noteOff(1, 60), 47);
            if (hubRequest.exchange(0)) {
                // End for an absent ID is idempotently accepted; a bad version
                // is rejected.
                for (int version : {1, 2}) {
                    const int data[20] = {version, 7, 0, 1,   0,   0, 60,
                                          100,     0, 0, 0,   0,   0, 0,
                                          0,       0, 0, 127, 127, 0};
                    midi.addEvent(
                        juce::MidiMessage::controllerEvent(1, 118, 83), 33);
                    int checksum = 0;
                    for (int i = 0; i < 20; ++i) {
                        checksum = (checksum + (i + 1) * data[i]) % 128;
                        midi.addEvent(
                            juce::MidiMessage::controllerEvent(1, 119, data[i]),
                            33);
                    }
                    midi.addEvent(
                        juce::MidiMessage::controllerEvent(1, 119, checksum),
                        33);
                    midi.addEvent(
                        juce::MidiMessage::controllerEvent(1, 118, 84), 33);
                }
            }
            processor->processBlock(buffer, midi);
            for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
                for (int i = 0; i < buffer.getNumSamples(); ++i) {
                    const float value = buffer.getSample(ch, i);
                    if (!std::isfinite(value))
                        badAudio = true;
                    if (std::abs(value) > peak.load())
                        peak = std::abs(value);
                }
            ++blocks;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };
    std::thread audio(runAudio);
    struct Join {
        std::atomic<bool> &stop;
        std::thread &audio;
        ~Join() {
            stop = true;
            if (audio.joinable())
                audio.join();
        }
    } join{stop, audio};
    const auto read = [&](const char *name) {
        double value = 0;
        size_t copied = 0, capacity = 0;
        int64_t logical = 0;
        require(za_sample_gfx_snapshot(processor.get(), name, &value, &copied,
                                       &capacity, &logical),
                "Missing Sample snapshot variable");
        return value;
    };
    const auto parameter = [&](int slider) -> juce::RangedAudioParameter & {
        auto *p = dynamic_cast<juce::RangedAudioParameter *>(
            processor->getParameters()[slider - 1]);
        require(p != nullptr, "Missing Sample parameter");
        return *p;
    };
    const bool native = za_sample_gfx_native();
    const auto ui = [&](const char *name) {
        return za_sample_gfx_ui(processor.get(), name);
    };
    const auto actual = [&](int slider) {
        return parameter(slider).convertFrom0to1(parameter(slider).getValue());
    };
    const auto set = [&](int slider, float value) {
        parameter(slider).setValueNotifyingHost(
            parameter(slider).convertTo0to1(value));
    };
    const auto until = [&](auto condition, const char *message) {
        for (int attempt = 0; attempt < 160 && !condition(); ++attempt)
            pump(25);
        require(condition(), message);
    };
    const auto knob = [&](int index) -> juce::Point<float> {
        const double cols = ui("knob_cols");
        require(cols > 0, "Missing native layout");
        return {(float)(ui("right_inner_x") +
                        std::fmod(index, cols) *
                            (ui("cell_w") + ui("knob_col_gap")) +
                        ui("cell_w") * 0.5),
                (float)(ui("knob_panel_top") +
                        std::floor(index / cols) *
                            (ui("cell_h") + ui("knob_row_gap")) +
                        ui("kb_r") + 6)};
    };
    const auto wheel = [&](juce::Component &view, juce::Point<float> point,
                           int modifiers) {
        juce::MouseWheelDetails details;
        details.deltaY = 1.0f;
        view.mouseWheelMove(event(view, point, modifiers), details);
        pump(250);
    };
    Gestures gestures;
    parameter(59).addListener(&gestures);
    struct RemoveListener {
        juce::RangedAudioParameter &parameter;
        Gestures &gestures;
        ~RemoveListener() { parameter.removeListener(&gestures); }
    } removeListener{parameter(59), gestures};
    for (int opening = 0; opening < 2; ++opening) {
        std::unique_ptr<juce::AudioProcessorEditor> editor(
            processor->createEditor());
        editor->addToDesktop(juce::ComponentPeer::windowHasTitleBar);
        editor->setVisible(true);
        editor->setSize(1480, 980);
        waitForFrame(*editor);
        if (opening == 0) {
            for (int attempt = 0; attempt < 60 && read("sample_count") != 0;
                 ++attempt)
                pump(50);
            require(read("sample_count") == 0,
                    "Sample baseline did not start with an empty bank");
            waitForFrame(*editor);
            save(*editor, directory, "sample-empty.png");
            std::vector<const char *> paths;
            for (const auto &path : bank)
                paths.push_back(path.c_str());
            za_sample_gfx_load_bank(processor.get(), paths.data(),
                                    (int)paths.size());
            for (int attempt = 0; attempt < 160 && read("sample_count") != 3;
                 ++attempt)
                pump(50);
            require(read("sample_count") == 3 && read("bank_loaded") == 1,
                    "Sample did not adopt all three files");
            if (native)
                until([&] { return ui("sample_count") == 3; },
                      "Native graphics did not adopt bank publication");
            noteRequest = 1;
            pump(200);
            require(peak.load() > 0.0001f,
                    "Loaded Sample bank did not produce MIDI audio");
            waitForFrame(*editor);
            save(*editor, directory, "sample-playing.png");
            noteRequest = -1;
            auto *view = canvas(*editor);
            // Playback button coordinates follow Sample's responsive panel
            // layout.
            double left = std::clamp(std::floor((view->getWidth() - 20) * 0.19),
                                     215.0, 290.0);
            double mid = std::clamp(std::floor((view->getWidth() - 20) * 0.27),
                                    300.0, 410.0);
            double proc = std::clamp(std::floor((view->getWidth() - 20) * 0.24),
                                     250.0, 360.0);
            double right = view->getWidth() - 50 - left - mid - proc;
            if (right < 330) {
                double deficit = 330 - right;
                const double a =
                    std::min(deficit * 0.55, std::max(0.0, proc - 235));
                proc -= a;
                deficit -= a;
                const double b =
                    std::min(deficit * 0.70, std::max(0.0, mid - 260));
                mid -= b;
                deficit -= b;
                left -= std::min(deficit, std::max(0.0, left - 190));
                right = view->getWidth() - 50 - left - mid - proc;
            }
            if (right > 410) {
                proc += std::min(right - 410, std::max(0.0, 360 - proc));
                right = view->getWidth() - 50 - left - mid - proc;
            }
            const double buttonWidth =
                std::floor((std::max(160.0, right - 28) - 10) * 0.5);
            const double oldMode =
                parameter(17).convertFrom0to1(parameter(17).getValue());
            click(*view,
                  {(float)(30 + left + mid + 14 + buttonWidth * 0.5), 161});
            require(parameter(17).convertFrom0to1(parameter(17).getValue()) ==
                        std::fmod(oldMode + 1, 4),
                    "Canvas playback click did not reach host parameter");
            parameter(6).setValueNotifyingHost(parameter(6).convertTo0to1(5));
            parameter(55).setValueNotifyingHost(
                parameter(55).convertTo0to1(0.3f));
            pump(200);
            require(std::abs(read("color_tilt_db") - 5) < 0.1 &&
                        std::abs(read("output_harmonics_amt") - 0.3) < 0.01,
                    "Color/Harmonics host edits did not reach DSP");
            if (native) {
                until(
                    [&] {
                        return std::abs(ui("color_tilt_db") - 5) < 0.1 &&
                               std::abs(ui("output_harmonics_amt") - 0.3) <
                                   0.01;
                    },
                    "UI mirrors ignored host automation");
                wheel(*view, knob(0), juce::ModifierKeys::shiftModifier);
                require(actual(47) > 0 && ui("formant_shift_semi") > 0,
                        "Hidden Formant wheel did not commit");
                set(59, 3 + 32 * 10);
                until([&] { return ui("v71_profile") == 3; },
                      "Packed profile host automation missing");
                wheel(*view, knob(7), juce::ModifierKeys::shiftModifier);
                until([&] { return actual(59) == 3 + 32 * 11; },
                      "Thrill wheel corrupted packed profile");
                set(59, 5 + 32 * 11);
                until([&] { return ui("v71_profile") == 5; },
                      "Host edit stuck behind local preview acknowledgement");
                // Actual JUCE menu overlay, keyboard selection, and audio
                // running during a modal call.
                const auto point = knob(12);
                view->mouseDown(event(*view, point,
                                      juce::ModifierKeys::leftButtonModifier |
                                          juce::ModifierKeys::ctrlModifier));
                until(
                    [&] {
                        return view->getNumChildComponents() &&
                               view->getChildComponent(0)->isVisible();
                    },
                    "Native profile menu did not open");
                auto *menu = view->getChildComponent(0);
                save(*editor, directory, "sample-native-menu.png");
                const auto beforeMenu = blocks.load();
                pump(150);
                require(blocks.load() > beforeMenu + 5,
                        "Modal graphics menu stalled audio");
                menu->keyPressed(juce::KeyPress(juce::KeyPress::downKey));
                menu->keyPressed(juce::KeyPress(juce::KeyPress::downKey));
                menu->keyPressed(juce::KeyPress(juce::KeyPress::returnKey));
                view->mouseUp(event(*view, point, 0));
                until([&] { return actual(59) == 2 + 32 * 11; },
                      "Menu selection lost packed Thrill field");
                view->grabKeyboardFocus();
                view->focusGained(juce::Component::focusChangedDirectly);
                const juce::Point<float> band{(float)ui("b2_x"),
                                              (float)ui("b2_y")};
                view->mouseDown(event(*view, band,
                                      juce::ModifierKeys::leftButtonModifier |
                                          juce::ModifierKeys::ctrlModifier));
                until([&] { return read("posteq_solo_target") == 3; },
                      "Native EQ solo did not reach audio boundary");
                save(*editor, directory, "sample-native-solo.png");
                view->focusLost(juce::Component::focusChangedDirectly);
                until([&] { return read("posteq_solo_target") == 0; },
                      "EQ solo stuck after focus loss");
                view->mouseUp(event(*view, band, 0));
                hubRequest = 1;
                until(
                    [&] {
                        return read("hub_log_accepted") > 0 &&
                               read("hub_log_rejected") > 0;
                    },
                    "Hub MIDI packets were not logged");
                require(!za_sample_gfx_faulted(processor.get()),
                        "Native publication address fault");
                save(*editor, directory, "sample-native-controls.png");
            }
            click(*view,
                  {(float)(std::max(450, view->getWidth() - 352) + 50), 39});
            waitForFrame(*editor);
            save(*editor, directory, "sample-midi-log.png");
            editor->setSize(1140, 900);
            waitForFrame(*editor);
            save(*editor, directory, "sample-log-resized.png");
        } else {
            require(read("sample_count") == 3, "Bank lost on editor reopen");
            if (native)
                require(ui("hub_ui_open") == 1, "MIDI log view lost on reopen");
            save(*editor, directory, "sample-reopened.png");
            if (native) {
                auto *view = canvas(*editor);
                click(
                    *view,
                    {(float)(std::max(450, view->getWidth() - 352) + 50), 39});
                until([&] { return ui("hub_ui_open") == 0; },
                      "Could not close MIDI log");
                // Grow past SAMPLE_POOL_MIN=64: all dependent heap bases
                // relocate.
                std::vector<std::string> larger = bank;
                for (int i = 3; i < 65; ++i) {
                    const auto file = directory.getChildFile(
                        "grow-" + juce::String(i) + ".wav");
                    require(juce::File(bank[(size_t)i % 3]).copyFileTo(file),
                            "Cannot extend bank fixture");
                    larger.push_back(file.getFullPathName().toStdString());
                }
                std::vector<const char *> paths;
                for (const auto &path : larger)
                    paths.push_back(path.c_str());
                za_sample_gfx_load_bank(processor.get(), paths.data(),
                                        (int)paths.size());
                until(
                    [&] {
                        return read("sample_count") == 65 &&
                               ui("sample_count") == 65;
                    },
                    "Native publication did not follow resized bank layout");
                save(*editor, directory, "sample-native-grown-bank.png");
                paths.resize(1);
                za_sample_gfx_load_bank(processor.get(), paths.data(), 1);
                until(
                    [&] {
                        return read("sample_count") == 1 &&
                               ui("sample_count") == 1;
                    },
                    "Native publication did not follow shrunken bank");
                stop = true;
                audio.join();
                processor->releaseResources();
                processor->prepareToPlay(44100, 512);
                stop = false;
                audio = std::thread(runAudio);
                until([&] { return ui("sample_count") == 1; },
                      "Reprepare lost native bank publication");
                save(*editor, directory, "sample-native-reprepared.png");
                view->grabKeyboardFocus();
                view->focusGained(juce::Component::focusChangedDirectly);
                const juce::Point<float> heldBand{(float)ui("b2_x"),
                                                  (float)ui("b2_y")};
                view->mouseDown(event(*view, heldBand,
                                      juce::ModifierKeys::leftButtonModifier |
                                          juce::ModifierKeys::ctrlModifier));
                until([&] { return read("posteq_solo_target") == 3; },
                      "Could not arm solo after reprepare");
                editor->setVisible(false);
                until([&] { return read("posteq_solo_target") == 0; },
                      "Solo stuck while editor hidden");
                view->mouseUp(event(*view, heldBand, 0));
                editor->setVisible(true);
                pump(150);
                // Close while the native worker is blocked in a real modal
                // menu.
                view->mouseDown(event(*view, knob(12),
                                      juce::ModifierKeys::leftButtonModifier |
                                          juce::ModifierKeys::ctrlModifier));
                until([&] { return view->getChildComponent(0)->isVisible(); },
                      "Close-menu fixture did not open");
            }
        }
        editor->setVisible(false);
        editor->removeFromDesktop();
        editor.reset();
        pump(100);
        const auto closed = za_sample_gfx_frames();
        pump(150);
        require(za_sample_gfx_frames() == closed,
                "Sample worker continued after editor close");
    }
    require(!badAudio.load(), "Sample produced nonfinite audio");
    if (native) {
        require(!za_sample_gfx_faulted(processor.get()),
                "Native graphics faulted after bank resize/reprepare");
        require(read("posteq_solo_target") == 0, "EQ solo armed after close");
        require(gestures.begins > 0 && gestures.begins == gestures.ends,
                "Unbalanced native host gestures");
    }
    stop = true;
    audio.join();
    double value = 0;
    size_t cells = 0, capacity = 0;
    int64_t logical = 0;
    require(za_sample_gfx_snapshot(processor.get(), "sample_count", &value,
                                   &cells, &capacity, &logical),
            "No final snapshot");
    if (native)
        require(za_sample_gfx_legacy() ? cells == 0 && capacity == 0 : cells == 1412 && capacity == 66944,
                "Native publication capacity drifted with DSP heap size");
    uint64_t copyCount = 0;
    double copyTotal = 0, copyMax = 0;
    za_sample_gfx_copy_stats(processor.get(), &copyCount, &copyTotal, &copyMax);
    std::cout << "PASS " << (native ? "native" : "EEL")
              << " Sample editor: bank=3; MIDI peak=" << peak.load()
              << "; canvas mode edit; Color/Harmonics automation; log; resize; "
                 "2 lifetimes; blocks="
              << blocks.load() << " frames=" << za_sample_gfx_frames()
              << " logical_heap_cells=" << logical
              << " snapshot_cells=" << cells
              << " snapshot_capacity_cells=" << capacity
              << " publication_mean_ms="
              << (copyCount ? copyTotal / copyCount : 0)
              << " publication_max_ms=" << copyMax
              << " gestures=" << gestures.begins.load() << '/'
              << gestures.ends.load() << '\n';
    if (native) {
        constexpr int instanceCount = 4;
        std::vector<std::unique_ptr<juce::AudioProcessor>> processors;
        std::vector<std::unique_ptr<juce::AudioProcessorEditor>> editors;
        std::vector<const char *> paths;
        for (int i = 0; i < 3; ++i)
            paths.push_back(bank[(size_t)i].c_str());
        for (int i = 0; i < instanceCount; ++i) {
            processors.emplace_back(createPluginFilter());
            processors.back()->prepareToPlay(48000, 512);
            za_sample_gfx_load_bank(processors.back().get(), paths.data(),
                                    (int)paths.size());
            editors.emplace_back(processors.back()->createEditor());
            editors.back()->addToDesktop(
                juce::ComponentPeer::windowHasTitleBar);
            editors.back()->setVisible(true);
        }
        std::atomic<bool> multiStop{false}, multiBad{false};
        std::atomic<uint64_t> multiBlocks{0};
        std::thread multiAudio([&] {
            juce::AudioBuffer<float> buffer(2, 512);
            juce::MidiBuffer midi;
            while (!multiStop.load()) {
                for (auto &p : processors) {
                    buffer.clear();
                    midi.clear();
                    if (multiBlocks.load() == 100)
                        midi.addEvent(
                            juce::MidiMessage::noteOn(1, 60, (juce::uint8)100),
                            17);
                    p->processBlock(buffer, midi);
                    for (int ch = 0; ch < 2; ++ch)
                        for (int sample = 0; sample < 512; ++sample)
                            if (!std::isfinite(buffer.getSample(ch, sample)))
                                multiBad = true;
                }
                ++multiBlocks;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
        struct MultiJoin {
            std::atomic<bool> &stop;
            std::thread &thread;
            ~MultiJoin() {
                stop = true;
                if (thread.joinable())
                    thread.join();
            }
        } multiJoin{multiStop, multiAudio};
        for (auto &p : processors)
            until(
                [&] { return za_sample_gfx_ui(p.get(), "sample_count") == 3; },
                "Parallel native instance failed to adopt its bank");
        pump(1300);
        multiStop = true;
        multiAudio.join();
        require(!multiBad.load(),
                "Parallel instances produced nonfinite audio");
        uint64_t count = 0;
        double total = 0, maximum = 0;
        size_t totalCapacity = 0;
        for (auto &p : processors) {
            require(!za_sample_gfx_faulted(p.get()),
                    "Parallel instance publication fault");
            double sampleCount = 0;
            size_t copied = 0, allocated = 0;
            int64_t logicalSize = 0;
            require(
                za_sample_gfx_snapshot(p.get(), "sample_count", &sampleCount,
                                       &copied, &allocated, &logicalSize) &&
                    sampleCount == 3 && (za_sample_gfx_legacy() ? copied == 0 && allocated == 0 : copied == 1420 && allocated == 66944),
                "Parallel publications crossed instance state or capacity");
            totalCapacity += allocated * 3;
            uint64_t n = 0;
            double t = 0, m = 0;
            za_sample_gfx_copy_stats(p.get(), &n, &t, &m);
            count += n;
            total += t;
            maximum = std::max(maximum, m);
        }
        std::cout << "PASS " << instanceCount
                  << " simultaneous native Sample editors: cycles="
                  << multiBlocks.load()
                  << " total_publication_capacity_cells=" << totalCapacity
                  << " publication_mean_ms=" << (count ? total / count : 0)
                  << " publication_max_ms=" << maximum << '\n';
        editors.clear();
        for (auto &p : processors)
            p->releaseResources();
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
