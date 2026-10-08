// SPDX-License-Identifier: Zlib
// Native AOT @gfx versus existing EEL renderer, using the same draw backend.
#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#ifdef ZA_SHOWCASE_REAL_JUCE
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_graphics/juce_graphics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#else
#include "juce_contract_stub.h"
#endif
#include "YSFXGfxInterpreter.h" // Supplies the amalgamated draw types; include order matters.
#include "NativeGfxPrototype.h"
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

extern "C" void jsfx_ensure_mem(DSPJSFX_State *state, int64_t count) {
    if (count <= state->memN)
        return;
    auto *data =
        (double *)std::realloc(state->mem, (size_t)count * sizeof(double));
    if (!data)
        throw std::bad_alloc();
    std::fill(data + state->memN, data + count, 0.0);
    state->mem = data;
    state->memN = count;
}

static void savePpm(const char *path, const juce::Image &image) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << image.getWidth() << " " << image.getHeight() << "\n255\n";
    for (int y = 0; y < image.getHeight(); ++y)
        for (int x = 0; x < image.getWidth(); ++x) {
            const auto p = image.getPixelAt(x, y).getARGB();
            const char rgb[] = {(char)(p >> 16), (char)(p >> 8), (char)p};
            out.write(rgb, 3);
        }
}

int main(int argc, char **argv) try {
#ifdef ZA_SHOWCASE_REAL_JUCE
    juce::ScopedJuceInitialiser_GUI gui;
#endif
    if (argc != 3)
        throw std::runtime_error("Pass source and output prefix");
    std::ifstream input(argv[1]);
    std::ostringstream text;
    text << input.rdbuf();
    if (!input || text.str().empty())
        throw std::runtime_error("Missing JSFX source");

    DSPJSFX_State dsp{};za::jsfx::StateVariables dsp_variables;dsp_variables.bind(dsp,DSPJSFX_VARS_COUNT);
    dsp.srate = 48000;
    dsp.samplesblock = 512;
    const std::regex slider(
        R"(slider([0-9]+):(?:([A-Za-z_][A-Za-z_0-9]*)=)?(-?[0-9.]+)<)");
    const auto source = text.str();
    for (auto it = std::sregex_iterator(source.begin(), source.end(), slider);
         it != std::sregex_iterator(); ++it) {
        const double value = std::stod((*it)[3]);
        dsp.sliders[std::stoi((*it)[1]) - 1] = value;
        const int index =
            jsfx_native_gfx::Frame::findIndex((*it)[2].str().c_str());
        if (index >= 0)
            dsp.vars[index] = value;
    }
    jsfx_init(&dsp);
    jsfx_slider(&dsp);
    // readVars excludes graphics-owned outputs. Serialize test metrics inside
    // EEL instead of enabling audio writeback for those locals.
    std::vector<const char *> metricNames;
    std::string referenceSource = source;
    for (const char *name : {"mw", "mh", "only_width", "alias", "bw", "bh"})
        if (jsfx_native_gfx::Frame::findIndex(name) >= 0)
            metricNames.push_back(name);
    if (!metricNames.empty()) {
        referenceSource += "\nsprintf(#native_probe_metrics,\"";
        for (size_t i = 0; i < metricNames.size(); ++i)
            referenceSource += "%.17g ";
        referenceSource += "\"";
        for (const char *name : metricNames)
            referenceSource += std::string(",") + name;
        referenceSource += ");\n";
    }
    jsfx_gfx::Interpreter eel(referenceSource.c_str());
    if (!eel.gfxCompiledOk())
        throw std::runtime_error(eel.getLastError().toRawUTF8());
    jsfx_native_gfx::Frame native;
    const bool expander = jsfx_native_gfx::Frame::findIndex("det_db_m") >= 0;
    const auto set = [&](const char *name, double value) {
        const int i = jsfx_native_gfx::Frame::findIndex(name);
        if (i >= 0)
            dsp.vars[i] = value;
    };

    for (const auto &size : {std::pair<int, int>{DSPJSFX_NATIVE_GFX_WIDTH,
                                                 DSPJSFX_NATIVE_GFX_HEIGHT},
                             {DSPJSFX_NATIVE_GFX_WIDTH * 3 / 2,
                              DSPJSFX_NATIVE_GFX_HEIGHT * 3 / 2}}) {
        const int w = size.first, h = size.second;
        for (int scenario = 0; scenario < (expander ? 3 : 1); ++scenario) {
            if (expander) {
                set("det_db_m", scenario == 0   ? -120
                                : scenario == 1 ? -18
                                                : -60);
                set("gr_db_m", scenario == 2 ? 18 : 0);
                set("expanding", scenario == 2 ? 1 : 0);
                set("gain_db_s", scenario == 2 ? -12 : 0);
            }
            juce::Image nativeImage(juce::Image::ARGB, w, h, true);
            juce::Image eelImage(juce::Image::ARGB, w, h, true);
            const auto originalDsp = dsp;
            std::vector<double> originalHeap;
            if (dsp.mem)
                originalHeap.assign(dsp.mem, dsp.mem + dsp.memN);
            native.begin(dsp, w, h);
            native.run();
            if (std::memcmp(dsp.vars, originalDsp.vars, sizeof dsp.vars) != 0)
                throw std::runtime_error(
                    "Native GFX changed the audio-owned state");
            if (!originalHeap.empty() &&
                std::memcmp(dsp.mem, originalHeap.data(),
                            originalHeap.size() * sizeof(double)) != 0)
                throw std::runtime_error(
                    "Native GFX changed the audio-owned heap");
            jsfx_gfx::GfxRenderSession nativeRenderer(nativeImage);
            nativeRenderer.finish(native.commands);
            std::vector<jsfx_gfx::DrawCmd> withoutText;
            bool hasText = false;
            for (const auto &command : native.commands)
                if (command.type == jsfx_gfx::DrawCmd::Type::Text) {
                    hasText = true;
                    if (command.font.getStringWidthFloat(command.text) <= 0)
                        throw std::runtime_error(
                            "JUCE did not resolve a font for native text");
                } else
                    withoutText.push_back(command);
            juce::Image noTextImage(juce::Image::ARGB, w, h, true);
            jsfx_gfx::GfxRenderSession noTextRenderer(noTextImage);
            noTextRenderer.finish(withoutText);

            jsfx_gfx::Interpreter::Snapshot snap;
            snap.vars = dsp.vars;
            snap.varsCount = DSPJSFX_VARS_COUNT;
            snap.sliders = dsp.sliders;
            snap.slidersCount = DSPJSFX_MAX_SLIDERS;
            snap.srate = dsp.srate;
            snap.samplesblock = dsp.samplesblock;
            jsfx_gfx::GfxRenderSession eelRenderer(eelImage);
            auto binding = eel.bindRenderer(eelRenderer);
            eel.renderFrame(w, h, snap);
            eelRenderer.finish(eel.getCommands());

            // Sample's qualification scene exercises omitted outputs, helper
            // lvalues, multiline/bitmap metrics, and two outputs aliasing.
            juce::String eelMetrics;
            if (!metricNames.empty() &&
                !eel.readStringVarUtf8("#native_probe_metrics", eelMetrics))
                throw std::runtime_error("Missing EEL text metrics");
            std::istringstream metrics(eelMetrics.toStdString());
            for (const char *name : metricNames) {
                const int index = jsfx_native_gfx::Frame::findIndex(name);
                double value = 0;
                if (!(metrics >> value) || native.state.vars[index] != value)
                    throw std::runtime_error(
                        std::string("Text measurement differs from EEL: ") +
                        name);
            }

            int mismatches = 0, nonBlack = 0, textPixels = 0;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const auto a = nativeImage.getPixelAt(x, y).getARGB();
                    const auto b = eelImage.getPixelAt(x, y).getARGB();
                    mismatches += a != b;
                    nonBlack += (a & 0xffffffu) != 0;
                    textPixels += a != noTextImage.getPixelAt(x, y).getARGB();
                }
            const auto path =
                std::string(argv[2]) + "-" + std::to_string(w) + "x" +
                std::to_string(h) +
                (expander ? "-state" + std::to_string(scenario) : "") + ".ppm";
            savePpm(path.c_str(), nativeImage);
            std::cout << w << "x" << h << " state=" << scenario
                      << " commands=" << native.commands.size()
                      << " mismatched_pixels=" << mismatches
                      << " nonblack=" << nonBlack
                      << " text_pixels=" << textPixels << "\n";
            if (hasText && textPixels < 200)
                throw std::runtime_error(
                    "Text commands did not produce visible text");
            if (mismatches || nonBlack < w * h / 2)
                throw std::runtime_error(
                    "Native GFX frame differs from the EEL reference");
        }
    }
    // Runtime contracts which drawing parity alone cannot exercise.
    jsfx_native_gfx::Frame contract;
    contract.strings[10] = "Grüße 日本語";
    jsfx_native_strcpy(&contract.state, 11, 10);
    if (contract.string(11) != contract.string(10))
        throw std::runtime_error("UTF-8 string copy failed");
    jsfx_native_strcat(&contract.state, 11, 11);
    if (contract.string(11) != contract.string(10) + contract.string(10))
        throw std::runtime_error("Aliased string append failed");
    jsfx_native_strncpy(&contract.state, 12, 11, 3);
    if (contract.string(12) != contract.string(11).substr(0, 3))
        throw std::runtime_error("Byte-counted string clipping failed");
    contract.strings[13] = std::string(16383, 'x');
    jsfx_native_strcat(&contract.state, 13, 10);
    if (jsfx_native_strlen(&contract.state, 13) != 16383)
        throw std::runtime_error("String limit failed");
    contract.focused = contract.visible = true;
    contract.keys.push_back('a');
    contract.keysDown.insert('a');
    if (jsfx_native_gfx_getchar(&contract.state, 65536) != 7 ||
        jsfx_native_gfx_getchar(&contract.state, 'a') != 1 ||
        jsfx_native_gfx_getchar(&contract.state, 0) != 'a' ||
        jsfx_native_gfx_getchar(&contract.state, 0) != 0)
        throw std::runtime_error("Native input queue/window flags failed");
    jsfx_native_slider_event(&contract.state, 0, 255, 1, 0);
    jsfx_native_slider_event(&contract.state, 0, 255, 1, 1);
    if (!contract.automateMask.test(255) || !contract.automateEndMask.test(255))
        throw std::runtime_error("Direct high-slider gesture failed");
    jsfx_native_slider_event(&contract.state, 2305843009213693952.0, -1, 0, 0);
    if (!contract.changeMask.test(61))
        throw std::runtime_error("High numeric slider mask failed");
    if (jsfx_native_read_mem(&contract.state, 12345) != 0 ||
        !contract.memoryFault)
        throw std::runtime_error("Unpublished memory read did not fail closed");
    std::cout
        << "PASS native strings/input/gesture/memory-boundary contracts\n";
    std::free(dsp.mem);
    std::cout << "PASS native AOT @gfx prototype\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
