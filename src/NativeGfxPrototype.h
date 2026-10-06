// SPDX-License-Identifier: Zlib
// Include once after JSFXDSP.h and YSFXGfxInterpreter.h. Native @gfx runs
// on a private execution context. LEGACY binds guest state to the live DSP heap;
// the publication mode retains worker-owned variables and bounded read views.
#pragma once
#include "JsfxSharedCells.h"
#include <atomic>
#include <cstdio>
#include <climits>
#include <chrono>
#include <unordered_map>
#include <string_view>
#include <unordered_set>
#include <deque>
#include <sstream>
#include "NativeJsfxStrings.h"
#if !DSPJSFX_HAS_NATIVE_GFX
#error "Build this bridge only with --native-gfx-prototype"
#endif
namespace jsfx_native_gfx {
using jsfx_gfx::boundedGfxInt;
// Both real guest references and private computed arguments use aligned
// double storage. Do not expose a plain C++ load/store on a guest reference.
struct Reference {
    double* ptr = nullptr;
    Reference() = default;
    Reference(double* p) : ptr(p) {}
    explicit operator bool() const noexcept { return ptr != nullptr; }
    bool operator==(const double* p) const noexcept { return ptr == p; }
    struct Value {
        double* ptr;
        operator double() const noexcept { return jsfxCellLoad(ptr); }
        Value& operator=(double v) noexcept { jsfxCellStore(ptr,v); return *this; }
        Value& operator=(const Value& v) noexcept { return *this=double(v); }
    };
    Value operator*() const noexcept { return {ptr}; }
};
#if ZA_NATIVE_GFX_TEST_RUNNER || ZA_SAMPLE_GFX_TEST_RUNNER
inline std::atomic<uint64_t> executedFrames{0};
inline std::atomic<uint64_t> publishedFrames{0};
#endif
// Each slot owns fixed capacity. Publishing gathers only the declared fields;
// record addresses remain those of JSFX, but there is no EEL-compatible heap.
struct Publication {
    struct View {
        int64_t base = 0, stride = 1;
        int records = 0, fieldCount = 0;
        std::array<int, 16> fields{};
        std::vector<double> data;
    };
    std::array<View, DSPJSFX_NATIVE_GFX_VIEW_COUNT> views;
    size_t copiedCells = 0;
    bool invalid = false;
    void allocate() {
        for (size_t i = 0; i < views.size(); ++i)
            views[i].data.resize((size_t)DSPJSFX_NATIVE_VIEWS[i].maxRecords *
                                 DSPJSFX_NATIVE_VIEWS[i].fieldCount);
    }
    void publish(const DSPJSFX_State &state) noexcept {
        copiedCells = 0;
        invalid = false;
        auto value = [&](int index, int64_t literal) -> int64_t {
            if (index < 0)
                return literal;
            const double d = state.vars[index];
            if (!std::isfinite(d) || d < 0 || d >= 9223372036854775808.0) {
                invalid = true;
                return 0;
            }
            return (int64_t)d;
        };
        for (size_t i = 0; i < views.size(); ++i) {
            const auto &desc = DSPJSFX_NATIVE_VIEWS[i];
            auto &view = views[i];
            view.base = value(desc.baseVar, desc.base);
            const auto count = value(desc.countVar, desc.count);
            view.stride = value(desc.strideVar, desc.stride);
            view.fieldCount = desc.fieldCount;
            std::copy_n(desc.fields, desc.fieldCount, view.fields.begin());
            view.records = (int)std::min<int64_t>(count, desc.maxRecords);
            if (count > desc.maxRecords || view.stride < 1)
                invalid = true;
            for (int f = 0; f < view.fieldCount; ++f)
                if (view.fields[(size_t)f] >= view.stride)
                    invalid = true;
            for (int r = 0; r < view.records; ++r)
                for (int f = 0; f < view.fieldCount; ++f) {
                    const auto field = view.fields[(size_t)f];
                    auto &dest = view.data[(size_t)r * view.fieldCount + f];
                    // Check multiplication and addition before forming an
                    // address.
                    const bool valid =
                        state.mem && view.stride > 0 &&
                        view.base < state.memN &&
                        field < state.memN - view.base &&
                        r <= (state.memN - 1 - view.base - field) / view.stride;
                    if (valid)
                        dest = state.mem[view.base + r * view.stride + field];
                    else {
                        dest = 0;
                        invalid = true;
                    }
                    ++copiedCells;
                }
        }
    }
    bool read(int64_t address, double &result) const noexcept {
        for (const auto &view : views) {
            if (view.stride < 1 || address < view.base)
                continue;
            const auto relative = address - view.base;
            const auto record = relative / view.stride;
            const auto field = relative % view.stride;
            if (record >= view.records)
                continue;
            for (int f = 0; f < view.fieldCount; ++f)
                if (field == view.fields[(size_t)f]) {
                    result = view.data[(size_t)record * view.fieldCount + f];
                    return true;
                }
        }
        return false;
    }
    size_t capacity() const noexcept {
        size_t result = 0;
        for (const auto &view : views)
            result += view.data.capacity();
        return result;
    }
};

class Frame {
  public:
    Frame() {
        for (int i = 0; i < DSPJSFX_VARS_COUNT; ++i)
            if (DSPJSFX_GFX_VAR_FLAGS[i] & DSPJSFX_GFX_VAR_FLAG_TO_GFX)
                publishedIndices.push_back(i);
        for (const char *name : {"gfx_r", "gfx_g", "gfx_b", "gfx_a", "gfx_a2"})
            set(name, 1);
        set("gfx_dest", -1);
        set("gfx_texth", 8);
        for (const auto &s : DSPJSFX_STRING_LITERALS)
            strings.emplace((double)s.handle, std::string((const char *)s.data,
                                                          (size_t)s.length));
        state.hostOwner = this;
        fileFormatManager.registerBasicFormats();
#if DSPJSFX_NATIVE_GFX_LEGACY
        state.nativeStrings = &ownedStrings;
        ownedStrings.graphics = this;
#endif
        commands.reserve(64);
    }
    DSPJSFX_State state{};
    const Publication *publication =
        nullptr; // Valid only while the slot is pinned.
    bool memoryFault = false, focused = false, visible = false;
    jsfx_gfx::AsyncMenuPort *menuPort = nullptr;
    jsfx_gfx::SliderMask changeMask, automateMask, automateEndMask;
    std::deque<int> keys;
    std::unordered_set<int> keysDown;
    std::vector<jsfx_gfx::DrawCmd> commands;
    jsfx_gfx::GfxImageBank imageBank;
    std::vector<int> publishedIndices;
    std::unordered_map<double, std::string> strings;
#if DSPJSFX_NATIVE_GFX_LEGACY
    Strings ownedStrings;
#endif
    struct FontSlot {
        juce::Font font;
        juce::String name;
        float size = 0;
        int flags = 0;
        uint64_t serial = 0;
    };
    std::array<FontSlot, 17> fonts;
    std::vector<FontSlot> fontCache;
    int currentFont = 0, frameWidth = 0, frameHeight = 0;
    void begin(const double *vars, int count, const double *sliders,
               double rate, double block, int width, int height) {
        commands.clear();
        changeMask.clear();
        automateMask.clear();
        automateEndMask.clear();
        // Keep graphics-owned variables and persistent helper locals across
        // frames.
        for (int i : publishedIndices)
            if (i < count)
                state.vars[i] = vars[i];
        jsfxWriteCells(state.sliders, sliders, DSPJSFX_MAX_SLIDERS);
        state.srate = rate;
        state.samplesblock = block;
        state.mem = nullptr;
        state.memN = 0;
        frameWidth = width;
        frameHeight = height;
        set("gfx_w", width);
        set("gfx_h", height);
        set("gfx_a", 1);
        set("gfx_a2", 1);
        set("gfx_dest", -1);
        beginDrawingFrame();
    }
    void begin(const DSPJSFX_State &s, int w, int h) {
#if DSPJSFX_HAS_TASKS
        state.taskContext = s.taskContext;
#endif
#if DSPJSFX_NATIVE_GFX_LEGACY
        bindLegacy(const_cast<DSPJSFX_State&>(s), w, h);
#else
        begin(s.vars, DSPJSFX_VARS_COUNT, s.sliders, s.srate, s.samplesblock, w,
              h);
#endif
    }
#if DSPJSFX_NATIVE_GFX_LEGACY
    void bindLegacy(DSPJSFX_State& shared, int w, int h) {
#if DSPJSFX_HAS_TASKS
        state.taskContext = shared.taskContext;
#endif
        if(state.sharedState != &shared)state.sharedState = &shared;
        state.atomicContext = shared.atomicContext;
        state.mem = shared.mem;
        state.memN = shared.memN;
        state.runtimeOpaque = shared.runtimeOpaque;
        if(state.nativeStrings != shared.nativeStrings)state.nativeStrings = shared.nativeStrings;
        state.memUsed = shared.memN;
        commands.clear(); changeMask.clear(); automateMask.clear(); automateEndMask.clear();
        frameWidth = w; frameHeight = h;
        set("gfx_w", w); set("gfx_h", h);
        set("gfx_a", 1); set("gfx_a2", 1); set("gfx_dest", -1);
        beginDrawingFrame();
    }
#endif
    DSPJSFX_State& scriptState() noexcept {
#if DSPJSFX_NATIVE_GFX_LEGACY
        if (state.sharedState) return *static_cast<DSPJSFX_State*>(state.sharedState);
#endif
        return state;
    }
    const DSPJSFX_State& scriptState() const noexcept {
        return const_cast<Frame*>(this)->scriptState();
    }
    void run() {
        jsfx_gfx_aot(&state);
#if ZA_NATIVE_GFX_TEST_RUNNER || ZA_SAMPLE_GFX_TEST_RUNNER
        executedFrames.fetch_add(1, std::memory_order_release);
#endif
    }
    static int findIndex(const char *name) noexcept {
        static const auto indices = [] {
            std::unordered_map<std::string_view, int> result;
            result.reserve(DSPJSFX_VARS_COUNT);
            for (const auto &var : DSPJSFX_VARS)
                result.emplace(var.name, var.index);
            return result;
        }();
        const auto it = indices.find(name);
        return it == indices.end() ? -1 : it->second;
    }
    void set(const char *name, double v) noexcept {
        const int i = findIndex(name);
        if (i < 0) return;
#if DSPJSFX_NATIVE_GFX_LEGACY
        const int alias = DSPJSFX_LEGACY_SLIDER_ALIASES[i];
        if (alias >= 0) { scriptState().sliders[alias] = v; return; }
#endif
        scriptState().vars[i] = v;
    }
    double get(const char *name) const noexcept {
        const int i = findIndex(name);
#if DSPJSFX_NATIVE_GFX_LEGACY
        if (i >= 0 && DSPJSFX_LEGACY_SLIDER_ALIASES[i] >= 0)
            return scriptState().sliders[DSPJSFX_LEGACY_SLIDER_ALIASES[i]];
#endif
        return i >= 0 ? double(scriptState().vars[i]) : 0;
    }
    double* cellPointer(const char* name) noexcept {
        const int i=findIndex(name);
        if(i<0)return nullptr;
#if DSPJSFX_NATIVE_GFX_LEGACY
        const int alias=DSPJSFX_LEGACY_SLIDER_ALIASES[i];
        if(alias>=0)return &scriptState().sliders[alias];
#endif
        return &scriptState().vars[i];
    }
    std::string string(double handle) const {
#if DSPJSFX_NATIVE_GFX_LEGACY
        if(auto* context=stringContext(const_cast<DSPJSFX_State*>(&state)))return context->read(handle);
#endif
        static const std::string empty;
        const auto it = strings.find(handle);
        return it == strings.end() ? empty : it->second;
    }
    void writeString(double handle,std::string text) {
#if DSPJSFX_NATIVE_GFX_LEGACY
        if(auto* context=stringContext(&state)){context->write(handle,std::move(text));return;}
#endif
        strings[handle]=std::move(text);
    }
    jsfx_gfx::DrawCmd command(jsfx_gfx::DrawCmd::Type type) {
        jsfx_gfx::DrawCmd c;
        c.type = type;
        c.dest = -1;
        c.imageBank = &imageBank;
        c.frameWidth = frameWidth;
        c.frameHeight = frameHeight;
        const auto channel = [](double v) -> uint32_t {
            const float f = (float)v;
            return std::isfinite(f)
                       ? (uint32_t)(std::clamp(f, 0.0f, 1.0f) * 255.0f)
                       : 0;
        };
        c.colour = juce::Colour(
            (channel(get("gfx_a2")) << 24) | (channel(get("gfx_r")) << 16) |
            (channel(get("gfx_g")) << 8) | channel(get("gfx_b")));
        const double a = get("gfx_a");
        c.opacity = std::isfinite(a) ? (float)std::clamp(a, -8.0, 8.0) : 0;
        return c;
    }
#include "NativeGfxDrawing.inc"
#include "NativeGfxFiles.inc"
#if DSPJSFX_NATIVE_GFX_LEGACY
#include "NativeJsfxMatch.inc"
#endif
};
static Frame *owner(DSPJSFX_State *s) noexcept {
#if DSPJSFX_NATIVE_GFX_LEGACY
    if(auto* context=stringContext(s))return context->graphics;
#endif
    return s ? static_cast<Frame *>(s->hostOwner) : nullptr;
}
} // namespace jsfx_native_gfx
extern "C" double jsfx_native_gfx_set(DSPJSFX_State *s, double r, double g,
                                      double b, double a) {
    if (auto *f = jsfx_native_gfx::owner(s)) {
        f->set("gfx_r", r);
        f->set("gfx_g", g);
        f->set("gfx_b", b);
        f->set("gfx_a", a);
        f->set("gfx_a2", 1);
        f->set("gfx_mode", 0);
    }
    return 0;
}
extern "C" double jsfx_native_gfx_rect(DSPJSFX_State *s, double x, double y,
                                       double w, double h, double fill) {
    auto *f = jsfx_native_gfx::owner(s);
    if (!f || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(w) ||
        !std::isfinite(h) || std::floor(w) <= 0 || std::floor(h) <= 0)
        return 0;
    auto c = f->command(jsfx_gfx::DrawCmd::Type::Rect);
    c.x = (float)std::floor(x);
    c.y = (float)std::floor(y);
    c.w = (float)std::floor(w);
    c.h = (float)std::floor(h);
    c.fill = fill > 0.5;
    f->commands.push_back(std::move(c));
    return 0;
}
extern "C" double jsfx_native_gfx_circle(DSPJSFX_State *s, double x, double y,
                                         double radius, double fill,
                                         double aa) {
    auto *f = jsfx_native_gfx::owner(s);
    if (!f || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(radius))
        return 0;
    auto c = f->command(jsfx_gfx::DrawCmd::Type::Circle);
    c.x = (float)x;
    c.y = (float)y;
    c.radius = (float)radius;
    c.fill = fill > 0.5;
    c.antialias = aa > 0.5;
    f->commands.push_back(std::move(c));
    return 0;
}
extern "C" double jsfx_native_gfx_line(DSPJSFX_State *s, double x, double y,
                                       double x2, double y2, double aa) {
    auto *f = jsfx_native_gfx::owner(s);
    if (!f || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(x2) ||
        !std::isfinite(y2))
        return 0;
    auto c = f->command(jsfx_gfx::DrawCmd::Type::Line);
    c.x = (float)std::floor(x);
    c.y = (float)std::floor(y);
    c.x2 = (float)std::floor(x2);
    c.y2 = (float)std::floor(y2);
    c.antialias = aa >= 0.5;
    f->commands.push_back(std::move(c));
    return 0;
}
extern "C" double jsfx_native_gfx_setfont(DSPJSFX_State *s, double rawId,
                                          double name, double size,
                                          double packedFlags, double argc) {
    auto *f = jsfx_native_gfx::owner(s);
    if (!f)
        return 0;
    const int id = jsfx_gfx::boundedGfxInt(std::floor(rawId));
    if (id < 1 || id > 16) {
        f->currentFont = 0;
        f->set("gfx_texth", 8);
        return 1;
    }
    auto &slot = f->fonts[(size_t)id];
    if (argc > 1) {
        const auto &bytes = f->string(name);
        const auto family =
            juce::String::fromUTF8(bytes.empty() ? "Arial" : bytes.c_str());
        const float height = argc > 2 && std::isfinite(size)
                                 ? (float)std::clamp(size, 1.0, 2048.0)
                                 : 10;
        unsigned int packed =
            std::isfinite(packedFlags)
                ? (unsigned int)std::fmod(std::max(0.0, packedFlags),
                                          4294967296.0)
                : 0;
        int flags = juce::Font::plain;
        while (packed) {
            switch (std::toupper((int)(packed & 255u))) {
            case 'B':
                flags |= juce::Font::bold;
                break;
            case 'I':
                flags |= juce::Font::italic;
                break;
            case 'U':
                flags |= juce::Font::underlined;
                break;
            default:
                break;
            }
            packed >>= 8;
        }
        if (!slot.serial || !(slot.name == family) || slot.size != height ||
            slot.flags != flags) {
            for (const auto &cached : f->fontCache)
                if (cached.name == family && cached.size == height &&
                    cached.flags == flags) {
                    slot = cached;
                    f->currentFont = id;
                    f->set("gfx_texth", slot.font.getHeight());
                    return 1;
                }
            static std::atomic<uint64_t> serial{1ULL << 48};
            slot.font = jsfx_gfx::makeGfxFont(family, height, flags);
            slot.name = family;
            slot.size = height;
            slot.flags = flags;
            slot.serial = serial.fetch_add(1, std::memory_order_relaxed);
            if (f->fontCache.size() == 64)
                f->fontCache.erase(f->fontCache.begin());
            f->fontCache.push_back(slot);
        }
    }
    f->currentFont = slot.serial ? id : 0;
    f->set("gfx_texth", f->currentFont ? slot.font.getHeight() : 8);
    return 1;
}
extern "C" double jsfx_native_gfx_measurestr(DSPJSFX_State *s, double handle,
                                             double *width, double *height) {
    auto *f = jsfx_native_gfx::owner(s);
    if (!f)
        return 0;
    const auto text = juce::String::fromUTF8(f->string(handle).c_str());
    double w = 0, h = 0;
    if (!f->currentFont) {
        int bitmapWidth = 0, bitmapHeight = 0;
        LICE_MeasureText(text.toRawUTF8(), &bitmapWidth, &bitmapHeight);
        w = bitmapWidth;
        h = bitmapHeight;
    } else {
        juce::StringArray lines;
        lines.addLines(text);
        const auto &font = f->fonts[(size_t)f->currentFont].font;
        if (lines.isEmpty())
            w = font.getStringWidthFloat(text);
        else
            for (const auto &line : lines)
                w = std::max(w, (double)font.getStringWidthFloat(line));
        h = std::max(1, lines.size()) * font.getHeight();
    }
    if (width)
        jsfxCellStore(width, w);
    if (height)
        jsfxCellStore(height, h);
    return handle;
}
extern "C" double jsfx_native_gfx_drawstr(DSPJSFX_State *s, double handle) {
    auto *f = jsfx_native_gfx::owner(s);
    if (!f)
        return 0;
    const auto text = juce::String::fromUTF8(f->string(handle).c_str());
    if (text.isEmpty())
        return handle;
    auto c = f->command(jsfx_gfx::DrawCmd::Type::Text);
    const auto &font = f->fonts[(size_t)f->currentFont];
    c.text = text;
    c.font = font.font;
    c.fontSerial = font.serial;
    c.bitmapFont = f->currentFont == 0;
    c.x = (float)jsfx_gfx::boundedGfxInt(std::floor(f->get("gfx_x")));
    c.y = (float)jsfx_gfx::boundedGfxInt(std::floor(f->get("gfx_y")));
    f->commands.push_back(std::move(c));
    juce::StringArray lines;
    lines.addLines(text);
    const int n = std::max(1, lines.size());
    const auto last = lines.isEmpty() ? text : lines[n - 1];
    int width = 0, height = 0;
    if (!f->currentFont)
        LICE_MeasureText(last.toRawUTF8(), &width, &height);
    f->set("gfx_x", f->get("gfx_x") + (f->currentFont
                                           ? font.font.getStringWidthFloat(last)
                                           : (float)width));
    f->set("gfx_y", f->get("gfx_y") +
                        (n - 1) * (f->currentFont ? font.font.getHeight() : 8));
    return handle;
}
extern "C" double jsfx_native_sprintf(DSPJSFX_State *s, double dest,
                                      double format, const double *args,
                                      int32_t count) {
    auto *f = jsfx_native_gfx::owner(s);
    if (!f)
        return 0;
    const auto fmt = f->string(format);
    std::string result;
    result.reserve(256);
    int arg = 0;
    for (size_t i = 0; i < fmt.size() && result.size() < 16384;) {
        if (fmt[i] != '%') {
            result += fmt[i++];
            continue;
        }
        const size_t start = i++;
        if (i < fmt.size() && fmt[i] == '%') {
            result += '%';
            ++i;
            continue;
        }
        while (i < fmt.size() && std::strchr("-+ 0#0123456789.", fmt[i]))
            ++i;
        if (i == fmt.size() || arg >= count)
            break;
        const char type = fmt[i++];
        const auto spec = fmt.substr(start, i - start);
        char buffer[2048]{};
        const double value = args[arg++];
        if (type == 's')
            ::snprintf(buffer, sizeof buffer, spec.c_str(),
                          f->string(value).c_str());
        else if (std::strchr("feEgG", type))
            ::snprintf(buffer, sizeof buffer, spec.c_str(), value);
        else if (type == 'd' || type == 'i' || type == 'c')
            ::snprintf(buffer, sizeof buffer, spec.c_str(),
                          jsfx_gfx::boundedGfxInt(value));
        else if (type == 'u' || type == 'x' || type == 'X')
            ::snprintf(buffer, sizeof buffer, spec.c_str(),
                          (unsigned int)jsfx_gfx::boundedGfxInt(value));
        else
            break;
        result += buffer;
    }
    f->strings[dest] = std::move(result);
    return dest;
}

extern "C" double jsfx_native_read_mem(DSPJSFX_State *s, int64_t address) {
    auto *f = jsfx_native_gfx::owner(s);
    double value = 0;
    if (!f || !f->publication || !f->publication->read(address, value)) {
        if (f)
            f->memoryFault = true;
        return 0;
    }
    return value;
}
extern "C" double jsfx_native_strcpy(DSPJSFX_State *s, double dest,
                                     double source) {
    if (auto *f = jsfx_native_gfx::owner(s))
        f->strings[dest] = f->string(source).substr(0, 16383);
    return dest;
}
extern "C" double jsfx_native_strncpy(DSPJSFX_State *s, double dest,
                                      double source, double count) {
    if (auto *f = jsfx_native_gfx::owner(s)) {
        const size_t n =
            std::isfinite(count) ? (size_t)std::clamp(count, 0.0, 16383.0) : 0;
        f->strings[dest] = f->string(source).substr(0, n);
    }
    return dest;
}
extern "C" double jsfx_native_strcat(DSPJSFX_State *s, double dest,
                                     double source) {
    if (auto *f = jsfx_native_gfx::owner(s)) {
        const auto copy =
            f->string(source); // dest==source and map rehash are safe.
        auto &output = f->strings[dest];
        output +=
            copy.substr(0, 16383 - std::min<size_t>(output.size(), 16383));
    }
    return dest;
}
extern "C" double jsfx_native_strlen(DSPJSFX_State *s, double handle) {
    if (auto *f = jsfx_native_gfx::owner(s))
        return (double)f->string(handle).size();
    return 0;
}
extern "C" double jsfx_native_time_precise(DSPJSFX_State *) {
    return std::chrono::duration<double>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}
extern "C" double jsfx_native_gfx_getchar(DSPJSFX_State *s, double code) {
    auto *f = jsfx_native_gfx::owner(s);
    if (!f)
        return 0;
    if (code == 65536)
        return 1 | (f->focused ? 2 : 0) | (f->visible ? 4 : 0);
    if (code != 0)
        return std::isfinite(code) && code >= INT_MIN && code <= INT_MAX &&
                       f->keysDown.count((int)code)
                   ? 1
                   : 0;
    if (f->keys.empty())
        return 0;
    const auto result = f->keys.front();
    f->keys.pop_front();
    return result;
}
extern "C" double jsfx_native_gfx_showmenu(DSPJSFX_State *s, double handle) {
    auto *f = jsfx_native_gfx::owner(s);
    return f && f->menuPort
               ? f->menuPort->showMenuModal(
                     juce::String::fromUTF8(f->string(handle).c_str()),
                     (int)f->get("gfx_x"), (int)f->get("gfx_y"))
               : 0;
}
extern "C" double jsfx_native_slider_event(DSPJSFX_State *s, double value,
                                           int32_t index, int32_t automate,
                                           double end) {
    auto *f = jsfx_native_gfx::owner(s);
    if (!f)
        return 0;
    jsfx_gfx::SliderMask mask;
    if (index >= 0 && index < DSPJSFX_MAX_SLIDERS)
        mask.set(index);
    else if (std::isfinite(value)) {
        const double rounded = std::round(value);
        if (rounded >= 0 && rounded < 18446744073709551616.0)
            mask.words[0] = (uint64_t)rounded;
    }
    if (automate) {
        if (end != 0)
            f->automateEndMask.merge(mask);
        else
            f->automateMask.merge(mask);
    } else
        f->changeMask.merge(mask);
    return 0;
}

#if DSPJSFX_NATIVE_GFX_LEGACY
extern "C" double jsfx_native_gfx_dispatch(DSPJSFX_State* state,int32_t opcode,double** args,int32_t count) {
    auto* frame=jsfx_native_gfx::owner(state);
    if(!frame || count<0 || count>1024)return 0;
    std::array<jsfx_native_gfx::Reference,32> smallval;
    std::vector<jsfx_native_gfx::Reference> large;
    if(count>32)large.resize((size_t)count);
    auto* refs=count>32?large.data():smallval.data();
    for(int i=0;i<count;++i)refs[i]=args[i];
    switch(opcode) {
    case DSPJSFX_GFX_SET: return jsfx_native_gfx::Frame::eel_gfx_set(frame,count,refs);
    case DSPJSFX_GFX_RECT: return jsfx_native_gfx::Frame::eel_gfx_rect(frame,count,refs);
    case DSPJSFX_GFX_RECTTO: return jsfx_native_gfx::Frame::eel_gfx_rectto(frame,count,refs);
    case DSPJSFX_GFX_LINE: return jsfx_native_gfx::Frame::eel_gfx_line(frame,count,refs);
    case DSPJSFX_GFX_LINETO: return jsfx_native_gfx::Frame::eel_gfx_lineto(frame,count,refs);
    case DSPJSFX_GFX_CIRCLE: return jsfx_native_gfx::Frame::eel_gfx_circle(frame,count,refs);
    case DSPJSFX_GFX_ROUNDRECT: return jsfx_native_gfx::Frame::eel_gfx_roundrect(frame,count,refs);
    case DSPJSFX_GFX_ARC: return jsfx_native_gfx::Frame::eel_gfx_arc(frame,count,refs);
    case DSPJSFX_GFX_TRIANGLE: return jsfx_native_gfx::Frame::eel_gfx_triangle(frame,count,refs);
    case DSPJSFX_GFX_SETIMGDIM: return jsfx_native_gfx::Frame::eel_gfx_setimgdim(frame,count,refs);
    case DSPJSFX_GFX_GETIMGDIM: return jsfx_native_gfx::Frame::eel_gfx_getimgdim(frame,count,refs);
    case DSPJSFX_GFX_LOADIMG: return jsfx_native_gfx::Frame::eel_gfx_loadimg(frame,count,refs);
    case DSPJSFX_GFX_BLIT: return jsfx_native_gfx::Frame::eel_gfx_blit(frame,count,refs);
    case DSPJSFX_GFX_BLITEXT: return jsfx_native_gfx::Frame::eel_gfx_blitext(frame,count,refs);
    case DSPJSFX_GFX_DELTABLIT: return jsfx_native_gfx::Frame::eel_gfx_deltablit(frame,count,refs);
    case DSPJSFX_GFX_TRANSFORMBLIT: return jsfx_native_gfx::Frame::eel_gfx_transformblit(frame,count,refs);
    case DSPJSFX_GFX_SETPIXEL: return jsfx_native_gfx::Frame::eel_gfx_setpixel(frame,count,refs);
    case DSPJSFX_GFX_GETPIXEL: return jsfx_native_gfx::Frame::eel_gfx_getpixel(frame,count,refs);
    case DSPJSFX_GFX_GRADRECT: return jsfx_native_gfx::Frame::eel_gfx_gradrect(frame,count,refs);
    case DSPJSFX_GFX_MULADDRECT: return jsfx_native_gfx::Frame::eel_gfx_muladdrect(frame,count,refs);
    case DSPJSFX_GFX_BLURTO: return jsfx_native_gfx::Frame::eel_gfx_blurto(frame,count,refs);
    case DSPJSFX_GFX_DRAWSTR: return jsfx_native_gfx::Frame::eel_gfx_drawstr(frame,count,refs);
    case DSPJSFX_GFX_PRINTF: return jsfx_native_gfx::Frame::eel_gfx_printf(frame,count,refs);
    case DSPJSFX_GFX_MEASURESTR: return jsfx_native_gfx::Frame::eel_gfx_measurestr(frame,count,refs);
    case DSPJSFX_GFX_DRAWCHAR: return jsfx_native_gfx::Frame::eel_gfx_drawchar(frame,count,refs);
    case DSPJSFX_GFX_DRAWNUMBER: return jsfx_native_gfx::Frame::eel_gfx_drawnumber(frame,count,refs);
    case DSPJSFX_GFX_MEASURECHAR: return jsfx_native_gfx::Frame::eel_gfx_measurechar(frame,count,refs);
    case DSPJSFX_GFX_SETFONT: return jsfx_native_gfx::Frame::eel_gfx_setfont(frame,count,refs);
    case DSPJSFX_GFX_GETFONT: return jsfx_native_gfx::Frame::eel_gfx_getfont(frame,count,refs);
    case DSPJSFX_GFX_SETCURSOR: return jsfx_native_gfx::Frame::eel_gfx_setcursor(frame,count,refs);
    case DSPJSFX_GFX_GETDROPFILE: return jsfx_native_gfx::Frame::eel_gfx_getdropfile(frame,count,refs);
    case DSPJSFX_FILE_OPEN: return jsfx_native_gfx::Frame::eel_file_open(frame,count,refs);
    case DSPJSFX_FILE_OPEN_MULTI: return jsfx_native_gfx::Frame::eel_file_open_multi(frame,count,refs);
    case DSPJSFX_FILE_CLOSE: return jsfx_native_gfx::Frame::eel_file_close(frame,count,refs);
    case DSPJSFX_FILE_REWIND: return jsfx_native_gfx::Frame::eel_file_rewind(frame,count,refs);
    case DSPJSFX_FILE_SEEK: return jsfx_native_gfx::Frame::eel_file_seek(frame,count,refs);
    case DSPJSFX_FILE_AVAIL: return jsfx_native_gfx::Frame::eel_file_avail(frame,count,refs);
    case DSPJSFX_FILE_TEXT: return jsfx_native_gfx::Frame::eel_file_text(frame,count,refs);
    case DSPJSFX_FILE_RIFF: return jsfx_native_gfx::Frame::eel_file_riff(frame,count,refs);
    case DSPJSFX_FILE_VAR: return jsfx_native_gfx::Frame::eel_file_var(frame,count,refs);
    case DSPJSFX_FILE_MEM: return jsfx_native_gfx::Frame::eel_file_mem(frame,count,refs);
    case DSPJSFX_FILE_STRING: return jsfx_native_gfx::Frame::eel_file_string(frame,count,refs);
    case DSPJSFX_FILE_MULTI_COUNT: return jsfx_native_gfx::Frame::eel_file_multi_count(frame,count,refs);
    case DSPJSFX_FILE_MULTI_SELECT: return jsfx_native_gfx::Frame::eel_file_multi_select(frame,count,refs);
    case DSPJSFX_MATCH: return frame->matchStrings(count,refs,false);
    case DSPJSFX_MATCHI: return frame->matchStrings(count,refs,true);
    default:return 0;
    }
}
#endif
