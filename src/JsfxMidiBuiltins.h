// Shared production MIDI queues, strings, short/long/SysEx and bus builtins.
#pragma once
namespace
{
static constexpr int kInitialMidiQueueCapacity = 512;
static constexpr int64_t kJsfxDynamicStringHandleBase = (int64_t) 1 << 48;

static inline int jsfxRoundToInt (double v) noexcept
{
    if (! std::isfinite (v))
        return 0;
    return (int) std::llround (v);
}

static inline int jsfxClampMidiByte (double v) noexcept
{
    return juce::jlimit (0, 255, jsfxRoundToInt (v));
}

static inline int64_t jsfxClampMemIndex (double v) noexcept
{
    const int64_t idx = jsfxRoundToIndex (v);
    return juce::jmax<int64_t> ((int64_t) 0, idx);
}

static inline int jsfxClampMidiOffset (const DSPJSFX_State* st, double v) noexcept
{
    const int maxOffset = (st != nullptr && st->currentBlockSize > 0) ? (st->currentBlockSize - 1) : 0;
    return juce::jlimit (0, juce::jmax (0, maxOffset), jsfxRoundToInt (v));
}

static inline int jsfxShortMessageLength (int statusByte) noexcept
{
    const int len = juce::MidiMessage::getMessageLengthFromFirstByte ((juce::uint8) (statusByte & 0xff));
    if (len < 1)
        return 1;
    return juce::jmin (3, len);
}

struct JsfxRuntimeMidiEvent
{
    int sampleOffset = 0;
    int bus = 0;
    int length = 0;
    std::array<juce::uint8, 3> shortBytes { { 0, 0, 0 } };
    std::vector<juce::uint8> longBytes;

    void assign (const juce::uint8* src, int len)
    {
        length = 0;
        shortBytes = { { 0, 0, 0 } };
        longBytes.clear();

        if (src == nullptr || len <= 0)
            return;

        if (len <= 3)
        {
            length = len;
            for (int i = 0; i < len; ++i)
                shortBytes[(size_t) i] = src[i];
            return;
        }

        longBytes.assign (src, src + len);
        length = (int) longBytes.size();
    }

    const juce::uint8* data() const noexcept
    {
        if (length <= 0)
            return nullptr;
        return length <= 3 ? shortBytes.data() : longBytes.data();
    }

    bool isSysEx() const noexcept
    {
        const auto* bytes = data();
        if (bytes == nullptr || length <= 0)
            return false;
        return bytes[0] == (juce::uint8) 0xF0 || (length > 1 && bytes[length - 1] == (juce::uint8) 0xF7);
    }
};

struct JsfxMidiRuntime
{
    std::vector<JsfxRuntimeMidiEvent> midiInEvents;
    std::vector<JsfxRuntimeMidiEvent> midiOutEvents;
    int midiInReadIndex = 0;
    std::unordered_map<int64_t, std::string> dynamicStrings;
#if DSPJSFX_NATIVE_GFX_LEGACY
    jsfx_native_gfx::Strings nativeStrings;
#endif
    int64_t nextDynamicStringHandle = kJsfxDynamicStringHandleBase;

    void beginBlock()
    {
        midiInEvents.clear();
        midiOutEvents.clear();
        midiInReadIndex = 0;
    }

    void resetAll()
    {
        beginBlock();
        dynamicStrings.clear();
#if DSPJSFX_NATIVE_GFX_LEGACY
        nativeStrings.reset();
#endif
        nextDynamicStringHandle = kJsfxDynamicStringHandleBase;
    }

    void reserveQueues (size_t minCapacity)
    {
        if (midiInEvents.capacity() < minCapacity)
            midiInEvents.reserve (minCapacity);
        if (midiOutEvents.capacity() < minCapacity)
            midiOutEvents.reserve (minCapacity);
    }
};

static inline JsfxMidiRuntime* getJsfxMidiRuntime (DSPJSFX_State* st) noexcept
{
    return (st != nullptr) ? static_cast<JsfxMidiRuntime*> (st->runtimeOpaque) : nullptr;
}

static inline const JsfxMidiRuntime* getJsfxMidiRuntime (const DSPJSFX_State* st) noexcept
{
    return (st != nullptr) ? static_cast<const JsfxMidiRuntime*> (st->runtimeOpaque) : nullptr;
}

static inline int64_t jsfxStringHandleFromDouble (double v) noexcept
{
    if (! std::isfinite (v))
        return 0;
    return (int64_t) std::llround (v);
}

static const DSPJSFX_StringLiteralDesc* findJsfxStringLiteralDesc (int64_t handle) noexcept
{
   #if DSPJSFX_STRING_LITERALS_COUNT > 0
    for (int i = 0; i < (int) DSPJSFX_STRING_LITERALS_COUNT; ++i)
    {
        if (DSPJSFX_STRING_LITERALS[i].handle == handle)
            return &DSPJSFX_STRING_LITERALS[i];
    }
   #else
    juce::ignoreUnused (handle);
   #endif
    return nullptr;
}

static inline void jsfxSetCurrentMidiBus (DSPJSFX_State* st, int bus) noexcept
{
    if (st == nullptr)
        return;

    if (st->ext_midi_bus == 0.0)
        st->midi_bus = 0.0;
    else
        st->midi_bus = (double) juce::jlimit (0, 15, bus);
}

static inline int jsfxGetCurrentMidiBus (const DSPJSFX_State* st) noexcept
{
    if (st == nullptr || st->ext_midi_bus == 0.0)
        return 0;
    return juce::jlimit (0, 15, jsfxRoundToInt (st->midi_bus));
}

static const juce::uint8* jsfxGetStringBytes (const DSPJSFX_State* st, double strHandle, int& outLength) noexcept
{
    outLength = 0;

    const int64_t handle = jsfxStringHandleFromDouble (strHandle);
    if (handle == 0)
        return nullptr;

    if (const auto* lit = findJsfxStringLiteralDesc (handle); lit != nullptr)
    {
        outLength = juce::jmax (0, (int) lit->length);
        return lit->data;
    }

#if DSPJSFX_NATIVE_GFX_LEGACY
    if (auto* context = jsfx_native_gfx::stringContext(const_cast<DSPJSFX_State*>(st))) {
        thread_local std::string copy;
        copy = context->read(strHandle);
        outLength = (int)copy.size();
        return reinterpret_cast<const juce::uint8*>(copy.data());
    }
#endif
    if (const auto* rt = getJsfxMidiRuntime (st); rt != nullptr)
    {
        const auto it = rt->dynamicStrings.find (handle);
        if (it != rt->dynamicStrings.end())
        {
            outLength = (int) it->second.size();
            return reinterpret_cast<const juce::uint8*> (it->second.data());
        }
    }

    return nullptr;
}

static std::string* jsfxGetMutableStringForSlot (DSPJSFX_State* st, double* slot) noexcept
{
    if (st == nullptr || slot == nullptr)
        return nullptr;

    auto* rt = getJsfxMidiRuntime (st);
    if (rt == nullptr)
        return nullptr;

    try
    {
        const int64_t handle = jsfxStringHandleFromDouble (jsfxCellLoad(slot));
        if (const auto it = rt->dynamicStrings.find (handle); it != rt->dynamicStrings.end())
            return &it->second;

        const int64_t newHandle = rt->nextDynamicStringHandle++;
        auto result = rt->dynamicStrings.emplace (newHandle, std::string());
        jsfxCellStore (slot, (double) newHandle);
        return &result.first->second;
    }
    catch (...)
    {
        return nullptr;
    }
}

static bool jsfxAssignStringBytes (DSPJSFX_State* st, double* slot, const juce::uint8* data, int len) noexcept
{
#if DSPJSFX_NATIVE_GFX_LEGACY
    if (st && slot) if (auto* context = jsfx_native_gfx::stringContext(st)) {
        std::string copy(data && len > 0 ? reinterpret_cast<const char*>(data) : "", data && len > 0 ? (size_t)len : 0);
        const double handle=jsfxCellLoad(slot);
        if (handle >= double(int64_t(1)<<39) && context->writable(handle)) context->write(handle,std::move(copy));
        else jsfxCellStore(slot,context->create(std::move(copy)));
        return true;
    }
#endif
    auto* dst = jsfxGetMutableStringForSlot (st, slot);
    if (dst == nullptr)
        return false;

    try
    {
        if (data == nullptr || len <= 0)
            dst->clear();
        else
            dst->assign (reinterpret_cast<const char*> (data), (size_t) len);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

extern "C" std::uint64_t jsfx_string_hash (DSPJSFX_State* st, double strHandle)
{
    int len = 0;
    const auto* bytes = jsfxGetStringBytes (st, strHandle, len);
    if (bytes == nullptr || len <= 0)
        return 0;

    std::uint64_t hash = 1469598103934665603ull;
    for (int i = 0; i < len; ++i)
    {
        hash ^= static_cast<std::uint64_t> (bytes[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

extern "C" int jsfx_string_assign_utf8 (DSPJSFX_State* st, double* slot, const char* data, int len)
{
    if (data == nullptr || len < 0)
        return 0;
    return jsfxAssignStringBytes (st,
                                  slot,
                                  reinterpret_cast<const juce::uint8*> (data),
                                  len) ? len : 0;
}

static bool jsfxReadBytesFromMem (const DSPJSFX_State* st, double bufIndex, int len, std::vector<juce::uint8>& out)
{
    out.clear();

    if (st == nullptr || st->mem == nullptr || len <= 0)
        return false;

    const int64_t base = jsfxClampMemIndex (bufIndex);
    const int64_t end = base + len;
    if (end < base || end > st->memN)
        return false;

    out.resize ((size_t) len);
    for (int i = 0; i < len; ++i)
        out[(size_t) i] = (juce::uint8) jsfxClampMidiByte (st->mem[base + i]);

    return true;
}

static int jsfxWriteBytesToMem (DSPJSFX_State* st, double bufIndex, const juce::uint8* data, int len) noexcept
{
    if (st == nullptr || data == nullptr || len <= 0)
        return 0;

    const int64_t base = jsfxClampMemIndex (bufIndex);
    const int64_t end = base + len;
    if (end < base)
        return 0;

    jsfx_ensure_mem (st, end);
    if (st->mem == nullptr || end > st->memN)
        return 0;

    for (int i = 0; i < len; ++i)
        st->mem[base + i] = (double) data[i];

    noteTrackedJsfxMemUsed (st, end);
    return len;
}

static inline bool jsfxLooksLikeSysEx (const juce::uint8* bytes, int len) noexcept
{
    if (bytes == nullptr || len <= 0)
        return false;
    return len > 3 || bytes[0] == (juce::uint8) 0xF0 || bytes[len - 1] == (juce::uint8) 0xF7;
}

static int jsfxPrepareVariableMidiBytes (std::vector<juce::uint8>& bytes, bool forceSysEx)
{
    if (bytes.empty())
        return 0;

    if (forceSysEx || jsfxLooksLikeSysEx (bytes.data(), (int) bytes.size()))
    {
        if (bytes.front() != (juce::uint8) 0xF0)
            bytes.insert (bytes.begin(), (juce::uint8) 0xF0);
        if (bytes.back() != (juce::uint8) 0xF7)
            bytes.push_back ((juce::uint8) 0xF7);
    }

    return (int) bytes.size();
}

static bool runtimeMidiEventsAlreadySortedByOffset (const std::vector<JsfxRuntimeMidiEvent>& events) noexcept
{
    if (events.size() <= 1)
        return true;

    int prev = events.front().sampleOffset;
    for (size_t i = 1; i < events.size(); ++i)
    {
        const int cur = events[i].sampleOffset;
        if (cur < prev)
            return false;
        prev = cur;
    }
    return true;
}

static void stableSortRuntimeMidiEventsByOffset (std::vector<JsfxRuntimeMidiEvent>& events)
{
    std::stable_sort (events.begin(), events.end(),
                      [] (const JsfxRuntimeMidiEvent& a, const JsfxRuntimeMidiEvent& b)
                      {
                          return a.sampleOffset < b.sampleOffset;
                      });
}

static bool jsfxQueueOutputEvent (DSPJSFX_State* st, int sampleOffset, int bus, const juce::uint8* data, int len)
{
    if (st == nullptr || data == nullptr || len <= 0)
        return false;

    auto* rt = getJsfxMidiRuntime (st);
    if (rt == nullptr)
        return false;

    try
    {
        JsfxRuntimeMidiEvent ev;
        ev.sampleOffset = juce::jmax (0, sampleOffset);
        ev.bus = juce::jlimit (0, 15, bus);
        ev.assign (data, len);
        if (ev.length <= 0 || ev.data() == nullptr)
            return false;

        rt->midiOutEvents.push_back (std::move (ev));
        st->midiOutCount = (int32_t) rt->midiOutEvents.size();
        return true;
    }
    catch (...)
    {
        ++st->midiOutDropped;
        return false;
    }
}

static bool jsfxQueueOutputEvent (DSPJSFX_State* st, const JsfxRuntimeMidiEvent& ev)
{
    return jsfxQueueOutputEvent (st, ev.sampleOffset, ev.bus, ev.data(), ev.length);
}

static const JsfxRuntimeMidiEvent* jsfxPopInputEvent (DSPJSFX_State* st) noexcept
{
    if (st == nullptr)
        return nullptr;

    auto* rt = getJsfxMidiRuntime (st);
    if (rt == nullptr)
        return nullptr;

    if (rt->midiInReadIndex < 0 || rt->midiInReadIndex >= (int) rt->midiInEvents.size())
        return nullptr;

    const auto* ev = &rt->midiInEvents[(size_t) rt->midiInReadIndex++];
    st->midiInReadIndex = rt->midiInReadIndex;
    return ev;
}

}
extern "C" int jsfx_midirecv (DSPJSFX_State* st, double* offset, double* msg1, double* msg2, double* msg3)
{
    if (st == nullptr)
        return 0;

    while (const auto* ev = jsfxPopInputEvent (st))
    {
        jsfxSetCurrentMidiBus (st, ev->bus);

        if (ev->length > 3 || ev->isSysEx())
        {
            (void) jsfxQueueOutputEvent (st, *ev);
            continue;
        }

        const auto* bytes = ev->data();
        if (bytes == nullptr || ev->length <= 0)
            continue;

        if (offset != nullptr) jsfxCellStore (offset, (double) ev->sampleOffset);
        if (msg1 != nullptr) jsfxCellStore (msg1, (double) bytes[0]);
        if (msg2 != nullptr) jsfxCellStore (msg2, (double) (ev->length > 1 ? bytes[1] : 0));
        if (msg3 != nullptr) jsfxCellStore (msg3, (double) (ev->length > 2 ? bytes[2] : 0));
        return 1;
    }

    return 0;
}

extern "C" int jsfx_midirecv_msg23 (DSPJSFX_State* st, double* offset, double* msg1, double* msg23)
{
    double m2 = 0.0;
    double m3 = 0.0;
    if (! jsfx_midirecv (st, offset, msg1, &m2, &m3))
        return 0;

    if (msg23 != nullptr)
        jsfxCellStore (msg23, m2 + (256.0 * m3));
    return 1;
}

extern "C" int jsfx_midirecv_buf (DSPJSFX_State* st, double* offset, double buf, double maxlen)
{
    if (st == nullptr)
        return 0;

    const int maxLen = juce::jmax (0, jsfxRoundToInt (maxlen));
    if (maxLen <= 0)
        return 0;

    while (const auto* ev = jsfxPopInputEvent (st))
    {
        jsfxSetCurrentMidiBus (st, ev->bus);

        const auto* bytes = ev->data();
        if (bytes == nullptr || ev->length <= 0)
            continue;

        if (ev->length > maxLen)
        {
            (void) jsfxQueueOutputEvent (st, *ev);
            continue;
        }

        if (offset != nullptr)
            jsfxCellStore (offset, (double) ev->sampleOffset);
        return jsfxWriteBytesToMem (st, buf, bytes, ev->length);
    }

    return 0;
}

extern "C" int jsfx_midirecv_str (DSPJSFX_State* st, double* offset, double* outStr)
{
    if (st == nullptr || outStr == nullptr)
        return 0;

    while (const auto* ev = jsfxPopInputEvent (st))
    {
        jsfxSetCurrentMidiBus (st, ev->bus);

        const auto* bytes = ev->data();
        if (bytes == nullptr || ev->length <= 0)
            continue;

        if (! jsfxAssignStringBytes (st, outStr, bytes, ev->length))
            return 0;

        if (offset != nullptr)
            jsfxCellStore (offset, (double) ev->sampleOffset);
        return ev->length;
    }

    return 0;
}

extern "C" int jsfx_midisend (DSPJSFX_State* st, double offset, double msg1, double msg2, double msg3)
{
    if (st == nullptr)
        return 0;

    const int status = jsfxClampMidiByte (msg1);
    const juce::uint8 raw[3] = {
        (juce::uint8) status,
        (juce::uint8) jsfxClampMidiByte (msg2),
        (juce::uint8) jsfxClampMidiByte (msg3)
    };
    const int len = jsfxShortMessageLength (status);

    if (! jsfxQueueOutputEvent (st, jsfxClampMidiOffset (st, offset), jsfxGetCurrentMidiBus (st), raw, len))
        return 0;
    return status;
}

extern "C" int jsfx_midisend_msg23 (DSPJSFX_State* st, double offset, double msg1, double msg23)
{
    const int packed = jsfxRoundToInt (msg23);
    const int msg2 = packed & 0xff;
    const int msg3 = (packed >> 8) & 0xff;
    return jsfx_midisend (st, offset, msg1, (double) msg2, (double) msg3);
}

extern "C" int jsfx_midisend_buf (DSPJSFX_State* st, double offset, double buf, double len)
{
    const int requestedLen = juce::jmax (0, jsfxRoundToInt (len));
    if (st == nullptr || requestedLen <= 0)
        return 0;

    std::vector<juce::uint8> bytes;
    if (! jsfxReadBytesFromMem (st, buf, requestedLen, bytes))
        return 0;

    const int actualLen = jsfxPrepareVariableMidiBytes (bytes, false);
    if (actualLen <= 0)
        return 0;

    return jsfxQueueOutputEvent (st, jsfxClampMidiOffset (st, offset), jsfxGetCurrentMidiBus (st), bytes.data(), actualLen)
             ? actualLen
             : 0;
}

extern "C" int jsfx_midisend_str (DSPJSFX_State* st, double offset, double strHandle)
{
    if (st == nullptr)
        return 0;

    int len = 0;
    const auto* src = jsfxGetStringBytes (st, strHandle, len);
    if (src == nullptr || len <= 0)
        return 0;

    try
    {
        std::vector<juce::uint8> bytes (src, src + len);
        const int actualLen = jsfxPrepareVariableMidiBytes (bytes, false);
        if (actualLen <= 0)
            return 0;

        return jsfxQueueOutputEvent (st, jsfxClampMidiOffset (st, offset), jsfxGetCurrentMidiBus (st), bytes.data(), actualLen)
                 ? actualLen
                 : 0;
    }
    catch (...)
    {
        ++st->midiOutDropped;
        return 0;
    }
}

extern "C" int jsfx_midisyx (DSPJSFX_State* st, double offset, double msgptr, double len)
{
    const int requestedLen = juce::jmax (0, jsfxRoundToInt (len));
    if (st == nullptr || requestedLen <= 0)
        return 0;

    std::vector<juce::uint8> bytes;
    if (! jsfxReadBytesFromMem (st, msgptr, requestedLen, bytes))
        return 0;

    const int actualLen = jsfxPrepareVariableMidiBytes (bytes, true);
    if (actualLen <= 0)
        return 0;

    return jsfxQueueOutputEvent (st, jsfxClampMidiOffset (st, offset), jsfxGetCurrentMidiBus (st), bytes.data(), actualLen)
             ? actualLen
             : 0;
}

extern "C" int jsfx_strlen (DSPJSFX_State* st, double strHandle)
{
    int len = 0;
    (void) jsfxGetStringBytes (st, strHandle, len);
    return len;
}

extern "C" int jsfx_str_getchar (DSPJSFX_State* st, double strHandle, double index)
{
    int len = 0;
    const auto* bytes = jsfxGetStringBytes (st, strHandle, len);
    if (bytes == nullptr || len <= 0)
        return 0;

    const int64_t idx = jsfxRoundToIndex (index);
    if (idx < 0 || idx >= len)
        return 0;

    return (int) bytes[(size_t) idx];
}

