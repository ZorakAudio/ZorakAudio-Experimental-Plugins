// SPDX-License-Identifier: Zlib
#pragma once
namespace za::jsfx {
// ZA_MIDI_RUNAWAY_FIX_V2: minimal active-note/sustain tracking for unsafe transport boundaries.
// This deliberately avoids sample-position heuristics and avoids blanket CC120 spam.
struct RuntimeMidiNoteTracker
{
    struct NoteState
    {
        bool held = false;
        bool sustained = false;
    };

    void clear() noexcept
    {
        sustainDown.fill (false);
        for (auto& channelNotes : notes)
            for (auto& note : channelNotes)
                note = {};
    }

    bool hasAnyUnsafeState() const noexcept
    {
        for (bool down : sustainDown)
            if (down)
                return true;

        for (const auto& channelNotes : notes)
            for (const auto& note : channelNotes)
                if (note.held || note.sustained)
                    return true;

        return false;
    }

    bool hasChannelUnsafeState (int channel) const noexcept
    {
        if (channel < 0 || channel >= 16)
            return false;

        if (sustainDown[(size_t) channel])
            return true;

        for (const auto& note : notes[(size_t) channel])
            if (note.held || note.sustained)
                return true;

        return false;
    }

    bool hasActiveNotesOnChannel (int channel) const noexcept
    {
        if (channel < 0 || channel >= 16)
            return false;

        for (const auto& note : notes[(size_t) channel])
            if (note.held || note.sustained)
                return true;

        return false;
    }

    static bool isChannelVoiceClearMessage (const juce::uint8* data, int len, int& channel) noexcept
    {
        channel = -1;

        if (data == nullptr || len < 3)
            return false;

        const int status = (int) data[0] & 0xff;
        if ((status & 0xf0) != 0xb0)
            return false;

        const int controller = (int) data[1] & 0x7f;
        if (controller != 120 && (controller < 123 || controller > 127)) // All Sound Off / Channel Mode note clear.
            return false;

        channel = status & 0x0f;
        return true;
    }

    void observeRawMidi (const juce::uint8* data, int len) noexcept
    {
        if (data == nullptr || len <= 0)
            return;

        const int status = (int) data[0] & 0xff;
        const int type = status & 0xf0;

        if (type == 0xf0)
            return;

        const int channel = status & 0x0f;
        if (channel < 0 || channel >= 16)
            return;

        if ((type == 0x90 || type == 0x80) && len >= 3)
        {
            const int noteNumber = (int) data[1] & 0x7f;
            const int velocity = (int) data[2] & 0x7f;
            auto& note = notes[(size_t) channel][(size_t) noteNumber];

            if (type == 0x90 && velocity > 0)
            {
                note.held = true;
                note.sustained = false;
                return;
            }

            if (sustainDown[(size_t) channel] && (note.held || note.sustained))
            {
                note.held = false;
                note.sustained = true;
                return;
            }

            note = {};
            return;
        }

        if (type == 0xb0 && len >= 3)
        {
            const int controller = (int) data[1] & 0x7f;
            const int value = (int) data[2] & 0x7f;

            if (controller == 64) // Damper/sustain pedal.
            {
                const bool down = value >= 64;
                sustainDown[(size_t) channel] = down;

                if (! down)
                    releaseSustainedNotesOnChannel (channel);

                return;
            }

            if (controller == 120 || (controller >= 123 && controller <= 127)) // All Sound Off / Channel Mode note clear.
            {
                sustainDown[(size_t) channel] = false;
                for (auto& note : notes[(size_t) channel])
                    note = {};
                return;
            }

            if (controller == 121) // Reset All Controllers: release sustain, but do not kill physically held notes.
            {
                sustainDown[(size_t) channel] = false;
                releaseSustainedNotesOnChannel (channel);
                return;
            }
        }
    }

    size_t activeNoteCountOnChannel (int channel) const noexcept
    {
        if (channel < 0 || channel >= 16)
            return 0;

        size_t count = 0;
        for (const auto& note : notes[(size_t) channel])
            if (note.held || note.sustained)
                ++count;

        return count;
    }

    size_t unsafeEventReserveCount (bool includeAllNotesOff) const noexcept
    {
        size_t count = 0;

        for (int ch = 0; ch < 16; ++ch)
        {
            if (! hasChannelUnsafeState (ch))
                continue;

            if (sustainDown[(size_t) ch])
                ++count;

            const auto active = activeNoteCountOnChannel (ch);
            count += active;

            if (includeAllNotesOff && active > 0)
                ++count;
        }

        return count;
    }

    void appendActiveNoteOffsForChannelToMidiBuffer (juce::MidiBuffer& midiMessages,
                                                     int sampleOffset,
                                                     int channel) const
    {
        if (channel < 0 || channel >= 16 || ! hasActiveNotesOnChannel (channel))
            return;

        const int clampedOffset = juce::jmax (0, sampleOffset);
        const int juceChannel = channel + 1;
        midiMessages.ensureSize (juce::jmax ((size_t) 128, activeNoteCountOnChannel (channel) * 3u));

        for (int noteNumber = 0; noteNumber < 128; ++noteNumber)
        {
            const auto& note = notes[(size_t) channel][(size_t) noteNumber];
            if (note.held || note.sustained)
                midiMessages.addEvent (juce::MidiMessage::noteOff (juceChannel, noteNumber), clampedOffset);
        }
    }

    // At a program/reset boundary, release the old owner's notes before
    // adding the new owner's events at the same timestamp. JUCE otherwise
    // appends same-time events and an old note-off can kill a new note-on.
    void prependUnsafeNoteEndsToMidiBuffer(juce::MidiBuffer& messages,int offset,bool allNotesOff) const {
        if(!hasAnyUnsafeState())return;
        juce::MidiBuffer ordered;
        appendUnsafeNoteEndsToMidiBuffer(ordered,offset,allNotesOff);
        ordered.addEvents(messages,0,-1,0);
        messages.swapWith(ordered);
    }

    void appendUnsafeNoteEndsToMidiBuffer (juce::MidiBuffer& midiMessages,
                                           int sampleOffset,
                                           bool includeAllNotesOff) const
    {
        if (! hasAnyUnsafeState())
            return;

        const int clampedOffset = juce::jmax (0, sampleOffset);
        midiMessages.ensureSize (juce::jmax ((size_t) 128, unsafeEventReserveCount (includeAllNotesOff) * 3u));

        for (int ch = 0; ch < 16; ++ch)
        {
            if (! hasChannelUnsafeState (ch))
                continue;

            const int juceChannel = ch + 1;
            const bool hasNotes = hasActiveNotesOnChannel (ch);

            if (sustainDown[(size_t) ch])
                midiMessages.addEvent (juce::MidiMessage::controllerEvent (juceChannel, 64, 0), clampedOffset);

            for (int noteNumber = 0; noteNumber < 128; ++noteNumber)
            {
                const auto& note = notes[(size_t) ch][(size_t) noteNumber];
                if (note.held || note.sustained)
                    midiMessages.addEvent (juce::MidiMessage::noteOff (juceChannel, noteNumber), clampedOffset);
            }

            if (includeAllNotesOff && hasNotes)
                midiMessages.addEvent (juce::MidiMessage::controllerEvent (juceChannel, 123, 0), clampedOffset);
        }
    }

    void appendActiveNoteOffsForChannelToRuntimeQueue (std::vector<JsfxRuntimeMidiEvent>& queue,
                                                       int sampleOffset,
                                                       int bus,
                                                       int channel) const
    {
        if (channel < 0 || channel >= 16 || ! hasActiveNotesOnChannel (channel))
            return;

        const int clampedOffset = juce::jmax (0, sampleOffset);
        const int clampedBus = juce::jlimit (0, 15, bus);

        try
        {
            queue.reserve (queue.size() + activeNoteCountOnChannel (channel));
        }
        catch (...)
        {
        }

        const auto noteOffStatus = (juce::uint8) (0x80 | channel);
        for (int noteNumber = 0; noteNumber < 128; ++noteNumber)
        {
            const auto& note = notes[(size_t) channel][(size_t) noteNumber];
            if (note.held || note.sustained)
                pushShortEventToRuntimeQueue (queue, clampedOffset, clampedBus, noteOffStatus, (juce::uint8) noteNumber, 0);
        }
    }

    void appendUnsafeNoteEndsToRuntimeQueue (std::vector<JsfxRuntimeMidiEvent>& queue,
                                             int sampleOffset,
                                             int bus,
                                             bool includeAllNotesOff) const
    {
        if (! hasAnyUnsafeState())
            return;

        const int clampedOffset = juce::jmax (0, sampleOffset);
        const int clampedBus = juce::jlimit (0, 15, bus);

        try
        {
            queue.reserve (queue.size() + unsafeEventReserveCount (includeAllNotesOff));
        }
        catch (...)
        {
        }

        for (int ch = 0; ch < 16; ++ch)
        {
            if (! hasChannelUnsafeState (ch))
                continue;

            const auto ccStatus = (juce::uint8) (0xb0 | ch);
            const auto noteOffStatus = (juce::uint8) (0x80 | ch);
            const bool hasNotes = hasActiveNotesOnChannel (ch);

            if (sustainDown[(size_t) ch])
                pushShortEventToRuntimeQueue (queue, clampedOffset, clampedBus, ccStatus, 64, 0);

            for (int noteNumber = 0; noteNumber < 128; ++noteNumber)
            {
                const auto& note = notes[(size_t) ch][(size_t) noteNumber];
                if (note.held || note.sustained)
                    pushShortEventToRuntimeQueue (queue, clampedOffset, clampedBus, noteOffStatus, (juce::uint8) noteNumber, 0);
            }

            if (includeAllNotesOff && hasNotes)
                pushShortEventToRuntimeQueue (queue, clampedOffset, clampedBus, ccStatus, 123, 0);
        }
    }

private:
    void releaseSustainedNotesOnChannel (int channel) noexcept
    {
        if (channel < 0 || channel >= 16)
            return;

        for (auto& note : notes[(size_t) channel])
            if (! note.held && note.sustained)
                note = {};
    }

    static void pushShortEventToRuntimeQueue (std::vector<JsfxRuntimeMidiEvent>& queue,
                                              int sampleOffset,
                                              int bus,
                                              juce::uint8 a,
                                              juce::uint8 b,
                                              juce::uint8 c)
    {
        try
        {
            JsfxRuntimeMidiEvent ev;
            ev.sampleOffset = sampleOffset;
            ev.bus = bus;
            const juce::uint8 bytes[3] { a, b, c };
            ev.assign (bytes, 3);
            if (ev.length > 0 && ev.data() != nullptr)
                queue.push_back (std::move (ev));
        }
        catch (...)
        {
        }
    }

    std::array<std::array<NoteState, 128>, 16> notes {};
    std::array<bool, 16> sustainDown {};
};
}


namespace za::jsfx {
class MidiHostRuntime {
public:
    static constexpr int kInitialMidiQueueCapacity=::kInitialMidiQueueCapacity;
    JsfxMidiRuntime midiRuntime;
    RuntimeMidiNoteTracker midiInputNoteTracker;
    RuntimeMidiNoteTracker midiOutputNoteTracker;
    bool midiTransportHaveLastPlaying=false,midiTransportLastPlaying=false;
    std::vector<DSPJSFX_MidiEvent> midiInStorage,midiOutStorage;
    void bindMidiRuntimeBuffers(DSPJSFX_State& st)
    {
        st.midiIn = midiInStorage.empty() ? nullptr : midiInStorage.data();
        st.midiInCapacity = (int32_t) midiInStorage.size();
        st.midiInCount = 0;
        st.midiInReadIndex = 0;

        st.midiOut = midiOutStorage.empty() ? nullptr : midiOutStorage.data();
        st.midiOutCapacity = (int32_t) midiOutStorage.size();
        st.midiOutCount = 0;

        st.runtimeOpaque = &midiRuntime;
#if DSPJSFX_NATIVE_GFX_LEGACY
        st.nativeStrings = &midiRuntime.nativeStrings;
#endif
        st.midi_bus = 0.0;
        st.ext_midi_bus = 0.0;
        st.currentSampleRate = st.srate;
        st.currentBlockSize = 0;
    }


    void resetMidiTransportStatusForCleanup() noexcept
    {
        midiTransportHaveLastPlaying = false;
        midiTransportLastPlaying = false;
    }


    void appendTrackedInputMidiCleanupToRuntimeQueue (int sampleOffset, bool includeAllNotesOff)
    {
        midiInputNoteTracker.appendUnsafeNoteEndsToRuntimeQueue (midiRuntime.midiInEvents,
                                                                 sampleOffset,
                                                                 0,
                                                                 includeAllNotesOff);
        midiInputNoteTracker.clear();
    }

    void appendTrackedOutputMidiCleanupToBuffer (juce::MidiBuffer& midiMessages,
                                                int sampleOffset,
                                                bool includeAllNotesOff)
    {
        midiOutputNoteTracker.appendUnsafeNoteEndsToMidiBuffer (midiMessages,
                                                                sampleOffset,
                                                                includeAllNotesOff);
        midiOutputNoteTracker.clear();
    }


    void importMidiToState (DSPJSFX_State& st,const juce::MidiBuffer& midiMessages,
                            int hostSamples,
                            int engineSamples,
                            int offsetScale,
                            bool prependUnsafeMidiCleanup,
                            bool includeAllNotesOffInPrependedCleanup)
    {
        hostSamples = juce::jmax (0, hostSamples);
        engineSamples = juce::jmax (0, engineSamples);
        offsetScale = juce::jlimit (1, 8, offsetScale);

        st.currentBlockSize = engineSamples;
        st.midiInCount = 0;
        st.midiInReadIndex = 0;
        st.midiOutCount = 0;
        st.midiInCountLastBlock = 0;
        st.midiOutCountLastBlock = 0;

        midiRuntime.beginBlock();
        midiRuntime.reserveQueues ((size_t) kInitialMidiQueueCapacity);

        if (prependUnsafeMidiCleanup)
            appendTrackedInputMidiCleanupToRuntimeQueue (0, includeAllNotesOffInPrependedCleanup);

        for (const auto metadata : midiMessages)
        {
            const auto msg = metadata.getMessage();
            const auto* raw = msg.getRawData();
            const int rawSize = msg.getRawDataSize();
            if (raw == nullptr || rawSize <= 0)
                continue;

            try
            {
                JsfxRuntimeMidiEvent ev;
                const int hostOffset = juce::jlimit (0, juce::jmax (0, hostSamples - 1), metadata.samplePosition);
                ev.sampleOffset = juce::jlimit (0, juce::jmax (0, engineSamples - 1), hostOffset * offsetScale);
                ev.bus = 0;
                ev.assign (reinterpret_cast<const juce::uint8*> (raw), rawSize);
                if (ev.length <= 0 || ev.data() == nullptr)
                    continue;

                int clearChannel = -1;
                if (RuntimeMidiNoteTracker::isChannelVoiceClearMessage (ev.data(), ev.length, clearChannel))
                    midiInputNoteTracker.appendActiveNoteOffsForChannelToRuntimeQueue (midiRuntime.midiInEvents,
                                                                                       ev.sampleOffset,
                                                                                       ev.bus,
                                                                                       clearChannel);

                midiInputNoteTracker.observeRawMidi (ev.data(), ev.length);
                midiRuntime.midiInEvents.push_back (std::move (ev));
            }
            catch (...)
            {
                ++st.midiInDropped;
            }
        }

        midiRuntime.midiInReadIndex = 0;
        st.midiInCount = (int32_t) midiRuntime.midiInEvents.size();
        st.midiInCountLastBlock = st.midiInCount;
        st.midiInPeak = juce::jmax (st.midiInPeak, st.midiInCount);
    }

    bool flushMidiFromState (DSPJSFX_State& st,juce::MidiBuffer& midiMessages, int offsetDivisor = 1, int hostSamples = -1)
    {
        if (midiRuntime.midiOutEvents.empty())
            return false;

        offsetDivisor = juce::jlimit (1, 8, offsetDivisor);

        if (! runtimeMidiEventsAlreadySortedByOffset (midiRuntime.midiOutEvents))
            stableSortRuntimeMidiEventsByOffset (midiRuntime.midiOutEvents);

        st.midiOutCount = (int32_t) midiRuntime.midiOutEvents.size();
        st.midiOutCountLastBlock = st.midiOutCount;
        st.midiOutPeak = juce::jmax (st.midiOutPeak, st.midiOutCount);

        size_t totalBytes = 0;
        for (const auto& ev : midiRuntime.midiOutEvents)
            totalBytes += (size_t) juce::jmax (1, ev.length);

        midiMessages.ensureSize ((size_t) juce::jmax ((int64_t) 128, (int64_t) totalBytes * 2));

        for (const auto& ev : midiRuntime.midiOutEvents)
        {
            const auto* raw = ev.data();
            if (raw == nullptr || ev.length <= 0)
                continue;

            int hostOffset = ev.sampleOffset;
            if (offsetDivisor > 1)
                hostOffset = (int) std::floor ((double) ev.sampleOffset / (double) offsetDivisor);

            if (hostSamples >= 0)
                hostOffset = juce::jlimit (0, juce::jmax (0, hostSamples - 1), hostOffset);
            else
                hostOffset = juce::jmax (0, hostOffset);

            int clearChannel = -1;
            if (RuntimeMidiNoteTracker::isChannelVoiceClearMessage (raw, ev.length, clearChannel))
                midiOutputNoteTracker.appendActiveNoteOffsForChannelToMidiBuffer (midiMessages,
                                                                                  hostOffset,
                                                                                  clearChannel);

            midiOutputNoteTracker.observeRawMidi (raw, ev.length);
            midiMessages.addEvent (raw, ev.length, hostOffset);
        }

        midiRuntime.midiOutEvents.clear();
        st.midiOutCount = 0;
        return true;
    }

    bool observeMidiTransport(bool valid,bool playing) noexcept {
        if(!valid){resetMidiTransportStatusForCleanup();return false;}
        bool changed=midiTransportHaveLastPlaying && midiTransportLastPlaying!=playing;
        midiTransportHaveLastPlaying=true;midiTransportLastPlaying=playing;return changed;
    }
    void prepareMidiRuntime(DSPJSFX_State& st,int block) {
        const auto reserveHint=std::max(kInitialMidiQueueCapacity,std::max(0,block));
        if((int)midiInStorage.size()<kInitialMidiQueueCapacity)midiInStorage.resize(kInitialMidiQueueCapacity);
        if((int)midiOutStorage.size()<kInitialMidiQueueCapacity)midiOutStorage.resize(kInitialMidiQueueCapacity);
        midiRuntime.reserveQueues(reserveHint);midiRuntime.resetAll();bindMidiRuntimeBuffers(st);
        midiInputNoteTracker.clear();midiOutputNoteTracker.clear();resetMidiTransportStatusForCleanup();
    }
};
}
