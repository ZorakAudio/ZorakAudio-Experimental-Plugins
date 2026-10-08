// SPDX-License-Identifier: Zlib
#pragma once
// Include once after defining JSFX_SAMPLE_POOL_OWNER(state).
extern "C" double jsfx_sample_pool_from_slot (DSPJSFX_State* state, double slot, double nameHandle)
{
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_pool_from_slot (state, slot, nameHandle);
    return 0.0;
}

extern "C" double jsfx_sample_pool_set_deferred (DSPJSFX_State* state, double pool, double deferred)
{
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_pool_set_deferred (pool, deferred);
    return 0.0;
}

extern "C" double jsfx_sample_pool_adopt (DSPJSFX_State* state, double pool)
{
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_pool_adopt (pool);
    return 0.0;
}

extern "C" double jsfx_sample_pool_set_mode (DSPJSFX_State* state, double pool, double mode)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_pool_set_mode (pool, mode);
    return 0.0;
}

extern "C" double jsfx_sample_pool_set_budget_mb (DSPJSFX_State* state, double pool, double mb)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_pool_set_budget_mb (pool, mb);
    return 0.0;
}

extern "C" double jsfx_sample_pool_commit (DSPJSFX_State* state, double pool)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_pool_commit (pool);
    return 0.0;
}

extern "C" double jsfx_sample_pool_state (DSPJSFX_State* state, double pool)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_pool_state (pool);
    return 0.0;
}

extern "C" double jsfx_sample_pool_selected (DSPJSFX_State* state, double pool)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_pool_selected (pool);
    return 0.0;
}

extern "C" double jsfx_sample_pool_loaded (DSPJSFX_State* state, double pool)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_pool_loaded (pool);
    return 0.0;
}

extern "C" double jsfx_sample_pool_failed (DSPJSFX_State* state, double pool)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_pool_failed (pool);
    return 0.0;
}

extern "C" double jsfx_sample_pool_ram_mb (DSPJSFX_State* state, double pool)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_pool_ram_mb (pool);
    return 0.0;
}

extern "C" double jsfx_sample_pool_generation (DSPJSFX_State* state, double pool)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_pool_generation (pool);
    return 0.0;
}

extern "C" double jsfx_sample_get (DSPJSFX_State* state, double pool, double index)
{
#if DSPJSFX_HAS_TASKS
    if(!jsfx_tasks::Runtime::sampleReadAllowed(state,pool)) return 0.;
#endif

    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_get (pool, index);
    return 0.0;
}

extern "C" double jsfx_sample_len (DSPJSFX_State* state, double pool, double sampleId)
{
#if DSPJSFX_HAS_TASKS
    if(!jsfx_tasks::Runtime::sampleReadAllowed(state,pool)) return 0.;
#endif

    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_len (pool, sampleId);
    return 0.0;
}

extern "C" double jsfx_sample_channels (DSPJSFX_State* state, double pool, double sampleId)
{
#if DSPJSFX_HAS_TASKS
    if(!jsfx_tasks::Runtime::sampleReadAllowed(state,pool)) return 0.;
#endif

    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_channels (pool, sampleId);
    return 0.0;
}

extern "C" double jsfx_sample_srate (DSPJSFX_State* state, double pool, double sampleId)
{
#if DSPJSFX_HAS_TASKS
    if(!jsfx_tasks::Runtime::sampleReadAllowed(state,pool)) return 0.;
#endif

    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_srate (pool, sampleId);
    return 0.0;
}

extern "C" double jsfx_sample_peak (DSPJSFX_State* state, double pool, double sampleId)
{
#if DSPJSFX_HAS_TASKS
    if(!jsfx_tasks::Runtime::sampleReadAllowed(state,pool)) return 0.;
#endif

    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_peak (pool, sampleId);
    return 0.0;
}

extern "C" double jsfx_sample_rms (DSPJSFX_State* state, double pool, double sampleId)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_rms (pool, sampleId);
    return 0.0;
}

extern "C" double jsfx_sample_name (DSPJSFX_State* state, double pool, double sampleId, double* outStr)
{
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_name (state, pool, sampleId, outStr);
    return 0.0;
}

extern "C" double jsfx_sample_read (DSPJSFX_State* state, double pool, double sampleId, double channel, double frame)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_read (pool, sampleId, channel, frame);
    return 0.0;
}

extern "C" double jsfx_sample_read_interp (DSPJSFX_State* state, double pool, double sampleId, double channel, double phase)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_read_interp (pool, sampleId, channel, phase);
    return 0.0;
}

extern "C" double jsfx_sample_read2 (DSPJSFX_State* state, double pool, double sampleId, double phase, double* outL, double* outR)
{
#if DSPJSFX_HAS_TASKS
    if (!jsfx_tasks::Runtime::sampleReadAllowed(state,pool)) {
        if(outL) jsfxCellStore(outL,0.);if(outR) jsfxCellStore(outR,0.);return 0.;
    }
#endif

    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_read2 (pool, sampleId, phase, outL, outR, false);
    if (outL) jsfxCellStore (outL, 0.0);
    if (outR) jsfxCellStore (outR, 0.0);
    return 0.0;
}

extern "C" double jsfx_sample_read2_interp (DSPJSFX_State* state, double pool, double sampleId, double phase, double* outL, double* outR)
{
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_read2 (pool, sampleId, phase, outL, outR, true);
    if (outL) jsfxCellStore (outL, 0.0);
    if (outR) jsfxCellStore (outR, 0.0);
    return 0.0;
}

extern "C" double jsfx_sample_preview_bins (DSPJSFX_State* state, double pool, double sampleId)
{
    juce::ignoreUnused (state);
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_preview_bins (pool, sampleId);
    return 0.0;
}

extern "C" double jsfx_sample_preview_read (DSPJSFX_State* state, double pool, double sampleId, double bin, double* mn, double* mx, double* rms)
{
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_preview_read (pool, sampleId, bin, mn, mx, rms);
    return 0.0;
}

extern "C" double jsfx_sample_export_mem (DSPJSFX_State* state, double pool, double sampleId, double dstBase, double srcFrame, double frameCount)
{
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_export_mem (state, pool, sampleId, dstBase, srcFrame, frameCount, false);
    return 0.0;
}

extern "C" double jsfx_sample_export_mem2 (DSPJSFX_State* state, double pool, double sampleId, double dstBase, double srcFrame, double frameCount)
{
    if (auto* owner = JSFX_SAMPLE_POOL_OWNER(state))
        return owner->rt_sample_export_mem (state, pool, sampleId, dstBase, srcFrame, frameCount, true);
    return 0.0;
}
