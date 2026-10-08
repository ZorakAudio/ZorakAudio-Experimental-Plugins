// SPDX-License-Identifier: Zlib
#pragma once
#include "DspJsfxSamplePool.h"
namespace za::jsfx {
template<class Owner> class SamplePoolOperations {
    DspJsfxSamplePool* getSamplePoolForHandle(double handle) {return static_cast<Owner*>(this)->getSamplePoolForHandle(handle);}
public:
    double rt_sample_pool_set_deferred (double poolHandle, double deferred) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
        {
            pool->setDeferred (deferred != 0.0);
            return 1.0;
        }
        return 0.0;
    }

    double rt_sample_pool_adopt (double poolHandle) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return static_cast<double> (pool->adoptPending());
        return 0.0;
    }

    double rt_sample_pool_set_mode (double poolHandle, double modeValue) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
        {
            pool->setMode ((int) std::llround (modeValue));
            return 1.0;
        }
        return 0.0;
    }

    double rt_sample_pool_set_budget_mb (double poolHandle, double mb) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
        {
            pool->setBudgetMB (mb);
            return 1.0;
        }
        return 0.0;
    }

    double rt_sample_pool_state (double poolHandle) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return (double) pool->state();
        return 0.0;
    }

    double rt_sample_pool_selected (double poolHandle) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return (double) pool->selected();
        return 0.0;
    }

    double rt_sample_pool_loaded (double poolHandle) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return (double) pool->loaded();
        return 0.0;
    }

    double rt_sample_pool_failed (double poolHandle) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return (double) pool->failed();
        return 0.0;
    }

    double rt_sample_pool_ram_mb (double poolHandle) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return pool->ramMB();
        return 0.0;
    }

    double rt_sample_pool_generation (double poolHandle) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return (double) pool->generation();
        return 0.0;
    }

    double rt_sample_get (double poolHandle, double indexValue) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return (double) pool->sampleIdAt ((int) std::llround (indexValue));
        return 0.0;
    }

    double rt_sample_len (double poolHandle, double sampleId) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return (double) pool->length ((std::uint64_t) std::llround (sampleId));
        return 0.0;
    }

    double rt_sample_channels (double poolHandle, double sampleId) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return (double) pool->channels ((std::uint64_t) std::llround (sampleId));
        return 0.0;
    }

    double rt_sample_srate (double poolHandle, double sampleId) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return (double) pool->sampleRate ((std::uint64_t) std::llround (sampleId));
        return 0.0;
    }

    double rt_sample_peak (double poolHandle, double sampleId) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return pool->peak ((std::uint64_t) std::llround (sampleId));
        return 0.0;
    }

    double rt_sample_rms (double poolHandle, double sampleId) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return pool->rms ((std::uint64_t) std::llround (sampleId));
        return 0.0;
    }

    double rt_sample_read (double poolHandle, double sampleId, double channel, double frame) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return pool->read ((std::uint64_t) std::llround (sampleId), (int) std::llround (channel), frame);
        return 0.0;
    }

    double rt_sample_read_interp (double poolHandle, double sampleId, double channel, double phase) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return pool->readInterp ((std::uint64_t) std::llround (sampleId), (int) std::llround (channel), phase);
        return 0.0;
    }

    double rt_sample_read2 (double poolHandle, double sampleId, double phase, double* outL, double* outR, bool interp) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle)) {
            double l = 0, r = 0;
            const bool ok = pool->read2 ((std::uint64_t) std::llround (sampleId), phase, &l, &r, interp);
            if (outL) jsfxCellStore(outL, l);
            if (outR) jsfxCellStore(outR, r);
            return ok ? 1.0 : 0.0;
        }
        if (outL != nullptr) jsfxCellStore (outL, 0.0);
        if (outR != nullptr) jsfxCellStore (outR, 0.0);
        return 0.0;
    }

    double rt_sample_preview_bins (double poolHandle, double sampleId) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle))
            return (double) pool->previewBins ((std::uint64_t) std::llround (sampleId));
        return 0.0;
    }

    double rt_sample_preview_read (double poolHandle, double sampleId, double bin, double* mn, double* mx, double* rms) noexcept
    {
        if (auto* pool = getSamplePoolForHandle (poolHandle)) {
            double lo = 0, hi = 0, level = 0;
            const bool ok = pool->previewRead ((std::uint64_t) std::llround (sampleId), (int) std::llround (bin), &lo, &hi, &level);
            if (mn) jsfxCellStore(mn, lo);
            if (mx) jsfxCellStore(mx, hi);
            if (rms) jsfxCellStore(rms, level);
            return ok ? 1.0 : 0.0;
        }
        return 0.0;
    }

    double rt_sample_name (DSPJSFX_State* state, double poolHandle, double sampleId, double* outStr) noexcept
    {
        if (outStr == nullptr)
            return 0.0;
        if (auto* pool = getSamplePoolForHandle (poolHandle))
        {
            std::string name;
            if (pool->name ((std::uint64_t) std::llround (sampleId), name))
                return (double) jsfx_string_assign_utf8 (state, outStr, name.c_str(), (int) name.size());
        }
        return 0.0;
    }

    double rt_sample_export_mem (DSPJSFX_State* state, double poolHandle, double sampleId, double dstBase, double srcFrame, double frameCount, bool stereo) noexcept
    {
        if (state == nullptr || state->mem == nullptr)
            return 0.0;
        auto* pool = getSamplePoolForHandle (poolHandle);
        if (pool == nullptr)
            return 0.0;

        const int dst = (int) std::llround (dstBase);
        const int start = (int) std::llround (srcFrame);
        const int count = (int) std::llround (frameCount);
        const int stride = stereo ? 2 : 1;
        if (dst < 0 || start < 0 || count <= 0)
            return 0.0;

        const auto needed = (int64_t) dst + (int64_t) count * stride;
        if (needed > state->memN)
            jsfx_ensure_mem (state, needed);
        if (state->mem == nullptr || needed > state->memN)
            return 0.0;

        const auto sid = (std::uint64_t) std::llround (sampleId);
        for (int i = 0; i < count; ++i)
        {
            if (stereo)
            {
                double l = 0.0, r = 0.0;
                pool->read2 (sid, (double) (start + i), &l, &r, false);
                state->mem[dst + i * 2 + 0] = l;
                state->mem[dst + i * 2 + 1] = r;
            }
            else
            {
                state->mem[dst + i] = pool->read (sid, 0, (double) (start + i));
            }
        }
        noteTrackedJsfxMemUsed (state, needed);
        return (double) count;
    }



};
}
