#pragma once

#include <algorithm>
#include <cmath>
#include <mutex>

// Extra JSFX @gfx/EEL2 compatibility builtins for DSP-only comm APIs.
//
// The lightweight @gfx VM compiles @init alongside @gfx so helper functions
// defined in @init remain visible to UI code. DSP communication APIs are owned
// by the AOT/audio runtime, not the @gfx VM. These inert stubs make comm-enabled
// scripts compile in the @gfx VM without giving UI code direct bus/gmem access.
//
// Keep this separate from YSFXGfxInterpreter.h so the monolithic interpreter can
// remain close to upstream while plugin-specific DSP-JSFX builtins evolve.
namespace jsfx_gfx_compat
{

static EEL_F NSEEL_CGEN_CALL eel_return_zero (void* opaque, INT_PTR np, EEL_F** parms)
{
    (void) opaque;
    (void) np;
    (void) parms;
    return 0.0;
}

static EEL_F NSEEL_CGEN_CALL eel_return_one (void* opaque, INT_PTR np, EEL_F** parms)
{
    (void) opaque;
    (void) np;
    (void) parms;
    return 1.0;
}

static EEL_F NSEEL_CGEN_CALL eel_gfx_drawnumber (void* opaque, INT_PTR np, EEL_F** parms)
{
    auto* self = (jsfx_gfx::GfxVm*) opaque;
    if (self == nullptr || np < 1 || parms == nullptr || parms[0] == nullptr)
        return 0.0;

    const double value = (double) *parms[0];
    int digits = 0;
    if (np >= 2 && parms[1] != nullptr)
        digits = (int) std::llround ((double) *parms[1]);

    juce::String text;
    if (std::isfinite (value))
    {
        if (digits >= 0 && digits <= 24)
            text = juce::String (value, digits);
        else
            text = juce::String (value, 15);
    }
    else if (std::isnan (value))
    {
        text = "nan";
    }
    else
    {
        text = value < 0.0 ? "-inf" : "inf";
    }

    return jsfx_gfx::GfxVm::emitTextCommand (self, text, 1, parms);
}

inline void registerBuiltins()
{
    static std::once_flag once;
    std::call_once (once, []()
    {
        // GfxVm owns the base EEL/WDL initialization and registers the normal
        // gfx_* builtins from inside its constructor. If this compatibility
        // table is registered before GfxVm has run, a later NSEEL_init() from
        // GfxVm can erase these entries on some WDL builds. Force the base VM
        // initialization first, then append the DSP-JSFX compatibility stubs.
        jsfx_gfx::GfxVm warmup;
        (void) warmup;

        NSEEL_addfunc_varparm_ex ("gfx_drawnumber", 1, 0, NSEEL_PProc_THIS, &eel_gfx_drawnumber, nullptr);

        // Identity / domain. Return harmless values inside @gfx.
        NSEEL_addfunc_varparm_ex ("instance_id", 0, 0, NSEEL_PProc_THIS, &eel_return_one, nullptr);
        NSEEL_addfunc_varparm_ex ("instance_uid", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("instance_set_name", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("instance_get_name", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("comm_join", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);

        // gmem compatibility. Scalar gmem[] compiles as ordinary EEL memory in
        // the UI VM; the DSP-owned shared-memory namespace is not exposed here.
        NSEEL_addfunc_varparm_ex ("gmem_attach", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("gmem_attach_size", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("gmem_size", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("gmem_get", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("gmem_put", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("gmem_fill", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("gmem_zero", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("gmem_copy", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("gmem_seq", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("gmem_page", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);

        // Message bus compatibility. These are DSP/audio-thread APIs; @gfx sees
        // mirrored vars/mem only. Return empty/no-op behavior in the UI VM.
        NSEEL_addfunc_varparm_ex ("msg_subscribe", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_unsubscribe", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_advertise", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_send", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_sendto", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_avail", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_kind", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_recv", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_send_buf", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_sendto_buf", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_recv_buf", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_length", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_dropped", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_clear", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_peer_count", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_peer_id", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_peer_name", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_peer_uid", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_peer_caps", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("msg_peer_alive", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);

        // Sample-pool compatibility. Real raw access is DSP-owned; @gfx should
        // read mirrored summary vars or preview-specific access once a UI bridge
        // is added. These stubs keep @init/@gfx compileable.
        NSEEL_addfunc_varparm_ex ("sample_pool_from_slot", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_pool_set_mode", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_pool_set_budget_mb", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_pool_commit", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_pool_state", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_pool_selected", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_pool_loaded", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_pool_failed", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_pool_ram_mb", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_pool_generation", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_get", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_len", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_channels", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_srate", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_peak", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_rms", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_name", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_read", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_read_interp", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_read2", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_read2_interp", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_preview_bins", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_preview_read", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_export_mem", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
        NSEEL_addfunc_varparm_ex ("sample_export_mem2", 0, 0, NSEEL_PProc_THIS, &eel_return_zero, nullptr);
    });
}

} // namespace jsfx_gfx_compat
