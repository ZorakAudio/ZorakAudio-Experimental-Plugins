// SPDX-License-Identifier: Zlib
// Define JSFX_FILE_RUNTIME_OWNER(state) before including once.
#pragma once
extern "C" double jsfx_file_open (DSPJSFX_State* state, double indexOrSlot, double mode)
{
    if (auto* owner = JSFX_FILE_RUNTIME_OWNER(state))
        return owner->rt_file_open (state, indexOrSlot, mode);

    return -1.0;
}

extern "C" double jsfx_file_open_multi (DSPJSFX_State* state, double indexOrSlot, double mode)
{
    if (auto* owner = JSFX_FILE_RUNTIME_OWNER(state))
        return owner->rt_file_open_multi (state, indexOrSlot, mode);

    return -1.0;
}

extern "C" double jsfx_file_close (DSPJSFX_State* state, double handle)
{
    if (auto* owner = JSFX_FILE_RUNTIME_OWNER(state))
        return owner->rt_file_close (state, handle);

    return 0.0;
}

extern "C" double jsfx_file_rewind (DSPJSFX_State* state, double handle)
{
    if (auto* owner = JSFX_FILE_RUNTIME_OWNER(state))
        return owner->rt_file_rewind (state, handle);

    return 0.0;
}

extern "C" double jsfx_file_seek (DSPJSFX_State* state, double handle, double offset)
{
    if (auto* owner = JSFX_FILE_RUNTIME_OWNER(state))
        return owner->rt_file_seek (state, handle, offset);

    return 0.0;
}

extern "C" double jsfx_file_avail (DSPJSFX_State* state, double handle)
{
    if (auto* owner = JSFX_FILE_RUNTIME_OWNER(state))
        return owner->rt_file_avail (state, handle);

    return 0.0;
}

extern "C" double jsfx_file_text (DSPJSFX_State* state, double handle)
{
    if (auto* owner = JSFX_FILE_RUNTIME_OWNER(state))
        return owner->rt_file_text (state, handle);

    return 0.0;
}

extern "C" double jsfx_file_riff (DSPJSFX_State* state, double handle, double* outNch, double* outSr)
{
    if (auto* owner = JSFX_FILE_RUNTIME_OWNER(state))
        return owner->rt_file_riff (state, handle, outNch, outSr);

    if (outNch) jsfxCellStore (outNch, 0.0);
    if (outSr)  jsfxCellStore (outSr, 0.0);
    return 0.0;
}

extern "C" double jsfx_file_var (DSPJSFX_State* state, double handle, double* outVar)
{
    if (auto* owner = JSFX_FILE_RUNTIME_OWNER(state))
        return owner->rt_file_var (state, handle, outVar);

    if (outVar) jsfxCellStore (outVar, 0.0);
    return 0.0;
}

extern "C" double jsfx_file_mem (DSPJSFX_State* state, double handle, double destIndex, double length)
{
    if (auto* owner = JSFX_FILE_RUNTIME_OWNER(state))
        return owner->rt_file_mem (state, handle, destIndex, length);

    return 0.0;
}

extern "C" double jsfx_file_multi_count (DSPJSFX_State* state, double handle)
{
    if (auto* owner = JSFX_FILE_RUNTIME_OWNER(state))
        return owner->rt_file_multi_count (state, handle);

    return 0.0;
}

extern "C" double jsfx_file_multi_select (DSPJSFX_State* state, double handle, double fileIndex)
{
    if (auto* owner = JSFX_FILE_RUNTIME_OWNER(state))
        return owner->rt_file_multi_select (state, handle, fileIndex);

    return 0.0;
}

