#include "DspJsfxRuntime.h"
#include "JSFXDSP.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

extern "C" void jsfx_ensure_mem(DSPJSFX_State* st, int64_t needed);
extern "C" std::uint64_t jsfx_string_hash(DSPJSFX_State* st, double strHandle);
extern "C" int jsfx_string_assign_utf8(DSPJSFX_State* st, double* slot, const char* data, int len);

namespace
{
using za::jsfx::DspJsfxRuntime;
using za::jsfx::kDspJsfxDefaultGmemCells;

static DspJsfxRuntime* runtimeFor(DSPJSFX_State* st) noexcept
{
    return DspJsfxRuntime::findForState(st);
}

static std::uint64_t toU64(double v) noexcept
{
    if (! std::isfinite(v) || v <= 0.0)
        return 0;
    if (v >= static_cast<double> (std::numeric_limits<std::uint64_t>::max()))
        return std::numeric_limits<std::uint64_t>::max();
    return static_cast<std::uint64_t> (std::llround(v));
}

static int toInt(double v) noexcept
{
    if (! std::isfinite(v))
        return 0;
    if (v <= static_cast<double> (std::numeric_limits<int>::min()))
        return std::numeric_limits<int>::min();
    if (v >= static_cast<double> (std::numeric_limits<int>::max()))
        return std::numeric_limits<int>::max();
    return static_cast<int> (std::llround(v));
}

static std::uint64_t hashedStringHandle(DSPJSFX_State* st, double handle) noexcept
{
    return jsfx_string_hash(st, handle);
}

static double* memPtrForReceive(DSPJSFX_State* st, double dstBase, int count) noexcept
{
    if (st == nullptr || count <= 0)
        return nullptr;
    const int dst = toInt(dstBase);
    if (dst < 0)
        return nullptr;
    const auto need = static_cast<int64_t> (dst) + static_cast<int64_t> (count);
    if (need > st->memN)
        jsfx_ensure_mem(st, need);
    if (st->mem == nullptr || need > st->memN)
        return nullptr;
    return st->mem + dst;
}
} // namespace

extern "C" double jsfx_instance_id(DSPJSFX_State* st)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return static_cast<double> (rt->instanceId());
    return 0.0;
}

extern "C" int jsfx_instance_uid(DSPJSFX_State* st, double* outStr)
{
    if (outStr == nullptr)
        return 0;
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return jsfx_string_assign_utf8(st, outStr, rt->instanceUid().c_str(), static_cast<int> (rt->instanceUid().size()));
    return 0;
}

extern "C" int jsfx_instance_set_name(DSPJSFX_State* st, double strHandle)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->setNameHandle(static_cast<std::int64_t> (std::llround(strHandle))) ? 1 : 0;
    return 0;
}

extern "C" int jsfx_instance_get_name(DSPJSFX_State* st, double* outStr)
{
    if (outStr == nullptr)
        return 0;
    if (auto* rt = runtimeFor(st); rt != nullptr)
    {
        *outStr = static_cast<double> (rt->instanceNameHandle());
        return 1;
    }
    return 0;
}

extern "C" int jsfx_comm_join(DSPJSFX_State* st, double domainHandle)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->joinDomain(hashedStringHandle(st, domainHandle)) ? 1 : 0;
    return 0;
}

extern "C" int jsfx_gmem_attach(DSPJSFX_State* st, double nameHandle)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->gmemAttach(hashedStringHandle(st, nameHandle), kDspJsfxDefaultGmemCells) ? 1 : 0;
    return 0;
}

extern "C" int jsfx_gmem_attach_size(DSPJSFX_State* st, double nameHandle, double cells)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
    {
        const std::uint64_t requested = std::max<std::uint64_t> (toU64(cells), kDspJsfxDefaultGmemCells);
        return rt->gmemAttach(hashedStringHandle(st, nameHandle), requested) ? 1 : 0;
    }
    return 0;
}

extern "C" double jsfx_gmem_size(DSPJSFX_State* st)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->gmemSize();
    return 0.0;
}

extern "C" double jsfx_gmem_load(DSPJSFX_State* st, double idx)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->gmemLoad(idx);
    return 0.0;
}

extern "C" double jsfx_gmem_store(DSPJSFX_State* st, double idx, double value)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->gmemStore(idx, value);
    return 0.0;
}

extern "C" int jsfx_gmem_get(DSPJSFX_State* st, double dstBase, double srcIdx, double count)
{
    if (st == nullptr)
        return 0;
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return static_cast<int> (rt->gmemGet(*st, dstBase, srcIdx, count));
    return 0;
}

extern "C" int jsfx_gmem_put(DSPJSFX_State* st, double dstIdx, double srcBase, double count)
{
    if (st == nullptr)
        return 0;
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return static_cast<int> (rt->gmemPut(*st, dstIdx, srcBase, count));
    return 0;
}

extern "C" int jsfx_gmem_fill(DSPJSFX_State* st, double dstIdx, double value, double count)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return static_cast<int> (rt->gmemFill(dstIdx, value, count));
    return 0;
}

extern "C" int jsfx_gmem_zero(DSPJSFX_State* st, double dstIdx, double count)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return static_cast<int> (rt->gmemZero(dstIdx, count));
    return 0;
}

extern "C" int jsfx_gmem_copy(DSPJSFX_State* st, double dstIdx, double srcIdx, double count)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return static_cast<int> (rt->gmemCopy(dstIdx, srcIdx, count));
    return 0;
}

extern "C" double jsfx_gmem_seq(DSPJSFX_State* st, double page)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->gmemSeq(page);
    return 0.0;
}

extern "C" double jsfx_gmem_page(DSPJSFX_State* st, double idx)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->gmemPage(idx);
    return 0.0;
}

extern "C" int jsfx_msg_subscribe(DSPJSFX_State* st, double chanHandle)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->subscribe(hashedStringHandle(st, chanHandle)) ? 1 : 0;
    return 0;
}

extern "C" int jsfx_msg_unsubscribe(DSPJSFX_State* st, double chanHandle)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->unsubscribe(hashedStringHandle(st, chanHandle)) ? 1 : 0;
    return 0;
}

extern "C" int jsfx_msg_advertise(DSPJSFX_State* st, double chanHandle, double caps)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->advertise(hashedStringHandle(st, chanHandle), toU64(caps)) ? 1 : 0;
    return 0;
}

extern "C" int jsfx_msg_send(DSPJSFX_State* st, double chanHandle, double tag, double a, double b, double c, double d)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->queueScalar(hashedStringHandle(st, chanHandle), tag, a, b, c, d) ? 1 : 0;
    return 0;
}

extern "C" int jsfx_msg_sendto(DSPJSFX_State* st, double targetId, double chanHandle, double tag, double a, double b, double c, double d)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->queueScalarTo(toU64(targetId), hashedStringHandle(st, chanHandle), tag, a, b, c, d) ? 1 : 0;
    return 0;
}

extern "C" double jsfx_msg_avail(DSPJSFX_State* st, double chanHandle)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return static_cast<double> (rt->avail(hashedStringHandle(st, chanHandle)));
    return 0.0;
}

extern "C" double jsfx_msg_kind(DSPJSFX_State* st, double chanHandle)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return static_cast<double> (rt->kind(hashedStringHandle(st, chanHandle)));
    return 0.0;
}

extern "C" int jsfx_msg_recv(DSPJSFX_State* st, double chanHandle, double* src, double* tag, double* a, double* b, double* c, double* d)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->recvScalar(hashedStringHandle(st, chanHandle), src, tag, a, b, c, d);
    return 0;
}

extern "C" int jsfx_msg_send_buf(DSPJSFX_State* st, double chanHandle, double tag, double srcBase, double len)
{
    if (st == nullptr || st->mem == nullptr)
        return 0;
    const int base = toInt(srcBase);
    const int n = toInt(len);
    if (base < 0 || n <= 0 || static_cast<int64_t> (base) + static_cast<int64_t> (n) > st->memN)
        return 0;
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->queueBuffer(hashedStringHandle(st, chanHandle), tag, st->mem + base, n) ? 1 : 0;
    return 0;
}

extern "C" int jsfx_msg_sendto_buf(DSPJSFX_State* st, double targetId, double chanHandle, double tag, double srcBase, double len)
{
    if (st == nullptr || st->mem == nullptr)
        return 0;
    const int base = toInt(srcBase);
    const int n = toInt(len);
    if (base < 0 || n <= 0 || static_cast<int64_t> (base) + static_cast<int64_t> (n) > st->memN)
        return 0;
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->queueBufferTo(toU64(targetId), hashedStringHandle(st, chanHandle), tag, st->mem + base, n) ? 1 : 0;
    return 0;
}

extern "C" int jsfx_msg_recv_buf(DSPJSFX_State* st, double chanHandle, double* src, double* tag, double dstBase, double maxLen)
{
    if (st == nullptr)
        return 0;
    if (auto* rt = runtimeFor(st); rt != nullptr)
    {
        const int capacity = std::max(0, toInt(maxLen));
        if (capacity <= 0)
            return 0;
        double* dst = memPtrForReceive(st, dstBase, capacity);
        if (dst == nullptr)
        {
            const int base = toInt(dstBase);
            if (base < 0)
                return 0;
            const auto need = static_cast<int64_t> (base) + static_cast<int64_t> (capacity);
            jsfx_ensure_mem(st, need);
            if (st->mem == nullptr || need > st->memN)
                return 0;
            dst = st->mem + base;
        }
        return rt->recvBuffer(hashedStringHandle(st, chanHandle), src, tag, dst, capacity, *st);
    }
    return 0;
}

extern "C" double jsfx_msg_length(DSPJSFX_State* st)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->lastMessageLength();
    return 0.0;
}

extern "C" double jsfx_msg_dropped(DSPJSFX_State* st, double chanHandle)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->dropped(hashedStringHandle(st, chanHandle));
    return 0.0;
}

extern "C" int jsfx_msg_clear(DSPJSFX_State* st, double chanHandle)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->clear(hashedStringHandle(st, chanHandle));
    return 0;
}

extern "C" double jsfx_msg_peer_count(DSPJSFX_State* st, double chanHandle, double role)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->peerCount(hashedStringHandle(st, chanHandle), toInt(role));
    return 0.0;
}

extern "C" double jsfx_msg_peer_id(DSPJSFX_State* st, double chanHandle, double role, double index)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->peerId(hashedStringHandle(st, chanHandle), toInt(role), toInt(index));
    return 0.0;
}

extern "C" int jsfx_msg_peer_name(DSPJSFX_State* st, double peerId, double* outStr)
{
    if (outStr == nullptr)
        return 0;
    if (auto* rt = runtimeFor(st); rt != nullptr)
    {
        std::int64_t handle = 0;
        if (! rt->peerNameHandle(toU64(peerId), &handle))
            return 0;
        *outStr = static_cast<double> (handle);
        return 1;
    }
    return 0;
}

extern "C" int jsfx_msg_peer_uid(DSPJSFX_State* st, double peerId, double* outStr)
{
    if (outStr == nullptr)
        return 0;
    if (auto* rt = runtimeFor(st); rt != nullptr)
    {
        std::string uid;
        if (! rt->peerUid(toU64(peerId), &uid))
            return 0;
        return jsfx_string_assign_utf8(st, outStr, uid.c_str(), static_cast<int> (uid.size()));
    }
    return 0;
}

extern "C" double jsfx_msg_peer_caps(DSPJSFX_State* st, double peerId)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->peerCaps(toU64(peerId));
    return 0.0;
}

extern "C" double jsfx_msg_peer_alive(DSPJSFX_State* st, double peerId)
{
    if (auto* rt = runtimeFor(st); rt != nullptr)
        return rt->peerAlive(toU64(peerId));
    return 0.0;
}
