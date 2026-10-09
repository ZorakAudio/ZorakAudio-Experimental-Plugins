// Exercise the production mapping's setup lock, names and attachment contract.
#include "DspJsfxSharedMemory.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
// Use the actual gmem layout and stem helper, not a second test implementation.
#include "../../src/DspJsfxGmem.cpp"
extern "C" void jsfx_ensure_mem(DSPJSFX_State*, int64_t)
{
    std::abort(); // This mapping-only fixture must not resize EEL memory.
}
#if !defined(_WIN32)
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

int main(int argc, char** argv)
{
    using namespace za::jsfx;
    constexpr std::size_t bytes = 4096;
    if (argc == 3)
    {
        DspJsfxSharedMemorySegment abandoned;
        bool created = false;
        assert(abandoned.openOrCreate(argv[2], bytes, &created) && created);
        *static_cast<std::uint64_t*>(abandoned.data()) = 0xabcdef1234567890ull;
        // Deliberately bypass destructors: OS cleanup must release the lock.
        std::_Exit(0);
    }

    const auto suffix = std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const auto stem = "gmem_v2_ffffffffffffffff_ffffffffffffffff_test_" + suffix;
    const auto name = makeSharedMemoryObjectName(stem);
    const auto ipcName = makeSharedMemoryObjectName("msg_v4_ffffffffffffffff");
#if defined(__APPLE__) || defined(ZA_JSFX_TEST_DARWIN_SHM)
    assert(name.size() <= 31 && ipcName.size() <= 31);
#elif defined(_WIN32)
    assert(ipcName == "Local\\za_jsfx_msg_v4_ffffffffffffffff");
#else
    assert(ipcName == "/za_jsfx_msg_v4_ffffffffffffffff");
#endif
    assert(name == makeSharedMemoryObjectName(stem));
    assert(name != makeSharedMemoryObjectName(stem + "_different_namespace"));
    assert(makeSharedMemoryObjectName("") == makeSharedMemoryObjectName("default"));
    assert(makeSharedMemoryObjectName("Case.Name") == makeSharedMemoryObjectName("case_name"));

    DspJsfxSharedMemorySegment first;
    bool created = false;
    assert(first.openOrCreate(stem, bytes, &created) && created);
    *static_cast<std::uint64_t*>(first.data()) = 0x123456789abcdef0ull;
    // Move ownership while initialization is still locked.
    DspJsfxSharedMemorySegment owner(std::move(first));
    assert(!first.isOpen() && owner.isOpen());
    bool timedOut = false;
    std::thread blocked([&] {
        DspJsfxSharedMemorySegment next;
        bool nextCreated = true;
        timedOut = !next.openOrCreate(stem, bytes, &nextCreated);
        assert(!nextCreated);
    });
    blocked.join();
    assert(timedOut); // No reader can attach to an unpublished header.
    owner.finishInitialization();
    owner.finishInitialization(); // Idempotent; releases no unrelated lock.

    DspJsfxSharedMemorySegment attached;
    created = true;
    assert(attached.openOrCreate(stem, bytes, &created) && !created);
    assert(*static_cast<std::uint64_t*>(attached.data()) == 0x123456789abcdef0ull);
    attached.finishInitialization();
    *static_cast<std::uint64_t*>(attached.data()) = 7;
    assert(*static_cast<std::uint64_t*>(owner.data()) == 7);
    DspJsfxSharedMemorySegment moved;
    moved = std::move(attached);
    assert(!attached.isOpen() && moved.isOpen());
    // A fresh attachment observes the established backing; the creator's view
    // can still be the original requested length on POSIX.
    const auto backedBytes = moved.size();
    moved.close();

    // Reject requests beyond ACTUAL backing, including OS page padding. Darwin
    // arm64 can back a 4096-byte request with a 16384-byte page, so bytes * 4
    // is a valid attachment there, not an oversized request.
    DspJsfxSharedMemorySegment oversized;
    assert(backedBytes >= bytes);
    assert(!oversized.openOrCreate(stem, backedBytes + 1, &created));
    assert(!oversized.isOpen());
    assert(*static_cast<std::uint64_t*>(owner.data()) == 7);
    assert(oversized.openOrCreate(stem, bytes, &created) && !created);
    assert(oversized.size() == backedBytes);
    oversized.finishInitialization();
    oversized.close();
    assert(!oversized.openOrCreate(stem, 0, &created) && !created);
    owner.close();

    // Long production gmem names preserve domain/namespace isolation and data
    // when another instance attaches to the initialized atomic-cell layout.
    const auto domain = static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    constexpr std::uint64_t firstNamespace = 0xfffffffffffffff1ull;
    constexpr std::uint64_t secondNamespace = 0xfffffffffffffff2ull;
    DspJsfxGmemAttachment writer, reader, separate;
    assert(writer.attach(domain, firstNamespace, 0, 101));
    assert(writer.cellCount() == kDspJsfxDefaultGmemCells);
    assert(writer.store(1025, 3.25, 101) == 3.25);
    assert(reader.attach(domain, firstNamespace, 0, 102));
    assert(reader.load(1025) == 3.25);
    assert(separate.attach(domain, secondNamespace, 0, 103));
    assert(separate.load(1025) == 0);
    separate.store(1025, -1.5, 103);
    assert(reader.load(1025) == 3.25);
    assert(reader.store(1025, 7.5, 102) == 7.5);
    assert(writer.load(1025) == 7.5);
    writer.detach();reader.detach();separate.detach();

#if !defined(_WIN32)
    // Across processes, a terminated initializer must not strand the lock.
    const auto abandonedStem = stem + "_abandoned";
    const auto child = fork();
    assert(child >= 0);
    if (child == 0)
    {
        execl(argv[0], argv[0], "--abandon", abandonedStem.c_str(), nullptr);
        _exit(127);
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    DspJsfxSharedMemorySegment recovered;
    assert(recovered.openOrCreate(abandonedStem, bytes, &created) && !created);
    assert(*static_cast<std::uint64_t*>(recovered.data()) == 0xabcdef1234567890ull);
    recovered.finishInitialization();
    recovered.close();
    assert(shm_unlink(makeSharedMemoryObjectName(abandonedStem).c_str()) == 0);
    assert(shm_unlink(name.c_str()) == 0);
    assert(shm_unlink(makeSharedMemoryObjectName(objectStem(domain, firstNamespace)).c_str()) == 0);
    assert(shm_unlink(makeSharedMemoryObjectName(objectStem(domain, secondNamespace)).c_str()) == 0);
#endif
    std::cout << "Shared-memory names, publication lock, moves, sizing, gmem isolation and cleanup passed\n";
}
