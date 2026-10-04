// SPDX-License-Identifier: Zlib
// Actual AOT entries, actual extracted production bulk helpers, and WDL oracle.
#include "JSFXDSP.h"
#include "JsfxLegacyAtomics.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#define EEL_TARGET_PORTABLE 1
#define EELSCRIPT_NO_LICE 1
#define EELSCRIPT_NO_NET 1
#define EELSCRIPT_NO_FILE 1
#define EELSCRIPT_NO_MDCT 1
#define EELSCRIPT_NO_PREPROC 1
#include "WDL/eel2/eelscript.h"
static std::mutex eelMutex, eelAtomicMutex;
extern "C" void NSEEL_HOSTSTUB_EnterMutex() { eelMutex.lock(); }
extern "C" void NSEEL_HOSTSTUB_LeaveMutex() { eelMutex.unlock(); }
#define EEL_ATOMIC_SET_SCOPE(opaque)
#define EEL_ATOMIC_ENTER eelAtomicMutex.lock()
#define EEL_ATOMIC_LEAVE eelAtomicMutex.unlock()
#include "WDL/eel2/eel_atomic.h"
namespace juce {
template <class... T> void ignoreUnused(const T &...) {}
} // namespace juce
#include "numeric_runtime.inc"
extern "C" void jsfx_ensure_mem(DSPJSFX_State *st, int64_t needed) {
  if (needed <= 0 || needed > st->memN || !st->mem)
    st->memoryFault = 1;
}
static void require(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
static int var(const char *name) {
  for (const auto &v : DSPJSFX_VARS)
    if (!std::strcmp(v.name, name))
      return v.index;
  throw std::runtime_error(std::string("missing variable ") + name);
}
static std::map<std::string, std::string> sections(const char *path) {
  std::ifstream f(path);
  require(bool(f), "missing fixture");
  std::map<std::string, std::string> result;
  std::string line, section;
  while (std::getline(f, line)) {
    if (line.starts_with('@'))
      section = line.substr(1, line.find(' ') - 1);
    else if (!section.empty())
      result[section] += line + '\n';
  }
  return result;
}
int main(int argc, char **argv) try {
  require(argc == 2, "pass legacy_shared_probe.jsfx");
  std::vector<DSPJSFX_Cell> memory(DSPJSFX_MAX_MEM_CELLS);
  std::mutex atomicMutex;
  DSPJSFX_State dsp{}, gfx{};
  dsp.mem = memory.data();
  dsp.memN = memory.size();
  dsp.atomicContext = &atomicMutex;
  dsp.srate = 48000;
  dsp.samplesblock = 64;
  gfx.sharedState = &dsp;
  gfx.mem = dsp.mem;
  gfx.memN = dsp.memN;
  gfx.atomicContext = &atomicMutex;
  const auto heap = dsp.mem;
  jsfx_init(&dsp);

  NSEEL_init();
  eelScriptInst::init();
  EEL_atomic_register();
  eelScriptInst ref;
  auto source = sections(argv[1]);
  std::map<std::string, NSEEL_CODEHANDLE> handles;
  for (auto n : {"init", "sample", "gfx"}) {
    const char *error = nullptr;
    handles[n] = ref.compile_code(source[n].c_str(), &error);
    if (!handles[n])
      throw std::runtime_error(error ? error : "WDL compile failure");
  }
  NSEEL_code_execute(handles["init"]);
  int compared = 0;
  for (auto n : {"init_local", "set_result", "add_result", "cas_old",
                 "cas_failed", "exch_other", "exch_result", "get_result",
                 "sqrt_literal", "sqrt_dynamic", "sqrt_expression"}) {
    require(double(dsp.vars[var(n)]) == *NSEEL_VM_regvar(ref.m_vm, n),
            "atomic return/local parity failed");
    ++compared;
  }
  dsp.sliders[0] = 5;
  *NSEEL_VM_regvar(ref.m_vm, "slider1") = 5;
  for (int i = 0; i < 2; ++i) {
    jsfx_gfx_aot(&gfx);
    NSEEL_code_execute(handles["gfx"]);
    jsfx_sample(&dsp);
    NSEEL_code_execute(handles["sample"]);
  }
  for (auto n : {"audio_value", "block_local", "gfx_local", "audio_instance",
                 "gfx_instance"}) {
    require(double(dsp.vars[var(n)]) == *NSEEL_VM_regvar(ref.m_vm, n),
            "cross-context guest-state parity failed");
    ++compared;
  }
  require(double(dsp.vars[var("audio_value")]) == 15,
          "GFX writes not visible to audio");
  require(double(dsp.mem[65535]) == 17, "last heap cell failed");
  require(double(dsp.sliders[1]) == 9 &&
              double(dsp.vars[var("alias_seen")]) == 18,
          "Named slider alias is not the canonical shared slider cell");

  // Numerically compare production FFT gather/compute/commit with WDL EEL.
  for (int i = 0; i < 128; ++i) {
    const double value = std::sin(i * .17) + .1 * std::cos(i * .7);
    dsp.mem[4096 + i] = value;
    int available = 0;
    *NSEEL_VM_getramptr(ref.m_vm, 4096 + i, &available) = value;
  }
  const char *error = nullptr;
  auto fft =
      ref.compile_code("fft_real(4096,128);fft_permute(4096,64);", &error);
  require(fft != nullptr, "WDL FFT compile failed");
  NSEEL_code_execute(fft);
  jsfx_fft_real(&gfx, 4096, 128);
  jsfx_fft_permute(&gfx, 4096, 64);
  double fftError = 0;
  for (int i = 0; i < 128; ++i) {
    int available = 0;
    fftError =
        std::max(fftError,
                 std::abs(double(dsp.mem[4096 + i]) -
                          *NSEEL_VM_getramptr(ref.m_vm, 4096 + i, &available)));
  }
  require(fftError < 1e-10, "FFT numerical parity failed");

  int bulkCases = 0;
  auto bulk = [&](const char *code, auto native, int base, int count) {
    for (int i = 0; i < count; ++i) {
      const double v = std::sin(i * .17) + .1 * std::cos(i * .7);
      dsp.mem[base + i] = v;
      int available = 0;
      *NSEEL_VM_getramptr(ref.m_vm, base + i, &available) = v;
    }
    auto handle = ref.compile_code(code, &error);
    require(handle != nullptr, "WDL bulk compile failed");
    NSEEL_code_execute(handle);
    native();
    for (int i = 0; i < count; ++i) {
      int available = 0;
      fftError = std::max(
          fftError,
          std::abs(double(dsp.mem[base + i]) -
                   *NSEEL_VM_getramptr(ref.m_vm, base + i, &available)));
    }
    require(fftError < 1e-10, code);
    ++bulkCases;
  };
  bulk(
      "fft(4096,64);fft_permute(4096,64);fft_ipermute(4096,64);ifft(4096,64);",
      [&] {
        jsfx_fft(&gfx, 4096, 64);
        jsfx_fft_permute(&gfx, 4096, 64);
        jsfx_fft_ipermute(&gfx, 4096, 64);
        jsfx_ifft(&gfx, 4096, 64);
      },
      4096, 128);
  bulk(
      "fft_real(4096,128);ifft_real(4096,128);",
      [&] {
        jsfx_fft_real(&gfx, 4096, 128);
        jsfx_ifft_real(&gfx, 4096, 128);
      },
      4096, 128);
  bulk(
      "convolve_c(4096,4160,32);",
      [&] { jsfx_convolve_c(&gfx, 4096, 4160, 32); }, 4096, 128);
  bulk(
      "convolve_c(4096,4096,32);",
      [&] { jsfx_convolve_c(&gfx, 4096, 4096, 32); }, 4096, 64);
  bulk(
      "memcpy(4100,4096,32);memcpy(4096,4100,32);",
      [&] {
        jsfx_memcpy(&gfx, 4100, 4096, 32);
        jsfx_memcpy(&gfx, 4096, 4100, 32);
      },
      4096, 128);
  bulk(
      "fft(0,32768);ifft(0,32768);",
      [&] {
        jsfx_fft(&gfx, 0, 32768);
        jsfx_ifft(&gfx, 0, 32768);
      },
      0, 65536);
  bulk(
      "fft_real(0,32768);ifft_real(0,32768);",
      [&] {
        jsfx_fft_real(&gfx, 0, 32768);
        jsfx_ifft_real(&gfx, 0, 32768);
      },
      0, 32768);
  dsp.mem[2] = dsp.mem[3] = 0;
  dsp.mem[65535] = 17;

  constexpr int iterations = 100000;
  auto runPair = [&](int count) {
    std::atomic<bool> start{false};
    auto worker = [&](bool ui) {
      while (!start.load(std::memory_order_acquire)) {
      };
      for (int i = 0; i < count; ++i) {
        if (ui)
          jsfx_gfx_aot(&gfx);
        else
          jsfx_sample(&dsp);
      }
    };
    std::thread audio(worker, false), ui(worker, true);
    start.store(true, std::memory_order_release);
    audio.join();
    ui.join();
  };
  dsp.vars[var("phase")] = 1;
  runPair(iterations);
  require(double(dsp.vars[var("counter")]) == 2 * iterations,
          "atomic_add lost updates");
  dsp.vars[var("phase")] = 2;
  runPair(iterations);
  require(double(dsp.vars[var("bad")]) == 0,
          "explicit publication ordering failed");
  for (int i = 100; i < 164; ++i)
    dsp.mem[i] = 0;
  dsp.vars[var("phase")] = 3;
  runPair(iterations);
  require(double(dsp.vars[var("audio_bad")]) == 0,
          "live bulk access produced invalid cell values");
  require(dsp.mem == heap && gfx.mem == heap, "heap address changed");
  dsp.vars[var("phase")] = 5;
  dsp.vars[var("doubled")] = 1;
  runPair(10);
  require(double(dsp.vars[var("doubled")]) == 1048576,
          "Atomic addend reference escaped the critical section");
  dsp.vars[var("phase")] = 4;
  jsfx_gfx_aot(&gfx);
  require(gfx.memoryFault == 1 && dsp.memoryFault == 0,
          "heap bounds did not fault the invoking context");
  require(double(dsp.mem[65535]) == 17,
          "out of bounds write damaged last cell");
  std::cout << "{\"wdl_scalar_checks\":" << compared
            << ",\"fft_max_error\":" << fftError
            << ",\"bulk_oracle_cases\":" << bulkCases
            << ",\"atomic_increments\":" << 2 * iterations
            << ",\"publication_errors\":0,\"bulk_errors\":0,"
               "\"aliased_atomic_operand\":true,\"heap_stable\":true,\"bounds_"
               "fault\":true}\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
