// SPDX-License-Identifier: Zlib
//
// JSFX @gfx interpreter bridge (portable EEL2) extracted from the YSFX/WDL toolchain.
//
// This file is intended to be INCLUDED (amalgamated) by JSFXJuceProcessor.cpp to keep
// integration as monolithic as possible.
//
// Dependencies:
//   - WDL (Cockos) headers/sources (zlib license)
//     Expected layout: ./WDL/...
//
// Notes:
//   - Uses EEL_TARGET_PORTABLE to avoid platform-specific JIT/assembly.
//   - Implements a minimal subset of JSFX gfx_* API by recording draw commands
//     and rendering them with JUCE.

#ifndef JSFX_YSFX_GFX_INTERPRETER_INCLUDED
#define JSFX_YSFX_GFX_INTERPRETER_INCLUDED

// -------------------------
// WDL / EEL2 configuration
// -------------------------
#ifndef EEL_TARGET_PORTABLE
#define EEL_TARGET_PORTABLE 1
#endif

// Keep eelscript lean: no file, net, mdct, lice.
#ifndef EELSCRIPT_NO_FILE
#define EELSCRIPT_NO_FILE 1
#endif
#ifndef EELSCRIPT_NO_NET
#define EELSCRIPT_NO_NET 1
#endif
#ifndef EELSCRIPT_NO_MDCT
#define EELSCRIPT_NO_MDCT 1
#endif
// NOTE: do NOT define EELSCRIPT_NO_EVAL.
// WDL's eelscript.h currently defines eval-cache helper methods unconditionally,
// but only declares their members when EELSCRIPT_NO_EVAL is *not* set.
// Defining EELSCRIPT_NO_EVAL therefore breaks compilation on MSVC.

// If the build system defines EELSCRIPT_NO_EVAL globally, undo it for this TU.
#ifdef EELSCRIPT_NO_EVAL
#undef EELSCRIPT_NO_EVAL
#endif
#ifndef EELSCRIPT_NO_PREPROC
#define EELSCRIPT_NO_PREPROC 1
#endif
#ifndef EELSCRIPT_NO_LICE
#define EELSCRIPT_NO_LICE 1
#endif

// -------------------------
// Include EEL2 core sources
// -------------------------
#include "WDL/eel2/ns-eel.h"
#include "WDL/eel2/eelscript.h"

// JUCE is expected to be included by the includer (JSFXJuceProcessor.cpp). If not,
// uncomment the next line.
// #include <JuceHeader.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Windows headers (directly or via JUCE) can define min/max macros.
// That breaks std::min/std::max and produces cryptic MSVC errors.
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// -------------------------
// EEL host stubs (thread safety for global EEL tables)
// -------------------------
static std::mutex g_eelGlobalMutex;
extern "C" void NSEEL_HOSTSTUB_EnterMutex() { g_eelGlobalMutex.lock(); }
extern "C" void NSEEL_HOSTSTUB_LeaveMutex() { g_eelGlobalMutex.unlock(); }

namespace jsfx_gfx
{
// If the AOT header wasn't regenerated with the variable table yet,
// provide a harmless fallback so this file still compiles.
#ifndef DSPJSFX_VARS_COUNT
typedef struct DSPJSFX_VarDesc { const char* name; int32_t index; } DSPJSFX_VarDesc;
static const int32_t DSPJSFX_VARS_COUNT = 0;
// MSVC does not allow zero-sized arrays. Keep a dummy element and expose COUNT=0.
static const DSPJSFX_VarDesc DSPJSFX_VARS[1] = { { "", -1 } };
#endif

// -------------------------
// JSFX section extraction (@gfx, @init, ...)
// -------------------------
struct JsfxSections
{
  std::string init;
  std::string slider;
  std::string block;
  std::string sample;
  std::string gfx;
  int gfxW = 0;
  int gfxH = 0;
  bool hasGfx = false;
};

static inline bool startsWithSection(const std::string& s, const char* sec)
{
  // case-insensitive match for "@sec"
  const size_t n = std::strlen(sec);
  if (s.size() < n + 1) return false;
  if (s[0] != '@') return false;
  for (size_t i = 0; i < n; ++i)
  {
    const char a = (char)std::tolower((unsigned char)s[i + 1]);
    const char b = (char)std::tolower((unsigned char)sec[i]);
    if (a != b) return false;
  }
  return true;
}

static JsfxSections extractJsfxSections(const char* jsfxText)
{
  JsfxSections out;
  if (!jsfxText) return out;

  enum class Sec { None, Init, Slider, Block, Sample, Gfx };
  Sec cur = Sec::None;

  std::string line;
  const char* p = jsfxText;
  while (*p)
  {
    // read one line (preserve newline)
    const char* start = p;
    while (*p && *p != '\n') ++p;
    const char* end = p;
    if (*p == '\n') ++p;
    line.assign(start, end);

    // Trim leading spaces for section detection
    size_t firstNonSpace = line.find_first_not_of(" \t\r");
    const std::string ltrim = (firstNonSpace == std::string::npos) ? std::string() : line.substr(firstNonSpace);

    if (!ltrim.empty() && ltrim[0] == '@')
    {
      if (startsWithSection(ltrim, "init"))  { cur = Sec::Init;  continue; }
      if (startsWithSection(ltrim, "slider")){ cur = Sec::Slider;continue; }
      if (startsWithSection(ltrim, "block")) { cur = Sec::Block; continue; }
      if (startsWithSection(ltrim, "sample")){ cur = Sec::Sample;continue; }
      if (startsWithSection(ltrim, "gfx"))
      {
        cur = Sec::Gfx;
        out.hasGfx = true;
        // Parse optional size: "@gfx <w> <h>"
        int w = 0, h = 0;
        // very permissive parse
        if (std::sscanf(ltrim.c_str(), "@gfx %d %d", &w, &h) == 2)
        {
          out.gfxW = w;
          out.gfxH = h;
        }
        continue;
      }
      // Unknown @section: stop capturing until next known section.
      cur = Sec::None;
      continue;
    }

    // Append to current section
    switch (cur)
    {
      case Sec::Init:   out.init.append(line).push_back('\n'); break;
      case Sec::Slider: out.slider.append(line).push_back('\n'); break;
      case Sec::Block:  out.block.append(line).push_back('\n'); break;
      case Sec::Sample: out.sample.append(line).push_back('\n'); break;
      case Sec::Gfx:    out.gfx.append(line).push_back('\n'); break;
      default: break;
    }
  }
  return out;
}


// -------------------------
// JSFX -> portable-EEL compatibility shim
//
// Some JSFX scripts use special lvalue forms like:
//   slider(i) = v;
//   spl(i)    = v;
// REAPER's JSFX dialect supports these, but portable EEL2 does not.
// We rewrite them into ordinary function calls:
//   slider(i, v);
//   spl(i, v);
// and provide slider()/spl() builtins below.
// This is a best-effort text transform (not a full parser), but it covers the
// common UI patterns used by many JSFX scripts.
// -------------------------
static inline bool isIdentChar(char c)
{
  return std::isalnum((unsigned char)c) || c == '_';
}

static std::string preprocessJsfxForPortableEel(const std::string& in)
{
  std::string out;
  out.reserve(in.size());

  bool inLineComment = false;
  bool inBlockComment = false;
  bool inString = false;
  char strQuote = 0;

  auto tryRewriteAssign = [&](size_t& i, const char* name) -> bool
  {
    const size_t nlen = std::strlen(name);
    if (i + nlen + 1 >= in.size()) return false;
    if (in.compare(i, nlen, name) != 0) return false;

    // Word boundary: avoid matching "myslider(...)" etc.
    if (i > 0 && isIdentChar(in[i - 1])) return false;
    if (i + nlen < in.size() && isIdentChar(in[i + nlen])) return false;

    const size_t parenStart = i + nlen;
    if (in[parenStart] != '(') return false;

    // Find matching ')', respecting nested parens and strings.
    size_t p = parenStart + 1;
    int depth = 1;
    bool s = false;
    char q = 0;

    while (p < in.size() && depth > 0)
    {
      const char c = in[p];

      if (s)
      {
        if (c == '\\' && p + 1 < in.size()) { p += 2; continue; }
        if (c == q) { s = false; ++p; continue; }
        ++p;
        continue;
      }

      if (c == '"' || c == '\'') { s = true; q = c; ++p; continue; }
      if (c == '(') { ++depth; ++p; continue; }
      if (c == ')') { --depth; ++p; continue; }
      ++p;
    }

    if (depth != 0) return false;

    const size_t parenEnd = p - 1; // index of ')'

    // Look for assignment after ")"
    size_t a = p;
    while (a < in.size() && std::isspace((unsigned char)in[a])) ++a;

    // Only rewrite plain "=", not "=="
    if (a >= in.size() || in[a] != '=') return false;
    if (a + 1 < in.size() && in[a + 1] == '=') return false;

    // Parse RHS up to ';' at top level
    size_t rhsStart = a + 1;
    while (rhsStart < in.size() && std::isspace((unsigned char)in[rhsStart])) ++rhsStart;

    size_t r = rhsStart;
    int par = 0, br = 0, cr = 0;
    bool rs = false;
    char rq = 0;

    while (r < in.size())
    {
      const char c = in[r];

      if (rs)
      {
        if (c == '\\' && r + 1 < in.size()) { r += 2; continue; }
        if (c == rq) { rs = false; ++r; continue; }
        ++r;
        continue;
      }

      // stop at end-of-statement
      if (c == ';' && par == 0 && br == 0 && cr == 0)
        break;

      if (c == '"' || c == '\'') { rs = true; rq = c; ++r; continue; }
      if (c == '(') { ++par; ++r; continue; }
      if (c == ')' && par > 0) { --par; ++r; continue; }
      if (c == '[') { ++br; ++r; continue; }
      if (c == ']' && br > 0) { --br; ++r; continue; }
      if (c == '{') { ++cr; ++r; continue; }
      if (c == '}' && cr > 0) { --cr; ++r; continue; }

      ++r;
    }

    const size_t rhsEnd = r;

    // Emit rewritten call
    out.append(name);
    out.push_back('(');
    out.append(in.substr(parenStart + 1, parenEnd - (parenStart + 1)));
    out.append(", ");
    out.append(in.substr(rhsStart, rhsEnd - rhsStart));
    out.push_back(')');

    // Preserve trailing ';' if present
    if (r < in.size() && in[r] == ';')
    {
      out.push_back(';');
      ++r;
    }

    i = r;
    return true;
  };

  for (size_t i = 0; i < in.size(); )
  {
    const char c = in[i];

    // Track comments/strings so we don't rewrite inside them.
    if (inLineComment)
    {
      out.push_back(c);
      ++i;
      if (c == '\n') inLineComment = false;
      continue;
    }
    if (inBlockComment)
    {
      out.push_back(c);
      if (c == '*' && i + 1 < in.size() && in[i + 1] == '/')
      {
        out.push_back('/');
        i += 2;
        inBlockComment = false;
      }
      else
      {
        ++i;
      }
      continue;
    }
    if (inString)
    {
      out.push_back(c);
      if (c == '\\' && i + 1 < in.size())
      {
        out.push_back(in[i + 1]);
        i += 2;
        continue;
      }
      if (c == strQuote) inString = false;
      ++i;
      continue;
    }

    // Enter comment/string states
    if (c == '/' && i + 1 < in.size() && in[i + 1] == '/')
    {
      out.push_back('/');
      out.push_back('/');
      i += 2;
      inLineComment = true;
      continue;
    }
    if (c == '/' && i + 1 < in.size() && in[i + 1] == '*')
    {
      out.push_back('/');
      out.push_back('*');
      i += 2;
      inBlockComment = true;
      continue;
    }
    if (c == '"' || c == '\'')
    {
      out.push_back(c);
      inString = true;
      strQuote = c;
      ++i;
      continue;
    }

    // Rewrite slider()/spl() assignments
    if (c == 's')
    {
      if (tryRewriteAssign(i, "slider")) continue;
      if (tryRewriteAssign(i, "spl"))    continue;
    }

    out.push_back(c);
    ++i;
  }

  return out;
}


// -------------------------
// Draw command list (JUCE playback)
// -------------------------
struct DrawCmd
{
  enum class Type { Rect, Line, Text, Circle, RoundRect, Arc, Triangle };
  Type type = Type::Rect;

  // Common
  juce::Colour colour { 0xff000000 };

  // Rect / round-rect
  float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;
  bool fill = true;
  float cornerRadius = 0.0f;

  // Line / arc endpoints / generic auxiliaries
  float x2 = 0.0f, y2 = 0.0f;

  // Text
  juce::Font font;
  juce::String text;

  // Circle / arc
  float radius = 0.0f;
  float angle1 = 0.0f;
  float angle2 = 0.0f;

  // Triangle / convex polygon (gfx_triangle)
  std::vector<juce::Point<float>> points;
};

// A sparse mem[] span mirrored into the @gfx VM.
struct MemSpanView
{
  const double* data = nullptr;
  int64_t base = 0;
  int count = 0;
};

// -------------------------
// EEL VM wrapper implementing gfx_* API by recording DrawCmds
// -------------------------
class GfxVm final : public eelScriptInst
{
public:
  // Resolve (and auto-create) an EEL variable by name.
  //
  // NSEEL_VM_regvar returns a pointer to the VM's backing storage for that variable.
  // This lets us bind JSFX globals (gfx_*, mouse_* etc.) into the VM.
  EEL_F* get_var(const char* name)
  {
    return m_vm ? NSEEL_VM_regvar(m_vm, name) : nullptr;
  }

  GfxVm()
  {
    // Ensure global init and builtins are registered.
    static std::once_flag s_initOnce;
    std::call_once(s_initOnce, []() {
      NSEEL_init();
      eelScriptInst::init();

      // Register our JSFX gfx builtins globally.
      registerGfxBuiltins();
    });

    // Bind core gfx variables.
    gfx_x     = get_var("gfx_x");
    gfx_y     = get_var("gfx_y");
    gfx_w     = get_var("gfx_w");
    gfx_h     = get_var("gfx_h");
    gfx_frame = get_var("gfx_frame");
    if (gfx_frame) *gfx_frame = 0.0;
    gfx_r     = get_var("gfx_r");
    gfx_g     = get_var("gfx_g");
    gfx_b     = get_var("gfx_b");
    gfx_a     = get_var("gfx_a");
    gfx_clear = get_var("gfx_clear");
    gfx_mode  = get_var("gfx_mode");

    mouse_x     = get_var("mouse_x");
    mouse_y     = get_var("mouse_y");
    mouse_cap   = get_var("mouse_cap");
    mouse_wheel = get_var("mouse_wheel");
    mouse_hwheel= get_var("mouse_hwheel");

    srate_var = get_var("srate");
    samplesblock_var = get_var("samplesblock");

    // Default values
    *gfx_x = 0.0;
    *gfx_y = 0.0;
    *gfx_r = 1.0;
    *gfx_g = 1.0;
    *gfx_b = 1.0;
    *gfx_a = 1.0;
    *gfx_clear = 0.0; // default clear-to-black (JSFX-style). Set gfx_clear=-1 to disable.

    if (srate_var) *srate_var = 44100.0;
    if (samplesblock_var) *samplesblock_var = 0.0;

    currentFont = juce::Font(juce::Font::getDefaultSansSerifFontName(), 12.0f, juce::Font::plain);
  }

  juce::Colour getCurrentColour() const
  {
    const float r = gfx_r ? (float)*gfx_r : 1.0f;
    const float g = gfx_g ? (float)*gfx_g : 1.0f;
    const float b = gfx_b ? (float)*gfx_b : 1.0f;
    const float a = gfx_a ? (float)*gfx_a : 1.0f;
    return juce::Colour::fromFloatRGBA(r, g, b, a);
  }

  // -------------------------------------------------------------------
  // Lazily clear the framebuffer on the first draw call of a frame.
  // This matches WDL/eel_lice behaviour: if the script doesn't draw anything
  // (common when throttling UI), the previous frame remains visible.
  // -------------------------------------------------------------------
  void setImageDirty()
  {
    if (framebufferDirty)
      return;

    framebufferDirty = true;

    if (gfx_clear && *gfx_clear > -1.0)
    {
      // JSFX packs RGB as: r + g*256 + b*65536  (see WDL eel_lice.h docs)
      const int rgb = (int)(*gfx_clear + 0.5);
      const int r = (rgb) & 0xff;
      const int g = (rgb >> 8) & 0xff;
      const int b = (rgb >> 16) & 0xff;

      DrawCmd cmd;
      cmd.type = DrawCmd::Type::Rect;
      cmd.colour = juce::Colour::fromRGB((juce::uint8)r, (juce::uint8)g, (juce::uint8)b);
      cmd.x = 0.0f;
      cmd.y = 0.0f;
      cmd.w = (float)frameW;
      cmd.h = (float)frameH;
      cmd.fill = true;
      commands.push_back(std::move(cmd));
    }
  }


  // -------------------------------------------------------------------
  // State set by host before executing @gfx
  // -------------------------------------------------------------------
  void beginFrame(int w, int h)
  {
    commands.clear();

    // Clear per-frame host interaction events.
    sliderChangeMask = 0;
    sliderAutomateMask = 0;
    sliderAutomateEndMask = 0;
    undoPointRequested = false;

    frameW = w;
    frameH = h;
    framebufferDirty = false;

    *gfx_w = (double)w;
    *gfx_h = (double)h;

    if (gfx_frame) *gfx_frame = frameCounter++;
  }

  void setMouse(float x, float y, int cap, float wheel, float hwheel)
  {
    *mouse_x = (double)x;
    *mouse_y = (double)y;
    *mouse_cap = (double)cap;
    *mouse_wheel = (double)wheel;
    *mouse_hwheel = (double)hwheel;
  }

  // -------------------------------------------------------------------
  // Host keyboard input (gfx_getchar)
  // -------------------------------------------------------------------
  void pushKey(int code)
  {
    if (code == 0)
      return;
    keyQueue.push_back(code);
  }

  void setKeyDown(int code, bool isDown)
  {
    if (code == 0)
      return;
    if (isDown)
      keysDown.insert(code);
    else
      keysDown.erase(code);
  }

  void clearKeys()
  {
    keyQueue.clear();
    keysDown.clear();
  }

  // -------------------------------------------------------------------
  // Host slider interaction events (sliderchange/slider_automate)
  // -------------------------------------------------------------------
  uint64_t popSliderChangeMask()       { const auto m = sliderChangeMask;      sliderChangeMask = 0; return m; }
  uint64_t popSliderAutomateMask()     { const auto m = sliderAutomateMask;    sliderAutomateMask = 0; return m; }
  uint64_t popSliderAutomateEndMask()  { const auto m = sliderAutomateEndMask; sliderAutomateEndMask = 0; return m; }
  bool popUndoPointRequested()         { const bool b = undoPointRequested;    undoPointRequested = false; return b; }

  void setTiming(double srate, double samplesblock)
  {
    if (srate_var) *srate_var = (EEL_F)srate;
    if (samplesblock_var) *samplesblock_var = (EEL_F)samplesblock;
  }

  // -------------------------------------------------------------------
  // Output commands
  // -------------------------------------------------------------------
  const std::vector<DrawCmd>& getCommands() const { return commands; }

  // -------------------------------------------------------------------
  // Host sync helpers
  // -------------------------------------------------------------------
  std::array<EEL_F*, 64> sliderPtrs {{}};
  void bindSliderPtrs()
  {
    for (int i = 0; i < 64; ++i)
    {
      const std::string nm = std::string("slider") + std::to_string(i + 1);
      sliderPtrs[(size_t)i] = get_var(nm.c_str());
    }
  }

  struct BoundVar { const char* name; int index; EEL_F* ptr; };
  std::vector<BoundVar> boundVars;
  void bindUserVars(const DSPJSFX_VarDesc* vars, int count)
  {
    boundVars.clear();
    boundVars.reserve((size_t)count);
    for (int i = 0; i < count; ++i)
    {
      const char* name = vars[i].name;
      const int idx = vars[i].index;
      if (!name) continue;
      BoundVar bv { name, idx, get_var(name) };
      boundVars.push_back(bv);
    }
  }

  void syncSliders(const double* sliders, int count)
  {
    const int n = std::min(count, 64);
    for (int i = 0; i < n; ++i)
      if (sliderPtrs[(size_t)i]) *sliderPtrs[(size_t)i] = sliders[i];
  }

  void readSliders(double* dst, int count) const
  {
    if (!dst) return;
    const int n = std::min(count, 64);
    for (int i = 0; i < n; ++i)
      dst[i] = sliderPtrs[(size_t)i] ? (double)*sliderPtrs[(size_t)i] : 0.0;
  }

  void syncVars(const double* vars, int count)
  {
    for (const auto& bv : boundVars)
    {
      if (bv.index >= 0 && bv.index < count && bv.ptr)
        *bv.ptr = vars[bv.index];
    }
  }

  void syncMemRange(const double* mem, int64_t base, int count)
  {
    if (!mem || count <= 0 || base < 0) return;

    int64_t pos64 = base;
    int copied = 0;
    while (copied < count)
    {
      if (pos64 > (int64_t) std::numeric_limits<unsigned int>::max())
        break;

      int validCount = 0;
      EEL_F* dst = NSEEL_VM_getramptr(m_vm, (unsigned int) pos64, &validCount);
      if (!dst || validCount <= 0) break;

      const int n = std::min(validCount, count - copied);
      std::memcpy(dst, mem + copied, (size_t) n * sizeof(EEL_F));
      copied += n;
      pos64 += (int64_t) n;
    }
  }

  void syncMem(const double* mem, int memN)
  {
    if (!mem || memN <= 0) return;

    // Avoid redundant VM RAM resize calls on steady-state frames.
    if (memN != memSize)
      NSEEL_VM_setramsize(m_vm, (unsigned int)memN);

    syncMemRange(mem, 0, memN);

    // Remember the effective RAM size so we can read it back later.
    memSize = memN;
  }

  void syncMemSpans(const MemSpanView* spans, int spanCount, int64_t logicalMemN)
  {
    if (!spans || spanCount <= 0)
      return;

    int64_t requiredMem = std::max<int64_t>(0, logicalMemN);
    for (int i = 0; i < spanCount; ++i)
    {
      const auto& span = spans[i];
      if (!span.data || span.count <= 0 || span.base < 0)
        continue;
      requiredMem = std::max<int64_t>(requiredMem, span.base + (int64_t) span.count);
    }

    requiredMem = std::min<int64_t>(requiredMem, (int64_t) std::numeric_limits<unsigned int>::max());

    if ((int) requiredMem != memSize)
      NSEEL_VM_setramsize(m_vm, (unsigned int) requiredMem);

    for (int i = 0; i < spanCount; ++i)
    {
      const auto& span = spans[i];
      if (!span.data || span.count <= 0)
        continue;
      syncMemRange(span.data, span.base, span.count);
    }

    memSize = (int) requiredMem;
  }

  // Read back bound user vars into a JSFX-style vars[] array.
  // Only variables that are actually bound into the EEL VM are written.
  void readVars(double* dst, int count) const
  {
    if (!dst || count <= 0) return;
    for (const auto& bv : boundVars)
    {
      if (bv.index >= 0 && bv.index < count && bv.ptr)
        dst[bv.index] = *bv.ptr;
    }
  }

  void readMemRange(int64_t base, double* dst, int count) const
  {
    if (!dst || count <= 0 || memSize <= 0 || base < 0) return;
    if (base >= (int64_t) memSize) return;

    const int64_t available = (int64_t) memSize - base;
    const int n = (int) std::min<int64_t>((int64_t) count, available);
    int copied = 0;
    int64_t pos64 = base;
    while (copied < n)
    {
      if (pos64 > (int64_t) std::numeric_limits<unsigned int>::max())
        break;

      int validCount = 0;
      EEL_F* src = NSEEL_VM_getramptr(m_vm, (unsigned int) pos64, &validCount);
      if (!src || validCount <= 0) break;
      const int m = std::min(validCount, n - copied);
      std::memcpy(dst + copied, src, (size_t) m * sizeof(EEL_F));
      copied += m;
      pos64 += (int64_t) m;
    }
  }

  // Read back the EEL RAM (JSFX mem[]) into dst.
  // This copies [0..min(count, memSize)) and leaves the rest unchanged.
  void readMem(double* dst, int count) const
  {
    readMemRange(0, dst, count);
  }

  // -------------------------------------------------------------------
  // EEL-exposed gfx builtins (static)
  // -------------------------------------------------------------------
  static void registerGfxBuiltins()
  {
    // IMPORTANT:
    //   - The 3rd parameter to NSEEL_addfunc_varparm_ex is a boolean "want_exact", NOT a max-arg count.
    //     Passing nonzero here forces an exact-arity function, which breaks JSFX calls like
    //     gfx_set(r,g,b,a) (4 params) or gfx_rect(x,y,w,h,fill) (5 params).
    //   - We also must use NSEEL_PProc_THIS so the callback receives the per-VM "this" pointer
    //     (set by eelScriptInst), which we use as our GfxVm instance.

    // Register into the global EEL function table.
    // Signature required: EEL_F (NSEEL_CGEN_CALL *)(void* opaque, INT_PTR np, EEL_F** parms)
    // want_exact=0 => varargs with minimum parameter count.
    NSEEL_addfunc_varparm_ex("gfx_set",        3, 0, NSEEL_PProc_THIS, &eel_gfx_set,        nullptr);
    NSEEL_addfunc_varparm_ex("gfx_rect",       4, 0, NSEEL_PProc_THIS, &eel_gfx_rect,       nullptr);
    NSEEL_addfunc_varparm_ex("gfx_rectto",     2, 0, NSEEL_PProc_THIS, &eel_gfx_rectto,     nullptr);
    NSEEL_addfunc_varparm_ex("gfx_circle",     3, 0, NSEEL_PProc_THIS, &eel_gfx_circle,     nullptr);
    NSEEL_addfunc_varparm_ex("gfx_roundrect",  5, 0, NSEEL_PProc_THIS, &eel_gfx_roundrect,  nullptr);
    NSEEL_addfunc_varparm_ex("gfx_arc",        5, 0, NSEEL_PProc_THIS, &eel_gfx_arc,        nullptr);
    NSEEL_addfunc_varparm_ex("gfx_triangle",   6, 0, NSEEL_PProc_THIS, &eel_gfx_triangle,   nullptr);
    NSEEL_addfunc_varparm_ex("gfx_line",       4, 0, NSEEL_PProc_THIS, &eel_gfx_line,       nullptr);
    NSEEL_addfunc_varparm_ex("gfx_lineto",     2, 0, NSEEL_PProc_THIS, &eel_gfx_lineto,     nullptr);
    NSEEL_addfunc_varparm_ex("gfx_drawstr",    1, 0, NSEEL_PProc_THIS, &eel_gfx_drawstr,    nullptr);
    NSEEL_addfunc_varparm_ex("gfx_printf",     1, 0, NSEEL_PProc_THIS, &eel_gfx_printf,     nullptr);
    NSEEL_addfunc_varparm_ex("gfx_setfont",    1, 0, NSEEL_PProc_THIS, &eel_gfx_setfont,    nullptr);
    NSEEL_addfunc_varparm_ex("gfx_measurestr", 1, 0, NSEEL_PProc_THIS, &eel_gfx_measurestr, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_getchar",    0, 0, NSEEL_PProc_THIS, &eel_gfx_getchar,    nullptr);

    // Minimal host interaction helpers used by many JSFX UIs.
    // See: https://www.reaper.fm/sdk/js/advfunc.php
    NSEEL_addfunc_varparm_ex("sliderchange",   1, 0, NSEEL_PProc_THIS, &eel_sliderchange,   nullptr);
    NSEEL_addfunc_varparm_ex("slider_automate",1, 0, NSEEL_PProc_THIS, &eel_slider_automate,nullptr);

    // JSFX dynamic access helpers (REAPER dialect)
    // Many scripts use slider(i) / slider(i)=v and spl(i) / spl(i)=v.
    // We implement portable equivalents (see preprocessJsfxForPortableEel()).
    NSEEL_addfunc_varparm_ex("slider",   1, 0, NSEEL_PProc_THIS, &eel_slider,   nullptr);
    NSEEL_addfunc_varparm_ex("spl",      1, 0, NSEEL_PProc_THIS, &eel_spl,      nullptr);
    NSEEL_addfunc_varparm_ex("freembuf", 1, 0, NSEEL_PProc_THIS, &eel_freembuf, nullptr);

  }

  static uint64_t sliderMaskFromArg(GfxVm* self, EEL_F* argPtr, double argValue)
  {
    if (self)
    {
      for (int i = 0; i < 64; ++i)
      {
        if (self->sliderPtrs[(size_t)i] == argPtr)
          return (uint64_t)1u << (uint64_t)i;
      }
    }

    // If not a direct slider var, treat as an integer bitmask.
    if (argValue <= 0.0)
      return 0;

    const int64_t m = (int64_t)std::llround(argValue);
    if (m <= 0)
      return 0;
    return (uint64_t)m;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_set(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 3) return 0.0;

    const float r = (float)*parms[0];
    const float g = (float)*parms[1];
    const float b = (float)*parms[2];
    const float a = (np >= 4) ? (float)*parms[3] : 1.0f;

    if (self->gfx_r) *self->gfx_r = r;
    if (self->gfx_g) *self->gfx_g = g;
    if (self->gfx_b) *self->gfx_b = b;
    if (self->gfx_a) *self->gfx_a = a;

    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_rect(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 4) return 0.0;

    self->setImageDirty();

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Rect;
    cmd.colour = self->getCurrentColour();
    cmd.x = (float)*parms[0];
    cmd.y = (float)*parms[1];
    cmd.w = (float)*parms[2];
    cmd.h = (float)*parms[3];
    cmd.fill = (np >= 5) ? (*parms[4] != 0.0) : true;

    self->commands.push_back(std::move(cmd));
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_rectto(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 2) return 0.0;

    self->setImageDirty();

    const float x1 = (float)(self->gfx_x ? *self->gfx_x : 0.0);
    const float y1 = (float)(self->gfx_y ? *self->gfx_y : 0.0);
    const float x2 = (float)*parms[0];
    const float y2 = (float)*parms[1];

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Rect;
    cmd.colour = self->getCurrentColour();
    cmd.x = std::min(x1, x2);
    cmd.y = std::min(y1, y2);
    cmd.w = std::fabs(x2 - x1);
    cmd.h = std::fabs(y2 - y1);
    cmd.fill = (np >= 3) ? (*parms[2] != 0.0) : true;
    self->commands.push_back(std::move(cmd));

    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_line(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 4) return 0.0;

    self->setImageDirty();

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Line;
    cmd.colour = self->getCurrentColour();
    cmd.x = (float)*parms[0];
    cmd.y = (float)*parms[1];
    cmd.x2 = (float)*parms[2];
    cmd.y2 = (float)*parms[3];

    self->commands.push_back(std::move(cmd));

    // Update pen position (JSFX convention)
    if (self->gfx_x) *self->gfx_x = cmd.x2;
    if (self->gfx_y) *self->gfx_y = cmd.y2;

    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_lineto(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 2) return 0.0;

    self->setImageDirty();

    const float x1 = (float)(self->gfx_x ? *self->gfx_x : 0.0);
    const float y1 = (float)(self->gfx_y ? *self->gfx_y : 0.0);
    const float x2 = (float)*parms[0];
    const float y2 = (float)*parms[1];

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Line;
    cmd.colour = self->getCurrentColour();
    cmd.x = x1;
    cmd.y = y1;
    cmd.x2 = x2;
    cmd.y2 = y2;

    self->commands.push_back(std::move(cmd));

    if (self->gfx_x) *self->gfx_x = x2;
    if (self->gfx_y) *self->gfx_y = y2;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_setfont(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1) return 0.0;

    const int fontId = (int)(*parms[0] + 0.5);

    juce::String fontName = juce::Font::getDefaultSansSerifFontName();
    float fontSize = 12.0f;
    int styleFlags = juce::Font::plain;

    if (np >= 2)
    {
      // parms[1] is a string handle
      EEL_STRING_MUTEXLOCK_SCOPE;
      const char* fn = EEL_STRING_GET_FOR_INDEX(*parms[1], nullptr);
      if (fn && *fn) fontName = juce::String(fn);
    }
    if (np >= 3)
      fontSize = (float)*parms[2];

    if (np >= 4)
    {
      const int flags = (int)(*parms[3] + 0.5);
      // JSFX flags are not 1:1 with JUCE; map a few common ones.
      // Bit 1 often used for bold, bit 2 for italic in many scripts.
      if (flags & 1) styleFlags |= juce::Font::bold;
      if (flags & 2) styleFlags |= juce::Font::italic;
    }

    juce::Font f(fontName, fontSize, styleFlags);
    self->fonts[fontId] = f;
    self->currentFontId = fontId;
    self->currentFont = f;

    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_drawstr(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1) return 0.0;

    self->setImageDirty();

    EEL_STRING_MUTEXLOCK_SCOPE;
    const char* str = EEL_STRING_GET_FOR_INDEX(*parms[0], nullptr);

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Text;
    cmd.colour = self->getCurrentColour();
    cmd.font = self->currentFont;
    cmd.text = juce::String::fromUTF8(str ? str : "");
    cmd.x = (float)(self->gfx_x ? *self->gfx_x : 0.0);
    cmd.y = (float)(self->gfx_y ? *self->gfx_y : 0.0);

    self->commands.push_back(cmd);

    // Advance pen position by string width (approx)
    const float advance = cmd.font.getStringWidthFloat(cmd.text);
    if (self->gfx_x) *self->gfx_x = (double)(cmd.x + advance);

    return 0.0;
  }

  

static EEL_F NSEEL_CGEN_CALL eel_gfx_printf(void* opaque, INT_PTR np, EEL_F** parms)
{
  auto* self = (GfxVm*)opaque;
  if (!self || np < 1) return 0.0;

    self->setImageDirty();

  juce::String textToDraw;
  {
    EEL_STRING_MUTEXLOCK_SCOPE;

    const char* fmt = EEL_STRING_GET_FOR_INDEX(*parms[0], nullptr);
    if (fmt == nullptr)
      fmt = "";

    std::string out;
    out.reserve(std::strlen(fmt) + 32);

    int argIndex = 1;

    for (size_t i = 0; fmt[i] != '\0'; ++i)
    {
      if (fmt[i] != '%')
      {
        out.push_back(fmt[i]);
        continue;
      }

      // %%
      if (fmt[i + 1] == '%')
      {
        out.push_back('%');
        ++i;
        continue;
      }

      const size_t specStart = i;
      size_t j = i + 1;

      // flags
      while (fmt[j] != '\0' && std::strchr("-+0 #", fmt[j]) != nullptr)
        ++j;

      // width
      while (fmt[j] != '\0' && std::isdigit((unsigned char)fmt[j]))
        ++j;

      // precision
      if (fmt[j] == '.')
      {
        ++j;
        while (fmt[j] != '\0' && std::isdigit((unsigned char)fmt[j]))
          ++j;
      }

      // length modifiers (ignored, but consumed so "%lld" etc doesn't break parsing)
      if (fmt[j] == 'h' || fmt[j] == 'l' || fmt[j] == 'L')
      {
        const char first = fmt[j];
        ++j;
        if ((first == 'h' || first == 'l') && fmt[j] == first)
          ++j;
      }

      const char spec = fmt[j];

      if (spec == '\0')
      {
        // Unterminated format specifier: append the rest verbatim and stop.
        out.append(fmt + specStart);
        break;
      }

      // Include conversion letter
      ++j;

      const std::string oneFmt(fmt + specStart, fmt + j);

      char buf[512];
      buf[0] = '\0';

      const double v = (argIndex < (int)np) ? (double)*parms[argIndex] : 0.0;

      if (spec == 's')
      {
        const char* s = EEL_STRING_GET_FOR_INDEX(v, nullptr);
        if (s == nullptr)
          s = "";
        ::snprintf(buf, sizeof(buf), oneFmt.c_str(), s);
        ++argIndex;
      }
      else if (spec == 'd' || spec == 'i')
      {
        const int iv = (int)std::llround(v);
        ::snprintf(buf, sizeof(buf), oneFmt.c_str(), iv);
        ++argIndex;
      }
      else if (spec == 'u' || spec == 'x' || spec == 'X' || spec == 'o')
      {
        const unsigned int uv = (unsigned int)std::llround(v);
        ::snprintf(buf, sizeof(buf), oneFmt.c_str(), uv);
        ++argIndex;
      }
      else if (spec == 'c')
      {
        const int cv = (int)std::llround(v);
        ::snprintf(buf, sizeof(buf), oneFmt.c_str(), cv);
        ++argIndex;
      }
      else
      {
        // Treat everything else as floating-point
        ::snprintf(buf, sizeof(buf), oneFmt.c_str(), v);
        ++argIndex;
      }

      out.append(buf);

      // Continue parsing at the end of this format specifier.
      i = j - 1;
    }

    textToDraw = juce::String::fromUTF8(out.c_str(), (int)out.size());
  }

  DrawCmd cmd;
  cmd.type = DrawCmd::Type::Text;
  cmd.colour = self->getCurrentColour();
  cmd.font = self->currentFont;
  cmd.text = textToDraw;
  cmd.x = (float)(self->gfx_x ? *self->gfx_x : 0.0);
  cmd.y = (float)(self->gfx_y ? *self->gfx_y : 0.0);

  self->commands.push_back(cmd);

  const float advance = cmd.font.getStringWidthFloat(cmd.text);
  if (self->gfx_x) *self->gfx_x = (double)(cmd.x + advance);

  return 0.0;
}

static EEL_F NSEEL_CGEN_CALL eel_gfx_measurestr(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1) return 0.0;

    EEL_STRING_MUTEXLOCK_SCOPE;
    const char* str = EEL_STRING_GET_FOR_INDEX(*parms[0], nullptr);
    const juce::String text = juce::String::fromUTF8(str ? str : "");

    const float w = self->currentFont.getStringWidthFloat(text);
    const float h = self->currentFont.getHeight();

    if (self->gfx_x) *self->gfx_x = (double)w;
    if (self->gfx_y) *self->gfx_y = (double)h;

    // JSFX compatibility: allow gfx_measurestr(str, w, h)
    if (np >= 2 && parms[1]) *parms[1] = (EEL_F)w;
    if (np >= 3 && parms[2]) *parms[2] = (EEL_F)h;

    return (EEL_F)w;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_circle(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 3) return 0.0;

    self->setImageDirty();

    DrawCmd cmd;
    cmd.type   = DrawCmd::Type::Circle;
    cmd.colour = self->getCurrentColour();
    cmd.x      = (float)*parms[0];
    cmd.y      = (float)*parms[1];
    cmd.radius = (float)*parms[2];
    cmd.fill   = (np >= 4) ? (*parms[3] != 0.0) : true;

    self->commands.push_back(std::move(cmd));
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_roundrect(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 5) return 0.0;

    self->setImageDirty();

    const float w = (float)*parms[2];
    const float h = (float)*parms[3];
    if (!(w > 0.0f) || !(h > 0.0f))
      return 0.0;

    DrawCmd cmd;
    cmd.type         = DrawCmd::Type::RoundRect;
    cmd.colour       = self->getCurrentColour();
    cmd.x            = (float)*parms[0];
    cmd.y            = (float)*parms[1];
    cmd.w            = w;
    cmd.h            = h;
    cmd.cornerRadius = std::max(0.0f, (float)*parms[4]);
    cmd.fill         = false; // JSFX gfx_roundrect draws an outline.

    self->commands.push_back(std::move(cmd));
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_arc(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 5) return 0.0;

    self->setImageDirty();

    const double cx = (double)*parms[0];
    const double cy = (double)*parms[1];
    const double r  = (double)*parms[2];
    const double a1 = (double)*parms[3];
    const double a2 = (double)*parms[4];

    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(r) ||
        !std::isfinite(a1) || !std::isfinite(a2) || r <= 0.0)
      return 0.0;

    DrawCmd cmd;
    cmd.type   = DrawCmd::Type::Arc;
    cmd.colour = self->getCurrentColour();
    cmd.x      = (float)cx;
    cmd.y      = (float)cy;
    cmd.radius = (float)r;
    cmd.angle1 = (float)a1;
    cmd.angle2 = (float)a2;
    cmd.fill   = false;

    self->commands.push_back(std::move(cmd));
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_triangle(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 6) return 0.0;

    self->setImageDirty();

    // gfx_triangle(x1,y1,x2,y2,x3,y3[,x4,y4...]) -- always filled.
    const int pairs = (int)(np / 2);
    if (pairs < 3) return 0.0;

    DrawCmd cmd;
    cmd.type   = DrawCmd::Type::Triangle;
    cmd.colour = self->getCurrentColour();
    cmd.fill   = true;

    cmd.points.reserve((size_t)pairs);

    for (INT_PTR i = 0; i + 1 < np; i += 2)
    {
      const double x = (double)*parms[i + 0];
      const double y = (double)*parms[i + 1];

      const float fx = std::isfinite(x) ? (float)x : 0.0f;
      const float fy = std::isfinite(y) ? (float)y : 0.0f;
      cmd.points.emplace_back(fx, fy);
    }

    if (cmd.points.size() >= 3)
      self->commands.push_back(std::move(cmd));

    return 0.0;
  }


  static EEL_F NSEEL_CGEN_CALL eel_gfx_getchar(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self)
      return 0.0;

    // gfx_getchar([char, unicodechar])
    // - If no parameter or zero is passed: pop from keyboard queue.
    // - If char is passed and nonzero: return whether that key is currently down.
    // (Unicode support is not implemented; second parameter is ignored.)
    if (np >= 1 && *parms[0] != 0.0)
    {
      const int code = (int)std::llround(*parms[0]);
      return self->keysDown.count(code) ? 1.0 : 0.0;
    }

    if (self->keyQueue.empty())
      return 0.0;

    const int code = self->keyQueue.front();
    self->keyQueue.pop_front();
    return (EEL_F)code;
  }

  static EEL_F NSEEL_CGEN_CALL eel_sliderchange(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1)
      return 0.0;

    const double v = (double)*parms[0];

    // IMPORTANT:
    // When called as sliderchange(slider3), the argument value can be negative
    // (slider ranges are arbitrary). So we must resolve slider-vs-mask by *pointer*,
    // not by numeric value.
    const uint64_t mask = sliderMaskFromArg(self, parms[0], v);
    if (mask != 0)
    {
      self->sliderChangeMask |= mask;
      return 0.0;
    }

    // In REAPER, sliderchange(-1) from @gfx adds an undo point.
    // In this standalone runtime, we just flag it so the host can choose what to do.
    if (v < 0.0)
      self->undoPointRequested = true;

    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_slider_automate(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1)
      return 0.0;

    const double v = (double)*parms[0];

    // IMPORTANT: slider values may be negative; see comment in eel_sliderchange.
    const uint64_t mask = sliderMaskFromArg(self, parms[0], v);
    if (mask == 0)
      return 0.0;

    const bool endTouch = (np >= 2 && *parms[1] != 0.0);
    if (endTouch)
      self->sliderAutomateEndMask |= mask;
    else
      self->sliderAutomateMask |= mask;

    return 0.0;
  }

  // -------------------------------------------------------------------
  // JSFX helpers: slider(i) / spl(i) dynamic access (portable implementation)
  //
  // Notes:
  // - slider(i) is 1-based (slider(1) == slider1).
  // - Many JSFX scripts also *assign* to slider(i) / spl(i). Portable EEL2 does
  //   not support function-call lvalues, so we rewrite those assignments to
  //   slider(i, v) / spl(i, v) in preprocessJsfxForPortableEel().
  // -------------------------------------------------------------------
  static EEL_F NSEEL_CGEN_CALL eel_slider(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1) return 0.0;

    const int idx = (int)std::llround((double)*parms[0]);
    if (idx < 1 || idx > 64)
    {
      // Setter form still returns the value (mirrors assignment-as-expression).
      return (np >= 2) ? *parms[1] : 0.0;
    }

    EEL_F* ptr = self->sliderPtrs[(size_t)(idx - 1)];
    if (!ptr)
      return (np >= 2) ? *parms[1] : 0.0;

    if (np >= 2)
    {
      *ptr = *parms[1];
      return *ptr;
    }

    return *ptr;
  }

  static EEL_F NSEEL_CGEN_CALL eel_spl(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;
    if (np < 1) return 0.0;

    // In REAPER, spl() accesses audio channel sample registers.
    // This lightweight @gfx interpreter does not expose audio, so:
    //   spl(i)    -> 0
    //   spl(i, v) -> returns v (ignored write)
    return (np >= 2) ? *parms[1] : 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_freembuf(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1) return 0.0;

    int64_t n = (int64_t)std::llround((double)*parms[0]);
    if (n < 0) n = 0;
    if (n > 0x7fffffffLL) n = 0x7fffffffLL;

    // Shrink/grow EEL RAM (mem[]).
    if (self->m_vm)
      NSEEL_VM_setramsize(self->m_vm, (unsigned int)n);

    self->memSize = (int)n;
    return 0.0;
  }



  // -------------------------------------------------------------------
  // VM-bound variables
  // -------------------------------------------------------------------
  EEL_F* gfx_x = nullptr;
  EEL_F* gfx_y = nullptr;
  EEL_F* gfx_w = nullptr;
  EEL_F* gfx_h = nullptr;
  EEL_F* gfx_frame = nullptr;
  EEL_F* gfx_r = nullptr;
  EEL_F* gfx_g = nullptr;
  EEL_F* gfx_b = nullptr;
  EEL_F* gfx_a = nullptr;
  EEL_F* gfx_clear = nullptr;
  EEL_F* gfx_mode = nullptr;

    double frameCounter = 0.0;

  int frameW = 0;
  int frameH = 0;
  bool framebufferDirty = false;

  EEL_F* mouse_x = nullptr;
  EEL_F* mouse_y = nullptr;
  EEL_F* mouse_cap = nullptr;
  EEL_F* mouse_wheel = nullptr;
  EEL_F* mouse_hwheel = nullptr;

  EEL_F* srate_var = nullptr;
  EEL_F* samplesblock_var = nullptr;

  // Current JSFX mem[] size (in doubles) synced into the EEL VM RAM.
  int memSize = 0;

  // Drawing state
  std::unordered_map<int, juce::Font> fonts;
  int currentFontId = 0;
  juce::Font currentFont;

  std::vector<DrawCmd> commands;

  // Host interaction event state
  uint64_t sliderChangeMask = 0;
  uint64_t sliderAutomateMask = 0;
  uint64_t sliderAutomateEndMask = 0;
  bool undoPointRequested = false;

  // Keyboard input
  std::deque<int> keyQueue;
  std::unordered_set<int> keysDown;
};

// -------------------------
// Public interpreter: parses JSFX source, binds vars/mem, runs @gfx
// -------------------------
class Interpreter
{
public:
  struct Snapshot
  {
    const double* sliders = nullptr; // [64]
    int slidersCount = 64;

    const double* vars = nullptr;
    int varsCount = 0;

    // Back-compat contiguous low mem window.
    const double* mem = nullptr;
    int memN = 0;

    // Sparse mirrored mem[] ranges. When present, these take precedence over mem/memN.
    const MemSpanView* memSpans = nullptr;
    int memSpanCount = 0;
    int64_t logicalMemN = 0;

    double srate = 0.0;
    double samplesblock = 0.0;
  };

  Interpreter(const char* jsfxSourceText)
  {
    sections = extractJsfxSections(jsfxSourceText);
    if (!sections.hasGfx)
      return;

    vm = std::make_unique<GfxVm>();

    // Bind sliders and user vars.
    vm->bindSliderPtrs();

    // DSPJSFX_VARS is a *symbol* emitted by dsp_jsfx_aot.py (static const array),
    // not a preprocessor macro. So `defined(DSPJSFX_VARS)` is always false.
    // The fallback table at the top of this file guarantees DSPJSFX_VARS exists anyway.
    vm->bindUserVars(DSPJSFX_VARS, (int)DSPJSFX_VARS_COUNT);

    // Compile relevant sections. We compile init + gfx so helper functions
    // defined in init are available to gfx.
    const char* err = nullptr;

    // JSFX dialect compatibility: rewrite slider(i)=v / spl(i)=v into portable EEL.
    juce::String initErr;
    if (!sections.init.empty())
    {
      const std::string initCode = preprocessJsfxForPortableEel(sections.init);
      code_init = vm->compile_code(initCode.c_str(), &err);

      if (!code_init)
      {
        const char* e = err ? err : NSEEL_code_getcodeerror(vm->m_vm);
        initErr = e ? e : "Unknown EEL compile error";
      }
    }

    err = nullptr;
    if (sections.hasGfx)
    {
      // Some scripts specify "@gfx" with no body. Treat it as a no-op rather than a hard error.
      const std::string gfxCode = preprocessJsfxForPortableEel(sections.gfx.empty() ? std::string("0;") : sections.gfx);
      code_gfx = vm->compile_code(gfxCode.c_str(), &err);

      if (!code_gfx)
      {
        const char* e = err ? err : NSEEL_code_getcodeerror(vm->m_vm);
        lastError = e ? e : "Unknown EEL compile error";

        if (initErr.isNotEmpty())
          lastError = "@init compile error (also):\n" + initErr + "\n\n@gfx compile error:\n" + lastError;
      }
    }

// We execute @init ONCE (on first frame) so scripts that configure gfx state
    // there (gfx_clear, fonts, precomputed UI tables, etc) behave as expected.
  }

  // Does the JSFX source contain an @gfx section at all?
  // (Independent of whether compilation succeeded.)
  bool hasGfxSection() const { return sections.hasGfx; }

  // Did @gfx compile successfully?
  bool gfxCompiledOk() const { return code_gfx != nullptr; }

  int preferredWidth() const { return sections.gfxW; }
  int preferredHeight() const { return sections.gfxH; }

  juce::String getLastError() const { return lastError; }

  void setMouse(float x, float y, int cap, float wheel, float hwheel)
  {
    mouseX = x; mouseY = y; mouseCap = cap; mouseWheel = wheel; mouseHWheel = hwheel;
  }

  // Keyboard input support for gfx_getchar().
  void pushKey(int code)
  {
    if (vm) vm->pushKey(code);
  }

  void setKeyDown(int code, bool isDown)
  {
    if (vm) vm->setKeyDown(code, isDown);
  }

  void clearKeys()
  {
    if (vm) vm->clearKeys();
  }

  void readSliders(double* dst, int count) const
  {
    if (vm) vm->readSliders(dst, count);
  }

  void readVars(double* dst, int count) const
  {
    if (vm) vm->readVars(dst, count);
  }

  void readMem(double* dst, int count) const
  {
    if (vm) vm->readMem(dst, count);
  }

  void readMemRange(int64_t base, double* dst, int count) const
  {
    if (vm) vm->readMemRange(base, dst, count);
  }

  uint64_t popSliderChangeMask()      { return vm ? vm->popSliderChangeMask()      : 0; }
  uint64_t popSliderAutomateMask()    { return vm ? vm->popSliderAutomateMask()    : 0; }
  uint64_t popSliderAutomateEndMask() { return vm ? vm->popSliderAutomateEndMask() : 0; }
  bool popUndoPointRequested()        { return vm ? vm->popUndoPointRequested()    : false; }

  void renderFrame(int width, int height, const Snapshot& snap)
  {
    if (!hasGfxSection() || !gfxCompiledOk()) return;

    // One-time init, with current snapshot state applied first.
    if (!initRan && code_init)
    {
      if (snap.sliders) vm->syncSliders(snap.sliders, snap.slidersCount);
      if (snap.vars)    vm->syncVars(snap.vars, snap.varsCount);
      if (snap.memSpans && snap.memSpanCount > 0) vm->syncMemSpans(snap.memSpans, snap.memSpanCount, snap.logicalMemN);
      else if (snap.mem)                     vm->syncMem(snap.mem, snap.memN);
      vm->setTiming(snap.srate, snap.samplesblock);
      NSEEL_code_execute(code_init);
      initRan = true;
    }

    // ------------------------------------------------------------
    // Sync state into VM.
    //
    // IMPORTANT:
    // We always sync sliders (they are the "public" parameter surface).
    //
    // For vars/mem, we must be careful: many JSFX UIs keep *interaction state*
    // (e.g. drag_id, prev_mouse_cap, hover flags, toggle state) in vars/mem.
    // The audio thread snapshot can lag behind UI edits (queued writes), and
    // blindly re-syncing vars/mem every frame can stomp the UI's most recent
    // state.
    //
    // Symptom: clicking a new control keeps dragging the previously-selected
    // control because prev_mouse_cap/drag state gets overwritten right before
    // the edge-detect logic runs.
    //
    // Fix: during a mouse interaction (any button down), preserve the VM's
    // vars/mem and only sync sliders. When no buttons are down, we resync
    // vars/mem from the snapshot so the UI can display audio-driven state.
    // ------------------------------------------------------------
    if (snap.sliders) vm->syncSliders(snap.sliders, snap.slidersCount);

    const bool anyMouseButtonDown = (mouseCap & (1 | 2 | 64)) != 0;
    if (!anyMouseButtonDown)
    {
      if (snap.vars)    vm->syncVars(snap.vars, snap.varsCount);
      if (snap.memSpans && snap.memSpanCount > 0) vm->syncMemSpans(snap.memSpans, snap.memSpanCount, snap.logicalMemN);
      else if (snap.mem)                     vm->syncMem(snap.mem, snap.memN);
    }

    vm->setTiming(snap.srate, snap.samplesblock);

    vm->setMouse(mouseX, mouseY, mouseCap, mouseWheel, mouseHWheel);

    vm->beginFrame(width, height);

    // Execute gfx code.
    NSEEL_code_execute(code_gfx);

    // reset wheels after one tick
    mouseWheel = 0.0f;
    mouseHWheel = 0.0f;
  }

  const std::vector<DrawCmd>& getCommands() const
  {
    static const std::vector<DrawCmd> kEmpty;
    if (!vm) return kEmpty;
    return vm->getCommands();
  }

private:
  JsfxSections sections;
  std::unique_ptr<GfxVm> vm;
  NSEEL_CODEHANDLE code_init = nullptr;
  NSEEL_CODEHANDLE code_gfx = nullptr;

  bool initRan = false;

  juce::String lastError;

  float mouseX = 0.0f;
  float mouseY = 0.0f;
  int mouseCap = 0;
  float mouseWheel = 0.0f;
  float mouseHWheel = 0.0f;
};

// -------------------------
// JUCE helper: paint commands
// -------------------------
static inline void paintCommands(juce::Graphics& g, const std::vector<DrawCmd>& cmds)
{
  for (const auto& cmd : cmds)
  {
    g.setColour(cmd.colour);
    switch (cmd.type)
    {
      case DrawCmd::Type::Rect:
        if (cmd.fill) g.fillRect(cmd.x, cmd.y, cmd.w, cmd.h);
        else          g.drawRect(cmd.x, cmd.y, cmd.w, cmd.h, 1.0f);
        break;
      case DrawCmd::Type::Line:
        g.drawLine(cmd.x, cmd.y, cmd.x2, cmd.y2, 1.0f);
        break;
      case DrawCmd::Type::Text:
        g.setFont(cmd.font);
        // JSFX draws text with top-left at (gfx_x,gfx_y)
        g.drawText(cmd.text, (int)cmd.x, (int)cmd.y, 10000, (int)cmd.font.getHeight() + 4,
                   juce::Justification::topLeft, false);
        break;
      case DrawCmd::Type::Circle:
      {
        const float d = cmd.radius * 2.0f;
        const float x = cmd.x - cmd.radius;
        const float y = cmd.y - cmd.radius;
        if (cmd.fill) g.fillEllipse(x, y, d, d);
        else          g.drawEllipse(x, y, d, d, 1.0f);
        break;
      }
      case DrawCmd::Type::RoundRect:
      {
        const juce::Rectangle<float> rc(cmd.x, cmd.y, cmd.w, cmd.h);
        if (cmd.fill) g.fillRoundedRectangle(rc, cmd.cornerRadius);
        else          g.drawRoundedRectangle(rc, cmd.cornerRadius, 1.0f);
        break;
      }
      case DrawCmd::Type::Arc:
      {
        const float span = std::abs(cmd.angle2 - cmd.angle1);
        if (cmd.radius > 0.0f && span > 0.0f)
        {
          const int segments = juce::jlimit(8, 512,
                                            (int)std::ceil(span * std::max(8.0f, cmd.radius * 0.35f)));
          juce::Path p;
          for (int i = 0; i <= segments; ++i)
          {
            const float t = (float)i / (float)segments;
            const float a = cmd.angle1 + (cmd.angle2 - cmd.angle1) * t;
            const float px = cmd.x + std::cos(a) * cmd.radius;
            const float py = cmd.y + std::sin(a) * cmd.radius;
            if (i == 0) p.startNewSubPath(px, py);
            else        p.lineTo(px, py);
          }
          g.strokePath(p, juce::PathStrokeType(1.0f));
        }
        break;
      }
      case DrawCmd::Type::Triangle:
      {
        if (cmd.points.size() >= 3)
        {
          juce::Path p;
          p.startNewSubPath(cmd.points[0]);
          for (size_t i = 1; i < cmd.points.size(); ++i)
            p.lineTo(cmd.points[i]);
          p.closeSubPath();
          g.fillPath(p);
        }
        break;
      }
    }
  }
}



} // namespace jsfx_gfx

// Undef config macros to reduce bleed into includer.
#undef EEL_TARGET_PORTABLE
#undef EELSCRIPT_NO_FILE
#undef EELSCRIPT_NO_NET
#undef EELSCRIPT_NO_MDCT
#undef EELSCRIPT_NO_EVAL
#undef EELSCRIPT_NO_PREPROC
#undef EELSCRIPT_NO_LICE

#endif // JSFX_YSFX_GFX_INTERPRETER_INCLUDED
