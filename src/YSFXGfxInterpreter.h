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
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <functional>
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

#include "JsfxGfxLice.h"

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

#ifndef DSPJSFX_GFX_VAR_FLAG_TO_GFX
#define DSPJSFX_GFX_VAR_FLAG_TO_GFX 1u
#endif
#ifndef DSPJSFX_GFX_VAR_FLAG_FROM_GFX
#define DSPJSFX_GFX_VAR_FLAG_FROM_GFX 2u
#endif
#ifndef DSPJSFX_GFX_VAR_FLAGS_COUNT
#define DSPJSFX_GFX_VAR_FLAGS_COUNT 0
static const uint8_t DSPJSFX_GFX_VAR_FLAGS[1] = { 0 };
#endif

static inline int64_t jsfxTruncIndexLikeAot (double v) noexcept
{
  return (int64_t) (v + 1.0e-5);
}

// -------------------------
// JSFX section extraction (@gfx, @init, ...)
// -------------------------
struct JsfxSections
{
  std::string init;
  std::string slider;
  std::string block;
  std::string sample;
  std::string serialize;
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

  enum class Sec { None, Init, Slider, Block, Sample, Serialize, Gfx };
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
      if (startsWithSection(ltrim, "serialize")) { cur = Sec::Serialize; continue; }
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
      case Sec::Sample:    out.sample.append(line).push_back('\n'); break;
      case Sec::Serialize: out.serialize.append(line).push_back('\n'); break;
      case Sec::Gfx:       out.gfx.append(line).push_back('\n'); break;
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


// Convert only after bounding; these helpers never use llround on an unbounded value.
static inline int boundedGfxInt(double value) noexcept
{
  constexpr double limit = 16777216.0; // Safe coordinate arithmetic, exactly representable as float.
  if (!std::isfinite(value)) return 0;
  return (int)std::max(-limit, std::min(limit, value));
}

static inline int gfxArcSegments(double radius, double span) noexcept
{
  if (!std::isfinite(radius) || !std::isfinite(span) || radius <= 0.0 || span <= 0.0)
    return 0;
  const double estimate = std::ceil(span * std::max(8.0, radius * 0.35));
  return (int)std::max(8.0, std::min(512.0, estimate)); // Clamp BEFORE conversion, including overflow to +Inf.
}

template <typename NumberAt, typename StringAt>
static std::string formatGfxPrintf(const char* fmt, int argc, NumberAt numberAt, StringAt stringAt)
{
  constexpr size_t maxOutput = 65536;
  constexpr int maxField = 512;
  std::string out;
  int argIndex = 0;
  const auto next = [&]() { return argIndex < argc ? numberAt(argIndex++) : 0.0; };
  const auto field = [](double v) { return std::isfinite(v) ? (int)std::min(512.0, std::abs(v)) : 0; };
  if (fmt == nullptr) return out;
  for (size_t i = 0; fmt[i] != '\0' && out.size() < maxOutput; ++i)
  {
    if (fmt[i] != '%') { out.push_back(fmt[i]); continue; }
    if (fmt[i + 1] == '%') { out.push_back('%'); ++i; continue; }
    const size_t start = i;
    size_t j = i + 1;
    std::string flags;
    while (fmt[j] && std::strchr("-+0 #", fmt[j])) flags.push_back(fmt[j++]);
    int width = 0;
    if (fmt[j] == '*')
    {
      const double v = next();
      width = field(v);
      if (v < 0.0) flags.push_back('-');
      ++j;
    }
    else
      while (std::isdigit((unsigned char)fmt[j]))
        width = std::min(maxField, width * 10 + (fmt[j++] - '0'));
    int precision = -1;
    if (fmt[j] == '.')
    {
      ++j;
      precision = 0;
      if (fmt[j] == '*')
      {
        const double v = next();
        precision = v < 0.0 ? -1 : field(v);
        ++j;
      }
      else
        while (std::isdigit((unsigned char)fmt[j]))
          precision = std::min(maxField, precision * 10 + (fmt[j++] - '0'));
    }
    // Normalize native lengths away. All integers below have their exact
    // int/unsigned-int ABI; strings are always narrow UTF-8, never wchar_t*.
    while (fmt[j] && std::strchr("hlLjzt", fmt[j])) ++j;
    if (fmt[j] == 'I')
    {
      ++j;
      while (std::isdigit((unsigned char)fmt[j])) ++j;
    }
    const char spec = fmt[j];
    if (!spec) { out.append(fmt + start); break; }
    i = j;
    if (!std::strchr("diuoxXcsfFeEgGaA", spec))
    {
      // In particular, %n and %p never reach snprintf.
      out.append(fmt + start, j - start + 1);
      (void)next();
      continue;
    }
    const double value = next();
    std::string nativeFmt = "%";
    const char* validFlags = spec == 's' || spec == 'c' ? "-" :
                            spec == 'd' || spec == 'i' ? "-+ 0" :
                            spec == 'u' ? "-0" : std::strchr("oxX", spec) ? "-#0" : "-+ #0";
    for (const char flag : flags)
      if (std::strchr(validFlags, flag) && nativeFmt.find(flag) == std::string::npos)
        nativeFmt.push_back(flag);
    if (width > 0) nativeFmt += std::to_string(width);
    if (precision >= 0 && spec != 'c') nativeFmt += "." + std::to_string(precision);
    nativeFmt.push_back(spec);
    // Unqualified snprintf: WDL redirects this name on Windows; POSIX uses libc.
    // std::snprintf would macro-expand to std::WDL_snprintf on Windows.
    char buf[512] {};
    if (spec == 's')
    {
      const char* text = stringAt(value);
      snprintf(buf, sizeof(buf), nativeFmt.c_str(), text ? text : "");
    }
    else if (spec == 'd' || spec == 'i')
    {
      const double v = std::isfinite(value) ? std::round(value) : 0.0;
      const int n = (int)std::max((double)std::numeric_limits<int>::min(),
                                 std::min((double)std::numeric_limits<int>::max(), v));
      snprintf(buf, sizeof(buf), nativeFmt.c_str(), n);
    }
    else if (std::strchr("uoxXc", spec))
    {
      double v = std::isfinite(value) ? std::fmod(std::round(value), 4294967296.0) : 0.0;
      if (v < 0.0) v += 4294967296.0;
      const unsigned int n = (unsigned int)v;
      if (spec == 'c') snprintf(buf, sizeof(buf), nativeFmt.c_str(), (int)(n & 255u));
      else snprintf(buf, sizeof(buf), nativeFmt.c_str(), n);
    }
    else
      snprintf(buf, sizeof(buf), nativeFmt.c_str(), value);
    out.append(buf, std::min(std::strlen(buf), maxOutput - out.size()));
  }
  if (out.size() > maxOutput) out.resize(maxOutput);
  return out;
}

// -------------------------
// Draw command list (JUCE playback)
// -------------------------
// Persistent JSFX straight-alpha image storage (private to the GFX worker).
//
// The normal @gfx path records commands and paints them after EEL execution. Keeping
// the images here lets gfx_dest / gfx_blit participate in that same ordered command
// stream without forcing JSFXJuceProcessor.cpp to know about individual image slots.
// The bank is owned by GfxVm; DrawCmd only carries a non-owning pointer to it.
struct GfxImageBank
{
  static constexpr int kNumImages = 128;
  static constexpr int kMaxImageDimension = 2048;

  // Image is only an allocation/lifetime holder here. Offscreen bytes are LICE's
  // straight ARGB, including RGB under alpha=0. Never pass these to juce::Graphics.
  std::array<juce::Image, kNumImages> images {};
  static constexpr size_t kMaxImageBytes = 256u * 1024u * 1024u;
  static constexpr int kMaxLoadedDimension = 8192;
  std::shared_ptr<struct GfxTextMaskCache> textMasks;

  void resizeImage(int index, int width, int height)
  {
    if (index < 0 || index >= kNumImages)
      return;

    width = juce::jlimit(0, kMaxImageDimension, width);
    height = juce::jlimit(0, kMaxImageDimension, height);

    if (width <= 0 || height <= 0)
    {
      images[(size_t) index] = juce::Image();
      return;
    }

    auto& image = images[(size_t) index];
    if (image.isValid() && image.getWidth() == width && image.getHeight() == height)
      return; // Repeated size requests must not allocate a new surface every frame.

    // The GFX worker uses CPU-backed surfaces, not a native GUI/GPU context.
    // A destination is flushed before it is sampled or resized during playback.
    image = juce::Image(juce::Image::ARGB, width, height, true, juce::SoftwareImageType());
  }


};

struct DrawCmd
{
  enum class Type { Rect, Line, Text, Circle, RoundRect, Arc, Triangle,
                    SetImageDim, Blit, Pixel, GradRect, MulAddRect, Blur,
                    DeltaBlit, TransformBlit, LoadImage };
  Type type = Type::Rect;

  // Destination: -1 = main framebuffer, 0..127 = offscreen image.
  int dest = -1;
  GfxImageBank* imageBank = nullptr;
  int frameWidth = 0;
  int frameHeight = 0;

  // Common
  juce::Colour colour { 0xff000000 };

  // Rect / round-rect / text bounds
  float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;
  bool fill = true;
  float cornerRadius = 0.0f;

  // Line / arc endpoints / generic auxiliaries
  float x2 = 0.0f, y2 = 0.0f;

  // Text
  juce::Font font;
  juce::String text;
  bool useTextBounds = false;
  juce::Justification textJustification = juce::Justification::topLeft;

  // Circle / arc
  float radius = 0.0f;
  float angle1 = 0.0f;
  float angle2 = 0.0f;

  // Triangle / convex polygon (gfx_triangle)
  std::vector<juce::Point<float>> points;

  // gfx_setimgdim
  int imageIndex = -1;
  int imageWidth = 0;
  int imageHeight = 0;

  // gfx_blit
  int source = -1;
  float srcX = 0.0f, srcY = 0.0f, srcW = 0.0f, srcH = 0.0f;
  float destX = 0.0f, destY = 0.0f, destW = 0.0f, destH = 0.0f;
  float rotation = 0.0f;
  float rotationXOffset = 0.0f;
  float rotationYOffset = 0.0f;
  float opacity = 1.0f;
  int blitMode = 0;
  bool antialias = true;
  bool bitmapFont = false;
  int textFlags = 0;
  uint64_t fontSerial = 0;
  bool useClipRect = true;
  int divisionsX = 0, divisionsY = 0;
  std::array<double, 12> values {};
  std::vector<double> transformPoints;
  juce::Image replacementImage; // Raw straight-alpha pixels, NOT a JUCE drawing surface.
};

#include "JsfxGfxRaster.h"


// A sparse mem[] span mirrored into the @gfx VM.
struct MemSpanView
{
  const double* data = nullptr;
  int64_t base = 0;
  int count = 0;
};

static constexpr int SHOWMENU_NB_NONE_VALUE     = 0;
static constexpr int SHOWMENU_NB_PENDING_VALUE  = -1;
static constexpr int SHOWMENU_NB_CANCELED_VALUE = -2;

struct AsyncMenuPort
{
  virtual ~AsyncMenuPort() = default;

  // Worker-thread modal menu call. This blocks the dedicated @gfx worker until
  // the UI-side menu is dismissed, while keeping the message thread responsive.
  // That preserves classic JSFX gfx_showmenu() semantics.
  virtual int showMenuModal(const juce::String& description, int x, int y) = 0;

  // Explicit non-blocking menu API.
  //
  // open() returns 1 on success, 0 if no menu was opened.
  // poll() returns one of:
  //   SHOWMENU_NB_NONE_VALUE      (0)  -> no active async menu / no pending result
  //   SHOWMENU_NB_PENDING_VALUE   (-1) -> async menu still open or waiting
  //   SHOWMENU_NB_CANCELED_VALUE  (-2) -> async menu canceled / clicked away
  //   > 0                               -> selected 1-based menu index
  // cancel() returns 1 if a pending/open async menu was canceled, 0 otherwise.
  virtual int showMenuNonBlockingOpen(const juce::String& description, int x, int y) = 0;
  virtual int showMenuNonBlockingPoll() = 0;
  virtual int showMenuNonBlockingCancel() = 0;
};

// -------------------------
// EEL VM wrapper implementing gfx_* API by recording DrawCmds
// -------------------------
class GfxVm : public eelScriptInst
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
    gfx_a2    = get_var("gfx_a2");
    gfx_clear = get_var("gfx_clear");
    gfx_mode  = get_var("gfx_mode");
    gfx_dest  = get_var("gfx_dest");
    gfx_texth = get_var("gfx_texth");
    gfx_ext_retina = get_var("gfx_ext_retina");
    gfx_ext_flags = get_var("gfx_ext_flags");

    mouse_x     = get_var("mouse_x");
    mouse_y     = get_var("mouse_y");
    mouse_cap   = get_var("mouse_cap");
    mouse_wheel = get_var("mouse_wheel");
    mouse_hwheel= get_var("mouse_hwheel");

    srate_var = get_var("srate");
    samplesblock_var = get_var("samplesblock");

    showmenu_nb_none_var = get_var("SHOWMENU_NB_NONE");
    showmenu_nb_pending_var = get_var("SHOWMENU_NB_PENDING");
    showmenu_nb_canceled_var = get_var("SHOWMENU_NB_CANCELED");

    // Default values
    *gfx_x = 0.0;
    *gfx_y = 0.0;
    *gfx_r = 1.0;
    *gfx_g = 1.0;
    *gfx_b = 1.0;
    *gfx_a = 1.0;
    if (gfx_a2)    *gfx_a2 = 1.0;
    if (gfx_mode)  *gfx_mode = 0.0;
    if (gfx_dest)  *gfx_dest = -1.0;
    if (gfx_texth) *gfx_texth = 8.0;
    *gfx_clear = 0.0; // default clear-to-black (JSFX-style). Set gfx_clear=-1 to disable.

    if (srate_var) *srate_var = 44100.0;
    if (samplesblock_var) *samplesblock_var = 0.0;

    refreshShowMenuNbConstants();

    currentFont = juce::Font(juce::Font::getDefaultSansSerifFontName(), 12.0f, juce::Font::plain);
  }

  ~GfxVm() override
  {
    setMenuPort(nullptr);
    runAtExitCode(); // Drain before commands/fonts/keys/string callback state die.
    if (m_vm != nullptr) NSEEL_VM_SetFunctionTable(m_vm, nullptr);
    NSEEL_freefunctiontable(&privateFunctionTable);
  }

  bool usePrivateFunctionTable()
  {
    const std::lock_guard<std::mutex> lock(g_eelGlobalMutex);
    if (!NSEEL_clonefunctiontable(&privateFunctionTable, nullptr)) return false;
    NSEEL_VM_SetFunctionTable(m_vm, &privateFunctionTable);
    return true;
  }

  virtual bool freembufIsNoop() const noexcept { return false; }
  virtual bool allowsNegativeSliderMasks() const noexcept { return false; }

  void refreshShowMenuNbConstants()
  {
    if (showmenu_nb_none_var) *showmenu_nb_none_var = (EEL_F) SHOWMENU_NB_NONE_VALUE;
    if (showmenu_nb_pending_var) *showmenu_nb_pending_var = (EEL_F) SHOWMENU_NB_PENDING_VALUE;
    if (showmenu_nb_canceled_var) *showmenu_nb_canceled_var = (EEL_F) SHOWMENU_NB_CANCELED_VALUE;
  }

  juce::Colour getCurrentColour() const
  {
    const float r = gfx_r ? (float) *gfx_r : 1.0f;
    const float g = gfx_g ? (float) *gfx_g : 1.0f;
    const float b = gfx_b ? (float) *gfx_b : 1.0f;
    const float a = gfx_a2 ? (float) *gfx_a2 : 1.0f;
    const auto channel = [](float v) -> uint32_t {
      return std::isfinite(v) ? (uint32_t)(std::max(0.0f, std::min(1.0f, v)) * 255.0f) : 0u;
    };
    return juce::Colour((channel(a)<<24) | (channel(r)<<16) | (channel(g)<<8) | channel(b));
  }

  static constexpr int kInvalidGfxDestination = -2;

  int currentGfxDestination() const noexcept
  {
    if (!gfx_dest)
      return -1;

    const double raw = (double) *gfx_dest;
    if (!std::isfinite(raw))
      return kInvalidGfxDestination;
    if (raw < 0.0)
      return -1;
    if (raw >= (double) GfxImageBank::kNumImages)
      return kInvalidGfxDestination;

    const int dest = (int) std::trunc(raw);
    return (dest >= 0 && dest < GfxImageBank::kNumImages) ? dest : kInvalidGfxDestination;
  }

  bool isDrawingToMainFramebuffer() const
  {
    return currentGfxDestination() == -1;
  }

  void stampCommand(DrawCmd& cmd, int dest) noexcept
  {
    cmd.dest = dest;
    cmd.imageBank = &imageBank;
    cmd.frameWidth = frameW;
    cmd.frameHeight = frameH;
    const double alpha = gfx_a ? (double)*gfx_a : 1.0;
    cmd.opacity = std::isfinite(alpha) ? (float)std::max(-8.0, std::min(8.0, alpha)) : 0.0f;
    cmd.blitMode = gfx_mode ? boundedGfxInt(*gfx_mode) : 0;
  }

  bool beginDrawCommand(DrawCmd& cmd)
  {
    const int dest = currentGfxDestination();
    if (dest == kInvalidGfxDestination)
      return false;

    if (dest >= 0) {
      ensureImageSlot(dest);
      if (imageWidths[(size_t)dest]<=0 || imageHeights[(size_t)dest]<=0) return false;
    }
    if (dest == -1)
      setImageDirty();

    stampCommand(cmd, dest);
    return true;
  }

  // -------------------------------------------------------------------
  // Lazily clear the framebuffer on the first draw call of a frame.
  // This matches WDL/eel_lice behaviour: if the script doesn't draw anything
  // (common when throttling UI), the previous frame remains visible.
  // -------------------------------------------------------------------
  void setImageDirty()
  {
    // gfx_clear applies to the main framebuffer. Drawing only to gfx_dest>=0
    // must not implicitly clear the visible UI.
    if (!isDrawingToMainFramebuffer() || framebufferDirty)
      return;

    framebufferDirty = true;

    if (gfx_clear && *gfx_clear > -1.0)
    {
      // JSFX packs RGB as: r + g*256 + b*65536  (see WDL eel_lice.h docs)
      const int rgb = boundedGfxInt(*gfx_clear + 0.5);
      const int r = (rgb) & 0xff;
      const int g = (rgb >> 8) & 0xff;
      const int b = (rgb >> 16) & 0xff;

      DrawCmd cmd;
      cmd.type = DrawCmd::Type::Rect;
      stampCommand(cmd, -1);
      cmd.opacity = 1.0f;
      cmd.blitMode = 0;
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
    pendingImageBytes = 0;

    // Clear per-frame host interaction events.
    sliderChangeMask = 0;
    sliderAutomateMask = 0;
    sliderAutomateEndMask = 0;
    undoPointRequested = false;

    frameW = w;
    frameH = h;
    framebufferDirty = false;
    frameRecording = true;
    if (gfx_dest) *gfx_dest = -1;
    if (gfx_a) *gfx_a = 1;
    if (gfx_a2) *gfx_a2 = 1;

    *gfx_w = (double)w;
    *gfx_h = (double)h;

    refreshShowMenuNbConstants();

    if (gfx_frame) *gfx_frame = frameCounter++;
  }

  void setMouse(float x, float y, int cap, float wheel, float hwheel)
  {
    *mouse_x = (double)x;
    *mouse_y = (double)y;
    *mouse_cap = (double)cap;
    *mouse_wheel += (double)wheel;
    *mouse_hwheel += (double)hwheel;
  }

  // -------------------------------------------------------------------
  // Host keyboard input (gfx_getchar)
  // -------------------------------------------------------------------
  void pushKey(int code)
  {
    if (code == 0)
      return;
    if (keyQueue.size() >= 1024) keyQueue.pop_front();
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

  void setHostTrackName(const juce::String& name, std::uint64_t sequence)
  {
    hostTrackName = name;
    hostTrackNameSequence = sequence;
  }

  // -------------------------------------------------------------------
  // Output commands
  // -------------------------------------------------------------------
  const std::vector<DrawCmd>& getCommands() const { return commands; }

  void setMenuPort(AsyncMenuPort* port) { asyncMenuPort = port; }

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

  // These are GFX/host-owned state, not DSP globals even when the AOT
  // variable table contains a same-named entry from shared @init helpers.
  // Do not filter arbitrary gfx_* names: scripts may use those for user data.
  static bool isGfxOwnedVariable(const char* name) noexcept
  {
    static constexpr const char* names[] = {
      "gfx_x", "gfx_y", "gfx_w", "gfx_h", "gfx_r", "gfx_g", "gfx_b",
      "gfx_a", "gfx_a2", "gfx_mode", "gfx_dest", "gfx_clear", "gfx_texth",
      "gfx_ext_retina", "gfx_ext_flags", "gfx_frame",
      "mouse_x", "mouse_y", "mouse_cap", "mouse_wheel", "mouse_hwheel"
    };
    if (!name) return false;
    for (const auto* owned : names)
    {
      const char* a = name;
      const char* b = owned;
      while (*a && *b && std::tolower((unsigned char)*a) == *b) { ++a; ++b; }
      if (!*a && !*b) return true; // EEL variable names are case-insensitive.
    }
    return false;
  }

  struct BoundVar { const char* name; int index; EEL_F* ptr; uint8_t flags; };
  std::vector<BoundVar> boundVars;
  void bindUserVars(const DSPJSFX_VarDesc* vars, const uint8_t* flags, int flagsCount, int count)
  {
    boundVars.clear();
    boundVars.reserve((size_t)count);
    for (int i = 0; i < count; ++i)
    {
      const char* name = vars[i].name;
      const int idx = vars[i].index;
      if (!name || isGfxOwnedVariable(name)) continue;
      const uint8_t dirFlags = (flags != nullptr && idx >= 0 && idx < flagsCount)
                                 ? flags[idx]
                                 : (uint8_t) (DSPJSFX_GFX_VAR_FLAG_TO_GFX | DSPJSFX_GFX_VAR_FLAG_FROM_GFX);
      BoundVar bv { name, idx, get_var(name), dirFlags };
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


  bool syncStringVarUtf8(const char* name, const juce::String& text)
  {
    if (!name || !*name || !m_string_context) return false;

    EEL_F alt = 0.0;
    EEL_F* ptr = nullptr;

    // String slider aliases such as #scene_bus are not numeric NSEEL vars.
    // Resolve them through the EEL string context, otherwise a UI edit can
    // accidentally write user string 0 instead of the named string storage.
    if (name[0] == '#')
      ptr = m_string_context->GetNamedVar(name, true, &alt);
    else
    {
      ptr = get_var(name);
      if (!ptr)
        ptr = m_string_context->GetNamedVar(name, true, &alt);
    }

    if (!ptr)
      return false;

    void* opaque = this;
    EEL_STRING_MUTEXLOCK_SCOPE
    EEL_STRING_STORAGECLASS* wr = nullptr;
    EEL_STRING_GET_FOR_WRITE(*ptr, &wr);
    if (!wr)
      return false;

    const auto utf8 = text.substring(0, 1024).toRawUTF8();
    wr->SetRaw(utf8, (int) std::strlen(utf8));
    return true;
  }

  bool readStringVarUtf8(const char* name, juce::String& out)
  {
    if (!name || !*name || !m_string_context) return false;

    EEL_F alt = 0.0;
    EEL_F* ptr = nullptr;
    if (name[0] == '#')
      ptr = m_string_context->GetNamedVar(name, false, &alt);
    else
    {
      ptr = get_var(name);
      if (!ptr)
        ptr = m_string_context->GetNamedVar(name, false, &alt);
    }

    if (!ptr)
      return false;

    void* opaque = this;
    EEL_STRING_MUTEXLOCK_SCOPE
    EEL_STRING_STORAGECLASS* wr = nullptr;
    const char* s = EEL_STRING_GET_FOR_INDEX(*ptr, &wr);
    if (!s)
      return false;

    const int len = wr ? wr->GetLength() : (int) std::strlen(s);
    out = juce::String::fromUTF8(s, len);
    return true;
  }

  void syncVars(const double* vars, int count)
  {
    for (const auto& bv : boundVars)
    {
      if ((bv.flags & DSPJSFX_GFX_VAR_FLAG_TO_GFX) == 0u)
        continue;
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

  void ensureMemSize(int64_t requiredMem)
  {
    if (!m_vm) return;
    const int requested = (int) std::max<int64_t>(0, std::min<int64_t>(requiredMem,
                                                  (int64_t) NSEEL_RAM_BLOCKS * NSEEL_RAM_ITEMSPERBLOCK));
    if (memSize <= 0) memSize = NSEEL_VM_setramsize(m_vm, 0);
    // A sparse snapshot is not permission to truncate private UI scratch.
    if (requested > memSize) memSize = NSEEL_VM_setramsize(m_vm, requested);
  }

  void syncMem(const double* mem, int memN)
  {
    if (!mem || memN <= 0) return;
    ensureMemSize(memN);
    syncMemRange(mem, 0, memN);
  }

  void syncMemSpans(const MemSpanView* spans, int spanCount, int64_t logicalMemN)
  {
    ensureMemSize(logicalMemN);
    if (!spans || spanCount <= 0) return;
    for (int i = 0; i < spanCount; ++i)
    {
      const auto& span = spans[i];
      if (!span.data || span.count <= 0 || span.base < 0
          || span.base > (int64_t) std::numeric_limits<int>::max() - span.count)
        continue;
      ensureMemSize(span.base + span.count);
      syncMemRange(span.data, span.base, span.count);
    }
  }

  // Read back bound user vars into a JSFX-style vars[] array.
  // Only variables that are actually bound into the EEL VM are written.
  void readVars(double* dst, int count) const
  {
    if (!dst || count <= 0) return;
    for (const auto& bv : boundVars)
    {
      if ((bv.flags & DSPJSFX_GFX_VAR_FLAG_FROM_GFX) == 0u)
        continue;
      if (bv.index >= 0 && bv.index < count && bv.ptr)
        dst[bv.index] = *bv.ptr;
    }
  }

  void readMemRange(int64_t base, double* dst, int count) const
  {
    if (!dst || count <= 0) return;
    // Unallocated EEL pages read as zero. Never allocate them just for a diff.
    std::fill(dst, dst + count, 0.0);
    if (!m_vm || memSize <= 0 || base < 0 || base >= memSize) return;
    const int n = (int) std::min<int64_t>(count, (int64_t) memSize - base);
    int copied = 0;
    while (copied < n)
    {
      const auto pos = (unsigned int)(base + copied);
      int validCount = 0;
      const EEL_F* src = NSEEL_VM_getramptr_noalloc(m_vm, pos, &validCount);
      const int pageRemaining = NSEEL_RAM_ITEMSPERBLOCK - (int)(pos % NSEEL_RAM_ITEMSPERBLOCK);
      const int m = std::min(n - copied, src && validCount > 0 ? validCount : pageRemaining);
      if (src && validCount > 0) std::memcpy(dst + copied, src, (size_t)m * sizeof(EEL_F));
      copied += m;
    }
  }

  // Read back the EEL RAM (JSFX mem[]) into dst.
  // This copies allocated cells and returns zero for absent/out-of-range cells.
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
    NSEEL_addfunc_varparm_ex("gfx_set",        1, 0, NSEEL_PProc_THIS, &eel_gfx_set,        nullptr);
    NSEEL_addfunc_varparm_ex("gfx_setimgdim",  3, 0, NSEEL_PProc_THIS, &eel_gfx_setimgdim,  nullptr);
    NSEEL_addfunc_varparm_ex("gfx_getimgdim",  3, 0, NSEEL_PProc_THIS, &eel_gfx_getimgdim,  nullptr);
    NSEEL_addfunc_varparm_ex("gfx_blit",       3, 0, NSEEL_PProc_THIS, &eel_gfx_blit,       nullptr);
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
    NSEEL_addfunc_varparm_ex("gfx_measurestr",      1, 0, NSEEL_PProc_THIS, &eel_gfx_measurestr,      nullptr);
    NSEEL_addfunc_varparm_ex("gfx_getchar",         0, 0, NSEEL_PProc_THIS, &eel_gfx_getchar,         nullptr);
    NSEEL_addfunc_varparm_ex("gfx_showmenu",        1, 0, NSEEL_PProc_THIS, &eel_gfx_showmenu,        nullptr);
    NSEEL_addfunc_varparm_ex("gfx_showmenu_nb_open",   1, 0, NSEEL_PProc_THIS, &eel_gfx_showmenu_nb_open,   nullptr);
    NSEEL_addfunc_varparm_ex("gfx_showmenu_nb_poll",   0, 0, NSEEL_PProc_THIS, &eel_gfx_showmenu_nb_poll,   nullptr);
    NSEEL_addfunc_varparm_ex("gfx_showmenu_nb_cancel", 0, 0, NSEEL_PProc_THIS, &eel_gfx_showmenu_nb_cancel, nullptr);

    NSEEL_addfunc_varparm_ex("gfx_loadimg", 2, 0, NSEEL_PProc_THIS, &eel_gfx_loadimg, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_setpixel", 3, 0, NSEEL_PProc_THIS, &eel_gfx_setpixel, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_getpixel", 3, 0, NSEEL_PProc_THIS, &eel_gfx_getpixel, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_drawnumber", 2, 0, NSEEL_PProc_THIS, &eel_gfx_drawnumber, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_drawchar", 1, 0, NSEEL_PProc_THIS, &eel_gfx_drawchar, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_measurechar", 3, 0, NSEEL_PProc_THIS, &eel_gfx_measurechar, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_getfont", 0, 0, NSEEL_PProc_THIS, &eel_gfx_getfont, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_blurto", 2, 0, NSEEL_PProc_THIS, &eel_gfx_blurto, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_gradrect", 8, 0, NSEEL_PProc_THIS, &eel_gfx_gradrect, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_muladdrect", 7, 0, NSEEL_PProc_THIS, &eel_gfx_muladdrect, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_blitext", 3, 0, NSEEL_PProc_THIS, &eel_gfx_blitext, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_deltablit", 9, 0, NSEEL_PProc_THIS, &eel_gfx_deltablit, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_transformblit", 8, 0, NSEEL_PProc_THIS, &eel_gfx_transformblit, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_setcursor", 1, 0, NSEEL_PProc_THIS, &eel_gfx_setcursor, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_getdropfile", 1, 0, NSEEL_PProc_THIS, &eel_gfx_getdropfile, nullptr);

    // Minimal host interaction helpers used by many JSFX UIs.
    // See: https://www.reaper.fm/sdk/js/advfunc.php
    NSEEL_addfunc_varparm_ex("sliderchange",   1, 0, NSEEL_PProc_THIS, &eel_sliderchange,   nullptr);
    NSEEL_addfunc_varparm_ex("slider_automate",1, 0, NSEEL_PProc_THIS, &eel_slider_automate,nullptr);
    NSEEL_addfunc_varparm_ex("slider_show",    1, 0, NSEEL_PProc_THIS, &eel_slider_show,    nullptr);

    // DSP-JSFX host track metadata helpers. These mirror the AOT/DSP API and
    // are intentionally registered in the real @gfx VM rather than the inert
    // comm-compat table so UI code can query the current host track name.
    NSEEL_addfunc_varparm_ex("track_name",                 1, 0, NSEEL_PProc_THIS, &eel_track_name,                 nullptr);
    NSEEL_addfunc_varparm_ex("host_track_name",            1, 0, NSEEL_PProc_THIS, &eel_track_name,                 nullptr);
    NSEEL_addfunc_varparm_ex("track_name_available",       0, 0, NSEEL_PProc_THIS, &eel_track_name_available,       nullptr);
    NSEEL_addfunc_varparm_ex("host_track_name_available",  0, 0, NSEEL_PProc_THIS, &eel_track_name_available,       nullptr);
    NSEEL_addfunc_varparm_ex("track_name_seq",             0, 0, NSEEL_PProc_THIS, &eel_track_name_seq,             nullptr);
    NSEEL_addfunc_varparm_ex("host_track_name_seq",        0, 0, NSEEL_PProc_THIS, &eel_track_name_seq,             nullptr);

    // JSFX dynamic access helpers (REAPER dialect)
    // Many scripts use slider(i) / slider(i)=v and spl(i) / spl(i)=v.
    // We implement portable equivalents (see preprocessJsfxForPortableEel()).
    NSEEL_addfunc_varparm_ex("slider",   1, 0, NSEEL_PProc_THIS, &eel_slider,   nullptr);
    NSEEL_addfunc_varparm_ex("spl",      1, 0, NSEEL_PProc_THIS, &eel_spl,      nullptr);
    NSEEL_addfunc_varparm_ex("freembuf", 1, 0, NSEEL_PProc_THIS, &eel_freembuf, nullptr);

    // Inert file_* stubs for @gfx.
    //
    // The @gfx interpreter compiles @init alongside @gfx so shared helper
    // functions remain visible to UI code. Some samplers define their
    // DSP-owned file slot loading helpers in @init and only mirror the
    // resulting state into @gfx via vars/mem. Registering harmless "no file"
    // builtins here lets those scripts compile and run without giving the
    // lightweight @gfx VM direct ownership of host file I/O.
    NSEEL_addfunc_varparm_ex("file_open",         1, 0, NSEEL_PProc_THIS, &eel_file_open,         nullptr);
    NSEEL_addfunc_varparm_ex("file_open_multi",   1, 0, NSEEL_PProc_THIS, &eel_file_open_multi,   nullptr);
    NSEEL_addfunc_varparm_ex("file_close",        1, 0, NSEEL_PProc_THIS, &eel_file_close,        nullptr);
    NSEEL_addfunc_varparm_ex("file_rewind",       1, 0, NSEEL_PProc_THIS, &eel_file_rewind,       nullptr);
    NSEEL_addfunc_varparm_ex("file_seek",         2, 0, NSEEL_PProc_THIS, &eel_file_seek,         nullptr);
    NSEEL_addfunc_varparm_ex("file_avail",        1, 0, NSEEL_PProc_THIS, &eel_file_avail,        nullptr);
    NSEEL_addfunc_varparm_ex("file_text",         1, 0, NSEEL_PProc_THIS, &eel_file_text,         nullptr);
    NSEEL_addfunc_varparm_ex("file_riff",         3, 0, NSEEL_PProc_THIS, &eel_file_riff,         nullptr);
    NSEEL_addfunc_varparm_ex("file_var",          2, 0, NSEEL_PProc_THIS, &eel_file_var,          nullptr);
    NSEEL_addfunc_varparm_ex("file_mem",          3, 0, NSEEL_PProc_THIS, &eel_file_mem,          nullptr);
    NSEEL_addfunc_varparm_ex("file_multi_count",  1, 0, NSEEL_PProc_THIS, &eel_file_multi_count,  nullptr);
    NSEEL_addfunc_varparm_ex("file_multi_select", 2, 0, NSEEL_PProc_THIS, &eel_file_multi_select, nullptr);

  }

  // Raster resources, ordered readback, and host state. No audio-thread I/O.
  using ImageLoader = std::function<juce::Image(const juce::String&)>;
  ImageLoader imageLoader;
  GfxRenderPort* renderPort = nullptr; // Scoped binding owned by the worker.
  std::array<juce::String,128> imageResourceNames {};
  std::array<bool,128> imageResourceAttempted {};
  size_t pendingImageBytes = 0;
  EEL_F* gfx_ext_retina = nullptr;
  EEL_F* gfx_ext_flags = nullptr;
  double displayScale = 1.0;
  int windowFlags = 1; // supported, not focused/visible until the host reports it
  bool cursorPending = false;
  int cursorResource = 32512;
  juce::String cursorName;
  std::vector<juce::String> droppedFiles;
  uint64_t nextFontSerial = 1;
  std::array<uint64_t,17> fontSerials {};
  std::array<juce::String,17> fontNames {};
  std::array<float,17> fontSizes {};
  std::array<int,17> fontFlags {};

  static int gfxByte(double v) noexcept
  {
    return std::isfinite(v) ? (int)(std::max(0.0,std::min(1.0,v))*255.0) : 0;
  }

  void setHostWindowState(bool focused,bool visible,double scale,int flags=0)
  {
    windowFlags=1|(focused?2:0)|(visible?4:0);
    displayScale=std::isfinite(scale)?std::max(1.0,std::min(4.0,scale)):1.0;
    if(gfx_ext_retina && *gfx_ext_retina>0)*gfx_ext_retina=displayScale;
    if(gfx_ext_flags)*gfx_ext_flags=(boundedGfxInt(*gfx_ext_flags)&~3)|(flags&3);
  }

  bool imageBudgetAllows(int slot,int w,int h) const
  {
    size_t bytes=(size_t)w*(size_t)h*4;
    if(bytes>GfxImageBank::kMaxImageBytes || bytes+pendingImageBytes>GfxImageBank::kMaxImageBytes)return false;
    for(int i=0;i<128;++i)if(i!=slot) {
      bytes+=(size_t)imageWidths[(size_t)i]*(size_t)imageHeights[(size_t)i]*4;
      if(bytes>GfxImageBank::kMaxImageBytes)return false;
    }
    return true;
  }

  bool replaceImage(int slot,juce::Image image,int w,int h)
  {
    if(slot<0||slot>=128||!imageBudgetAllows(slot,w,h))return false;
    DrawCmd cmd;cmd.type=DrawCmd::Type::LoadImage;stampCommand(cmd,-1);
    cmd.imageIndex=slot;cmd.imageWidth=w;cmd.imageHeight=h;cmd.replacementImage=std::move(image);
    if(frameRecording) {
      pendingImageBytes+=(size_t)w*h*4;
      commands.push_back(std::move(cmd));
    } else imageBank.images[(size_t)slot]=cmd.replacementImage;
    imageWidths[(size_t)slot]=w;imageHeights[(size_t)slot]=h;
    imageResourceAttempted[(size_t)slot]=true;
    return true;
  }

  bool loadImageSlot(int slot,const juce::String& name)
  {
    if(slot<0||slot>=128||!imageLoader||name.isEmpty())return false;
    try {
      auto image=imageLoader(name);
      if(!image.isValid()||image.getWidth()>GfxImageBank::kMaxLoadedDimension
          ||image.getHeight()>GfxImageBank::kMaxLoadedDimension)return false;
      const int w=image.getWidth(),h=image.getHeight();
      return replaceImage(slot,std::move(image),w,h);
    } catch(const std::exception&) { return false; }
  }

  void ensureImageSlot(int slot)
  {
    if(slot<0||slot>=128||imageResourceAttempted[(size_t)slot])return;
    imageResourceAttempted[(size_t)slot]=true;
    if(imageResourceNames[(size_t)slot].isNotEmpty())loadImageSlot(slot,imageResourceNames[(size_t)slot]);
  }

  bool readGfxTable(double address,int count,std::vector<double>& out,bool requireOneBlock=false)
  {
    out.clear();
    if(!std::isfinite(address)||address<0||address>2147483647.0||count<0||count>8192)return false;
    const uint64_t base=(uint64_t)(address+0.5);
    if(base+(uint64_t)count>2147483647u)return false;
    if(requireOneBlock && (base%65536)+(uint64_t)count>65536)return false;
    out.reserve((size_t)count);
    for(int i=0;i<count;) {
      int valid=0;
      EEL_F* p=NSEEL_VM_getramptr(m_vm,(unsigned int)(base+(uint64_t)i),&valid);
      if(!p||valid<=0){out.clear();return false;}
      const int n=std::min(count-i,valid);
      for(int j=0;j<n;++j){if(!std::isfinite(p[j])){out.clear();return false;}out.push_back(p[j]);}
      i+=n;
    }
    return true;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_loadimg(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<2)return -1.0;
    const int slot=gfxImageIndexFromValue(*parms[0]);if(slot<0)return -1.0;
    juce::String name;
    { EEL_STRING_MUTEXLOCK_SCOPE;const char* p=EEL_STRING_GET_FOR_INDEX(*parms[1],nullptr);if(p)name=juce::String::fromUTF8(p); }
    return self->loadImageSlot(slot,name)?(EEL_F)slot:-1.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_setpixel(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<3)return 0.0;
    DrawCmd c;c.type=DrawCmd::Type::Pixel;if(!self->beginDrawCommand(c))return *parms[0];
    // LICE/JSFX setpixel uses alpha=255 for the source colour; gfx_a2 does not apply.
    c.colour=juce::Colour(0xff000000u|((uint32_t)gfxByte(*parms[0])<<16)|((uint32_t)gfxByte(*parms[1])<<8)|(uint32_t)gfxByte(*parms[2]));
    c.x=(float)boundedGfxInt(*self->gfx_x);c.y=(float)boundedGfxInt(*self->gfx_y);
    self->commands.push_back(std::move(c));return *parms[0];
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_getpixel(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<3)return 0.0;
    const int dest=self->currentGfxDestination();if(dest==-2)return *parms[0];
    if(dest>=0)self->ensureImageSlot(dest);
    uint32_t pixel=0;
    // Reads never call setImageDirty(): reading before the first drawing operation
    // must return previous-frame pixels, not trigger gfx_clear.
    if(self->renderPort) {
      self->renderPort->flush(self->commands);
      pixel=self->renderPort->readPixel(dest,self->imageBank,boundedGfxInt(*self->gfx_x),boundedGfxInt(*self->gfx_y));
    }
    *parms[0]=((pixel>>16)&255)/255.0;*parms[1]=((pixel>>8)&255)/255.0;*parms[2]=(pixel&255)/255.0;
    return *parms[0];
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_gradrect(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<8)return 0;
    DrawCmd c;c.type=DrawCmd::Type::GradRect;
    for(int i=0;i<12;++i){c.values[(size_t)i]=i+4<np?*parms[i+4]:0.0;if(!std::isfinite(c.values[(size_t)i])||std::abs(c.values[(size_t)i])>32760)return 0;}
    if(!self->beginDrawCommand(c))return 0;
    c.x=(float)std::floor(*parms[0]);c.y=(float)std::floor(*parms[1]);c.w=(float)std::floor(*parms[2]);c.h=(float)std::floor(*parms[3]);
    self->commands.push_back(std::move(c));return 0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_muladdrect(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<7)return 0;
    DrawCmd c;c.type=DrawCmd::Type::MulAddRect;
    for(int i=0;i<8;++i){c.values[(size_t)i]=i+4<np?*parms[i+4]:(i==3?1.0:0.0);if(!std::isfinite(c.values[(size_t)i])||std::abs(c.values[(size_t)i])>128)return 0;}
    if(!self->beginDrawCommand(c))return 0;
    c.x=(float)std::floor(*parms[0]);c.y=(float)std::floor(*parms[1]);c.w=(float)std::floor(*parms[2]);c.h=(float)std::floor(*parms[3]);
    self->commands.push_back(std::move(c));return 0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_blurto(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<2)return 0;
    DrawCmd c;c.type=DrawCmd::Type::Blur;if(!self->beginDrawCommand(c))return 0;
    const double x=*self->gfx_x,y=*self->gfx_y,x2=*parms[0],y2=*parms[1];
    c.x=(float)std::min(x,x2);c.y=(float)std::min(y,y2);c.w=(float)std::abs(x2-x);c.h=(float)std::abs(y2-y);
    self->commands.push_back(std::move(c));*self->gfx_x=x2;*self->gfx_y=y2;return *parms[0];
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_blitext(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<3)return 0;
    std::vector<double> table;if(!self->readGfxTable(*parms[1],10,table))return *parms[0];
    EEL_F a[13]={*parms[0],1,*parms[2]}; EEL_F* p[13];
    for(int i=0;i<10;++i)a[i+3]=table[(size_t)i];for(int i=0;i<13;++i)p[i]=a+i;
    eel_gfx_blit(opaque,13,p);return *parms[0];
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_deltablit(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<9)return 0;
    const int source=std::isfinite(*parms[0])&&*parms[0]<0?-1:gfxImageIndexFromValue(*parms[0]);
    if(source<0 && !(std::isfinite(*parms[0])&&*parms[0]<0))return 0;
    if(source>=0)self->ensureImageSlot(source);
    DrawCmd c;c.type=DrawCmd::Type::DeltaBlit;if(!self->beginDrawCommand(c))return 0;
    c.source=source;c.srcX=(float)*parms[1];c.srcY=(float)*parms[2];c.srcW=(float)*parms[3];c.srcH=(float)*parms[4];
    c.destX=(float)*parms[5];c.destY=(float)*parms[6];c.destW=(float)*parms[7];c.destH=(float)*parms[8];
    for(int i=0;i<6;++i)c.values[(size_t)i]=i+9<np?*parms[i+9]:(i==0||i==3?1:0);
    c.useClipRect=np<16||*parms[15]>0.5;self->commands.push_back(std::move(c));return 0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_transformblit(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<8)return 0;
    DrawCmd c;c.type=DrawCmd::Type::TransformBlit;
    c.divisionsX=boundedGfxInt(*parms[5]+0.5);c.divisionsY=boundedGfxInt(*parms[6]+0.5);
    if(c.divisionsX<2||c.divisionsX>64||c.divisionsY<2||c.divisionsY>64)return 0;
    if(!self->readGfxTable(*parms[7],2*c.divisionsX*c.divisionsY,c.transformPoints,true))return 0;
    const int source=std::isfinite(*parms[0])&&*parms[0]<0?-1:gfxImageIndexFromValue(*parms[0]);
    if(source<0 && !(std::isfinite(*parms[0])&&*parms[0]<0))return 0;
    if(source>=0)self->ensureImageSlot(source);
    if(!self->beginDrawCommand(c))return 0;
    c.source=source;c.destX=(float)std::floor(*parms[1]);c.destY=(float)std::floor(*parms[2]);c.destW=(float)std::floor(*parms[3]);c.destH=(float)std::floor(*parms[4]);
    self->commands.push_back(std::move(c));return 0;
  }

  static juce::String gfxCharacter(double value)
  {
    const int cp=boundedGfxInt(value);
    if(cp<=0||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff))return {};
    char b[5]={};int n=0;
    if(cp<128)b[n++]=(char)cp;
    else if(cp<2048){b[n++]=(char)(0xc0|(cp>>6));b[n++]=(char)(0x80|(cp&63));}
    else if(cp<65536){b[n++]=(char)(0xe0|(cp>>12));b[n++]=(char)(0x80|((cp>>6)&63));b[n++]=(char)(0x80|(cp&63));}
    else {b[n++]=(char)(0xf0|(cp>>18));b[n++]=(char)(0x80|((cp>>12)&63));b[n++]=(char)(0x80|((cp>>6)&63));b[n++]=(char)(0x80|(cp&63));}
    return juce::String::fromUTF8(b,n);
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_drawchar(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<1)return 0;
    const int ch=boundedGfxInt(*parms[0]+0.5);
    return emitTextCommand(self,gfxCharacter(ch==10||ch==13?32:ch),1,parms);
  }
  static EEL_F NSEEL_CGEN_CALL eel_gfx_drawnumber(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<2)return 0;
    const int digits=std::max(0,std::min(16,boundedGfxInt(*parms[1]+0.5)));
    char b[512]={};snprintf(b,sizeof(b),"%.*f",digits,(double)*parms[0]);
    return emitTextCommand(self,juce::String::fromUTF8(b),1,parms);
  }
  static EEL_F NSEEL_CGEN_CALL eel_gfx_measurechar(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<3)return 0;
    const auto text=gfxCharacter(*parms[0]);
    *parms[1]=self->currentFontId==0?8:self->currentFont.getStringWidthFloat(text);
    *parms[2]=self->currentFontId==0?8:self->currentFont.getHeight();return *parms[0];
  }
  static EEL_F NSEEL_CGEN_CALL eel_gfx_getfont(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self)return -1;
    if(np>0)writeUtf8ToStringArgument(opaque,parms[0],self->fontNames[(size_t)self->currentFontId]);
    // WDL reports -1 for the built-in bitmap font; 0..15 for setfont slots 1..16.
    return self->currentFontId-1;
  }
  static EEL_F NSEEL_CGEN_CALL eel_gfx_setcursor(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<1)return 0;
    const int resource=boundedGfxInt(*parms[0]);if(resource==0)return 0;
    juce::String name;
    if(np>=2){EEL_STRING_MUTEXLOCK_SCOPE;const char* p=EEL_STRING_GET_FOR_INDEX(*parms[1],nullptr);if(p)name=juce::String::fromUTF8(p);}
    if(self->cursorResource!=resource || !(self->cursorName==name)) {
      self->cursorResource=resource;self->cursorName=name;self->cursorPending=true;
    }
    return 0;
  }
  static EEL_F NSEEL_CGEN_CALL eel_gfx_getdropfile(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<1)return 0;
    const int index=boundedGfxInt(*parms[0]);
    if(index<0){self->droppedFiles.clear();return 0;}
    if(index>=(int)self->droppedFiles.size())return 0;
    if(np>1)writeUtf8ToStringArgument(opaque,parms[1],self->droppedFiles[(size_t)index]);
    return 1;
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

    // GFX retains negative-value undo requests; the DSP shadow uses full masks.
    if (!std::isfinite(argValue)) return 0;
    const double rounded = std::round(argValue);
    if (rounded >= 0.0 && rounded < 18446744073709551616.0)
      return (uint64_t)rounded;
    if (self && self->allowsNegativeSliderMasks()
        && rounded < 0.0 && rounded >= -9223372036854775808.0)
      return (uint64_t)(int64_t)rounded;
    return 0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_set(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1) return 0.0;

    if (self->gfx_r) *self->gfx_r = *parms[0];
    if (self->gfx_g) *self->gfx_g = (np > 1) ? *parms[1] : *parms[0];
    if (self->gfx_b) *self->gfx_b = (np > 2) ? *parms[2] : *parms[0];
    if (self->gfx_a) *self->gfx_a = (np > 3) ? *parms[3] : 1.0;
    if (self->gfx_mode) *self->gfx_mode = (np > 4) ? *parms[4] : 0.0;
    if (np > 5 && self->gfx_dest) *self->gfx_dest = *parms[5];
    if (self->gfx_a2) *self->gfx_a2 = (np > 6) ? *parms[6] : 1.0;

    return 0.0;
  }


  static int gfxImageIndexFromValue(double value) noexcept
  {
    if (!std::isfinite(value) || value < 0.0 || value >= (double) GfxImageBank::kNumImages)
      return -1;
    return (int) std::trunc(value);
  }

  static int gfxImageDimensionFromValue(double value) noexcept
  {
    if (!std::isfinite(value))
      return 0;
    value = std::max(0.0, std::min((double) GfxImageBank::kMaxImageDimension, value));
    return (int) std::floor(value);
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_setimgdim(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<3)return 0;
    const int slot=gfxImageIndexFromValue(*parms[0]);if(slot<0)return 0;
    int w=gfxImageDimensionFromValue(*parms[1]),h=gfxImageDimensionFromValue(*parms[2]);
    if(w<=0||h<=0)w=h=0;
    if(self->imageWidths[(size_t)slot]==w && self->imageHeights[(size_t)slot]==h) {
      self->imageResourceAttempted[(size_t)slot]=true;return 1;
    }
    if(!self->imageBudgetAllows(slot,w,h))return 0;
    try {
      juce::Image image;
      if(w>0&&h>0)image=juce::Image(juce::Image::ARGB,w,h,true,juce::SoftwareImageType());
      if(w>0 && !image.isValid())return 0;
      return self->replaceImage(slot,std::move(image),w,h)?1.0:0.0;
    } catch(const std::exception&) { return 0; }
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_getimgdim(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 3)
      return 0.0;

    int width = 0;
    int height = 0;
    const double imageValue = (double) *parms[0];

    if (std::isfinite(imageValue) && imageValue < 0.0)
    {
      // Handy compatibility extension: -1 describes the main framebuffer.
      width = self->frameW;
      height = self->frameH;
    }
    else
    {
      const int image = gfxImageIndexFromValue(imageValue);
      if (image >= 0)
      {
        self->ensureImageSlot(image);
        width = self->imageWidths[(size_t) image];
        height = self->imageHeights[(size_t) image];
      }
    }

    if (parms[1]) *parms[1] = (EEL_F) width;
    if (parms[2]) *parms[2] = (EEL_F) height;
    return (EEL_F) imageValue;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_blit(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 3)
      return 0.0;

    const double sourceValue = (double) *parms[0];
    int source = -1;
    int sourceWidth = self->frameW;
    int sourceHeight = self->frameH;

    if (!(std::isfinite(sourceValue) && sourceValue < 0.0))
    {
      source = gfxImageIndexFromValue(sourceValue);
      if (source < 0)
        return (EEL_F) sourceValue;
      self->ensureImageSlot(source);
      sourceWidth = self->imageWidths[(size_t) source];
      sourceHeight = self->imageHeights[(size_t) source];
    }

    if (sourceWidth <= 0 || sourceHeight <= 0)
      return (EEL_F) sourceValue;

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Blit;
    if (!self->beginDrawCommand(cmd))
      return (EEL_F) sourceValue;

    const double scale = std::isfinite((double) *parms[1]) ? (double) *parms[1] : 1.0;
    const double rotation = std::isfinite((double) *parms[2]) ? (double) *parms[2] : 0.0;

    cmd.source = source;
    cmd.rotation = (float) rotation;
    // beginDrawCommand captured signed gfx_a (negative additive = subtractive).
    cmd.blitMode = self->gfx_mode ? boundedGfxInt(std::floor((double) *self->gfx_mode)) : 0;

    cmd.srcX = (np >= 4 && std::isfinite((double) *parms[3])) ? (float) *parms[3] : 0.0f;
    cmd.srcY = (np >= 5 && std::isfinite((double) *parms[4])) ? (float) *parms[4] : 0.0f;
    cmd.srcW = (np >= 6 && std::isfinite((double) *parms[5])) ? (float) *parms[5] : (float) sourceWidth;
    cmd.srcH = (np >= 7 && std::isfinite((double) *parms[6])) ? (float) *parms[6] : (float) sourceHeight;

    cmd.destX = (np >= 8 && std::isfinite((double) *parms[7]))
                  ? (float) *parms[7]
                  : (float) (self->gfx_x ? *self->gfx_x : 0.0);
    cmd.destY = (np >= 9 && std::isfinite((double) *parms[8]))
                  ? (float) *parms[8]
                  : (float) (self->gfx_y ? *self->gfx_y : 0.0);

    cmd.destW = (np >= 10 && std::isfinite((double) *parms[9]))
                  ? (float) *parms[9]
                  : (float) ((double) cmd.srcW * scale);
    cmd.destH = (np >= 11 && std::isfinite((double) *parms[10]))
                  ? (float) *parms[10]
                  : (float) ((double) cmd.srcH * scale);

    cmd.rotationXOffset = (np >= 12 && std::isfinite((double) *parms[11])) ? (float) *parms[11] : 0.0f;
    cmd.rotationYOffset = (np >= 13 && std::isfinite((double) *parms[12])) ? (float) *parms[12] : 0.0f;

    if (!std::isfinite(cmd.srcX) || !std::isfinite(cmd.srcY)
        || !std::isfinite(cmd.srcW) || !std::isfinite(cmd.srcH)
        || !std::isfinite(cmd.destX) || !std::isfinite(cmd.destY)
        || !std::isfinite(cmd.destW) || !std::isfinite(cmd.destH)
        || !std::isfinite(cmd.rotation))
      return (EEL_F) sourceValue;

    if (cmd.srcW == 0.0f || cmd.srcH == 0.0f || cmd.destW == 0.0f || cmd.destH == 0.0f)
      return (EEL_F) sourceValue;

    self->commands.push_back(std::move(cmd));
    return (EEL_F) sourceValue;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_rect(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 4) return 0.0;

    const float w = (float) std::floor(*parms[2]);
    const float h = (float) std::floor(*parms[3]);
    if (!(w > 0.0f) || !(h > 0.0f))
      return 0.0;

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Rect;
    if (!self->beginDrawCommand(cmd)) return 0.0;
    cmd.colour = self->getCurrentColour();
    cmd.x = (float) std::floor(*parms[0]);
    cmd.y = (float) std::floor(*parms[1]);
    cmd.w = w;
    cmd.h = h;
    cmd.fill = (np >= 5) ? (*parms[4] > 0.5) : true;

    self->commands.push_back(std::move(cmd));
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_rectto(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 2) return 0.0;

    const float x1 = (float) std::floor(self->gfx_x ? *self->gfx_x : 0.0);
    const float y1 = (float) std::floor(self->gfx_y ? *self->gfx_y : 0.0);
    const float x2 = (float) std::floor(*parms[0]);
    const float y2 = (float) std::floor(*parms[1]);

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Rect;
    if (!self->beginDrawCommand(cmd))
    {
      if (self->gfx_x) *self->gfx_x = *parms[0];
      if (self->gfx_y) *self->gfx_y = *parms[1];
      return *parms[0];
    }
    cmd.colour = self->getCurrentColour();
    cmd.x = std::min(x1, x2);
    cmd.y = std::min(y1, y2);
    cmd.w = std::fabs(x2 - x1);
    cmd.h = std::fabs(y2 - y1);
    cmd.fill = true;
    self->commands.push_back(std::move(cmd));

    if (self->gfx_x) *self->gfx_x = *parms[0];
    if (self->gfx_y) *self->gfx_y = *parms[1];
    return *parms[0];
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_line(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 4) return 0.0;

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Line;
    cmd.antialias = np > 4 ? *parms[4] >= 0.5 : true;
    if (!self->beginDrawCommand(cmd)) return 0.0;
    cmd.colour = self->getCurrentColour();
    cmd.x = (float) std::floor(*parms[0]);
    cmd.y = (float) std::floor(*parms[1]);
    cmd.x2 = (float) std::floor(*parms[2]);
    cmd.y2 = (float) std::floor(*parms[3]);

    self->commands.push_back(std::move(cmd));
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_lineto(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 2) return 0.0;

    const float x1 = (float) std::floor(self->gfx_x ? *self->gfx_x : 0.0);
    const float y1 = (float) std::floor(self->gfx_y ? *self->gfx_y : 0.0);
    const float x2 = (float) std::floor(*parms[0]);
    const float y2 = (float) std::floor(*parms[1]);

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Line;
    cmd.antialias = np > 2 ? *parms[2] >= 0.5 : true;
    if (!self->beginDrawCommand(cmd))
    {
      if (self->gfx_x) *self->gfx_x = *parms[0];
      if (self->gfx_y) *self->gfx_y = *parms[1];
      return *parms[0];
    }
    cmd.colour = self->getCurrentColour();
    cmd.x = x1;
    cmd.y = y1;
    cmd.x2 = x2;
    cmd.y2 = y2;

    self->commands.push_back(std::move(cmd));

    if (self->gfx_x) *self->gfx_x = *parms[0];
    if (self->gfx_y) *self->gfx_y = *parms[1];
    return *parms[0];
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_setfont(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self||np<1)return 0;
    const int id=boundedGfxInt(std::floor(*parms[0]));
    if(id<1||id>16){self->currentFontId=0;*self->gfx_texth=8;return 1;}
    if(np>1) {
      juce::String name;
      {EEL_STRING_MUTEXLOCK_SCOPE;const char* p=EEL_STRING_GET_FOR_INDEX(*parms[1],nullptr);name=juce::String::fromUTF8(p&&*p?p:"Arial");}
      const float size=np>2&&std::isfinite(*parms[2])?(float)std::max(1.0,std::min(2048.0,*parms[2])):10.0f;
      unsigned int packed=np>3&&std::isfinite(*parms[3])?(unsigned int)std::fmod(std::max(0.0,*parms[3]),4294967296.0):0;
      int flags=juce::Font::plain;
      while(packed){switch(std::toupper((int)(packed&255u))){case 'B':flags|=juce::Font::bold;break;case 'I':flags|=juce::Font::italic;break;case 'U':flags|=juce::Font::underlined;break;default:break;}packed>>=8;}
      if(!self->fontSerials[(size_t)id]||!(self->fontNames[(size_t)id]==name)||self->fontSizes[(size_t)id]!=size||self->fontFlags[(size_t)id]!=flags){
        self->fonts[id]=self->getCachedFont(name,size,flags,&self->fontSerials[(size_t)id]);self->fontNames[(size_t)id]=name;self->fontSizes[(size_t)id]=size;self->fontFlags[(size_t)id]=flags;
      }
    }
    auto f=self->fonts.find(id);
    if(f==self->fonts.end()){self->currentFontId=0;*self->gfx_texth=8;return 1;}
    self->currentFontId=id;self->currentFont=f->second;*self->gfx_texth=self->currentFont.getHeight();return 1;
  }


  static juce::Justification textJustificationFromFlags(int flags)
  {
    const bool right = (flags & 0x0002) != 0;
    const bool hcenter = (flags & 0x0001) != 0;
    const bool bottom = (flags & 0x0008) != 0;
    const bool vcenter = (flags & 0x0004) != 0;

    if (hcenter && vcenter) return juce::Justification::centred;
    if (hcenter && bottom)  return juce::Justification::centredBottom;
    if (hcenter)            return juce::Justification::centredTop;
    if (right && vcenter)   return juce::Justification::centredRight;
    if (right && bottom)    return juce::Justification::bottomRight;
    if (right)              return juce::Justification::topRight;
    if (vcenter)            return juce::Justification::centredLeft;
    if (bottom)             return juce::Justification::bottomLeft;
    return juce::Justification::topLeft;
  }

  static int countTextLines(const juce::String& text)
  {
    int lines = 1;
    for (int i = 0; i < text.length(); ++i)
      if (text[i] == '\n')
        ++lines;
    return lines;
  }

  static float measureTextWidth(const juce::Font& font, const juce::String& text)
  {
    juce::StringArray split;
    split.addLines(text);
    if (split.isEmpty())
      return font.getStringWidthFloat(text);

    float width = 0.0f;
    for (int i = 0; i < split.size(); ++i)
      width = std::max(width, font.getStringWidthFloat(split[i]));
    return width;
  }

  static void updateTextPenPosition(GfxVm* self, const juce::String& text)
  {
    if (!self)
      return;

    const double x0 = self->gfx_x ? *self->gfx_x : 0.0;
    const double y0 = self->gfx_y ? *self->gfx_y : 0.0;

    juce::StringArray split;
    split.addLines(text);
    const int numLines = std::max(1, split.size());
    const juce::String lastLine = split.isEmpty() ? text : split[numLines - 1];
    int bitmapWidth=0,bitmapHeight=0;
    if(self->currentFontId==0)LICE_MeasureText(lastLine.toRawUTF8(),&bitmapWidth,&bitmapHeight);
    const float advance = self->currentFontId==0 ? (float)bitmapWidth : self->currentFont.getStringWidthFloat(lastLine);

    if (self->gfx_x)
      *self->gfx_x = x0 + advance;
    if (self->gfx_y)
      *self->gfx_y = y0 + (double) ((numLines - 1) * (self->currentFontId==0 ? 8.0f : self->currentFont.getHeight()));
  }

  static EEL_F emitTextCommand(GfxVm* self, const juce::String& text, INT_PTR np, EEL_F** parms)
  {
    if (!self || text.isEmpty())
      return np > 0 ? *parms[0] : 0.0;

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Text;
    if (!self->beginDrawCommand(cmd))
      return np > 0 ? *parms[0] : 0.0;
    cmd.colour = self->getCurrentColour();
    cmd.font = self->currentFont;
    cmd.bitmapFont = self->currentFontId == 0;
    cmd.fontSerial = self->fontSerials[(size_t)self->currentFontId];
    cmd.text = text;
    cmd.x = (float) boundedGfxInt(std::floor(self->gfx_x ? *self->gfx_x : 0.0));
    cmd.y = (float) boundedGfxInt(std::floor(self->gfx_y ? *self->gfx_y : 0.0));

    if (np >= 4)
    {
      const int flags = boundedGfxInt(std::round(*parms[1]));
      cmd.useTextBounds = true;
      cmd.textFlags = flags;
      cmd.w = (float) std::max(0, boundedGfxInt(std::floor(*parms[2] - cmd.x)));
      cmd.h = (float) std::max(0, boundedGfxInt(std::floor(*parms[3] - cmd.y)));
      cmd.textJustification = textJustificationFromFlags(flags);
    }

    self->commands.push_back(cmd);
    updateTextPenPosition(self, text);
    return np > 0 ? *parms[0] : 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_drawstr(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1) return 0.0;

    EEL_STRING_MUTEXLOCK_SCOPE;
    const char* str = EEL_STRING_GET_FOR_INDEX(*parms[0], nullptr);
    const juce::String text = juce::String::fromUTF8(str ? str : "");
    return emitTextCommand(self, text, np, parms);
  }

  

static EEL_F NSEEL_CGEN_CALL eel_gfx_printf(void* opaque, INT_PTR np, EEL_F** parms)
{
  auto* self = (GfxVm*)opaque;
  if (!self || np < 1) return 0.0;
  juce::String text;
  {
    EEL_STRING_MUTEXLOCK_SCOPE;
    const char* fmt = EEL_STRING_GET_FOR_INDEX(*parms[0], nullptr);
    const auto out = formatGfxPrintf(fmt, (int)np - 1,
      [&](int i) { return (double)*parms[i + 1]; },
      [&](double value) { return EEL_STRING_GET_FOR_INDEX(value, nullptr); });
    text = juce::String::fromUTF8(out.c_str(), (int)out.size());
  }
  return emitTextCommand(self, text, 1, parms);
}

static EEL_F NSEEL_CGEN_CALL eel_gfx_measurestr(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1) return 0.0;

    EEL_STRING_MUTEXLOCK_SCOPE;
    const char* str = EEL_STRING_GET_FOR_INDEX(*parms[0], nullptr);
    const juce::String text = juce::String::fromUTF8(str ? str : "");

    int bitmapWidth=0,bitmapHeight=0;
    if(self->currentFontId==0)LICE_MeasureText(text.toRawUTF8(),&bitmapWidth,&bitmapHeight);
    const float w = self->currentFontId==0 ? (float)bitmapWidth : measureTextWidth(self->currentFont, text);
    juce::StringArray lines; lines.addLines(text);
    const float h = self->currentFontId==0 ? (float)bitmapHeight : (float)std::max(1,lines.size())*self->currentFont.getHeight();

    if (np >= 2 && parms[1]) *parms[1] = (EEL_F) w;
    if (np >= 3 && parms[2]) *parms[2] = (EEL_F) h;

    return *parms[0];
  }


  // Worker-blocking gfx_showmenu bridge.
  //
  // JSFX expects gfx_showmenu() to behave modally: the call returns the user's
  // selection (or 0 on cancel) to the same call site. The explicit
  // gfx_showmenu_nb_* family below provides true async semantics for new UIs.
  static bool decodeMenuDescription(void* opaque, EEL_F* menuExpr, juce::String& outDescription)
  {
    if (opaque == nullptr || menuExpr == nullptr)
      return false;

    EEL_STRING_MUTEXLOCK_SCOPE;
    const char* str = EEL_STRING_GET_FOR_INDEX(*menuExpr, nullptr);
    if (str == nullptr || *str == '\0')
      return false;

    outDescription = juce::String::fromUTF8(str);
    return ! outDescription.isEmpty();
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_showmenu(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1 || self->asyncMenuPort == nullptr)
      return 0.0;

    juce::String description;
    if (! decodeMenuDescription(opaque, parms[0], description))
      return 0.0;

    const int x = (int) std::llround(self->gfx_x ? (double) *self->gfx_x : 0.0);
    const int y = (int) std::llround(self->gfx_y ? (double) *self->gfx_y : 0.0);
    return (EEL_F) self->asyncMenuPort->showMenuModal(description, x, y);
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_showmenu_nb_open(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1 || self->asyncMenuPort == nullptr)
      return 0.0;

    juce::String description;
    if (! decodeMenuDescription(opaque, parms[0], description))
      return 0.0;

    const int x = (int) std::llround(self->gfx_x ? (double) *self->gfx_x : 0.0);
    const int y = (int) std::llround(self->gfx_y ? (double) *self->gfx_y : 0.0);
    return (EEL_F) self->asyncMenuPort->showMenuNonBlockingOpen(description, x, y);
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_showmenu_nb_poll(void* opaque, INT_PTR np, EEL_F** parms)
  {
    juce::ignoreUnused(np, parms);
    auto* self = (GfxVm*)opaque;
    if (!self || self->asyncMenuPort == nullptr)
      return (EEL_F) SHOWMENU_NB_NONE_VALUE;
    return (EEL_F) self->asyncMenuPort->showMenuNonBlockingPoll();
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_showmenu_nb_cancel(void* opaque, INT_PTR np, EEL_F** parms)
  {
    juce::ignoreUnused(np, parms);
    auto* self = (GfxVm*)opaque;
    if (!self || self->asyncMenuPort == nullptr)
      return 0.0;
    return (EEL_F) self->asyncMenuPort->showMenuNonBlockingCancel();
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_circle(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 3) return 0.0;

    DrawCmd cmd;
    cmd.type   = DrawCmd::Type::Circle;
    if (!self->beginDrawCommand(cmd)) return 0.0;
    cmd.colour = self->getCurrentColour();
    cmd.x      = (float) *parms[0];
    cmd.y      = (float) *parms[1];
    cmd.radius = (float) *parms[2];
    cmd.fill   = (np >= 4) ? (*parms[3] > 0.5) : false;
    cmd.antialias = np > 4 ? *parms[4] > 0.5 : true;

    self->commands.push_back(std::move(cmd));
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_roundrect(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 5) return 0.0;

    const float w = (float)*parms[2];
    const float h = (float)*parms[3];
    if (!(w > 0.0f) || !(h > 0.0f))
      return 0.0;

    DrawCmd cmd;
    cmd.type         = DrawCmd::Type::RoundRect;
    if (!self->beginDrawCommand(cmd)) return 0.0;
    cmd.colour       = self->getCurrentColour();
    cmd.x            = (float)*parms[0];
    cmd.y            = (float)*parms[1];
    cmd.w            = w;
    cmd.h            = h;
    cmd.cornerRadius = std::max(0.0f, (float)*parms[4]);
    cmd.fill         = false; // JSFX gfx_roundrect draws an outline.
    cmd.antialias = np > 5 ? *parms[5] > 0.5 : true;

    self->commands.push_back(std::move(cmd));
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_arc(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 5) return 0.0;

    const double cx = (double)*parms[0];
    const double cy = (double)*parms[1];
    const double r  = (double)*parms[2];
    const double a1 = (double)*parms[3];
    const double a2 = (double)*parms[4];

    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(r) ||
        !std::isfinite(a1) || !std::isfinite(a2) || r <= 0.0)
      return 0.0;
    const double floatMax = (double)std::numeric_limits<float>::max();
    if (std::abs(cx) > floatMax || std::abs(cy) > floatMax || r > floatMax
        || std::abs(a1) > floatMax || std::abs(a2) > floatMax)
      return 0.0;

    DrawCmd cmd;
    cmd.type   = DrawCmd::Type::Arc;
    cmd.antialias = np > 5 ? *parms[5] > 0.5 : true;
    if (!self->beginDrawCommand(cmd)) return 0.0;
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

    // gfx_triangle(x1,y1,x2,y2,x3,y3[,x4,y4...]) -- always filled.
    const int pairs = (int)(np / 2);
    if (pairs < 3) return 0.0;

    DrawCmd cmd;
    cmd.type   = DrawCmd::Type::Triangle;
    if (!self->beginDrawCommand(cmd)) return 0.0;
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


  static EEL_F NSEEL_CGEN_CALL eel_gfx_getchar(void* opaque,INT_PTR np,EEL_F** parms)
  {
    auto* self=(GfxVm*)opaque;if(!self)return 0;
    if(np>1 && parms[1])*parms[1]=0;
    if(np>0 && *parms[0]!=0) {
      if(*parms[0]==65536)return self->windowFlags;
      const double v=*parms[0];
      if(!std::isfinite(v)||v<std::numeric_limits<int>::min()||v>std::numeric_limits<int>::max())return 0;
      return self->keysDown.count((int)v)?1:0;
    }
    if(self->keyQueue.empty())return 0;
    const int code=self->keyQueue.front();self->keyQueue.pop_front();
    if(np>1 && parms[1] && ((uint32_t)code>>24)==(uint32_t)'u')*parms[1]=(uint32_t)code&0xffffffu;
    return code;
  }

  static bool writeUtf8ToStringArgument(void* opaque, EEL_F* slot, const juce::String& text)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || slot == nullptr || !self->m_string_context)
      return false;

    EEL_STRING_MUTEXLOCK_SCOPE
    EEL_STRING_STORAGECLASS* wr = nullptr;
    EEL_STRING_GET_FOR_WRITE(*slot, &wr);
    if (!wr)
      return false;

    const juce::String limited = text.substring(0, 1024);
    const char* utf8 = limited.toRawUTF8();
    wr->SetRaw(utf8, (int) std::strlen(utf8));
    return true;
  }

  static EEL_F NSEEL_CGEN_CALL eel_track_name(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1 || parms == nullptr || parms[0] == nullptr)
      return 0.0;

    if (self->hostTrackName.isEmpty())
      return 0.0;

    return writeUtf8ToStringArgument(opaque, parms[0], self->hostTrackName) ? 1.0 : 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_track_name_available(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void) np;
    (void) parms;

    auto* self = (GfxVm*) opaque;
    return (self != nullptr && self->hostTrackName.isNotEmpty()) ? 1.0 : 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_track_name_seq(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void) np;
    (void) parms;

    auto* self = (GfxVm*) opaque;
    return self != nullptr ? (EEL_F) (double) self->hostTrackNameSequence : 0.0;
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

  static EEL_F NSEEL_CGEN_CALL eel_slider_show(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 1)
      return 0.0;

    const double v = (double) *parms[0];
    const uint64_t mask = sliderMaskFromArg (self, parms[0], v);
    if (mask == 0)
      return 0.0;

    if (np >= 2)
    {
      const double show = (double) *parms[1];
      if (show == -1.0)
        self->sliderVisibleMask ^= mask;
      else if (show <= 0.0)
        self->sliderVisibleMask &= ~mask;
      else
        self->sliderVisibleMask |= mask;
    }

    return (EEL_F) (double) (self->sliderVisibleMask & mask);
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

    const int idx = (int) jsfxTruncIndexLikeAot ((double) *parms[0]);
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

    int64_t n = jsfxTruncIndexLikeAot ((double) *parms[0]);
    if (n < 0) n = 0;
    if (n > 0x7fffffffLL) n = 0x7fffffffLL;

    if (self->freembufIsNoop())
      return 0.0;

    // Shrink/grow EEL RAM (mem[]).
    if (self->m_vm)
      NSEEL_VM_setramsize(self->m_vm, (unsigned int)n);

    self->memSize = (int)n;
    return 0.0;
  }

  // -------------------------------------------------------------------
  // Inert DSP file_* stubs for the lightweight @gfx VM.
  //
  // The DSP runtime owns real file slot loading and mirrors the resulting data
  // into @gfx-visible vars/mem. These implementations deliberately expose
  // "missing file" behaviour so shared @init helper chains that mention
  // file_open()/file_open_multi()/... can compile and execute safely inside the
  // @gfx interpreter without performing file I/O.
  // -------------------------------------------------------------------
  static EEL_F NSEEL_CGEN_CALL eel_file_open(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;
    (void)np;
    (void)parms;
    return -1.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_file_open_multi(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;
    (void)np;
    (void)parms;
    return -1.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_file_close(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;
    (void)np;
    (void)parms;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_file_rewind(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;
    (void)np;
    (void)parms;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_file_seek(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;
    (void)np;
    (void)parms;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_file_avail(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;
    (void)np;
    (void)parms;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_file_text(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;
    (void)np;
    (void)parms;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_file_riff(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;

    if (np >= 2 && parms[1] != nullptr)
      *parms[1] = 0.0;

    if (np >= 3 && parms[2] != nullptr)
      *parms[2] = 0.0;

    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_file_var(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;

    if (np >= 2 && parms[1] != nullptr)
      *parms[1] = 0.0;

    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_file_mem(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;
    (void)np;
    (void)parms;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_file_multi_count(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;
    (void)np;
    (void)parms;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_file_multi_select(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque;
    (void)np;
    (void)parms;
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
  EEL_F* gfx_a2 = nullptr;
  EEL_F* gfx_clear = nullptr;
  EEL_F* gfx_mode = nullptr;
  EEL_F* gfx_dest = nullptr;
  EEL_F* gfx_texth = nullptr;

    double frameCounter = 0.0;

  int frameW = 0;
  int frameH = 0;
  bool framebufferDirty = false;

  EEL_F* mouse_x = nullptr;
  EEL_F* mouse_y = nullptr;
  EEL_F* mouse_cap = nullptr;
  EEL_F* mouse_wheel = nullptr;
  EEL_F* mouse_hwheel = nullptr;

  EEL_F* showmenu_nb_none_var = nullptr;
  EEL_F* showmenu_nb_pending_var = nullptr;
  EEL_F* showmenu_nb_canceled_var = nullptr;

  EEL_F* srate_var = nullptr;
  EEL_F* samplesblock_var = nullptr;

  juce::String hostTrackName;
  std::uint64_t hostTrackNameSequence = 0;

  // Current JSFX mem[] size (in doubles) synced into the EEL VM RAM.
  int memSize = 0;

  eel_function_table privateFunctionTable {};
  AsyncMenuPort* asyncMenuPort = nullptr;

  struct FontCacheEntry
  {
    juce::String name;
    float size;
    int flags;
    juce::Font font;
    uint64_t identity;
  };
  std::vector<FontCacheEntry> fontCache;
  size_t nextFontCacheSlot = 0;

  juce::Font getCachedFont(const juce::String& name, float size, int flags, uint64_t* identity = nullptr)
  {
    for (const auto& e : fontCache)
      if (e.size == size && e.flags == flags && e.name == name) { if (identity) *identity=e.identity; return e.font; }
    juce::Font font(name, size, flags);
    FontCacheEntry entry { name, size, flags, font, nextFontSerial++ };
    if (identity) *identity=entry.identity;
    constexpr size_t capacity = 64;
    if (fontCache.size() < capacity) fontCache.push_back(std::move(entry));
    else
    {
      fontCache[nextFontCacheSlot] = std::move(entry);
      nextFontCacheSlot = (nextFontCacheSlot + 1) % capacity;
    }
    return font;
  }

  // Drawing state
  GfxImageBank imageBank;
  std::array<int, GfxImageBank::kNumImages> imageWidths {};
  std::array<int, GfxImageBank::kNumImages> imageHeights {};
  bool frameRecording = false;

  std::unordered_map<int, juce::Font> fonts;
  int currentFontId = 0;
  juce::Font currentFont;

  std::vector<DrawCmd> commands;

  // Host interaction event state
  uint64_t sliderChangeMask = 0;
  uint64_t sliderAutomateMask = 0;
  uint64_t sliderAutomateEndMask = 0;
  uint64_t sliderVisibleMask = ~UINT64_C (0);
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

    juce::String hostTrackName;
    std::uint64_t hostTrackNameSeq = 0;
  };

  Interpreter(const char* jsfxSourceText, GfxVm::ImageLoader loader = {})
  {
    sections = extractJsfxSections(jsfxSourceText);
    if (!sections.hasGfx)
      return;

    vm = std::make_unique<GfxVm>();
    vm->imageLoader=std::move(loader);
    const std::string source=jsfxSourceText?jsfxSourceText:"";
    size_t lineStart=0;
    while(lineStart<source.size()) {
      auto end=source.find('\n',lineStart);if(end==std::string::npos)end=source.size();
      std::string line=source.substr(lineStart,end-lineStart);lineStart=end+1;
      const auto start=line.find_first_not_of(" \t\r");if(start==std::string::npos)continue;
      line.erase(0,start);if(line[0]=='@')break;
      if(line.compare(0,9,"filename:")!=0)continue;
      const auto comma=line.find(',',9);if(comma==std::string::npos)continue;
      char* tail=nullptr;const long slot=std::strtol(line.c_str()+9,&tail,10);
      if(tail!=line.c_str()+comma||slot<0||slot>=128)continue;
      auto name=line.substr(comma+1);const auto first=name.find_first_not_of(" \t\r");
      if(first==std::string::npos)continue;name=name.substr(first);name.erase(name.find_last_not_of(" \t\r")+1);
      vm->imageResourceNames[(size_t)slot]=juce::String::fromUTF8(name.c_str());
    }

    // Bind sliders and user vars.
    vm->bindSliderPtrs();

    // DSPJSFX_VARS is a *symbol* emitted by dsp_jsfx_aot.py (static const array),
    // not a preprocessor macro. So `defined(DSPJSFX_VARS)` is always false.
    // The fallback table at the top of this file guarantees DSPJSFX_VARS exists anyway.
    vm->bindUserVars(DSPJSFX_VARS, DSPJSFX_GFX_VAR_FLAGS, (int) DSPJSFX_GFX_VAR_FLAGS_COUNT, (int) DSPJSFX_VARS_COUNT);

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


  bool syncStringVarUtf8(const char* name, const juce::String& text)
  {
    return vm ? vm->syncStringVarUtf8(name, text) : false;
  }

  bool readStringVarUtf8(const char* name, juce::String& out)
  {
    return vm ? vm->readStringVarUtf8(name, out) : false;
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

  void setMenuPort(AsyncMenuPort* port)
  {
    if (vm) vm->setMenuPort(port);
  }

  class RenderBinding {
  public:
    RenderBinding(GfxVm* v,GfxRenderPort* p):vm(v),previous(v?v->renderPort:nullptr){if(vm)vm->renderPort=p;}
    ~RenderBinding(){if(vm)vm->renderPort=previous;}
    RenderBinding(const RenderBinding&)=delete;
    RenderBinding& operator=(const RenderBinding&)=delete;
  private:GfxVm* vm;GfxRenderPort* previous;
  };
  RenderBinding bindRenderer(GfxRenderPort& port){return RenderBinding(vm.get(),&port);}
  void setHostWindowState(bool focused,bool visible,double scale,int flags=0)
  {if(vm)vm->setHostWindowState(focused,visible,scale,flags);}
  bool wantsRetina() const {return vm&&vm->gfx_ext_retina&&*vm->gfx_ext_retina>0;}
  bool takeCursorRequest(int& resource,juce::String& name)
  {if(!vm||!vm->cursorPending)return false;resource=vm->cursorResource;name=vm->cursorName;vm->cursorPending=false;return true;}
  void addDroppedFile(const juce::String& path){if(vm&&vm->droppedFiles.size()<1024)vm->droppedFiles.push_back(path);}

  void prepareFrame(int width, int height, const Snapshot& snap)
  {
    if (!hasGfxSection() || !gfxCompiledOk()) return;

    juce::ScopedNoDenormals noDenormals;
    const auto start = profilingEnabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    vm->ensureMemSize(snap.logicalMemN);
    // Record init draws/allocations in order rather than clearing them afterwards.
    vm->beginFrame(width, height);
    vm->setHostTrackName(snap.hostTrackName, snap.hostTrackNameSeq);

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
      if(vm->gfx_ext_retina && *vm->gfx_ext_retina>0)*vm->gfx_ext_retina=vm->displayScale;
    }

    // ------------------------------------------------------------
    // Sync state into VM.
    //
    // IMPORTANT:
    // We always sync sliders (they are the "public" parameter surface).
    //
    // Vars/mem sync policy is decided by the caller. The UI worker omits
    // vars/mem on button-edge frames, and while waiting for a fresh audio
    // snapshot after UI-authored writes. If a snapshot supplies vars/mem,
    // apply them unconditionally so held mouse buttons do not freeze
    // audio-driven visuals.
    // ------------------------------------------------------------
    if (snap.sliders) vm->syncSliders(snap.sliders, snap.slidersCount);

    if (snap.vars)    vm->syncVars(snap.vars, snap.varsCount);
    if (snap.memSpans && snap.memSpanCount > 0) vm->syncMemSpans(snap.memSpans, snap.memSpanCount, snap.logicalMemN);
    else if (snap.mem)                     vm->syncMem(snap.mem, snap.memN);

    vm->setTiming(snap.srate, snap.samplesblock);

    vm->setMouse(mouseX, mouseY, mouseCap, mouseWheel, mouseHWheel);

    framePrepared = true;
    syncMilliseconds = profilingEnabled ? std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() : 0.0;
  }

  void executeFrame()
  {
    if (!framePrepared || !vm || !code_gfx) return;
    framePrepared = false;
    juce::ScopedNoDenormals noDenormals;
    const auto start = profilingEnabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    NSEEL_code_execute(code_gfx);
    vm->frameRecording = false;
    gfxMilliseconds = profilingEnabled ? std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() : 0.0;
    mouseWheel = 0.0f;
    mouseHWheel = 0.0f;
  }

  void renderFrame(int width, int height, const Snapshot& snap)
  {
    prepareFrame(width, height, snap);
    executeFrame();
  }

  void setProfilingEnabled(bool enabled) noexcept { profilingEnabled = enabled; }
  double getSyncMilliseconds() const noexcept { return syncMilliseconds; }
  double getGfxMilliseconds() const noexcept { return gfxMilliseconds; }

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
  bool framePrepared = false;
  bool profilingEnabled = false;
  double syncMilliseconds = 0.0;
  double gfxMilliseconds = 0.0;

  juce::String lastError;

  float mouseX = 0.0f;
  float mouseY = 0.0f;
  int mouseCap = 0;
  float mouseWheel = 0.0f;
  float mouseHWheel = 0.0f;
};

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
