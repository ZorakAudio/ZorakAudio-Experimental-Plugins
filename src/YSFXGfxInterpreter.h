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

#if JUCE_MAC || JUCE_LINUX
 #include <dlfcn.h>
#endif
#if JUCE_WINDOWS
 #ifndef NOMINMAX
  #define NOMINMAX 1
 #endif
 #include <windows.h>
#endif

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
  std::vector<std::pair<int, std::string>> filenameSlots;
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

    if (!ltrim.empty() && ltrim.rfind("filename:", 0) == 0)
    {
      const size_t comma = ltrim.find(',');
      if (comma != std::string::npos)
      {
        const std::string indexText = ltrim.substr(std::strlen("filename:"), comma - std::strlen("filename:"));
        const std::string pathText = ltrim.substr(comma + 1);
        try
        {
          const int slot = std::stoi(indexText);
          if (slot >= 0)
            out.filenameSlots.emplace_back(slot, pathText);
        }
        catch (...)
        {
        }
      }
      continue;
    }

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

static inline void paintCommands(juce::Graphics& g, const std::vector<DrawCmd>& cmds);

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
// EEL VM wrapper implementing gfx_* API by rendering into JUCE Images
// -------------------------
#ifndef ZA_JSFX_SOURCE_ROOT
#define ZA_JSFX_SOURCE_ROOT ""
#endif
#ifndef ZA_JSFX_RESOURCE_DIR
#define ZA_JSFX_RESOURCE_DIR ""
#endif

static inline float clampUnitFloat(double v) noexcept
{
  if (v <= 0.0) return 0.0f;
  if (v >= 1.0) return 1.0f;
  return (float) v;
}

static inline int roundToNearestInt(double v) noexcept
{
  return (int) std::llround(v);
}

static inline juce::String trimJsfxPath(juce::String s)
{
  s = s.trim();
  if (s.length() >= 2)
  {
    const bool quotedWithDoubleQuotes = s.startsWithChar('"') && s.endsWithChar('"');
    const bool quotedWithSingleQuotes = s.startsWithChar('\'') && s.endsWithChar('\'');
    if (quotedWithDoubleQuotes || quotedWithSingleQuotes)
      s = s.substring(1, s.length() - 1);
  }
  return s.trim();
}

struct PixelF
{
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  float a = 0.0f;
};

static inline PixelF pixelFromColour(const juce::Colour& c)
{
  return PixelF { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(), c.getFloatAlpha() };
}

static inline juce::Colour colourFromPixel(const PixelF& p)
{
  return juce::Colour::fromFloatRGBA(clampUnitFloat(p.r), clampUnitFloat(p.g), clampUnitFloat(p.b), clampUnitFloat(p.a));
}

static inline PixelF readPixelSafe(const juce::Image& img, int x, int y)
{
  if (img.isNull() || x < 0 || y < 0 || x >= img.getWidth() || y >= img.getHeight())
    return {};
  return pixelFromColour(img.getPixelAt(x, y));
}

static inline PixelF lerpPixel(const PixelF& a, const PixelF& b, float t)
{
  return PixelF {
    a.r + (b.r - a.r) * t,
    a.g + (b.g - a.g) * t,
    a.b + (b.b - a.b) * t,
    a.a + (b.a - a.a) * t,
  };
}

static inline PixelF sampleImage(const juce::Image& img, double x, double y, bool nearest)
{
  if (img.isNull())
    return {};

  if (nearest)
    return readPixelSafe(img, roundToNearestInt(x), roundToNearestInt(y));

  const int x0 = (int) std::floor(x);
  const int y0 = (int) std::floor(y);
  const float fx = (float) (x - (double) x0);
  const float fy = (float) (y - (double) y0);

  const PixelF p00 = readPixelSafe(img, x0,     y0);
  const PixelF p10 = readPixelSafe(img, x0 + 1, y0);
  const PixelF p01 = readPixelSafe(img, x0,     y0 + 1);
  const PixelF p11 = readPixelSafe(img, x0 + 1, y0 + 1);

  const PixelF a = lerpPixel(p00, p10, fx);
  const PixelF b = lerpPixel(p01, p11, fx);
  return lerpPixel(a, b, fy);
}

static inline PixelF compositePixel(const PixelF& dst, PixelF src, double gfxAlpha, double alphaWrite, int gfxMode)
{
  const bool additive = (gfxMode & 1) != 0;
  const bool ignoreSourceAlpha = (gfxMode & 2) != 0;

  const float srcCoverage = ignoreSourceAlpha ? 1.0f : clampUnitFloat(src.a);
  const float signedGlobal = (float) gfxAlpha;
  const float magnitude = std::abs(signedGlobal) * srcCoverage;
  const float a = clampUnitFloat(magnitude);
  const float writtenAlpha = clampUnitFloat(alphaWrite);

  PixelF out = dst;

  if (additive)
  {
    const float sign = (signedGlobal >= 0.0f) ? 1.0f : -1.0f;
    out.r = clampUnitFloat(dst.r + src.r * a * sign);
    out.g = clampUnitFloat(dst.g + src.g * a * sign);
    out.b = clampUnitFloat(dst.b + src.b * a * sign);
    out.a = juce::jlimit(0.0f, 1.0f, dst.a + writtenAlpha * a);
    return out;
  }

  out.r = src.r * a + dst.r * (1.0f - a);
  out.g = src.g * a + dst.g * (1.0f - a);
  out.b = src.b * a + dst.b * (1.0f - a);
  out.a = writtenAlpha * a + dst.a * (1.0f - a);
  out.r = clampUnitFloat(out.r);
  out.g = clampUnitFloat(out.g);
  out.b = clampUnitFloat(out.b);
  out.a = clampUnitFloat(out.a);
  return out;
}

static inline double readVmRamScalar(NSEEL_VMCTX vm, int64_t index, double fallback = 0.0)
{
  if (vm == nullptr || index < 0 || index > (int64_t) std::numeric_limits<unsigned int>::max())
    return fallback;

  int validCount = 0;
  EEL_F* ptr = NSEEL_VM_getramptr(vm, (unsigned int) index, &validCount);
  if (ptr == nullptr || validCount <= 0)
    return fallback;
  return (double) ptr[0];
}

static juce::File jsfxModuleFileFromThisBinary()
{
 #if JUCE_WINDOWS
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&jsfxModuleFileFromThisBinary),
                         &module) && module != nullptr)
  {
    wchar_t path[4096] = {};
    const DWORD len = GetModuleFileNameW(module, path, (DWORD) std::size(path));
    if (len > 0)
      return juce::File(juce::String(path, (int) len));
  }
 #elif JUCE_MAC || JUCE_LINUX
  Dl_info info {};
  if (dladdr((const void*) &jsfxModuleFileFromThisBinary, &info) != 0 && info.dli_fname != nullptr)
    return juce::File(juce::String::fromUTF8(info.dli_fname));
 #endif

  return {};
}

static inline void appendUniqueFile(std::vector<juce::File>& files, const juce::File& file)
{
  if (file == juce::File())
    return;

  const juce::String p = file.getFullPathName();
  if (p.isEmpty())
    return;

  for (const auto& existing : files)
    if (existing == file)
      return;

  files.push_back(file);
}

static std::vector<juce::File> buildDefaultJsfxResourceRoots()
{
  std::vector<juce::File> roots;

  const juce::String buildResourceDir = juce::String::fromUTF8(ZA_JSFX_RESOURCE_DIR).trim();
  const juce::String buildSourceRoot  = juce::String::fromUTF8(ZA_JSFX_SOURCE_ROOT).trim();

  if (buildResourceDir.isNotEmpty())
    appendUniqueFile(roots, juce::File(buildResourceDir));

  if (buildSourceRoot.isNotEmpty())
  {
    const juce::File srcRoot(buildSourceRoot);
    appendUniqueFile(roots, srcRoot);
    appendUniqueFile(roots, srcRoot.getChildFile("resource"));
  }

  const juce::File moduleFile = jsfxModuleFileFromThisBinary();
  if (moduleFile != juce::File())
  {
    const juce::File moduleDir = moduleFile.getParentDirectory();
    const juce::String stem = moduleFile.getFileNameWithoutExtension();

    appendUniqueFile(roots, moduleDir);
    appendUniqueFile(roots, moduleDir.getChildFile("resource"));
    appendUniqueFile(roots, moduleDir.getChildFile(stem + ".resources"));
    appendUniqueFile(roots, moduleDir.getParentDirectory().getChildFile(stem + ".resources"));
    appendUniqueFile(roots, moduleDir.getParentDirectory().getChildFile("resource"));
    appendUniqueFile(roots, moduleFile.getSiblingFile(stem + ".resources"));
   #if JUCE_MAC
    appendUniqueFile(roots, moduleDir.getParentDirectory().getSiblingFile("Resources"));
    appendUniqueFile(roots, moduleDir.getParentDirectory().getParentDirectory().getChildFile("Resources"));
   #endif
  }

  appendUniqueFile(roots, juce::File::getCurrentWorkingDirectory());
  return roots;
}

static juce::MouseCursor::StandardCursorType mapJsfxCursorType(int resourceId, juce::String customName)
{
  customName = customName.trim().toLowerCase();

  if (customName == "ibeam" || customName == "i-beam" || customName == "text")
    return juce::MouseCursor::IBeamCursor;
  if (customName == "cross" || customName == "crosshair")
    return juce::MouseCursor::CrosshairCursor;
  if (customName == "hand" || customName == "pointinghand")
    return juce::MouseCursor::PointingHandCursor;
  if (customName == "wait" || customName == "busy")
    return juce::MouseCursor::WaitCursor;
  if (customName == "left_right" || customName == "leftright" || customName == "sizewe")
    return juce::MouseCursor::LeftRightResizeCursor;
  if (customName == "up_down" || customName == "updown" || customName == "sizens")
    return juce::MouseCursor::UpDownResizeCursor;
  if (customName == "top_left_corner" || customName == "sizenwse")
    return juce::MouseCursor::TopLeftCornerResizeCursor;
  if (customName == "top_right_corner" || customName == "sizenesw")
    return juce::MouseCursor::TopRightCornerResizeCursor;
  if (customName == "dragginghand" || customName == "sizeall" || customName == "move")
    return juce::MouseCursor::DraggingHandCursor;
  if (customName == "copying")
    return juce::MouseCursor::CopyingCursor;
  if (customName == "normal" || customName == "arrow")
    return juce::MouseCursor::NormalCursor;
  if (customName == "no" || customName == "forbidden")
    return juce::MouseCursor::NoCursor;

  switch (resourceId)
  {
    case 32513: return juce::MouseCursor::IBeamCursor;
    case 32514: return juce::MouseCursor::WaitCursor;
    case 32515: return juce::MouseCursor::CrosshairCursor;
    case 32649: return juce::MouseCursor::PointingHandCursor;
    case 32644: return juce::MouseCursor::UpDownResizeCursor;
    case 32645: return juce::MouseCursor::LeftRightResizeCursor;
    case 32642: return juce::MouseCursor::TopLeftCornerResizeCursor;
    case 32643: return juce::MouseCursor::TopRightCornerResizeCursor;
    case 32646: return juce::MouseCursor::DraggingHandCursor;
    default:    return juce::MouseCursor::NormalCursor;
  }
}

class GfxVm : public eelScriptInst
{
public:
  EEL_F* get_var(const char* name)
  {
    return m_vm ? NSEEL_VM_regvar(m_vm, name) : nullptr;
  }

  GfxVm()
  {
    static std::once_flag s_initOnce;
    std::call_once(s_initOnce, []() {
      NSEEL_init();
      eelScriptInst::init();
      registerGfxBuiltins();
    });

    gfx_x     = get_var("gfx_x");
    gfx_y     = get_var("gfx_y");
    gfx_w     = get_var("gfx_w");
    gfx_h     = get_var("gfx_h");
    gfx_frame = get_var("gfx_frame");
    gfx_r     = get_var("gfx_r");
    gfx_g     = get_var("gfx_g");
    gfx_b     = get_var("gfx_b");
    gfx_a     = get_var("gfx_a");
    gfx_clear = get_var("gfx_clear");
    gfx_mode  = get_var("gfx_mode");
    gfx_dest  = get_var("gfx_dest");
    gfx_a2    = get_var("gfx_a2");
    gfx_texth = get_var("gfx_texth");
    gfx_ext_retina = get_var("gfx_ext_retina");
    gfx_ext_flags = get_var("gfx_ext_flags");

    mouse_x      = get_var("mouse_x");
    mouse_y      = get_var("mouse_y");
    mouse_cap    = get_var("mouse_cap");
    mouse_wheel  = get_var("mouse_wheel");
    mouse_hwheel = get_var("mouse_hwheel");

    srate_var = get_var("srate");
    samplesblock_var = get_var("samplesblock");

    showmenu_nb_none_var = get_var("SHOWMENU_NB_NONE");
    showmenu_nb_pending_var = get_var("SHOWMENU_NB_PENDING");
    showmenu_nb_canceled_var = get_var("SHOWMENU_NB_CANCELED");

    if (gfx_x) *gfx_x = 0.0;
    if (gfx_y) *gfx_y = 0.0;
    if (gfx_frame) *gfx_frame = 0.0;
    if (gfx_r) *gfx_r = 1.0;
    if (gfx_g) *gfx_g = 1.0;
    if (gfx_b) *gfx_b = 1.0;
    if (gfx_a) *gfx_a = 1.0;
    if (gfx_clear) *gfx_clear = 0.0;
    if (gfx_mode) *gfx_mode = 0.0;
    if (gfx_dest) *gfx_dest = -1.0;
    if (gfx_a2) *gfx_a2 = 1.0;
    if (gfx_ext_retina) *gfx_ext_retina = 1.0;
    if (gfx_ext_flags) *gfx_ext_flags = 0.0;

    if (srate_var) *srate_var = 44100.0;
    if (samplesblock_var) *samplesblock_var = 0.0;

    refreshShowMenuNbConstants();

    currentFont = juce::Font(juce::Font::getDefaultSansSerifFontName(), 12.0f, juce::Font::plain);
    fonts[0] = currentFont;
    updateTextMetrics();

    resourceSearchRoots = buildDefaultJsfxResourceRoots();
  }

  virtual bool freembufIsNoop() const noexcept { return false; }

  void refreshShowMenuNbConstants()
  {
    if (showmenu_nb_none_var) *showmenu_nb_none_var = (EEL_F) SHOWMENU_NB_NONE_VALUE;
    if (showmenu_nb_pending_var) *showmenu_nb_pending_var = (EEL_F) SHOWMENU_NB_PENDING_VALUE;
    if (showmenu_nb_canceled_var) *showmenu_nb_canceled_var = (EEL_F) SHOWMENU_NB_CANCELED_VALUE;
  }

  void setFilenameAliases(const std::vector<std::pair<int, std::string>>& aliases)
  {
    filenameAliases.clear();
    for (const auto& item : aliases)
    {
      if (item.first < 0 || item.first > 127)
        continue;
      filenameAliases[item.first] = trimJsfxPath(juce::String::fromUTF8(item.second.c_str()));
    }
  }

  void setRetinaScale(double scale)
  {
    if (gfx_ext_retina)
      *gfx_ext_retina = scale > 0.0 ? (EEL_F) scale : 1.0;
  }

  void setExtFlags(int flags)
  {
    if (gfx_ext_flags)
      *gfx_ext_flags = (EEL_F) flags;
  }

  void setWindowInfoFlags(int flags)
  {
    windowInfoFlags = flags;
  }

  int getRequestedCursorType() const noexcept
  {
    return requestedCursorType;
  }

  juce::Image copyFramebuffer()
  {
    flushPendingCommands();
    if (mainFramebuffer.isNull())
      return {};
    return mainFramebuffer;
  }

  void beginFrame(int w, int h)
  {
    sliderChangeMask = 0;
    sliderAutomateMask = 0;
    sliderAutomateEndMask = 0;
    undoPointRequested = false;

    frameW = juce::jmax(1, w);
    frameH = juce::jmax(1, h);
    framebufferDirty = false;
    commands.clear();

    if (mainFramebuffer.isNull() || mainFramebuffer.getWidth() != frameW || mainFramebuffer.getHeight() != frameH)
      mainFramebuffer = juce::Image(juce::Image::ARGB, frameW, frameH, true);

    if (gfx_w) *gfx_w = (EEL_F) frameW;
    if (gfx_h) *gfx_h = (EEL_F) frameH;
    if (gfx_frame) *gfx_frame = frameCounter++;

    refreshShowMenuNbConstants();
  }

  void setMouse(float x, float y, int cap, float wheel, float hwheel)
  {
    if (mouse_x) *mouse_x = (EEL_F) x;
    if (mouse_y) *mouse_y = (EEL_F) y;
    if (mouse_cap) *mouse_cap = (EEL_F) cap;
    if (mouse_wheel) *mouse_wheel = (EEL_F) wheel;
    if (mouse_hwheel) *mouse_hwheel = (EEL_F) hwheel;
  }

  void pushKey(int code)
  {
    if (code != 0)
      keyQueue.push_back(code);
  }

  void setKeyDown(int code, bool isDown)
  {
    if (code == 0)
      return;
    if (isDown) keysDown.insert(code);
    else keysDown.erase(code);
  }

  void clearKeys()
  {
    keyQueue.clear();
    keysDown.clear();
  }

  uint64_t popSliderChangeMask()       { const auto m = sliderChangeMask;      sliderChangeMask = 0; return m; }
  uint64_t popSliderAutomateMask()     { const auto m = sliderAutomateMask;    sliderAutomateMask = 0; return m; }
  uint64_t popSliderAutomateEndMask()  { const auto m = sliderAutomateEndMask; sliderAutomateEndMask = 0; return m; }
  bool popUndoPointRequested()         { const bool b = undoPointRequested;    undoPointRequested = false; return b; }

  void setTiming(double srate, double samplesblock)
  {
    if (srate_var) *srate_var = (EEL_F) srate;
    if (samplesblock_var) *samplesblock_var = (EEL_F) samplesblock;
  }

  const std::vector<DrawCmd>& getCommands() const
  {
    return commands;
  }

  void setMenuPort(AsyncMenuPort* port) { asyncMenuPort = port; }

  std::array<EEL_F*, 64> sliderPtrs {{}};
  void bindSliderPtrs()
  {
    for (int i = 0; i < 64; ++i)
    {
      const std::string nm = std::string("slider") + std::to_string(i + 1);
      sliderPtrs[(size_t) i] = get_var(nm.c_str());
    }
  }

  struct BoundVar { const char* name; int index; EEL_F* ptr; uint8_t flags; };
  std::vector<BoundVar> boundVars;
  void bindUserVars(const DSPJSFX_VarDesc* vars, const uint8_t* flags, int flagsCount, int count)
  {
    boundVars.clear();
    boundVars.reserve((size_t) count);
    for (int i = 0; i < count; ++i)
    {
      const char* name = vars[i].name;
      const int idx = vars[i].index;
      if (!name) continue;
      const uint8_t dirFlags = (flags != nullptr && idx >= 0 && idx < flagsCount)
                                 ? flags[idx]
                                 : (uint8_t) (DSPJSFX_GFX_VAR_FLAG_TO_GFX | DSPJSFX_GFX_VAR_FLAG_FROM_GFX);
      boundVars.push_back({ name, idx, get_var(name), dirFlags });
    }
  }

  void syncSliders(const double* sliders, int count)
  {
    const int n = std::min(count, 64);
    for (int i = 0; i < n; ++i)
      if (sliderPtrs[(size_t) i]) *sliderPtrs[(size_t) i] = sliders[i];
  }

  void readSliders(double* dst, int count) const
  {
    if (!dst) return;
    const int n = std::min(count, 64);
    for (int i = 0; i < n; ++i)
      dst[i] = sliderPtrs[(size_t) i] ? (double) *sliderPtrs[(size_t) i] : 0.0;
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

  void syncMem(const double* mem, int memN)
  {
    if (!mem || memN <= 0) return;

    if (memN != memSize)
      NSEEL_VM_setramsize(m_vm, (unsigned int) memN);

    syncMemRange(mem, 0, memN);
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

  void readMem(double* dst, int count) const
  {
    readMemRange(0, dst, count);
  }


  bool canDeferSimpleMainSurfaceDraw() const noexcept
  {
    if (currentDestinationSlot() != -1)
      return false;

    if (currentMode() != 0)
      return false;

    if (gfx_a2 != nullptr && std::abs(*gfx_a2 - 1.0) > 1.0e-9)
      return false;

    return true;
  }

  void noteDeferredDraw()
  {
    if (framebufferDirty)
      return;

    framebufferDirty = true;

    if (gfx_clear && *gfx_clear > -1.0)
    {
      const int rgb = (int) std::llround(*gfx_clear);

      DrawCmd cmd;
      cmd.type = DrawCmd::Type::Rect;
      cmd.colour = juce::Colour::fromRGB((juce::uint8) (rgb & 0xff),
                                         (juce::uint8) ((rgb >> 8) & 0xff),
                                         (juce::uint8) ((rgb >> 16) & 0xff));
      cmd.x = 0.0f;
      cmd.y = 0.0f;
      cmd.w = (float) frameW;
      cmd.h = (float) frameH;
      cmd.fill = true;
      commands.push_back(std::move(cmd));
    }
  }

  void flushPendingCommands()
  {
    if (commands.empty())
      return;

    if (mainFramebuffer.isNull() || mainFramebuffer.getWidth() != frameW || mainFramebuffer.getHeight() != frameH)
      mainFramebuffer = juce::Image(juce::Image::ARGB, juce::jmax(1, frameW), juce::jmax(1, frameH), true);

    juce::Graphics g(mainFramebuffer);
    paintCommands(g, commands);
    commands.clear();
  }

  juce::Colour getCurrentColour() const
  {
    return juce::Colour::fromFloatRGBA(gfx_r ? (float) *gfx_r : 1.0f,
                                       gfx_g ? (float) *gfx_g : 1.0f,
                                       gfx_b ? (float) *gfx_b : 1.0f,
                                       gfx_a ? (float) *gfx_a : 1.0f);
  }

private:
  struct TextMetrics
  {
    float width = 0.0f;
    float height = 0.0f;
    float lastLineWidth = 0.0f;
    int lineCount = 1;
  };

  juce::File resolvePath(const juce::String& path) const
  {
    const juce::String cleaned = trimJsfxPath(path);
    if (cleaned.isEmpty())
      return {};

    juce::File direct = juce::File::isAbsolutePath(cleaned)
                         ? juce::File(cleaned)
                         : juce::File::getCurrentWorkingDirectory().getChildFile(cleaned);
    if (direct.existsAsFile())
      return direct;

    for (const auto& root : resourceSearchRoots)
    {
      if (root == juce::File())
        continue;

      juce::File candidate = root.getChildFile(cleaned);
      if (candidate.existsAsFile())
        return candidate;

      if (root.isDirectory())
      {
        candidate = root.getChildFile("resource").getChildFile(cleaned);
        if (candidate.existsAsFile())
          return candidate;
      }
    }

    return direct;
  }

  bool ensureAliasLoaded(int slot)
  {
    if (slot < 0 || slot > 127)
      return false;

    auto it = images.find(slot);
    if (it != images.end() && !it->second.isNull())
      return true;

    const auto aliasIt = filenameAliases.find(slot);
    if (aliasIt == filenameAliases.end())
      return false;

    const juce::File file = resolvePath(aliasIt->second);
    if (!file.existsAsFile())
      return false;

    juce::Image img = juce::ImageFileFormat::loadFrom(file);
    if (img.isNull())
      return false;

    images[slot] = img.convertedToFormat(juce::Image::ARGB);
    imageSourceFiles[slot] = file;
    return true;
  }

  juce::Image* getSourceImage(int slot, juce::Image* tempSnapshotIfSameAsDest = nullptr)
  {
    if (slot == -1)
    {
      flushPendingCommands();
      return tempSnapshotIfSameAsDest != nullptr ? tempSnapshotIfSameAsDest : &mainFramebuffer;
    }

    if (slot < 0 || slot > 127)
      return nullptr;

    ensureAliasLoaded(slot);
    auto it = images.find(slot);
    if (it == images.end() || it->second.isNull())
      return nullptr;
    return &it->second;
  }

  juce::Image& writableImageForSlot(int slot, int minWidth = 1, int minHeight = 1)
  {
    if (slot < 0)
    {
      ensureMainFramebufferForWrite();
      return mainFramebuffer;
    }

    ensureAliasLoaded(slot);
    auto& img = images[slot];
    if (img.isNull())
    {
      const int w = juce::jlimit(1, 2048, juce::jmax(frameW, minWidth));
      const int h = juce::jlimit(1, 2048, juce::jmax(frameH, minHeight));
      img = juce::Image(juce::Image::ARGB, w, h, true);
    }
    return img;
  }

  juce::Image& writableTargetImage(int minWidth = 1, int minHeight = 1)
  {
    const int slot = currentDestinationSlot();
    return writableImageForSlot(slot, minWidth, minHeight);
  }

  juce::Rectangle<int> clipRectToImage(const juce::Rectangle<int>& rect, const juce::Image& image) const
  {
    return rect.getIntersection(juce::Rectangle<int>(0, 0, image.getWidth(), image.getHeight()));
  }

  void ensureMainFramebufferForWrite()
  {
    if (mainFramebuffer.isNull())
      mainFramebuffer = juce::Image(juce::Image::ARGB, juce::jmax(1, frameW), juce::jmax(1, frameH), true);

    flushPendingCommands();

    if (!framebufferDirty)
    {
      framebufferDirty = true;
      if (gfx_clear && *gfx_clear > -1.0)
      {
        const int rgb = (int) std::llround(*gfx_clear);
        juce::Graphics g(mainFramebuffer);
        g.fillAll(juce::Colour::fromRGB((juce::uint8) (rgb & 0xff),
                                        (juce::uint8) ((rgb >> 8) & 0xff),
                                        (juce::uint8) ((rgb >> 16) & 0xff)));
      }
    }
  }

  int currentDestinationSlot() const
  {
    if (gfx_dest == nullptr)
      return -1;
    const int slot = (int) std::llround(*gfx_dest);
    return (slot >= 0 && slot <= 127) ? slot : -1;
  }

  float currentAlphaWrite() const
  {
    return gfx_a2 ? clampUnitFloat(*gfx_a2) : 1.0f;
  }

  int currentMode() const
  {
    return gfx_mode ? (int) std::llround(*gfx_mode) : 0;
  }

  void updateTextMetrics()
  {
    if (gfx_texth)
      *gfx_texth = (EEL_F) currentFont.getHeight();
  }

  static juce::String decodeJsfxCharFlags(double value)
  {
    const uint32_t packed = (uint32_t) std::llround(value);
    char buf[8] = {};
    int n = 0;
    for (int i = 0; i < 4; ++i)
    {
      const char c = (char) ((packed >> (8 * i)) & 0xffu);
      if (c == 0)
        break;
      buf[n++] = (char) std::tolower((unsigned char) c);
    }
    return juce::String::fromUTF8(buf, n);
  }

  TextMetrics measureTextInternal(const juce::String& text) const
  {
    TextMetrics m;
    juce::StringArray lines;
    lines.addLines(text);
    if (lines.isEmpty())
      lines.add (juce::String());

    m.lineCount = juce::jmax(1, lines.size());
    m.height = currentFont.getHeight() * (float) m.lineCount;
    m.width = 0.0f;
    m.lastLineWidth = 0.0f;

    for (int i = 0; i < lines.size(); ++i)
    {
      const float w = currentFont.getStringWidthFloat(lines[i]);
      m.width = juce::jmax(m.width, w);
      if (i == lines.size() - 1)
        m.lastLineWidth = w;
    }

    return m;
  }

  void advancePenAfterText(const juce::String& text)
  {
    if (!gfx_x || !gfx_y)
      return;

    const auto metrics = measureTextInternal(text);
    if (metrics.lineCount <= 1)
    {
      *gfx_x += metrics.lastLineWidth;
      return;
    }

    const double startX = *gfx_x;
    *gfx_x = startX + metrics.lastLineWidth;
    *gfx_y += currentFont.getHeight() * (double) (metrics.lineCount - 1);
  }

  void drawTextInternal(const juce::String& text, int flags = 0, double right = 0.0, double bottom = 0.0, bool haveBounds = false)
  {
    const float x = gfx_x ? (float) *gfx_x : 0.0f;
    const float y = gfx_y ? (float) *gfx_y : 0.0f;

    if (!haveBounds && flags == 0 && canDeferSimpleMainSurfaceDraw())
    {
      noteDeferredDraw();

      DrawCmd cmd;
      cmd.type = DrawCmd::Type::Text;
      cmd.colour = getCurrentColour();
      cmd.font = currentFont;
      cmd.text = text;
      cmd.x = x;
      cmd.y = y;
      commands.push_back(std::move(cmd));

      advancePenAfterText(text);
      return;
    }

    juce::Image& target = writableTargetImage();
    juce::Graphics g(target);
    g.setColour(getCurrentColour());
    g.setFont(currentFont);

    const auto metrics = measureTextInternal(text);

    const bool ignoreRightBottom = (flags & 256) != 0;
    const float areaW = haveBounds ? (float) std::max(0.0, right - (double) x) : metrics.width;
    const float areaH = haveBounds ? (float) std::max(0.0, bottom - (double) y) : metrics.height;

    const juce::Rectangle<int> clipRect((int) std::floor(x), (int) std::floor(y),
                                        (int) std::ceil(haveBounds ? areaW : metrics.width + 2.0f),
                                        (int) std::ceil(haveBounds ? areaH : metrics.height + 2.0f));

    if (haveBounds && !ignoreRightBottom)
      g.reduceClipRegion(clipRectToImage(clipRect, target));

    juce::StringArray lines;
    lines.addLines(text);
    if (lines.isEmpty())
      lines.add (juce::String());

    const float lineH = currentFont.getHeight();
    const float totalH = lineH * (float) juce::jmax(1, lines.size());
    const float usedW = haveBounds ? areaW : metrics.width;
    const float usedH = haveBounds ? areaH : metrics.height;

    float blockY = y;
    if (flags & 8)       blockY = y + (usedH - totalH);
    else if (flags & 4)  blockY = y + (usedH - totalH) * 0.5f;

    for (int i = 0; i < lines.size(); ++i)
    {
      const auto& line = lines[i];
      const float lineW = currentFont.getStringWidthFloat(line);
      float lineX = x;
      if (flags & 2)      lineX = x + (usedW - lineW);
      else if (flags & 1) lineX = x + (usedW - lineW) * 0.5f;
      g.drawSingleLineText(line, roundToNearestInt(lineX), roundToNearestInt(blockY + lineH * (float) i + currentFont.getAscent()));
    }

    advancePenAfterText(text);
  }

  void drawPathFilled(const juce::Path& path, bool fill)
  {
    juce::Image& target = writableTargetImage();
    juce::Graphics g(target);
    g.setColour(getCurrentColour());
    if (fill) g.fillPath(path);
    else      g.strokePath(path, juce::PathStrokeType(1.0f));
  }

  void drawPrimitiveRect(float x, float y, float w, float h, bool fill)
  {
    if (canDeferSimpleMainSurfaceDraw())
    {
      noteDeferredDraw();

      DrawCmd cmd;
      cmd.type = DrawCmd::Type::Rect;
      cmd.colour = getCurrentColour();
      cmd.x = x;
      cmd.y = y;
      cmd.w = w;
      cmd.h = h;
      cmd.fill = fill;
      commands.push_back(std::move(cmd));
      return;
    }

    juce::Image& target = writableTargetImage(juce::jmax(1, roundToNearestInt(x + w)), juce::jmax(1, roundToNearestInt(y + h)));
    juce::Graphics g(target);
    g.setColour(getCurrentColour());
    if (fill) g.fillRect(x, y, w, h);
    else      g.drawRect(x, y, w, h, 1.0f);
  }

  void drawPrimitiveLine(float x1, float y1, float x2, float y2)
  {
    if (canDeferSimpleMainSurfaceDraw())
    {
      noteDeferredDraw();

      DrawCmd cmd;
      cmd.type = DrawCmd::Type::Line;
      cmd.colour = getCurrentColour();
      cmd.x = x1;
      cmd.y = y1;
      cmd.x2 = x2;
      cmd.y2 = y2;
      commands.push_back(std::move(cmd));
      return;
    }

    juce::Image& target = writableTargetImage();
    juce::Graphics g(target);
    g.setColour(getCurrentColour());
    g.drawLine(x1, y1, x2, y2, 1.0f);
  }

  void performBlit(int sourceSlot, double rotation,
                   double srcx, double srcy, double srcw, double srch,
                   double destx, double desty, double destw, double desth,
                   double rotxoffs, double rotyoffs)
  {
    juce::Image& destImg = writableTargetImage(juce::jmax(1, roundToNearestInt(destx + destw)), juce::jmax(1, roundToNearestInt(desty + desth)));

    juce::Image sourceSnapshot;
    juce::Image* srcImg = nullptr;

    if ((sourceSlot == -1 && currentDestinationSlot() == -1) || sourceSlot == currentDestinationSlot())
    {
      juce::Image* original = getSourceImage(sourceSlot);
      if (original == nullptr || original->isNull())
        return;
      sourceSnapshot = original->createCopy();
      srcImg = &sourceSnapshot;
    }
    else
    {
      srcImg = getSourceImage(sourceSlot);
    }

    if (srcImg == nullptr || srcImg->isNull())
      return;

    const juce::Rectangle<int> destBounds = clipRectToImage(
        juce::Rectangle<int>((int) std::floor(destx), (int) std::floor(desty),
                             juce::jmax(1, (int) std::ceil(destw)), juce::jmax(1, (int) std::ceil(desth))),
        destImg);

    if (destBounds.isEmpty())
      return;

    const bool nearest = (currentMode() & 4) != 0;
    const juce::Point<double> pivot(destx + destw * 0.5 + rotxoffs,
                                    desty + desth * 0.5 + rotyoffs);
    const double c = std::cos(rotation);
    const double s = std::sin(rotation);

    for (int y = destBounds.getY(); y < destBounds.getBottom(); ++y)
    {
      for (int x = destBounds.getX(); x < destBounds.getRight(); ++x)
      {
        const double px = (double) x + 0.5;
        const double py = (double) y + 0.5;

        const double relx = px - pivot.x;
        const double rely = py - pivot.y;
        const double qx = pivot.x + (relx * c + rely * s);
        const double qy = pivot.y + (-relx * s + rely * c);

        const double u = srcx + ((qx - destx) / destw) * srcw;
        const double v = srcy + ((qy - desty) / desth) * srch;

        if (u < srcx || v < srcy || u >= srcx + srcw || v >= srcy + srch)
          continue;

        const PixelF src = sampleImage(*srcImg, u, v, nearest);
        const PixelF dst = readPixelSafe(destImg, x, y);
        const PixelF out = compositePixel(dst, src, gfx_a ? *gfx_a : 1.0, currentAlphaWrite(), currentMode());
        destImg.setPixelAt(x, y, colourFromPixel(out));
      }
    }
  }

  void performDeltaBlit(int sourceSlot,
                        double srcx, double srcy, double srcw, double srch,
                        double destx, double desty, double destw, double desth,
                        double dsdx, double dtdx, double dsdy, double dtdy,
                        double dsdxdy, double dtdxdy)
  {
    juce::Image& destImg = writableTargetImage(juce::jmax(1, roundToNearestInt(destx + destw)), juce::jmax(1, roundToNearestInt(desty + desth)));
    juce::Image sourceSnapshot;
    juce::Image* srcImg = nullptr;

    if ((sourceSlot == -1 && currentDestinationSlot() == -1) || sourceSlot == currentDestinationSlot())
    {
      juce::Image* original = getSourceImage(sourceSlot);
      if (original == nullptr || original->isNull())
        return;
      sourceSnapshot = original->createCopy();
      srcImg = &sourceSnapshot;
    }
    else
    {
      srcImg = getSourceImage(sourceSlot);
    }

    if (srcImg == nullptr || srcImg->isNull())
      return;

    const juce::Rectangle<int> destBounds = clipRectToImage(
        juce::Rectangle<int>((int) std::floor(destx), (int) std::floor(desty),
                             juce::jmax(1, (int) std::ceil(destw)), juce::jmax(1, (int) std::ceil(desth))),
        destImg);

    const bool nearest = (currentMode() & 4) != 0;

    for (int y = destBounds.getY(); y < destBounds.getBottom(); ++y)
    {
      const double dy = (double) y - desty;
      const double lineDsdx = dsdx + dsdxdy * dy;
      const double lineDtdx = dtdx + dtdxdy * dy;
      const double baseS = srcx + dsdy * dy;
      const double baseT = srcy + dtdy * dy;

      for (int x = destBounds.getX(); x < destBounds.getRight(); ++x)
      {
        const double dx = (double) x - destx;
        const double u = baseS + lineDsdx * dx;
        const double v = baseT + lineDtdx * dx;

        if (u < srcx || v < srcy || u >= srcx + srcw || v >= srcy + srch)
          continue;

        const PixelF src = sampleImage(*srcImg, u, v, nearest);
        const PixelF dst = readPixelSafe(destImg, x, y);
        destImg.setPixelAt(x, y, colourFromPixel(compositePixel(dst, src, gfx_a ? *gfx_a : 1.0, currentAlphaWrite(), currentMode())));
      }
    }
  }

  void performTransformBlit(int sourceSlot, double destx, double desty, double destw, double desth,
                            int divW, int divH, int64_t tableBase)
  {
    if (divW < 2 || divH < 2)
      return;

    juce::Image& destImg = writableTargetImage(juce::jmax(1, roundToNearestInt(destx + destw)), juce::jmax(1, roundToNearestInt(desty + desth)));
    juce::Image sourceSnapshot;
    juce::Image* srcImg = nullptr;

    if ((sourceSlot == -1 && currentDestinationSlot() == -1) || sourceSlot == currentDestinationSlot())
    {
      juce::Image* original = getSourceImage(sourceSlot);
      if (original == nullptr || original->isNull())
        return;
      sourceSnapshot = original->createCopy();
      srcImg = &sourceSnapshot;
    }
    else
    {
      srcImg = getSourceImage(sourceSlot);
    }

    if (srcImg == nullptr || srcImg->isNull())
      return;

    const juce::Rectangle<int> destBounds = clipRectToImage(
        juce::Rectangle<int>((int) std::floor(destx), (int) std::floor(desty),
                             juce::jmax(1, (int) std::ceil(destw)), juce::jmax(1, (int) std::ceil(desth))),
        destImg);

    const bool nearest = (currentMode() & 4) != 0;
    const int cellsX = divW - 1;
    const int cellsY = divH - 1;

    for (int y = destBounds.getY(); y < destBounds.getBottom(); ++y)
    {
      const double ny = desth > 1.0 ? ((double) y - desty) / juce::jmax(1.0, desth - 1.0) : 0.0;
      const double gy = juce::jlimit(0.0, (double) cellsY, ny * (double) cellsY);
      const int iy = juce::jlimit(0, cellsY - 1, (int) std::floor(gy));
      const double fy = juce::jlimit(0.0, 1.0, gy - (double) iy);

      for (int x = destBounds.getX(); x < destBounds.getRight(); ++x)
      {
        const double nx = destw > 1.0 ? ((double) x - destx) / juce::jmax(1.0, destw - 1.0) : 0.0;
        const double gx = juce::jlimit(0.0, (double) cellsX, nx * (double) cellsX);
        const int ix = juce::jlimit(0, cellsX - 1, (int) std::floor(gx));
        const double fx = juce::jlimit(0.0, 1.0, gx - (double) ix);

        const int64_t idx00 = tableBase + 2 * (int64_t) (iy * divW + ix);
        const int64_t idx10 = tableBase + 2 * (int64_t) (iy * divW + (ix + 1));
        const int64_t idx01 = tableBase + 2 * (int64_t) ((iy + 1) * divW + ix);
        const int64_t idx11 = tableBase + 2 * (int64_t) ((iy + 1) * divW + (ix + 1));

        const juce::Point<double> st00(readVmRamScalar(m_vm, idx00), readVmRamScalar(m_vm, idx00 + 1));
        const juce::Point<double> st10(readVmRamScalar(m_vm, idx10), readVmRamScalar(m_vm, idx10 + 1));
        const juce::Point<double> st01(readVmRamScalar(m_vm, idx01), readVmRamScalar(m_vm, idx01 + 1));
        const juce::Point<double> st11(readVmRamScalar(m_vm, idx11), readVmRamScalar(m_vm, idx11 + 1));

        const juce::Point<double> top(st00.x + (st10.x - st00.x) * fx, st00.y + (st10.y - st00.y) * fx);
        const juce::Point<double> bottom(st01.x + (st11.x - st01.x) * fx, st01.y + (st11.y - st01.y) * fx);
        const double u = top.x + (bottom.x - top.x) * fy;
        const double v = top.y + (bottom.y - top.y) * fy;

        const PixelF src = sampleImage(*srcImg, u, v, nearest);
        const PixelF dst = readPixelSafe(destImg, x, y);
        destImg.setPixelAt(x, y, colourFromPixel(compositePixel(dst, src, gfx_a ? *gfx_a : 1.0, currentAlphaWrite(), currentMode())));
      }
    }
  }

  void blurRegionTo(double x2, double y2)
  {
    juce::Image& target = writableTargetImage();
    const int x1 = roundToNearestInt(gfx_x ? *gfx_x : 0.0);
    const int y1 = roundToNearestInt(gfx_y ? *gfx_y : 0.0);
    const juce::Rectangle<int> rect = clipRectToImage(
        juce::Rectangle<int>::leftTopRightBottom(std::min(x1, roundToNearestInt(x2)),
                                                 std::min(y1, roundToNearestInt(y2)),
                                                 std::max(x1, roundToNearestInt(x2)) + 1,
                                                 std::max(y1, roundToNearestInt(y2)) + 1),
        target);

    if (!rect.isEmpty())
    {
      juce::Image src = target.createCopy();
      for (int y = rect.getY(); y < rect.getBottom(); ++y)
      {
        for (int x = rect.getX(); x < rect.getRight(); ++x)
        {
          PixelF sum {};
          int n = 0;
          for (int oy = -1; oy <= 1; ++oy)
            for (int ox = -1; ox <= 1; ++ox)
            {
              const PixelF p = readPixelSafe(src, x + ox, y + oy);
              sum.r += p.r; sum.g += p.g; sum.b += p.b; sum.a += p.a; ++n;
            }
          if (n > 0)
          {
            sum.r /= (float) n;
            sum.g /= (float) n;
            sum.b /= (float) n;
            sum.a /= (float) n;
            target.setPixelAt(x, y, colourFromPixel(sum));
          }
        }
      }
    }

    if (gfx_x) *gfx_x = x2;
    if (gfx_y) *gfx_y = y2;
  }

public:
  static void registerGfxBuiltins()
  {
    NSEEL_addfunc_varparm_ex("gfx_set",         1, 0, NSEEL_PProc_THIS, &eel_gfx_set,         nullptr);
    NSEEL_addfunc_varparm_ex("gfx_rect",        4, 0, NSEEL_PProc_THIS, &eel_gfx_rect,        nullptr);
    NSEEL_addfunc_varparm_ex("gfx_rectto",      2, 0, NSEEL_PProc_THIS, &eel_gfx_rectto,      nullptr);
    NSEEL_addfunc_varparm_ex("gfx_setpixel",    3, 0, NSEEL_PProc_THIS, &eel_gfx_setpixel,    nullptr);
    NSEEL_addfunc_varparm_ex("gfx_getpixel",    1, 0, NSEEL_PProc_THIS, &eel_gfx_getpixel,    nullptr);
    NSEEL_addfunc_varparm_ex("gfx_drawnumber",  2, 0, NSEEL_PProc_THIS, &eel_gfx_drawnumber,  nullptr);
    NSEEL_addfunc_varparm_ex("gfx_drawchar",    1, 0, NSEEL_PProc_THIS, &eel_gfx_drawchar,    nullptr);
    NSEEL_addfunc_varparm_ex("gfx_drawstr",     1, 0, NSEEL_PProc_THIS, &eel_gfx_drawstr,     nullptr);
    NSEEL_addfunc_varparm_ex("gfx_measurestr",  1, 0, NSEEL_PProc_THIS, &eel_gfx_measurestr,  nullptr);
    NSEEL_addfunc_varparm_ex("gfx_setfont",     1, 0, NSEEL_PProc_THIS, &eel_gfx_setfont,     nullptr);
    NSEEL_addfunc_varparm_ex("gfx_getfont",     0, 0, NSEEL_PProc_THIS, &eel_gfx_getfont,     nullptr);
    NSEEL_addfunc_varparm_ex("gfx_printf",      1, 0, NSEEL_PProc_THIS, &eel_gfx_printf,      nullptr);
    NSEEL_addfunc_varparm_ex("gfx_blurto",      2, 0, NSEEL_PProc_THIS, &eel_gfx_blurto,      nullptr);
    NSEEL_addfunc_varparm_ex("gfx_blit",        3, 0, NSEEL_PProc_THIS, &eel_gfx_blit,        nullptr);
    NSEEL_addfunc_varparm_ex("gfx_blitext",     3, 0, NSEEL_PProc_THIS, &eel_gfx_blitext,     nullptr);
    NSEEL_addfunc_varparm_ex("gfx_getimgdim",   1, 0, NSEEL_PProc_THIS, &eel_gfx_getimgdim,   nullptr);
    NSEEL_addfunc_varparm_ex("gfx_setimgdim",   3, 0, NSEEL_PProc_THIS, &eel_gfx_setimgdim,   nullptr);
    NSEEL_addfunc_varparm_ex("gfx_loadimg",     2, 0, NSEEL_PProc_THIS, &eel_gfx_loadimg,     nullptr);
    NSEEL_addfunc_varparm_ex("gfx_gradrect",    8, 0, NSEEL_PProc_THIS, &eel_gfx_gradrect,    nullptr);
    NSEEL_addfunc_varparm_ex("gfx_muladdrect",  7, 0, NSEEL_PProc_THIS, &eel_gfx_muladdrect,  nullptr);
    NSEEL_addfunc_varparm_ex("gfx_deltablit",  15, 0, NSEEL_PProc_THIS, &eel_gfx_deltablit,   nullptr);
    NSEEL_addfunc_varparm_ex("gfx_transformblit", 8, 0, NSEEL_PProc_THIS, &eel_gfx_transformblit, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_circle",      3, 0, NSEEL_PProc_THIS, &eel_gfx_circle,      nullptr);
    NSEEL_addfunc_varparm_ex("gfx_roundrect",   5, 0, NSEEL_PProc_THIS, &eel_gfx_roundrect,   nullptr);
    NSEEL_addfunc_varparm_ex("gfx_arc",         5, 0, NSEEL_PProc_THIS, &eel_gfx_arc,         nullptr);
    NSEEL_addfunc_varparm_ex("gfx_triangle",    6, 0, NSEEL_PProc_THIS, &eel_gfx_triangle,    nullptr);
    NSEEL_addfunc_varparm_ex("gfx_line",        4, 0, NSEEL_PProc_THIS, &eel_gfx_line,        nullptr);
    NSEEL_addfunc_varparm_ex("gfx_lineto",      2, 0, NSEEL_PProc_THIS, &eel_gfx_lineto,      nullptr);
    NSEEL_addfunc_varparm_ex("gfx_getchar",     0, 0, NSEEL_PProc_THIS, &eel_gfx_getchar,     nullptr);
    NSEEL_addfunc_varparm_ex("gfx_showmenu",    1, 0, NSEEL_PProc_THIS, &eel_gfx_showmenu,    nullptr);
    NSEEL_addfunc_varparm_ex("gfx_showmenu_nb_open",   1, 0, NSEEL_PProc_THIS, &eel_gfx_showmenu_nb_open,   nullptr);
    NSEEL_addfunc_varparm_ex("gfx_showmenu_nb_poll",   0, 0, NSEEL_PProc_THIS, &eel_gfx_showmenu_nb_poll,   nullptr);
    NSEEL_addfunc_varparm_ex("gfx_showmenu_nb_cancel", 0, 0, NSEEL_PProc_THIS, &eel_gfx_showmenu_nb_cancel, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_setcursor",   1, 0, NSEEL_PProc_THIS, &eel_gfx_setcursor,   nullptr);

    NSEEL_addfunc_varparm_ex("sliderchange",    1, 0, NSEEL_PProc_THIS, &eel_sliderchange,    nullptr);
    NSEEL_addfunc_varparm_ex("slider_automate", 1, 0, NSEEL_PProc_THIS, &eel_slider_automate, nullptr);
    NSEEL_addfunc_varparm_ex("slider_show",     1, 0, NSEEL_PProc_THIS, &eel_slider_show,     nullptr);
    NSEEL_addfunc_varparm_ex("slider",          1, 0, NSEEL_PProc_THIS, &eel_slider,          nullptr);
    NSEEL_addfunc_varparm_ex("spl",             1, 0, NSEEL_PProc_THIS, &eel_spl,             nullptr);
    NSEEL_addfunc_varparm_ex("freembuf",        1, 0, NSEEL_PProc_THIS, &eel_freembuf,        nullptr);
  }

  static uint64_t sliderMaskFromArg(GfxVm* self, EEL_F* argPtr, double argValue)
  {
    if (self)
      for (int i = 0; i < 64; ++i)
        if (self->sliderPtrs[(size_t) i] == argPtr)
          return (uint64_t) 1u << (uint64_t) i;

    if (argValue <= 0.0)
      return 0;
    const int64_t m = (int64_t) std::llround(argValue);
    return m > 0 ? (uint64_t) m : 0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_set(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1) return 0.0;

    const double r = (double) *parms[0];
    const double g = (np >= 2) ? (double) *parms[1] : r;
    const double b = (np >= 3) ? (double) *parms[2] : r;
    const double a = (np >= 4) ? (double) *parms[3] : 1.0;
    const double mode = (np >= 5) ? (double) *parms[4] : 0.0;

    if (self->gfx_r) *self->gfx_r = (EEL_F) r;
    if (self->gfx_g) *self->gfx_g = (EEL_F) g;
    if (self->gfx_b) *self->gfx_b = (EEL_F) b;
    if (self->gfx_a) *self->gfx_a = (EEL_F) a;
    if (self->gfx_mode) *self->gfx_mode = (EEL_F) mode;
    if (np >= 6 && self->gfx_dest) *self->gfx_dest = *parms[5];
    if (self->gfx_a2) *self->gfx_a2 = (np >= 7) ? *parms[6] : 1.0;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_rect(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 4) return 0.0;
    const bool fill = (np < 5) || (*parms[4] != 0.0);
    self->drawPrimitiveRect((float) *parms[0], (float) *parms[1], (float) *parms[2], (float) *parms[3], fill);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_rectto(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 2) return 0.0;

    const float x1 = (float) (self->gfx_x ? *self->gfx_x : 0.0);
    const float y1 = (float) (self->gfx_y ? *self->gfx_y : 0.0);
    const float x2 = (float) *parms[0];
    const float y2 = (float) *parms[1];
    self->drawPrimitiveRect(std::min(x1, x2), std::min(y1, y2), std::abs(x2 - x1), std::abs(y2 - y1), true);
    if (self->gfx_x) *self->gfx_x = x2;
    if (self->gfx_y) *self->gfx_y = y2;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_setpixel(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 3) return 0.0;

    juce::Image& target = self->writableTargetImage();
    const int x = roundToNearestInt(self->gfx_x ? *self->gfx_x : 0.0);
    const int y = roundToNearestInt(self->gfx_y ? *self->gfx_y : 0.0);
    if (x < 0 || y < 0 || x >= target.getWidth() || y >= target.getHeight())
      return 0.0;

    PixelF src { clampUnitFloat(*parms[0]), clampUnitFloat(*parms[1]), clampUnitFloat(*parms[2]), self->gfx_a ? clampUnitFloat(*self->gfx_a) : 1.0f };
    PixelF dst = readPixelSafe(target, x, y);
    target.setPixelAt(x, y, colourFromPixel(compositePixel(dst, src, 1.0, self->currentAlphaWrite(), self->currentMode())));
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_getpixel(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1) return 0.0;

    juce::Image& target = self->writableTargetImage();
    const int x = roundToNearestInt(self->gfx_x ? *self->gfx_x : 0.0);
    const int y = roundToNearestInt(self->gfx_y ? *self->gfx_y : 0.0);
    const PixelF px = readPixelSafe(target, x, y);
    if (np >= 1 && parms[0]) *parms[0] = px.r;
    if (np >= 2 && parms[1]) *parms[1] = px.g;
    if (np >= 3 && parms[2]) *parms[2] = px.b;
    return 1.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_drawnumber(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 2) return 0.0;
    const int digits = juce::jlimit(0, 16, roundToNearestInt(*parms[1]));
    self->drawTextInternal(juce::String((double) *parms[0], digits));
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_drawchar(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1) return 0.0;
    const int code = roundToNearestInt(*parms[0]);
    juce::String s;
    if (code > 0)
      s = juce::String::charToString(static_cast<juce::juce_wchar>(code));
    self->drawTextInternal(s);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_setfont(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1) return 0.0;

    const int fontId = roundToNearestInt(*parms[0]);
    if (fontId <= 0)
    {
      self->currentFontId = 0;
      self->currentFont = self->fonts[0];
      self->updateTextMetrics();
      return 0.0;
    }

    juce::String fontName = juce::Font::getDefaultSansSerifFontName();
    float fontSize = self->currentFont.getHeight() > 0.0f ? self->currentFont.getHeight() : 12.0f;
    bool underline = false;
    int styleFlags = juce::Font::plain;

    if (np >= 2)
    {
      EEL_STRING_MUTEXLOCK_SCOPE;
      const char* fn = EEL_STRING_GET_FOR_INDEX(*parms[1], nullptr);
      if (fn != nullptr && *fn != 0)
        fontName = juce::String::fromUTF8(fn);
    }
    if (np >= 3)
      fontSize = (float) juce::jlimit(1.0, 200.0, (double) *parms[2]);
    if (np >= 4)
    {
      const juce::String flags = decodeJsfxCharFlags(*parms[3]);
      if (flags.containsChar('b')) styleFlags |= juce::Font::bold;
      if (flags.containsChar('i')) styleFlags |= juce::Font::italic;
      underline = flags.containsChar('u');
    }

    juce::Font f(fontName, fontSize, styleFlags);
    if (underline)
      f.setUnderline(true);

    self->fonts[fontId] = f;
    self->currentFontId = fontId;
    self->currentFont = f;
    self->updateTextMetrics();
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_getfont(void* opaque, INT_PTR np, EEL_F** parms)
  {
    juce::ignoreUnused(np, parms);
    auto* self = (GfxVm*) opaque;
    return self ? (EEL_F) self->currentFontId : 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_drawstr(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1) return 0.0;

    juce::String text;
    {
      EEL_STRING_MUTEXLOCK_SCOPE;
      const char* str = EEL_STRING_GET_FOR_INDEX(*parms[0], nullptr);
      text = juce::String::fromUTF8(str ? str : "");
    }

    if (np >= 4)
      self->drawTextInternal(text, roundToNearestInt(*parms[1]), (double) *parms[2], (double) *parms[3], true);
    else
      self->drawTextInternal(text);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_printf(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1) return 0.0;

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
        if (fmt[i + 1] == '%')
        {
          out.push_back('%');
          ++i;
          continue;
        }

        const size_t specStart = i;
        size_t j = i + 1;
        while (fmt[j] != '\0' && std::strchr("-+0 #", fmt[j]) != nullptr) ++j;
        while (fmt[j] != '\0' && std::isdigit((unsigned char) fmt[j])) ++j;
        if (fmt[j] == '.')
        {
          ++j;
          while (fmt[j] != '\0' && std::isdigit((unsigned char) fmt[j])) ++j;
        }
        if (fmt[j] == 'h' || fmt[j] == 'l' || fmt[j] == 'L')
        {
          const char first = fmt[j++];
          if ((first == 'h' || first == 'l') && fmt[j] == first)
            ++j;
        }

        const char spec = fmt[j];
        if (spec == '\0')
        {
          out.append(fmt + specStart);
          break;
        }
        ++j;

        const std::string oneFmt(fmt + specStart, fmt + j);
        char buf[512] = {};
        const double v = (argIndex < (int) np) ? (double) *parms[argIndex] : 0.0;

        if (spec == 's')
        {
          const char* s = EEL_STRING_GET_FOR_INDEX(v, nullptr);
          ::snprintf(buf, sizeof(buf), oneFmt.c_str(), s ? s : "");
          ++argIndex;
        }
        else if (spec == 'd' || spec == 'i')
        {
          ::snprintf(buf, sizeof(buf), oneFmt.c_str(), (int) std::llround(v));
          ++argIndex;
        }
        else if (spec == 'u' || spec == 'x' || spec == 'X' || spec == 'o')
        {
          ::snprintf(buf, sizeof(buf), oneFmt.c_str(), (unsigned int) std::llround(v));
          ++argIndex;
        }
        else if (spec == 'c')
        {
          ::snprintf(buf, sizeof(buf), oneFmt.c_str(), (int) std::llround(v));
          ++argIndex;
        }
        else
        {
          ::snprintf(buf, sizeof(buf), oneFmt.c_str(), v);
          ++argIndex;
        }

        out.append(buf);
        i = j - 1;
      }

      textToDraw = juce::String::fromUTF8(out.c_str(), (int) out.size());
    }

    self->drawTextInternal(textToDraw);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_measurestr(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1) return 0.0;

    juce::String text;
    {
      EEL_STRING_MUTEXLOCK_SCOPE;
      const char* str = EEL_STRING_GET_FOR_INDEX(*parms[0], nullptr);
      text = juce::String::fromUTF8(str ? str : "");
    }

    const auto m = self->measureTextInternal(text);
    if (np >= 2 && parms[1]) *parms[1] = (EEL_F) m.width;
    if (np >= 3 && parms[2]) *parms[2] = (EEL_F) m.height;
    return (EEL_F) m.width;
  }

  static bool decodeMenuDescription(void* opaque, EEL_F* menuExpr, juce::String& outDescription)
  {
    if (opaque == nullptr || menuExpr == nullptr)
      return false;
    EEL_STRING_MUTEXLOCK_SCOPE;
    const char* str = EEL_STRING_GET_FOR_INDEX(*menuExpr, nullptr);
    if (str == nullptr || *str == '\0')
      return false;
    outDescription = juce::String::fromUTF8(str);
    return outDescription.isNotEmpty();
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_showmenu(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1 || self->asyncMenuPort == nullptr)
      return 0.0;

    juce::String description;
    if (! decodeMenuDescription(opaque, parms[0], description))
      return 0.0;

    const int x = roundToNearestInt(self->gfx_x ? *self->gfx_x : 0.0);
    const int y = roundToNearestInt(self->gfx_y ? *self->gfx_y : 0.0);
    return (EEL_F) self->asyncMenuPort->showMenuModal(description, x, y);
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_showmenu_nb_open(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1 || self->asyncMenuPort == nullptr)
      return (EEL_F) SHOWMENU_NB_NONE_VALUE;

    juce::String description;
    if (! decodeMenuDescription(opaque, parms[0], description))
      return (EEL_F) SHOWMENU_NB_NONE_VALUE;

    const int x = roundToNearestInt(self->gfx_x ? *self->gfx_x : 0.0);
    const int y = roundToNearestInt(self->gfx_y ? *self->gfx_y : 0.0);
    return (EEL_F) self->asyncMenuPort->showMenuNonBlockingOpen(description, x, y);
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_showmenu_nb_poll(void* opaque, INT_PTR np, EEL_F** parms)
  {
    juce::ignoreUnused(np, parms);
    auto* self = (GfxVm*) opaque;
    if (!self || self->asyncMenuPort == nullptr)
      return (EEL_F) SHOWMENU_NB_NONE_VALUE;
    return (EEL_F) self->asyncMenuPort->showMenuNonBlockingPoll();
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_showmenu_nb_cancel(void* opaque, INT_PTR np, EEL_F** parms)
  {
    juce::ignoreUnused(np, parms);
    auto* self = (GfxVm*) opaque;
    if (!self || self->asyncMenuPort == nullptr)
      return 0.0;
    return (EEL_F) self->asyncMenuPort->showMenuNonBlockingCancel();
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_blurto(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 2) return 0.0;
    self->blurRegionTo(*parms[0], *parms[1]);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_blit(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 3) return 0.0;

    const int source = roundToNearestInt(*parms[0]);
    const double scale = (double) *parms[1];
    const double rotation = (double) *parms[2];

    juce::Image* srcImg = self->getSourceImage(source);
    if (srcImg == nullptr || srcImg->isNull())
      return 0.0;

    double srcx = 0.0;
    double srcy = 0.0;
    double srcw = (double) srcImg->getWidth();
    double srch = (double) srcImg->getHeight();
    double destx = self->gfx_x ? (double) *self->gfx_x : 0.0;
    double desty = self->gfx_y ? (double) *self->gfx_y : 0.0;
    double destw = srcw * scale;
    double desth = srch * scale;
    double rotxoffs = 0.0;
    double rotyoffs = 0.0;

    if (np >= 7)
    {
      srcx = (double) *parms[3];
      srcy = (double) *parms[4];
      srcw = (double) *parms[5];
      srch = (double) *parms[6];
    }
    if (np >= 9)
    {
      destx = (double) *parms[7];
      desty = (double) *parms[8];
    }
    if (np >= 11)
    {
      destw = (double) *parms[9];
      desth = (double) *parms[10];
    }
    if (np >= 13)
    {
      rotxoffs = (double) *parms[11];
      rotyoffs = (double) *parms[12];
    }

    if (srcw <= 0.0 || srch <= 0.0 || destw == 0.0 || desth == 0.0)
      return 0.0;

    self->performBlit(source, rotation, srcx, srcy, srcw, srch, destx, desty, destw, desth, rotxoffs, rotyoffs);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_blitext(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 3) return 0.0;

    const int source = roundToNearestInt(*parms[0]);
    const int64_t base = jsfxTruncIndexLikeAot(*parms[1]);
    const double rotation = (double) *parms[2];

    const double srcx = readVmRamScalar(self->m_vm, base + 0);
    const double srcy = readVmRamScalar(self->m_vm, base + 1);
    const double srcw = readVmRamScalar(self->m_vm, base + 2);
    const double srch = readVmRamScalar(self->m_vm, base + 3);
    const double destx = readVmRamScalar(self->m_vm, base + 4);
    const double desty = readVmRamScalar(self->m_vm, base + 5);
    const double destw = readVmRamScalar(self->m_vm, base + 6);
    const double desth = readVmRamScalar(self->m_vm, base + 7);
    const double rotxoffs = readVmRamScalar(self->m_vm, base + 8);
    const double rotyoffs = readVmRamScalar(self->m_vm, base + 9);

    if (srcw <= 0.0 || srch <= 0.0 || destw == 0.0 || desth == 0.0)
      return 0.0;

    self->performBlit(source, rotation, srcx, srcy, srcw, srch, destx, desty, destw, desth, rotxoffs, rotyoffs);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_getimgdim(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1) return 0.0;

    const int slot = roundToNearestInt(*parms[0]);
    self->ensureAliasLoaded(slot);
    auto it = self->images.find(slot);
    const double w = (it != self->images.end() && !it->second.isNull()) ? (double) it->second.getWidth() : 0.0;
    const double h = (it != self->images.end() && !it->second.isNull()) ? (double) it->second.getHeight() : 0.0;
    if (np >= 2 && parms[1]) *parms[1] = (EEL_F) w;
    if (np >= 3 && parms[2]) *parms[2] = (EEL_F) h;
    return (EEL_F) w;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_setimgdim(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 3) return -1.0;
    const int slot = roundToNearestInt(*parms[0]);
    if (slot < 0 || slot > 127)
      return -1.0;

    const int w = juce::jlimit(0, 2048, roundToNearestInt(*parms[1]));
    const int h = juce::jlimit(0, 2048, roundToNearestInt(*parms[2]));
    if (w <= 0 || h <= 0)
    {
      self->images.erase(slot);
      return (EEL_F) slot;
    }

    self->images[slot] = juce::Image(juce::Image::ARGB, w, h, true);
    return (EEL_F) slot;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_loadimg(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 2) return -1.0;

    const int slot = roundToNearestInt(*parms[0]);
    if (slot < 0 || slot > 127)
      return -1.0;

    juce::String pathText;
    {
      EEL_STRING_MUTEXLOCK_SCOPE;
      const char* s = EEL_STRING_GET_FOR_INDEX(*parms[1], nullptr);
      pathText = juce::String::fromUTF8(s ? s : "");
    }

    const juce::File file = self->resolvePath(pathText);
    if (!file.existsAsFile())
      return -1.0;

    juce::Image img = juce::ImageFileFormat::loadFrom(file);
    if (img.isNull())
      return -1.0;

    self->images[slot] = img.convertedToFormat(juce::Image::ARGB);
    self->imageSourceFiles[slot] = file;
    return (EEL_F) slot;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_gradrect(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 8) return 0.0;

    juce::Image& target = self->writableTargetImage();
    const int x0 = roundToNearestInt(*parms[0]);
    const int y0 = roundToNearestInt(*parms[1]);
    const int w = juce::jmax(0, roundToNearestInt(*parms[2]));
    const int h = juce::jmax(0, roundToNearestInt(*parms[3]));
    const juce::Rectangle<int> rect = self->clipRectToImage({x0, y0, w, h}, target);
    if (rect.isEmpty())
      return 0.0;

    const double r = *parms[4], g = *parms[5], b = *parms[6], a = *parms[7];
    const double drdx = (np >= 9)  ? *parms[8]  : 0.0;
    const double dgdx = (np >= 10) ? *parms[9]  : 0.0;
    const double dbdx = (np >= 11) ? *parms[10] : 0.0;
    const double dadx = (np >= 12) ? *parms[11] : 0.0;
    const double drdy = (np >= 13) ? *parms[12] : 0.0;
    const double dgdy = (np >= 14) ? *parms[13] : 0.0;
    const double dbdy = (np >= 15) ? *parms[14] : 0.0;
    const double dady = (np >= 16) ? *parms[15] : 0.0;

    for (int y = rect.getY(); y < rect.getBottom(); ++y)
    {
      for (int x = rect.getX(); x < rect.getRight(); ++x)
      {
        const double fx = (double) (x - x0);
        const double fy = (double) (y - y0);
        PixelF src {
          clampUnitFloat(r + drdx * fx + drdy * fy),
          clampUnitFloat(g + dgdx * fx + dgdy * fy),
          clampUnitFloat(b + dbdx * fx + dbdy * fy),
          clampUnitFloat(a + dadx * fx + dady * fy)
        };
        const PixelF dst = readPixelSafe(target, x, y);
        target.setPixelAt(x, y, colourFromPixel(compositePixel(dst, src, 1.0, self->currentAlphaWrite(), self->currentMode())));
      }
    }
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_muladdrect(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 7) return 0.0;

    juce::Image& target = self->writableTargetImage();
    const juce::Rectangle<int> rect = self->clipRectToImage(
        { roundToNearestInt(*parms[0]), roundToNearestInt(*parms[1]),
          juce::jmax(0, roundToNearestInt(*parms[2])), juce::jmax(0, roundToNearestInt(*parms[3])) }, target);
    if (rect.isEmpty())
      return 0.0;

    const float mulR = clampUnitFloat(*parms[4]);
    const float mulG = clampUnitFloat(*parms[5]);
    const float mulB = clampUnitFloat(*parms[6]);
    const float mulA = (np >= 8) ? clampUnitFloat(*parms[7]) : 1.0f;
    const float addR = (np >= 9) ? (float) *parms[8] : 0.0f;
    const float addG = (np >= 10) ? (float) *parms[9] : 0.0f;
    const float addB = (np >= 11) ? (float) *parms[10] : 0.0f;
    const float addA = (np >= 12) ? (float) *parms[11] : 0.0f;

    for (int y = rect.getY(); y < rect.getBottom(); ++y)
      for (int x = rect.getX(); x < rect.getRight(); ++x)
      {
        PixelF p = readPixelSafe(target, x, y);
        p.r = clampUnitFloat(p.r * mulR + addR);
        p.g = clampUnitFloat(p.g * mulG + addG);
        p.b = clampUnitFloat(p.b * mulB + addB);
        p.a = clampUnitFloat(p.a * mulA + addA);
        target.setPixelAt(x, y, colourFromPixel(p));
      }
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_deltablit(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 15) return 0.0;

    self->performDeltaBlit(roundToNearestInt(*parms[0]),
                           *parms[1], *parms[2], *parms[3], *parms[4],
                           *parms[5], *parms[6], *parms[7], *parms[8],
                           *parms[9], *parms[10], *parms[11], *parms[12],
                           *parms[13], *parms[14]);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_transformblit(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 8) return 0.0;

    self->performTransformBlit(roundToNearestInt(*parms[0]),
                               *parms[1], *parms[2], *parms[3], *parms[4],
                               juce::jlimit(2, 64, roundToNearestInt(*parms[5])),
                               juce::jlimit(2, 64, roundToNearestInt(*parms[6])),
                               jsfxTruncIndexLikeAot(*parms[7]));
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_circle(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 3) return 0.0;

    const float r = (float) *parms[2];
    const bool fill = (np >= 4) ? (*parms[3] != 0.0) : true;

    if (self->canDeferSimpleMainSurfaceDraw())
    {
      self->noteDeferredDraw();

      DrawCmd cmd;
      cmd.type = DrawCmd::Type::Circle;
      cmd.colour = self->getCurrentColour();
      cmd.x = (float) *parms[0];
      cmd.y = (float) *parms[1];
      cmd.radius = r;
      cmd.fill = fill;
      self->commands.push_back(std::move(cmd));
      return 0.0;
    }

    juce::Image& target = self->writableTargetImage();
    juce::Graphics g(target);
    g.setColour(self->getCurrentColour());
    const float d = r * 2.0f;
    const float x = (float) *parms[0] - r;
    const float y = (float) *parms[1] - r;
    if (fill) g.fillEllipse(x, y, d, d); else g.drawEllipse(x, y, d, d, 1.0f);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_roundrect(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 5) return 0.0;

    if (self->canDeferSimpleMainSurfaceDraw())
    {
      self->noteDeferredDraw();

      DrawCmd cmd;
      cmd.type = DrawCmd::Type::RoundRect;
      cmd.colour = self->getCurrentColour();
      cmd.x = (float) *parms[0];
      cmd.y = (float) *parms[1];
      cmd.w = (float) *parms[2];
      cmd.h = (float) *parms[3];
      cmd.cornerRadius = juce::jmax(0.0f, (float) *parms[4]);
      cmd.fill = (np >= 6) ? (*parms[5] != 0.0) : false;
      self->commands.push_back(std::move(cmd));
      return 0.0;
    }

    juce::Image& target = self->writableTargetImage();
    juce::Graphics g(target);
    g.setColour(self->getCurrentColour());
    const juce::Rectangle<float> rect((float) *parms[0], (float) *parms[1], (float) *parms[2], (float) *parms[3]);
    const float radius = juce::jmax(0.0f, (float) *parms[4]);
    const bool fill = (np >= 6) ? (*parms[5] != 0.0) : false;
    if (fill) g.fillRoundedRectangle(rect, radius);
    else      g.drawRoundedRectangle(rect, radius, 1.0f);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_arc(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 5) return 0.0;

    const float cx = (float) *parms[0];
    const float cy = (float) *parms[1];
    const float r = juce::jmax(0.0f, (float) *parms[2]);
    const float a1 = (float) *parms[3];
    const float a2 = (float) *parms[4];

    if (self->canDeferSimpleMainSurfaceDraw())
    {
      self->noteDeferredDraw();

      DrawCmd cmd;
      cmd.type = DrawCmd::Type::Arc;
      cmd.colour = self->getCurrentColour();
      cmd.x = cx;
      cmd.y = cy;
      cmd.radius = r;
      cmd.angle1 = a1;
      cmd.angle2 = a2;
      self->commands.push_back(std::move(cmd));
      return 0.0;
    }

    juce::Path p;
    const int segments = juce::jlimit(8, 512, (int) std::ceil(std::abs(a2 - a1) * std::max(8.0f, r * 0.35f)));
    for (int i = 0; i <= segments; ++i)
    {
      const float t = (float) i / (float) juce::jmax(1, segments);
      const float a = a1 + (a2 - a1) * t;
      const float px = cx + std::cos(a) * r;
      const float py = cy + std::sin(a) * r;
      if (i == 0) p.startNewSubPath(px, py); else p.lineTo(px, py);
    }
    self->drawPathFilled(p, false);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_triangle(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 6) return 0.0;

    if (self->canDeferSimpleMainSurfaceDraw())
    {
      self->noteDeferredDraw();

      DrawCmd cmd;
      cmd.type = DrawCmd::Type::Triangle;
      cmd.colour = self->getCurrentColour();
      for (INT_PTR i = 0; i + 1 < np; i += 2)
        cmd.points.emplace_back((float) *parms[i], (float) *parms[i + 1]);
      self->commands.push_back(std::move(cmd));
      return 0.0;
    }

    juce::Path p;
    p.startNewSubPath((float) *parms[0], (float) *parms[1]);
    for (INT_PTR i = 2; i + 1 < np; i += 2)
      p.lineTo((float) *parms[i], (float) *parms[i + 1]);
    p.closeSubPath();
    self->drawPathFilled(p, true);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_line(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 4) return 0.0;
    self->drawPrimitiveLine((float) *parms[0], (float) *parms[1], (float) *parms[2], (float) *parms[3]);
    if (self->gfx_x) *self->gfx_x = *parms[2];
    if (self->gfx_y) *self->gfx_y = *parms[3];
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_lineto(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 2) return 0.0;
    const float x1 = (float) (self->gfx_x ? *self->gfx_x : 0.0);
    const float y1 = (float) (self->gfx_y ? *self->gfx_y : 0.0);
    const float x2 = (float) *parms[0];
    const float y2 = (float) *parms[1];
    self->drawPrimitiveLine(x1, y1, x2, y2);
    if (self->gfx_x) *self->gfx_x = x2;
    if (self->gfx_y) *self->gfx_y = y2;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_getchar(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self)
      return 0.0;

    if (np >= 1)
    {
      const int code = roundToNearestInt(*parms[0]);
      if (code == 65536)
        return (EEL_F) self->windowInfoFlags;
      if (code != 0)
        return self->keysDown.count(code) ? 1.0 : 0.0;
    }

    if (np >= 2 && parms[1])
      *parms[1] = 0.0;

    if (self->keyQueue.empty())
      return 0.0;

    const int code = self->keyQueue.front();
    self->keyQueue.pop_front();
    return (EEL_F) code;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_setcursor(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1) return 0.0;

    juce::String name;
    if (np >= 2)
    {
      EEL_STRING_MUTEXLOCK_SCOPE;
      const char* s = EEL_STRING_GET_FOR_INDEX(*parms[1], nullptr);
      name = juce::String::fromUTF8(s ? s : "");
    }

    const int resourceId = roundToNearestInt(*parms[0]);
    if (resourceId != 0 || name.isNotEmpty())
      self->requestedCursorType = (int) mapJsfxCursorType(resourceId, name);
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_sliderchange(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1)
      return 0.0;

    const double v = (double) *parms[0];
    const uint64_t mask = sliderMaskFromArg(self, parms[0], v);
    if (mask != 0)
    {
      self->sliderChangeMask |= mask;
      return 0.0;
    }

    if (v < 0.0)
      self->undoPointRequested = true;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_slider_automate(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1)
      return 0.0;

    const uint64_t mask = sliderMaskFromArg(self, parms[0], (double) *parms[0]);
    if (mask == 0)
      return 0.0;

    const bool endTouch = (np >= 2 && *parms[1] != 0.0);
    if (endTouch) self->sliderAutomateEndMask |= mask;
    else          self->sliderAutomateMask |= mask;
    return 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_slider_show(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1)
      return 0.0;

    const uint64_t mask = sliderMaskFromArg(self, parms[0], (double) *parms[0]);
    if (mask == 0)
      return 0.0;

    if (np >= 2)
    {
      const double show = (double) *parms[1];
      if (show == -1.0)      self->sliderVisibleMask ^= mask;
      else if (show <= 0.0)  self->sliderVisibleMask &= ~mask;
      else                   self->sliderVisibleMask |= mask;
    }

    return (EEL_F) (double) (self->sliderVisibleMask & mask);
  }

  static EEL_F NSEEL_CGEN_CALL eel_slider(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1) return 0.0;

    const int idx = (int) jsfxTruncIndexLikeAot((double) *parms[0]);
    if (idx < 1 || idx > 64)
      return (np >= 2) ? *parms[1] : 0.0;

    EEL_F* ptr = self->sliderPtrs[(size_t) (idx - 1)];
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
    juce::ignoreUnused(opaque);
    return (np >= 2) ? *parms[1] : 0.0;
  }

  static EEL_F NSEEL_CGEN_CALL eel_freembuf(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*) opaque;
    if (!self || np < 1) return 0.0;

    int64_t n = jsfxTruncIndexLikeAot((double) *parms[0]);
    if (n < 0) n = 0;
    if (n > 0x7fffffffLL) n = 0x7fffffffLL;

    if (self->freembufIsNoop())
      return 0.0;

    if (self->m_vm)
      NSEEL_VM_setramsize(self->m_vm, (unsigned int) n);
    self->memSize = (int) n;
    return 0.0;
  }

public:
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
  EEL_F* gfx_dest = nullptr;
  EEL_F* gfx_a2 = nullptr;
  EEL_F* gfx_texth = nullptr;
  EEL_F* gfx_ext_retina = nullptr;
  EEL_F* gfx_ext_flags = nullptr;

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

  int memSize = 0;
  AsyncMenuPort* asyncMenuPort = nullptr;

  std::unordered_map<int, juce::Font> fonts;
  int currentFontId = 0;
  juce::Font currentFont;
  juce::Image mainFramebuffer;
  std::vector<DrawCmd> commands;
  std::unordered_map<int, juce::Image> images;
  std::unordered_map<int, juce::String> filenameAliases;
  std::unordered_map<int, juce::File> imageSourceFiles;
  std::vector<juce::File> resourceSearchRoots;

  uint64_t sliderChangeMask = 0;
  uint64_t sliderAutomateMask = 0;
  uint64_t sliderAutomateEndMask = 0;
  uint64_t sliderVisibleMask = ~UINT64_C(0);
  bool undoPointRequested = false;

  std::deque<int> keyQueue;
  std::unordered_set<int> keysDown;
  int windowInfoFlags = 1;
  int requestedCursorType = (int) juce::MouseCursor::NormalCursor;
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
    vm->setFilenameAliases(sections.filenameSlots);

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

  void setRetinaScale(double scale)
  {
    if (vm) vm->setRetinaScale(scale);
  }

  void setWindowInfoFlags(int flags)
  {
    if (vm) vm->setWindowInfoFlags(flags);
  }

  void setExtFlags(int flags)
  {
    if (vm) vm->setExtFlags(flags);
  }

  int getRequestedCursorType() const
  {
    return vm ? vm->getRequestedCursorType() : (int) juce::MouseCursor::NormalCursor;
  }

  juce::Image copyFramebuffer()
  {
    return vm ? vm->copyFramebuffer() : juce::Image();
  }

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

  juce::Image copyCurrentFrame()
  {
    return vm ? vm->copyFramebuffer() : juce::Image();
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
