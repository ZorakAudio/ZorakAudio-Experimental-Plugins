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
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
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
// Draw command list (JUCE playback)
// -------------------------
struct DrawCmd
{
  enum class Type { Rect, Line, Text, Circle };
  Type type = Type::Rect;

  // Common
  juce::Colour colour { 0xff000000 };

  // Rect
  float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;
  bool fill = true;

  // Line
  float x2 = 0.0f, y2 = 0.0f;
    
  // Text
  juce::Font font;
  juce::String text;
  
  // Circle
  float radius = 0.0f;
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
    *gfx_clear = -1.0; // disabled unless script sets it

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
  // State set by host before executing @gfx
  // -------------------------------------------------------------------
  void beginFrame(int w, int h)
  {
    commands.clear();

    *gfx_w = (double)w;
    *gfx_h = (double)h;

    // Apply gfx_clear if set (JSFX convention: 0xRRGGBB)
    if (*gfx_clear >= 0.0)
    {
      const int rgb = (int)(*gfx_clear + 0.5);
      const int r = (rgb >> 16) & 0xff;
      const int g = (rgb >> 8) & 0xff;
      const int b = (rgb) & 0xff;
      DrawCmd cmd;
      cmd.type = DrawCmd::Type::Rect;
      cmd.colour = juce::Colour::fromRGB((juce::uint8)r, (juce::uint8)g, (juce::uint8)b);
      cmd.x = 0.0f;
      cmd.y = 0.0f;
      cmd.w = (float)w;
      cmd.h = (float)h;
      cmd.fill = true;
      commands.push_back(std::move(cmd));
    }
  }

  void setMouse(float x, float y, int cap, float wheel, float hwheel)
  {
    *mouse_x = (double)x;
    *mouse_y = (double)y;
    *mouse_cap = (double)cap;
    *mouse_wheel = (double)wheel;
    *mouse_hwheel = (double)hwheel;
  }

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

  void syncVars(const double* vars, int count)
  {
    for (const auto& bv : boundVars)
    {
      if (bv.index >= 0 && bv.index < count && bv.ptr)
        *bv.ptr = vars[bv.index];
    }
  }

  void syncMem(const double* mem, int memN)
  {
    if (!mem || memN <= 0) return;

    // Set RAM size and copy in chunks.
    NSEEL_VM_setramsize(m_vm, (unsigned int)memN);

    int pos = 0;
    while (pos < memN)
    {
      int validCount = 0;
      EEL_F* dst = NSEEL_VM_getramptr(m_vm, (unsigned int)pos, &validCount);
      if (!dst || validCount <= 0) break;
      const int n = std::min(validCount, memN - pos);
      std::memcpy(dst, mem + pos, (size_t)n * sizeof(EEL_F));
      pos += n;
    }
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
    NSEEL_addfunc_varparm_ex("gfx_line",       4, 0, NSEEL_PProc_THIS, &eel_gfx_line,       nullptr);
    NSEEL_addfunc_varparm_ex("gfx_lineto",     2, 0, NSEEL_PProc_THIS, &eel_gfx_lineto,     nullptr);
    NSEEL_addfunc_varparm_ex("gfx_drawstr",    1, 0, NSEEL_PProc_THIS, &eel_gfx_drawstr,    nullptr);
    NSEEL_addfunc_varparm_ex("gfx_setfont",    1, 0, NSEEL_PProc_THIS, &eel_gfx_setfont,    nullptr);
    NSEEL_addfunc_varparm_ex("gfx_measurestr", 1, 0, NSEEL_PProc_THIS, &eel_gfx_measurestr, nullptr);
    NSEEL_addfunc_varparm_ex("gfx_getchar",    0, 0, NSEEL_PProc_THIS, &eel_gfx_getchar,    nullptr);
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

    EEL_STRING_MUTEXLOCK_SCOPE;
    const char* str = EEL_STRING_GET_FOR_INDEX(*parms[0], nullptr);

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Text;
    cmd.colour = self->getCurrentColour();
    cmd.font = self->currentFont;
    cmd.text = juce::String(str ? str : "");
    cmd.x = (float)(self->gfx_x ? *self->gfx_x : 0.0);
    cmd.y = (float)(self->gfx_y ? *self->gfx_y : 0.0);

    self->commands.push_back(cmd);

    // Advance pen position by string width (approx)
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
    const juce::String text(str ? str : "");

    const float w = self->currentFont.getStringWidthFloat(text);
    const float h = self->currentFont.getHeight();

    if (self->gfx_x) *self->gfx_x = (double)w;
    if (self->gfx_y) *self->gfx_y = (double)h;

    return (EEL_F)w;
  }

  static EEL_F NSEEL_CGEN_CALL eel_gfx_circle(void* opaque, INT_PTR np, EEL_F** parms)
  {
    auto* self = (GfxVm*)opaque;
    if (!self || np < 3) return 0.0;

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

  static EEL_F NSEEL_CGEN_CALL eel_gfx_getchar(void* opaque, INT_PTR np, EEL_F** parms)
  {
    (void)opaque; (void)np; (void)parms;
    return 0.0; // “no key”
  }

  // -------------------------------------------------------------------
  // VM-bound variables
  // -------------------------------------------------------------------
  EEL_F* gfx_x = nullptr;
  EEL_F* gfx_y = nullptr;
  EEL_F* gfx_w = nullptr;
  EEL_F* gfx_h = nullptr;
  EEL_F* gfx_r = nullptr;
  EEL_F* gfx_g = nullptr;
  EEL_F* gfx_b = nullptr;
  EEL_F* gfx_a = nullptr;
  EEL_F* gfx_clear = nullptr;
  EEL_F* gfx_mode = nullptr;

  EEL_F* mouse_x = nullptr;
  EEL_F* mouse_y = nullptr;
  EEL_F* mouse_cap = nullptr;
  EEL_F* mouse_wheel = nullptr;
  EEL_F* mouse_hwheel = nullptr;

  EEL_F* srate_var = nullptr;
  EEL_F* samplesblock_var = nullptr;

  // Drawing state
  std::unordered_map<int, juce::Font> fonts;
  int currentFontId = 0;
  juce::Font currentFont;

  std::vector<DrawCmd> commands;
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

    const double* mem = nullptr;
    int memN = 0;

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
#if defined(DSPJSFX_VARS) && defined(DSPJSFX_VARS_COUNT)
    vm->bindUserVars(DSPJSFX_VARS, (int)DSPJSFX_VARS_COUNT);
#else
    vm->bindUserVars(nullptr, 0);
#endif

    // Compile relevant sections. We compile init + gfx so helper functions
    // defined in init are available to gfx.
    const char* err = nullptr;

    if (!sections.init.empty())
      code_init = vm->compile_code(sections.init.c_str(), &err);

    err = nullptr;
    if (sections.hasGfx)
    {
      // Some scripts specify "@gfx" with no body. Treat it as a no-op rather than a hard error.
      const char* gfxCode = sections.gfx.empty() ? "0;" : sections.gfx.c_str();
      code_gfx = vm->compile_code(gfxCode, &err);

      if (!code_gfx)
      {
        const char* e = err ? err : NSEEL_code_getcodeerror(vm->m_vm);
        lastError = e ? e : "Unknown EEL compile error";
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

  void renderFrame(int width, int height, const Snapshot& snap)
  {
    if (!hasGfxSection() || !gfxCompiledOk()) return;

    // One-time init, with current snapshot state applied first.
    if (!initRan && code_init)
    {
      if (snap.sliders) vm->syncSliders(snap.sliders, snap.slidersCount);
      if (snap.vars)    vm->syncVars(snap.vars, snap.varsCount);
      if (snap.mem)     vm->syncMem(snap.mem, snap.memN);
      vm->setTiming(snap.srate, snap.samplesblock);
      NSEEL_code_execute(code_init);
      initRan = true;
    }

    // Sync state into VM.
    if (snap.sliders) vm->syncSliders(snap.sliders, snap.slidersCount);
    if (snap.vars)    vm->syncVars(snap.vars, snap.varsCount);
    if (snap.mem)     vm->syncMem(snap.mem, snap.memN);

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
