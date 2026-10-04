// SPDX-License-Identifier: Zlib
// C++ drawing semantics adapted from YSFXGfxInterpreter.h. Included inside
// Frame: no EEL VM, code handle, RAM accessor, or string context is created.
using DrawCmd = jsfx_gfx::DrawCmd;
using GfxImageBank = jsfx_gfx::GfxImageBank;
Reference gfx_r, gfx_g, gfx_b, gfx_a, gfx_a2, gfx_x, gfx_y, gfx_w, gfx_h,
          gfx_dest, gfx_mode, gfx_texth, gfx_clear;
std::array<int,128> imageWidths{}, imageHeights{};
std::array<bool,128> imageResourceAttempted{};
std::array<juce::String,128> imageResourceNames{};
std::function<juce::Image(const juce::String&)> imageLoader;
jsfx_gfx::GfxRenderPort* renderPort = nullptr;
size_t pendingImageBytes = 0;
bool framebufferDirty = false, frameRecording = false;
uint64_t frameCounter = 0;
void configureResources(const char* source) {
  std::istringstream input(source?source:"");std::string line;
  while(std::getline(input,line)) {
    const auto start=line.find_first_not_of(" \t\r");if(start==std::string::npos)continue;
    line.erase(0,start);if(line[0]=='@')break;
    if(line.compare(0,9,"filename:")!=0)continue;
    const auto comma=line.find(',',9);if(comma==std::string::npos)continue;
    char* tail=nullptr;const long slot=std::strtol(line.c_str()+9,&tail,10);
    if(tail!=line.c_str()+comma||slot<0||slot>=128)continue;
    auto name=line.substr(comma+1);
    const auto semi=name.find(';');if(semi!=std::string::npos)name.erase(semi);
    const auto first=name.find_first_not_of(" \t\r");if(first==std::string::npos)continue;
    name=name.substr(first);name.erase(name.find_last_not_of(" \t\r")+1);
    imageResourceNames[(size_t)slot]=juce::String::fromUTF8(name.c_str());
    fileResourceNames[(size_t)slot]=imageResourceNames[(size_t)slot];
    fileResourceDeclared[(size_t)slot]=true;
  }
}
bool cursorPending = false;
int cursorResource = 32512;
juce::String cursorName;
std::vector<juce::String> activeDroppedFiles;
std::deque<std::vector<juce::String>> pendingDroppedFileBatches;
void bindDrawingGlobals() {
  gfx_r = Reference(cellPointer("gfx_r"));
  gfx_g = Reference(cellPointer("gfx_g"));
  gfx_b = Reference(cellPointer("gfx_b"));
  gfx_a = Reference(cellPointer("gfx_a"));
  gfx_a2 = Reference(cellPointer("gfx_a2"));
  gfx_x = Reference(cellPointer("gfx_x"));
  gfx_y = Reference(cellPointer("gfx_y"));
  gfx_w = Reference(cellPointer("gfx_w"));
  gfx_h = Reference(cellPointer("gfx_h"));
  gfx_dest = Reference(cellPointer("gfx_dest"));
  gfx_mode = Reference(cellPointer("gfx_mode"));
  gfx_texth = Reference(cellPointer("gfx_texth"));
  gfx_clear = Reference(cellPointer("gfx_clear"));
}
void beginDrawingFrame() {
  bindDrawingGlobals();
  pendingImageBytes=0; framebufferDirty=false; frameRecording=true;
  set("gfx_frame", (double)frameCounter++);
}
bool readGfxTable(double address,int count,std::vector<double>& out,bool requireOneBlock=false) {
  out.clear();
  if(!std::isfinite(address)||address<0||address>2147483647.0||count<0||count>8192)return false;
  const uint64_t base=(uint64_t)(address+0.5);
  if(requireOneBlock && base%65536+(uint64_t)count>65536)return false;
  auto& shared=scriptState();
  if(!shared.mem || base+(uint64_t)count>(uint64_t)shared.memN)return false;
  out.reserve((size_t)count);
  for(int i=0;i<count;++i) {
    double value=shared.mem[base+(size_t)i];
    if(!std::isfinite(value)){out.clear();return false;}
    out.push_back(value);
  }
  return true;
}
static constexpr int kInvalidGfxDestination=-2;
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
    cmd.frameWidth = frameWidth;
    cmd.frameHeight = frameHeight;
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
      cmd.w = (float)frameWidth;
      cmd.h = (float)frameHeight;
      cmd.fill = true;
      commands.push_back(std::move(cmd));
    }
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

  static int gfxByte(double v) noexcept
  {
    return std::isfinite(v) ? (int)(std::max(0.0,std::min(1.0,v))*255.0) : 0;
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

  static double eel_gfx_set(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
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

  static double eel_gfx_rect(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
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

  static double eel_gfx_rectto(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
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

  static double eel_gfx_line(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
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

  static double eel_gfx_lineto(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
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

  static double eel_gfx_circle(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
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

  static double eel_gfx_roundrect(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
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

  static double eel_gfx_arc(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
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

  static double eel_gfx_triangle(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
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

    for (intptr_t i = 0; i + 1 < np; i += 2)
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

  static double eel_gfx_setimgdim(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self=(Frame*)opaque;if(!self||np<3)return 0;
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

  static double eel_gfx_getimgdim(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*) opaque;
    if (!self || np < 3)
      return 0.0;

    int width = 0;
    int height = 0;
    const double imageValue = (double) *parms[0];

    if (std::isfinite(imageValue) && imageValue < 0.0)
    {
      // Handy compatibility extension: -1 describes the main framebuffer.
      width = self->frameWidth;
      height = self->frameHeight;
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

    if (parms[1]) *parms[1] = (double) width;
    if (parms[2]) *parms[2] = (double) height;
    return (double) imageValue;
  }

  static double eel_gfx_blit(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*) opaque;
    if (!self || np < 3)
      return 0.0;

    const double sourceValue = (double) *parms[0];
    int source = -1;
    int sourceWidth = self->frameWidth;
    int sourceHeight = self->frameHeight;

    if (!(std::isfinite(sourceValue) && sourceValue < 0.0))
    {
      source = gfxImageIndexFromValue(sourceValue);
      if (source < 0)
        return (double) sourceValue;
      self->ensureImageSlot(source);
      sourceWidth = self->imageWidths[(size_t) source];
      sourceHeight = self->imageHeights[(size_t) source];
    }

    if (sourceWidth <= 0 || sourceHeight <= 0)
      return (double) sourceValue;

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Blit;
    if (!self->beginDrawCommand(cmd))
      return (double) sourceValue;

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
      return (double) sourceValue;

    if (cmd.srcW == 0.0f || cmd.srcH == 0.0f || cmd.destW == 0.0f || cmd.destH == 0.0f)
      return (double) sourceValue;

    self->commands.push_back(std::move(cmd));
    return (double) sourceValue;
  }

  static double eel_gfx_blitext(void* opaque,intptr_t np,Reference* parms)
  {
    auto* self=(Frame*)opaque;if(!self||np<3)return 0;
    std::vector<double> table;if(!self->readGfxTable(*parms[1],10,table))return *parms[0];
    double a[13]={*parms[0],1,*parms[2]}; Reference p[13];
    for(int i=0;i<10;++i)a[i+3]=table[(size_t)i];for(int i=0;i<13;++i)p[i]=a+i;
    eel_gfx_blit(opaque,13,p);return *parms[0];
  }

  static double eel_gfx_deltablit(void* opaque,intptr_t np,Reference* parms)
  {
    auto* self=(Frame*)opaque;if(!self||np<9)return 0;
    const int source=std::isfinite(*parms[0])&&*parms[0]<0?-1:gfxImageIndexFromValue(*parms[0]);
    if(source<0 && !(std::isfinite(*parms[0])&&*parms[0]<0))return 0;
    if(source>=0)self->ensureImageSlot(source);
    DrawCmd c;c.type=DrawCmd::Type::DeltaBlit;if(!self->beginDrawCommand(c))return 0;
    c.source=source;c.srcX=(float)*parms[1];c.srcY=(float)*parms[2];c.srcW=(float)*parms[3];c.srcH=(float)*parms[4];
    c.destX=(float)*parms[5];c.destY=(float)*parms[6];c.destW=(float)*parms[7];c.destH=(float)*parms[8];
    for(int i=0;i<6;++i)c.values[(size_t)i]=i+9<np?*parms[i+9]:(i==0||i==3?1:0);
    c.useClipRect=np<16||*parms[15]>0.5;self->commands.push_back(std::move(c));return 0;
  }

  static double eel_gfx_transformblit(void* opaque,intptr_t np,Reference* parms)
  {
    auto* self=(Frame*)opaque;if(!self||np<8)return 0;
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

  static double eel_gfx_setpixel(void* opaque,intptr_t np,Reference* parms)
  {
    auto* self=(Frame*)opaque;if(!self||np<3)return 0.0;
    DrawCmd c;c.type=DrawCmd::Type::Pixel;if(!self->beginDrawCommand(c))return *parms[0];
    // LICE/JSFX setpixel uses alpha=255 for the source colour; gfx_a2 does not apply.
    c.colour=juce::Colour(0xff000000u|((uint32_t)gfxByte(*parms[0])<<16)|((uint32_t)gfxByte(*parms[1])<<8)|(uint32_t)gfxByte(*parms[2]));
    c.x=(float)boundedGfxInt(*self->gfx_x);c.y=(float)boundedGfxInt(*self->gfx_y);
    self->commands.push_back(std::move(c));return *parms[0];
  }

  static double eel_gfx_getpixel(void* opaque,intptr_t np,Reference* parms)
  {
    auto* self=(Frame*)opaque;if(!self||np<3)return 0.0;
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

  static double eel_gfx_gradrect(void* opaque,intptr_t np,Reference* parms)
  {
    auto* self=(Frame*)opaque;if(!self||np<8)return 0;
    DrawCmd c;c.type=DrawCmd::Type::GradRect;
    for(int i=0;i<12;++i){c.values[(size_t)i]=i+4<np?*parms[i+4]:0.0;if(!std::isfinite(c.values[(size_t)i])||std::abs(c.values[(size_t)i])>32760)return 0;}
    if(!self->beginDrawCommand(c))return 0;
    c.x=(float)std::floor(*parms[0]);c.y=(float)std::floor(*parms[1]);c.w=(float)std::floor(*parms[2]);c.h=(float)std::floor(*parms[3]);
    self->commands.push_back(std::move(c));return 0;
  }

  static double eel_gfx_muladdrect(void* opaque,intptr_t np,Reference* parms)
  {
    auto* self=(Frame*)opaque;if(!self||np<7)return 0;
    DrawCmd c;c.type=DrawCmd::Type::MulAddRect;
    for(int i=0;i<8;++i){c.values[(size_t)i]=i+4<np?*parms[i+4]:(i==3?1.0:0.0);if(!std::isfinite(c.values[(size_t)i])||std::abs(c.values[(size_t)i])>128)return 0;}
    if(!self->beginDrawCommand(c))return 0;
    c.x=(float)std::floor(*parms[0]);c.y=(float)std::floor(*parms[1]);c.w=(float)std::floor(*parms[2]);c.h=(float)std::floor(*parms[3]);
    self->commands.push_back(std::move(c));return 0;
  }

  static double eel_gfx_blurto(void* opaque,intptr_t np,Reference* parms)
  {
    auto* self=(Frame*)opaque;if(!self||np<2)return 0;
    DrawCmd c;c.type=DrawCmd::Type::Blur;if(!self->beginDrawCommand(c))return 0;
    const double x=*self->gfx_x,y=*self->gfx_y,x2=*parms[0],y2=*parms[1];
    c.x=(float)std::min(x,x2);c.y=(float)std::min(y,y2);c.w=(float)std::abs(x2-x);c.h=(float)std::abs(y2-y);
    self->commands.push_back(std::move(c));*self->gfx_x=x2;*self->gfx_y=y2;return *parms[0];
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

  static double eel_gfx_drawchar(void* opaque,intptr_t np,Reference* parms)
  {
    auto* self=(Frame*)opaque;if(!self||np<1)return 0;
    const int ch=boundedGfxInt(*parms[0]+0.5);
    return emitTextCommand(self,gfxCharacter(ch==10||ch==13?32:ch),1,parms);
  }

  static double eel_gfx_drawnumber(void* opaque,intptr_t np,Reference* parms)
  {
    auto* self=(Frame*)opaque;if(!self||np<2)return 0;
    const int digits=std::max(0,std::min(16,boundedGfxInt(*parms[1]+0.5)));
    char b[512]={};snprintf(b,sizeof(b),"%.*f",digits,(double)*parms[0]);
    return emitTextCommand(self,juce::String::fromUTF8(b),1,parms);
  }

  static double eel_gfx_measurechar(void* opaque,intptr_t np,Reference* parms)
  {
    auto* self=(Frame*)opaque;if(!self||np<3)return 0;
    const auto text=gfxCharacter(*parms[0]);
    *parms[1]=self->currentFont==0?8:self->fonts[(size_t)self->currentFont].font.getStringWidthFloat(text);
    *parms[2]=self->currentFont==0?8:self->fonts[(size_t)self->currentFont].font.getHeight();return *parms[0];
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

  static void updateTextPenPosition(Frame* self, const juce::String& text)
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
    if(self->currentFont==0)LICE_MeasureText(lastLine.toRawUTF8(),&bitmapWidth,&bitmapHeight);
    const float advance = self->currentFont==0 ? (float)bitmapWidth : self->fonts[(size_t)self->currentFont].font.getStringWidthFloat(lastLine);

    if (self->gfx_x)
      *self->gfx_x = x0 + advance;
    if (self->gfx_y)
      *self->gfx_y = y0 + (double) ((numLines - 1) * (self->currentFont==0 ? 8.0f : self->fonts[(size_t)self->currentFont].font.getHeight()));
  }

  static double emitTextCommand(Frame* self, const juce::String& text, intptr_t np, Reference* parms)
  {
    if (!self || text.isEmpty())
      return np > 0 ? *parms[0] : 0.0;

    DrawCmd cmd;
    cmd.type = DrawCmd::Type::Text;
    if (!self->beginDrawCommand(cmd))
      return np > 0 ? *parms[0] : 0.0;
    cmd.colour = self->getCurrentColour();
    cmd.font = self->fonts[(size_t)self->currentFont].font;
    cmd.bitmapFont = self->currentFont == 0;
    cmd.fontSerial = self->fonts[(size_t)self->currentFont].serial;
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

static double eel_gfx_loadimg(void* opaque,intptr_t np,Reference* parms) {
  auto* self=static_cast<Frame*>(opaque); if(!self||np<2)return -1;
  int slot=gfxImageIndexFromValue(*parms[0]);if(slot<0)return -1;
  return self->loadImageSlot(slot,juce::String::fromUTF8(self->string(*parms[1]).c_str()))?slot:-1;
}
static double eel_gfx_drawstr(void* opaque,intptr_t np,Reference* parms) {
  auto* self=static_cast<Frame*>(opaque); if(!self||np<1)return 0;
  return emitTextCommand(self,juce::String::fromUTF8(self->string(*parms[0]).c_str()),np,parms);
}
static double eel_gfx_printf(void* opaque,intptr_t np,Reference* parms) {
  auto* self=static_cast<Frame*>(opaque); if(!self||np<1)return 0;
  const auto fmt=self->string(*parms[0]);
  std::string stringArgument;
  const auto text=jsfx_gfx::formatGfxPrintf(fmt.c_str(),(int)np-1,
    [&](int i){return double(*parms[i+1]);},
    [&](double value){stringArgument=self->string(value);return stringArgument.c_str();});
  return emitTextCommand(self,juce::String::fromUTF8(text.c_str()),1,parms);
}
static double eel_gfx_measurestr(void* opaque,intptr_t np,Reference* parms) {
  auto* self=static_cast<Frame*>(opaque); if(!self||np<1)return 0;
  jsfx_native_gfx_measurestr(&self->state,*parms[0],np>1?parms[1].ptr:nullptr,np>2?parms[2].ptr:nullptr);
  return *parms[0];
}
static double eel_gfx_setfont(void* opaque,intptr_t np,Reference* parms) {
  auto* self=static_cast<Frame*>(opaque); if(!self||np<1)return 0;
  return jsfx_native_gfx_setfont(&self->state,*parms[0],np>1?double(*parms[1]):0,
    np>2?double(*parms[2]):0,np>3?double(*parms[3]):0,(double)np);
}
static double eel_gfx_getfont(void* opaque,intptr_t np,Reference* parms) {
  auto* self=static_cast<Frame*>(opaque); if(!self)return -1;
  if(np>0)self->writeString(*parms[0],self->fonts[(size_t)self->currentFont].name.toRawUTF8());
  return self->currentFont-1;
}
static double eel_gfx_setcursor(void* opaque,intptr_t np,Reference* parms) {
  auto* self=static_cast<Frame*>(opaque); if(!self||np<1)return 0;
  const int resource=jsfx_gfx::boundedGfxInt(*parms[0]);if(!resource)return 0;
  const auto name=np>1?juce::String::fromUTF8(self->string(*parms[1]).c_str()):juce::String();
  if(self->cursorResource!=resource || !(self->cursorName==name)) {
    self->cursorResource=resource;self->cursorName=name;self->cursorPending=true;
  }
  return 0;
}
static double eel_gfx_getdropfile(void* opaque,intptr_t np,Reference* parms) {
  auto* self=static_cast<Frame*>(opaque); if(!self||np<1)return 0;
  const int index=jsfx_gfx::boundedGfxInt(*parms[0]);
  if(index<0){self->activeDroppedFiles.clear();return 0;}
  if(self->activeDroppedFiles.empty()&&!self->pendingDroppedFileBatches.empty()) {
    self->activeDroppedFiles=std::move(self->pendingDroppedFileBatches.front());
    self->pendingDroppedFileBatches.pop_front();
  }
  if(index>=(int)self->activeDroppedFiles.size())return 0;
  if(np>1)self->writeString(*parms[1],self->activeDroppedFiles[(size_t)index].toRawUTF8());
  return 1;
}
