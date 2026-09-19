// SPDX-License-Identifier: Zlib
// Included INSIDE namespace jsfx_gfx, after DrawCmd. See JsfxGfxLice.h.
// Private image slots contain STRAIGHT (not premultiplied) ARGB. Only the main
// framebuffer is published to JUCE, with opaque alpha, as in JSFX's screen.
// This preserves RGB even beneath alpha=0 (gfx_mode&2 must still see that RGB).
#ifndef ZA_JSFX_GFX_RASTER_INCLUDED
#define ZA_JSFX_GFX_RASTER_INCLUDED

struct GfxRenderStats
{
  size_t commandCount = 0;
  unsigned int graphicsContexts = 0; // Actual JUCE glyph-mask contexts, not surface maps.
  unsigned int surfaceMaps = 0, textCacheHits = 0, textCacheMisses = 0;
  unsigned int blits = 0, selfBlitCopies = 0, readbacks = 0, checkpoints = 0;
  bool mainFramebufferChanged = false;
};

struct GfxRenderPort
{
  virtual ~GfxRenderPort() = default;
  virtual void flush(const std::vector<DrawCmd>& commands) = 0;
  virtual uint32_t readPixel(int dest, GfxImageBank& bank, int x, int y) = 0;
};

struct GfxTextMaskCache
{
  struct Entry {
    juce::String text;
    uint64_t serial = 0;
    int width = 0, height = 0;
    uint64_t lastUse = 0;
    std::vector<unsigned char> coverage;
  };
  std::vector<Entry> entries;
  size_t bytes = 0;
  uint64_t clock = 0;
  static constexpr size_t maxBytes = 8u * 1024u * 1024u;
  static constexpr size_t maxEntries = 256;
};

static inline juce::Image copyRawGfxImage(const juce::Image& source)
{
  if (!source.isValid()) return {};
  juce::Image result(juce::Image::ARGB, source.getWidth(), source.getHeight(), true, juce::SoftwareImageType());
  if (!result.isValid()) return {};
  juce::Image::BitmapData from(source, juce::Image::BitmapData::readOnly);
  juce::Image::BitmapData to(result, juce::Image::BitmapData::writeOnly);
  if (from.pixelStride != 4 || to.pixelStride != 4) return {};
  for (int y=0; y<source.getHeight(); ++y)
    std::memcpy(to.getLinePointer(y), from.getLinePointer(y), (size_t)source.getWidth()*4);
  return result;
}

static inline bool fullGfxReplacement(const DrawCmd& c, int w, int h)
{
  return c.type == DrawCmd::Type::Rect && c.fill && c.opacity == 1.0f
      && (c.blitMode & 0x71) == 0 && c.colour.isOpaque()
      && c.x <= 0 && c.y <= 0 && (double)c.x+c.w >= w && (double)c.y+c.h >= h;
}
static inline bool isGfxBlit(DrawCmd::Type t)
{
  return t==DrawCmd::Type::Blit || t==DrawCmd::Type::DeltaBlit || t==DrawCmd::Type::TransformBlit;
}
static inline bool canDiscardPreviousFramebuffer(const std::vector<DrawCmd>& cmds, int w, int h)
{
  for (const auto& c:cmds) {
    if (c.type==DrawCmd::Type::SetImageDim || c.type==DrawCmd::Type::LoadImage) continue;
    if (isGfxBlit(c.type) && c.source==-1) return false;
    if (c.dest==-1) return fullGfxReplacement(c,w,h);
  }
  return true;
}

class GfxRenderSession final : public GfxRenderPort
{
public:
  // prepareMain(false) inherits history; prepareMain(true) may discard it.
  // It is invoked lazily, so ordinary full-redraw UIs still make zero history copies.
  GfxRenderSession(juce::Image& target, std::function<void(bool)> prepareMain = {})
    : framebuffer(target), prepare(std::move(prepareMain)) {}
  ~GfxRenderSession() override { releaseMaps(); }
  GfxRenderSession(const GfxRenderSession&) = delete;
  GfxRenderSession& operator=(const GfxRenderSession&) = delete;

  void flush(const std::vector<DrawCmd>& commands) override
  {
    if (nextCommand >= commands.size()) return;
    ++stats.checkpoints;
    while (nextCommand < commands.size()) {
      const auto& c=commands[nextCommand++];
      ++stats.commandCount;
      paint(c);
    }
  }

  uint32_t readPixel(int dest, GfxImageBank& bank, int x, int y) override
  {
    ++stats.readbacks;
    if (auto* b=surface(dest,&bank,false)) return LICE_GetPixel(b,x,y);
    return 0;
  }

  GfxRenderStats finish(const std::vector<DrawCmd>& commands)
  {
    flush(commands);
    releaseMaps();
    if (stats.mainFramebufferChanged && framebuffer.isValid()) {
      // The screen's alpha is not used by JSFX, including screen-source blits.
      // Opaque ARGB is also a valid premultiplied JUCE image: no colour conversion,
      // no second framebuffer, and no copied publication buffer is required.
      juce::Image::BitmapData bits(framebuffer,juce::Image::BitmapData::readWrite);
      if (bits.pixelStride==4)
        for (int y=0;y<framebuffer.getHeight();++y) {
          auto* row=reinterpret_cast<uint32_t*>(bits.getLinePointer(y));
          for (int x=0;x<framebuffer.getWidth();++x) row[x]|=0xff000000u;
        }
    }
    return stats;
  }
  const GfxRenderStats& getStats() const noexcept { return stats; }

private:
  struct MappedSurface {
    juce::Image::BitmapData data;
    LICE_WrapperBitmap bitmap;
    explicit MappedSurface(juce::Image& im)
      : data(im,juce::Image::BitmapData::readWrite),
        bitmap(reinterpret_cast<LICE_pixel*>(data.getLinePointer(0)),
               data.pixelStride==4 ? im.getWidth():0, im.getHeight(),data.lineStride/4,false) {}
  };
  void releaseMaps()
  {
    mainMap.reset();
    for (auto& m:maps) m.reset();
  }
  LICE_IBitmap* surface(int dest,GfxImageBank* bank,bool discard)
  {
    if (dest==-1) {
      if (!mainReady) { if (prepare) prepare(discard); mainReady=true; }
      if (!framebuffer.isValid()) return nullptr;
      if (!mainMap) { mainMap=std::make_unique<MappedSurface>(framebuffer); ++stats.surfaceMaps; }
      return &mainMap->bitmap;
    }
    if (!bank || dest<0 || dest>=GfxImageBank::kNumImages) return nullptr;
    if (bank!=activeBank) { for (auto& m:maps)m.reset(); activeBank=bank; }
    auto& im=bank->images[(size_t)dest];
    if (!im.isValid()) return nullptr;
    auto& m=maps[(size_t)dest];
    if (!m) { m=std::make_unique<MappedSurface>(im); ++stats.surfaceMaps; }
    return &m->bitmap;
  }
  static int mode(const DrawCmd& c, bool blit=false)
  {
    const int extended=(c.blitMode>>4)&15;
    int m=extended>0 && extended<=5 ? extended : (c.blitMode&1 ? LICE_BLIT_MODE_ADD:LICE_BLIT_MODE_COPY);
    if (blit) {
      if (c.source!=-1 && !(c.blitMode&2)) m|=LICE_BLIT_USE_ALPHA;
      if (!(c.blitMode&4)) m|=LICE_BLIT_FILTER_BILINEAR;
    }
    return m;
  }
  static bool sane(double x,double limit=1048576.0) { return std::isfinite(x) && std::abs(x)<=limit; }
  static bool visibleRect(LICE_IBitmap* d,double x,double y,double w,double h)
  {
    return sane(x)&&sane(y)&&sane(w)&&sane(h)&&w>0&&h>0
        && x<d->getWidth() && y<d->getHeight() && x+w>0 && y+h>0;
  }

  const GfxTextMaskCache::Entry* textMask(GfxTextMaskCache& cache,const DrawCmd& c,const juce::String& line)
  {
    for(auto& e:cache.entries) if(e.serial==c.fontSerial && e.text==line) {
      e.lastUse=++cache.clock; ++stats.textCacheHits; return &e;
    }
    const int w=std::max(1,boundedGfxInt(std::ceil(c.font.getStringWidthFloat(line)))+4);
    const int h=std::max(1,boundedGfxInt(std::ceil(c.font.getHeight()))+4);
    if(w>16384 || h>2048 || (size_t)w*h>GfxTextMaskCache::maxBytes) return nullptr;
    while(!cache.entries.empty() && (cache.entries.size()>=GfxTextMaskCache::maxEntries
          || cache.bytes+(size_t)w*h>GfxTextMaskCache::maxBytes)) {
      auto oldest=std::min_element(cache.entries.begin(),cache.entries.end(),
                      [](const auto& a,const auto& b){return a.lastUse<b.lastUse;});
      cache.bytes-=oldest->coverage.size(); cache.entries.erase(oldest);
    }
    GfxTextMaskCache::Entry e;
    e.text=line;e.serial=c.fontSerial;e.width=w;e.height=h;e.lastUse=++cache.clock;
    e.coverage.resize((size_t)w*h);
    juce::Image mask(juce::Image::ARGB,w,h,true,juce::SoftwareImageType());
    {
      juce::Graphics g(mask); ++stats.graphicsContexts;
      g.setColour(juce::Colours::white);g.setFont(c.font);
      g.drawText(line,0,0,w,boundedGfxInt(c.font.getHeight()),juce::Justification::topLeft,false);
    }
    {
      juce::Image::BitmapData bits(mask,juce::Image::BitmapData::readOnly);
      for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        uint32_t p=0;std::memcpy(&p,bits.getLinePointer(y)+x*bits.pixelStride,4);
        e.coverage[(size_t)y*w+x]=(unsigned char)(p>>24);
      }
    }
    cache.bytes+=e.coverage.size();cache.entries.push_back(std::move(e));++stats.textCacheMisses;
    return &cache.entries.back();
  }

  void text(LICE_IBitmap* dest,const DrawCmd& c)
  {
    const int lineHeight=c.bitmapFont ? 8 : std::max(1,boundedGfxInt(c.font.getHeight()));
    juce::StringArray lines; lines.addLines(c.text);
    if(lines.isEmpty()) return;
    const int totalHeight=lineHeight*lines.size();
    int y=boundedGfxInt(c.y);
    if(c.useTextBounds) {
      if(c.textFlags&8)y+=boundedGfxInt(c.h)-totalHeight;
      else if(c.textFlags&4)y+=(boundedGfxInt(c.h)-totalHeight)/2;
    }
    const bool clip=c.useTextBounds && !(c.textFlags&256);
    const auto rect=juce::Rectangle<int>(boundedGfxInt(c.x),boundedGfxInt(c.y),boundedGfxInt(c.w),boundedGfxInt(c.h))
                      .getIntersection(juce::Rectangle<int>(0,0,dest->getWidth(),dest->getHeight()));
    if(clip && rect.isEmpty())return;
    LICE_SubBitmap sub(dest,clip?rect.getX():0,clip?rect.getY():0,
                      clip?rect.getWidth():dest->getWidth(),clip?rect.getHeight():dest->getHeight());
    auto* out=clip ? static_cast<LICE_IBitmap*>(&sub):dest;
    if(c.imageBank && !c.imageBank->textMasks)c.imageBank->textMasks=std::make_shared<GfxTextMaskCache>();
    auto& cache=c.imageBank ? *c.imageBank->textMasks : fallbackTextCache;
    for(int i=0;i<lines.size();++i,y+=lineHeight) {
      int x=boundedGfxInt(c.x);
      int width=0,unused=0;
      if(c.bitmapFont)LICE_MeasureText(lines[i].toRawUTF8(),&width,&unused);
      else width=boundedGfxInt(std::ceil(c.font.getStringWidthFloat(lines[i])));
      if(c.useTextBounds) {
        if(c.textFlags&2)x+=boundedGfxInt(c.w)-width;
        else if(c.textFlags&1)x+=(boundedGfxInt(c.w)-width)/2;
      }
      const int dx=x-(clip?rect.getX():0),dy=y-(clip?rect.getY():0);
      if(!visibleRect(out,dx,dy,std::max(1,width+4),lineHeight+4))continue;
      if(c.bitmapFont)LICE_DrawText(out,dx,dy,lines[i].toRawUTF8(),c.colour.getARGB(),c.opacity,mode(c));
      else if(auto* e=textMask(cache,c,lines[i]))
        LICE_DrawGlyphEx(out,dx,dy,c.colour.getARGB(),e->coverage.data(),e->width,e->width,e->height,c.opacity,mode(c));
    }
  }

  void paint(const DrawCmd& c)
  {
    if(c.type==DrawCmd::Type::SetImageDim || c.type==DrawCmd::Type::LoadImage) {
      if(c.imageBank && c.imageIndex>=0 && c.imageIndex<GfxImageBank::kNumImages) {
        if(activeBank==c.imageBank)maps[(size_t)c.imageIndex].reset();
        if(c.type==DrawCmd::Type::LoadImage || c.replacementImage.isValid()
            || c.imageWidth==0 || c.imageHeight==0)
          c.imageBank->images[(size_t)c.imageIndex]=c.replacementImage;
        else c.imageBank->resizeImage(c.imageIndex,c.imageWidth,c.imageHeight);
      }
      return;
    }
    // Resolve source before requesting a discardable destination. A screen read
    // must inherit history, even when the following draw covers the full canvas.
    LICE_IBitmap* src=nullptr;
    if(isGfxBlit(c.type)) {
      src=surface(c.source,c.imageBank,false);
      if(!src)return;
    }
    auto* d=surface(c.dest,c.imageBank,fullGfxReplacement(c,framebuffer.getWidth(),framebuffer.getHeight()));
    if(!d || d->getWidth()<=0 || d->getHeight()<=0)return;
    std::unique_ptr<LICE_MemBitmap> copied;
    if(src==d) {
      copied=std::make_unique<LICE_MemBitmap>(src->getWidth(),src->getHeight());
      if(!copied->getBits())return;
      LICE_Blit(copied.get(),src,0,0,0,0,src->getWidth(),src->getHeight(),1.0f,LICE_BLIT_MODE_COPY);
      src=copied.get();++stats.selfBlitCopies;
    }
    const int m=mode(c),bm=mode(c,true);
    const LICE_pixel col=c.colour.getARGB();
    const int x=boundedGfxInt(c.x),y=boundedGfxInt(c.y),w=boundedGfxInt(c.w),h=boundedGfxInt(c.h);
    bool drawn=true;
    switch(c.type) {
      case DrawCmd::Type::Rect:
        if(!visibleRect(d,c.x,c.y,c.w,c.h))return;
        if(c.fill)LICE_FillRect(d,x,y,w,h,col,c.opacity,m);
        else LICE_DrawRect(d,x,y,w-1,h-1,col,c.opacity,m);
        break;
      case DrawCmd::Type::Pixel:
        LICE_PutPixel(d,x,y,col,c.opacity,m); break;
      case DrawCmd::Type::Line:
        if(!sane(c.x)||!sane(c.y)||!sane(c.x2)||!sane(c.y2))return;
        { int x1=x,y1=y,x2=boundedGfxInt(c.x2),y2=boundedGfxInt(c.y2);
          if(!LICE_ClipLine(&x1,&y1,&x2,&y2,0,0,d->getWidth(),d->getHeight()))return;
          LICE_Line(d,x1,y1,x2,y2,col,c.opacity,m,c.antialias); } break;
      case DrawCmd::Type::Circle:
        if(!sane(c.radius,32760)||c.radius<0||!visibleRect(d,c.x-c.radius,c.y-c.radius,2*c.radius+2,2*c.radius+2))return;
        if(c.fill)LICE_FillCircle(d,c.x,c.y,c.radius,col,c.opacity,m,c.antialias);
        else LICE_Circle(d,c.x,c.y,c.radius,col,c.opacity,m,c.antialias);
        break;
      case DrawCmd::Type::RoundRect:
        if(!visibleRect(d,c.x,c.y,c.w,c.h)||!sane(c.cornerRadius,32760))return;
        LICE_RoundRect(d,c.x,c.y,c.w,c.h,boundedGfxInt(c.cornerRadius),col,c.opacity,m,c.antialias);break;
      case DrawCmd::Type::Arc:
        if(!sane(c.radius,32760)||c.radius<0||!sane(c.angle1)||!sane(c.angle2)
            ||!visibleRect(d,c.x-c.radius,c.y-c.radius,2*c.radius+2,2*c.radius+2))return;
        LICE_Arc(d,c.x,c.y,c.radius,c.angle1,c.angle2,col,c.opacity,m,c.antialias);break;
      case DrawCmd::Type::Triangle: {
        if(c.points.size()<3 || c.points.size()>1024)return;
        std::vector<int> xs,ys;xs.reserve(c.points.size());ys.reserve(c.points.size());
        for(const auto& p:c.points) {if(!sane(p.x)||!sane(p.y))return;xs.push_back(boundedGfxInt(p.x));ys.push_back(boundedGfxInt(p.y));}
        if(xs.size()==3)LICE_FillTriangle(d,xs[0],ys[0],xs[1],ys[1],xs[2],ys[2],col,c.opacity,m);
        else LICE_FillConvexPolygon(d,xs.data(),ys.data(),(int)xs.size(),col,c.opacity,m);break;
      }
      case DrawCmd::Type::Text: text(d,c);break;
      case DrawCmd::Type::GradRect:
        if(!visibleRect(d,c.x,c.y,c.w,c.h))return;
        for(int channel=0;channel<4;++channel) {
          if(!sane(c.values[(size_t)channel+4],64)||!sane(c.values[(size_t)channel+8],64))return;
          for(double xx:{0.0,(double)w+1})for(double yy:{0.0,(double)h+1})
            if(!sane(c.values[(size_t)channel]+xx*c.values[(size_t)channel+4]+yy*c.values[(size_t)channel+8],64))return;
        }
        LICE_GradRect(d,x,y,w,h,(float)c.values[0],(float)c.values[1],(float)c.values[2],(float)c.values[3],
          (float)c.values[4],(float)c.values[5],(float)c.values[6],(float)c.values[7],
          (float)c.values[8],(float)c.values[9],(float)c.values[10],(float)c.values[11],m);break;
      case DrawCmd::Type::MulAddRect:
        if(!visibleRect(d,c.x,c.y,c.w,c.h))return;
        LICE_MultiplyAddRect(d,x,y,w,h,(float)c.values[0],(float)c.values[1],(float)c.values[2],(float)c.values[3],
          (float)(c.values[4]*255),(float)(c.values[5]*255),(float)(c.values[6]*255),(float)(c.values[7]*255));break;
      case DrawCmd::Type::Blur:
        if(!visibleRect(d,c.x,c.y,c.w,c.h))return;
        LICE_Blur(d,d,x,y,x,y,w,h);break;
      case DrawCmd::Type::Blit:
      case DrawCmd::Type::DeltaBlit:
      case DrawCmd::Type::TransformBlit: {
        if(!sane(c.destX)||!sane(c.destY)||!sane(c.destW)||!sane(c.destH)
            ||!sane(c.srcX,32760)||!sane(c.srcY,32760)||!sane(c.srcW,32760)||!sane(c.srcH,32760)
            ||!sane(c.rotation)||!sane(c.rotationXOffset,32760)||!sane(c.rotationYOffset,32760))return;
        const int dx=boundedGfxInt(c.destX),dy=boundedGfxInt(c.destY),dw=boundedGfxInt(c.destW),dh=boundedGfxInt(c.destH);
        if(!dw||!dh)return;
        if(c.type==DrawCmd::Type::TransformBlit) {
          if(c.divisionsX<2||c.divisionsX>64||c.divisionsY<2||c.divisionsY>64
              ||c.transformPoints.size()!=(size_t)c.divisionsX*c.divisionsY*2)return;
          for(double v:c.transformPoints)if(!sane(v,32760))return;
          LICE_TransformBlit2(d,src,dx,dy,dw,dh,const_cast<double*>(c.transformPoints.data()),c.divisionsX,c.divisionsY,c.opacity,bm);
        } else if(c.type==DrawCmd::Type::DeltaBlit) {
          for(int i=0;i<6;++i)if(!sane(c.values[(size_t)i],32760))return;
          // Bound LICE's fixed-point conversion BEFORE it executes, including
          // the mixed derivative at the far corners of the destination.
          for(double xx:{0.0,(double)dw}) for(double yy:{0.0,(double)dh})
            if(!sane(c.srcX+xx*c.values[0]+yy*c.values[2]+xx*yy*c.values[4],32760)
              ||!sane(c.srcY+xx*c.values[1]+yy*c.values[3]+xx*yy*c.values[5],32760))return;
          LICE_DeltaBlit(d,src,dx,dy,dw,dh,c.srcX,c.srcY,c.srcW,c.srcH,
              c.values[0],c.values[1],c.values[2],c.values[3],c.values[4],c.values[5],c.useClipRect,c.opacity,bm);
        } else if(std::abs(c.rotation)>1e-9f)
          LICE_RotatedBlit(d,src,dx,dy,dw,dh,c.srcX,c.srcY,c.srcW,c.srcH,c.rotation,true,c.opacity,bm,c.rotationXOffset,c.rotationYOffset);
        else LICE_ScaledBlit(d,src,dx,dy,dw,dh,c.srcX,c.srcY,c.srcW,c.srcH,c.opacity,bm);
        ++stats.blits;break;
      }
      default:drawn=false;break;
    }
    if(drawn && c.dest==-1)stats.mainFramebufferChanged=true;
  }

  juce::Image& framebuffer;
  std::function<void(bool)> prepare;
  bool mainReady=false;
  size_t nextCommand=0;
  std::unique_ptr<MappedSurface> mainMap;
  std::array<std::unique_ptr<MappedSurface>,GfxImageBank::kNumImages> maps {};
  GfxImageBank* activeBank=nullptr;
  GfxTextMaskCache fallbackTextCache;
  GfxRenderStats stats;
};

static inline GfxRenderStats paintCommands(juce::Image& framebuffer,const std::vector<DrawCmd>& commands)
{
  GfxRenderSession session(framebuffer);
  return session.finish(commands);
}
#endif
