// SPDX-License-Identifier: Zlib
// TEST DOUBLE ONLY. Models Image sharing, raster ordering and simple rectangles.
// This is NOT JUCE, not a font/AA oracle, and not a performance benchmark.
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
namespace juce {
using uint8 = uint8_t;
template<class... T> void ignoreUnused(const T&...) {}
template<class T> T jlimit(T lo,T hi,T x) {return std::max(lo,std::min(hi,x));}
struct ScopedNoDenormals {};
class String {
public:
 std::string value;
 String()=default; String(const char* p):value(p?p:""){} String(std::string v):value(std::move(v)){}
 String(double v,int places){char b[128];std::snprintf(b,sizeof b,"%.*f",places,v);value=b;}
 static String fromUTF8(const char* p,int n=-1){return n<0?String(p):String(std::string(p,(size_t)n));}
 int length()const{return (int)value.size();} bool isEmpty()const{return value.empty();}
 bool isNotEmpty()const{return !isEmpty();} char operator[](int i)const{return value[(size_t)i];}
 const char* toRawUTF8()const{return value.c_str();}
 String substring(int start,int end)const{return String(value.substr((size_t)start,(size_t)std::max(0,end-start)));}
 bool operator==(const String& o)const{return value==o.value;}
 friend String operator+(const String&a,const String&b){return String(a.value+b.value);}
};
class StringArray {
 std::vector<String> items;
public:
 void addLines(const String&s){size_t p=0;while(p<s.value.size()){auto e=s.value.find('\n',p);if(e==std::string::npos)e=s.value.size();items.emplace_back(s.value.substr(p,e-p));p=e+1;}}
 bool isEmpty()const{return items.empty();} int size()const{return (int)items.size();}
 String operator[](int i)const{return items[(size_t)i];}
};
class Font {
 float size=12;
public:
 enum {plain=0,bold=1,italic=2,underlined=4};
 Font()=default;Font(const String&,float s,int):size(s){}
 static String getDefaultSansSerifFontName(){return "sans";}
 float getHeight()const{return size;} float getStringWidthFloat(const String&s)const{return (float)s.length()*size*.5f;}
};
class Colour {
 uint32_t rgba=0xff000000u;
public:
 Colour()=default;Colour(uint32_t v):rgba(v){}
 static Colour fromRGB(uint8 r,uint8 g,uint8 b){return Colour(0xff000000u|(uint32_t(r)<<16)|(uint32_t(g)<<8)|b);}
 static Colour fromFloatRGBA(float r,float g,float b,float a){return Colour((uint32_t(jlimit(0.f,1.f,a)*255+.5f)<<24)|(uint32_t(jlimit(0.f,1.f,r)*255+.5f)<<16)|(uint32_t(jlimit(0.f,1.f,g)*255+.5f)<<8)|uint32_t(jlimit(0.f,1.f,b)*255+.5f));}
 bool isOpaque()const{return (rgba>>24)==255;} uint32_t getARGB()const{return rgba;}
 bool operator==(const Colour&o)const{return rgba==o.rgba;}
};
namespace Colours { inline const Colour black(0xff000000),red(0xffff0000),blue(0xff0000ff),green(0xff00ff00),white(0xffffffff); }
template<class T> struct Point{T x{},y{};Point()=default;Point(T a,T b):x(a),y(b){}};
template<class T> class Rectangle {
 T x{},y{},w{},h{};
public:
 Rectangle()=default;Rectangle(T a,T b,T c,T d):x(a),y(b),w(c),h(d){}
 T getX()const{return x;}T getY()const{return y;}T getWidth()const{return w;}T getHeight()const{return h;}
 bool isEmpty()const{return w<=0||h<=0;}
 Rectangle getIntersection(const Rectangle&o)const{T a=std::max(x,o.x),b=std::max(y,o.y);return {a,b,std::max(T(0),std::min(x+w,o.x+o.w)-a),std::max(T(0),std::min(y+h,o.y+o.h)-b)};}
};
struct Justification {
 enum {topLeft,centred,centredBottom,centredTop,centredRight,bottomRight,topRight,centredLeft,bottomLeft};
 int v;Justification(int i=topLeft):v(i){}
};
struct AffineTransform {float a,b,c,d,e,f;AffineTransform(float aa,float bb,float cc,float dd,float ee,float ff):a(aa),b(bb),c(cc),d(dd),e(ee),f(ff){}};
struct Path {void startNewSubPath(float,float){}void startNewSubPath(Point<float>){}void lineTo(float,float){}void lineTo(Point<float>){}void closeSubPath(){}};
struct PathStrokeType{explicit PathStrokeType(float){}};
struct SoftwareImageType {};
class Image {
 struct Data {int width,height,writers=0;std::vector<uint32_t> pixels;Data(int w,int h):width(w),height(h),pixels((size_t)w*h){}};
 std::shared_ptr<Data> data;int ox=0,oy=0,w=0,h=0;
 friend class Graphics;
public:
 enum PixelFormat {UnknownFormat,RGB,ARGB,SingleChannel};
 Image()=default;Image(PixelFormat,int width,int height,bool):data(std::make_shared<Data>(width,height)),w(width),h(height){}
 Image(PixelFormat f,int a,int b,bool clear,SoftwareImageType):Image(f,a,b,clear){}
 bool isValid()const{return data!=nullptr;}bool isNull()const{return !isValid();}
 int getWidth()const{return w;}int getHeight()const{return h;}PixelFormat getFormat()const{return ARGB;}
 Rectangle<int> getBounds()const{return {0,0,w,h};}
 int getReferenceCount()const{return (int)data.use_count();}
 bool operator==(const Image&o)const{return data==o.data;}
 Colour getPixelAt(int x,int y)const{return Colour(data->pixels[(size_t)(oy+y)*data->width+ox+x]);}
 void setPixelAt(int x,int y,Colour c){data->pixels[(size_t)(oy+y)*data->width+ox+x]=c.getARGB();}
 void clear(Rectangle<int> r,Colour c=Colour(0)){r=r.getIntersection(getBounds());for(int y=r.getY();y<r.getY()+r.getHeight();++y)for(int x=r.getX();x<r.getX()+r.getWidth();++x)setPixelAt(x,y,c);}
 Image createCopy()const{assert(data->writers==0);Image out(ARGB,w,h,true);for(int y=0;y<h;++y)for(int x=0;x<w;++x)out.setPixelAt(x,y,getPixelAt(x,y));return out;}
 Image getClippedImage(Rectangle<int> r)const{r=r.getIntersection(getBounds());Image out=*this;out.ox+=r.getX();out.oy+=r.getY();out.w=r.getWidth();out.h=r.getHeight();return out;}
 struct BitmapData;
};
struct Image::BitmapData {
  enum ReadWriteMode{readOnly,writeOnly,readWrite};
  Image image;int pixelStride=4,lineStride=0;
  BitmapData(const Image&i,ReadWriteMode):image(i),lineStride(i.data->width*4){assert(i.data->writers==0);}
  uint8* getLinePointer(int y)const{return reinterpret_cast<uint8*>(image.data->pixels.data()+(size_t)(image.oy+y)*image.data->width+image.ox);}
 };

class Graphics {
 Image image;Colour colour;float opacity=1;std::vector<float> saves;
 void blend(int x,int y,Colour c,float a=1){if(x<0||y<0||x>=image.w||y>=image.h)return;auto src=c.getARGB();float sa=(src>>24)/255.f*a;auto dst=image.getPixelAt(x,y).getARGB();auto ch=[&](int s){return (uint32_t)jlimit(0.f,255.f,((src>>s)&255)*sa+((dst>>s)&255)*(1-sa)+.5f);};uint32_t alpha=(uint32_t)jlimit(0.f,255.f,sa*255+(dst>>24)*(1-sa)+.5f);image.setPixelAt(x,y,Colour((alpha<<24)|(ch(16)<<16)|(ch(8)<<8)|ch(0)));}
public:
 enum ResamplingQuality{lowResamplingQuality,mediumResamplingQuality};
 explicit Graphics(const Image&i):image(i){assert(i.isValid());++image.data->writers;}
 ~Graphics(){--image.data->writers;}
 bool clipRegionIntersects(Rectangle<int> r)const{return !r.getIntersection(image.getBounds()).isEmpty();}
 void setColour(Colour c){colour=c;opacity=1;}void setFont(const Font&){}void setOpacity(float o){opacity=o;}
 void setImageResamplingQuality(ResamplingQuality){}
 void saveState(){saves.push_back(opacity);}void restoreState(){opacity=saves.back();saves.pop_back();}
 void fillRect(float x,float y,float w,float h){auto r=Rectangle<int>((int)std::floor(x),(int)std::floor(y),(int)std::ceil(w),(int)std::ceil(h)).getIntersection(image.getBounds());for(int yy=r.getY();yy<r.getY()+r.getHeight();++yy)for(int xx=r.getX();xx<r.getX()+r.getWidth();++xx)blend(xx,yy,colour);}
 void drawRect(float x,float y,float w,float h,float){fillRect(x,y,w,1);fillRect(x,y+h,w,1);fillRect(x,y,1,h);fillRect(x+w,y,1,h);}
 void drawLine(float,float,float,float,float){}void drawText(const String&,Rectangle<int>,Justification,bool){}
 void drawText(const String&,int,int,int,int,Justification,bool){}void fillEllipse(float,float,float,float){}void drawEllipse(float,float,float,float,float){}
 void fillRoundedRectangle(Rectangle<float>,float){}void drawRoundedRectangle(Rectangle<float>,float,float){}
 void strokePath(const Path&,PathStrokeType){}void fillPath(const Path&){}
 void drawImage(const Image&src,int dx,int dy,int dw,int dh,int sx,int sy,int sw,int sh,bool){assert(src.data->writers==0);auto r=Rectangle<int>(dx,dy,dw,dh).getIntersection(image.getBounds());for(int y=r.getY();y<r.getY()+r.getHeight();++y)for(int x=r.getX();x<r.getX()+r.getWidth();++x){int a=sx+(int)((double)(x-dx)*sw/dw),b=sy+(int)((double)(y-dy)*sh/dh);if(a>=0&&b>=0&&a<src.w&&b<src.h)blend(x,y,src.getPixelAt(a,b),opacity);}}
 void drawImageTransformed(const Image&src,const AffineTransform&t,bool){assert(src.data->writers==0);float det=t.a*t.e-t.b*t.d;if(std::abs(det)<1e-10)return;for(int y=0;y<image.h;++y)for(int x=0;x<image.w;++x){float xx=x+.5f-t.c,yy=y+.5f-t.f;int sx=(int)std::floor((t.e*xx-t.b*yy)/det),sy=(int)std::floor((-t.d*xx+t.a*yy)/det);if(sx>=0&&sy>=0&&sx<src.w&&sy<src.h)blend(x,y,src.getPixelAt(sx,sy),opacity);}}
};
}
