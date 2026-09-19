// SPDX-License-Identifier: Zlib
// Bounded PNG-to-straight-ARGB decoder. Unlike a premultiplied Image decoder,
// this retains RGB under alpha=0 for JSFX's ignore-source-alpha copy mode.
// Inflate is supplied by the host; production uses JUCE's zlib stream.
#pragma once
#include <array>
#include <algorithm>
#include <cstdlib>
#include <utility>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
namespace jsfx_gfx_resources {
struct PngPixels {int width=0,height=0;std::vector<uint32_t> argb;};
inline uint32_t pngBE32(const unsigned char* p){return (uint32_t(p[0])<<24)|(uint32_t(p[1])<<16)|(uint32_t(p[2])<<8)|p[3];}
inline uint32_t pngCRC(const unsigned char* p,size_t n)
{
  static const auto table=[](){std::array<uint32_t,256> t{};for(uint32_t i=0;i<256;++i){uint32_t v=i;for(int k=0;k<8;++k)v=(v>>1)^(0xedb88320u&uint32_t(-int(v&1)));t[i]=v;}return t;}();
  uint32_t crc=~0u;while(n--)crc=table[(crc^*p++)&255]^(crc>>8);return ~crc;
}
inline uint32_t pngAdler(const std::vector<unsigned char>& data)
{
  uint32_t a=1,b=0;size_t i=0;
  while(i<data.size()){const size_t end=std::min(i+5552,data.size());for(;i<end;++i){a+=data[i];b+=a;}a%=65521;b%=65521;}
  return (b<<16)|a;
}
inline int pngPaeth(int a,int b,int c){int p=a+b-c,pa=std::abs(p-a),pb=std::abs(p-b),pc=std::abs(p-c);return pa<=pb&&pa<=pc?a:pb<=pc?b:c;}

template<class Inflate>
bool decodePng(const void* bytes,size_t size,PngPixels& output,Inflate inflate)
{
  output={};
  static const unsigned char signature[8]={137,80,78,71,13,10,26,10};
  if(!bytes||size<33||size>64u*1024u*1024u||std::memcmp(bytes,signature,8))return false;
  const auto* data=static_cast<const unsigned char*>(bytes);
  uint32_t width=0,height=0;int depth=0,type=-1,interlace=0,channels=0;
  std::vector<unsigned char> compressed,palette,transparency;
  bool gotHeader=false,gotData=false,endedData=false,gotEnd=false,gotPalette=false,gotTransparency=false;
  size_t pos=8;
  while(pos+12<=size) {
    const uint32_t n=pngBE32(data+pos);if(n>size-pos-12)return false;
    const auto* tag=data+pos+4;const auto* p=tag+4;
    if(pngCRC(tag,(size_t)n+4)!=pngBE32(p+n))return false;
    if(!gotHeader && std::memcmp(tag,"IHDR",4))return false;
    if(!std::memcmp(tag,"IHDR",4)) {
      if(gotHeader||n!=13)return false;gotHeader=true;width=pngBE32(p);height=pngBE32(p+4);depth=p[8];type=p[9];interlace=p[12];
      if(!width||!height||width>8192||height>8192||(uint64_t)width*height>16777216u||p[10]||p[11]||interlace>1)return false;
      channels=type==0?1:type==2?3:type==3?1:type==4?2:type==6?4:0;
      if(!channels)return false;
      const bool allowed=type==0?(depth==1||depth==2||depth==4||depth==8||depth==16):type==3?(depth==1||depth==2||depth==4||depth==8):(depth==8||depth==16);
      if(!allowed)return false;
    } else if(!std::memcmp(tag,"PLTE",4)) {
      if(gotData||gotPalette||n==0||n>768||n%3||type==0||type==4)return false;
      palette.assign(p,p+n);gotPalette=true;
    } else if(!std::memcmp(tag,"tRNS",4)) {
      if(gotData||gotTransparency||type==4||type==6)return false;
      if((type==0&&n!=2)||(type==2&&n!=6)||(type==3&&(!gotPalette||n>palette.size()/3)))return false;
      transparency.assign(p,p+n);gotTransparency=true;
    } else if(!std::memcmp(tag,"IDAT",4)) {
      if(endedData||(type==3&&!gotPalette)||compressed.size()+n>64u*1024u*1024u)return false;
      gotData=true;compressed.insert(compressed.end(),p,p+n);
    } else if(!std::memcmp(tag,"IEND",4)) {
      if(n||!gotData)return false;gotEnd=true;pos+=12;break;
    } else {
      if(!(tag[0]&32))return false; // Unknown critical chunk.
      if(gotData)endedData=true;
    }
    pos+=(size_t)n+12;
  }
  if(!gotEnd||compressed.size()<6||pos!=size)return false;
  if((compressed[0]&15)!=8||(compressed[0]>>4)>7||(compressed[1]&32)||((int)compressed[0]*256+compressed[1])%31)return false;
  const int sx[7]={0,4,0,2,0,1,0},sy[7]={0,0,4,0,2,0,1},dx[7]={8,8,4,4,2,2,1},dy[7]={8,8,8,4,4,2,2};
  const int passes=interlace?7:1;
  size_t expected=0;
  for(int pass=0;pass<passes;++pass) {
    const uint32_t x0=interlace?sx[pass]:0,y0=interlace?sy[pass]:0,xs=interlace?dx[pass]:1,ys=interlace?dy[pass]:1;
    const uint32_t pw=width>x0?(width-x0+xs-1)/xs:0,ph=height>y0?(height-y0+ys-1)/ys:0;
    if(pw&&ph)expected+=(1+((size_t)pw*channels*depth+7)/8)*ph;
  }
  if(expected>144u*1024u*1024u)return false;
  std::vector<unsigned char> raw;
  if(!inflate(compressed,expected,raw)||raw.size()!=expected||pngAdler(raw)!=pngBE32(compressed.data()+compressed.size()-4))return false;
  PngPixels result;result.width=(int)width;result.height=(int)height;result.argb.resize((size_t)width*height);
  size_t cursor=0;
  for(int pass=0;pass<passes;++pass) {
    const uint32_t x0=interlace?sx[pass]:0,y0=interlace?sy[pass]:0,xs=interlace?dx[pass]:1,ys=interlace?dy[pass]:1;
    const uint32_t pw=width>x0?(width-x0+xs-1)/xs:0,ph=height>y0?(height-y0+ys-1)/ys:0;
    if(!pw||!ph)continue;
    const size_t rowBytes=((size_t)pw*channels*depth+7)/8,bpp=std::max(1,(channels*depth+7)/8);
    std::vector<unsigned char> prev(rowBytes),row(rowBytes);
    for(uint32_t y=0;y<ph;++y) {
      const unsigned int filter=raw[cursor++];if(filter>4)return false;
      for(size_t i=0;i<rowBytes;++i) {
        const int a=i>=bpp?row[i-bpp]:0,b=prev[i],c=i>=bpp?prev[i-bpp]:0;
        const int add=filter==0?0:filter==1?a:filter==2?b:filter==3?(a+b)/2:pngPaeth(a,b,c);
        row[i]=(unsigned char)(raw[cursor++]+add);
      }
      const auto sample=[&](size_t i)->int{
        if(depth==16)return ((int)row[i*2]<<8)|row[i*2+1];
        if(depth==8)return row[i];
        const size_t bit=i*depth;return (row[bit/8]>>(8-depth-(bit%8)))&((1<<depth)-1);
      };
      const auto eight=[&](int v){return depth==16?v>>8:depth==8?v:v*255/((1<<depth)-1);};
      const auto key=[&](size_t i){return (int)transparency[i]*256+transparency[i+1];};
      for(uint32_t x=0;x<pw;++x) {
        const size_t i=(size_t)x*channels;int r=0,g=0,b=0,a=255;
        if(type==3){int idx=sample(i);if((size_t)idx>=palette.size()/3)return false;r=palette[(size_t)idx*3];g=palette[(size_t)idx*3+1];b=palette[(size_t)idx*3+2];if((size_t)idx<transparency.size())a=transparency[(size_t)idx];}
        else if(type==0||type==4){int v=sample(i);r=g=b=eight(v);if(type==4)a=eight(sample(i+1));else if(!transparency.empty()&&v==key(0))a=0;}
        else {const int rv=sample(i),gv=sample(i+1),bv=sample(i+2);r=eight(rv);g=eight(gv);b=eight(bv);if(type==6)a=eight(sample(i+3));else if(!transparency.empty()&&rv==key(0)&&gv==key(2)&&bv==key(4))a=0;}
        result.argb[(size_t)(y0+y*ys)*width+x0+x*xs]=(uint32_t(a)<<24)|(uint32_t(r)<<16)|(uint32_t(g)<<8)|uint32_t(b);
      }
      prev.swap(row);
    }
  }
  if(cursor!=raw.size())return false;
  output=std::move(result);return true;
}
} // namespace jsfx_gfx_resources
