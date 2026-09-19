// SPDX-License-Identifier: Zlib
#pragma once
#include "JsfxGfxPng.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#if defined(_WIN32)
 #include <windows.h>
#else
 #include <dlfcn.h>
#endif

namespace jsfx_gfx_resources {
struct EmbeddedResource {const char* name;const unsigned char* bytes;size_t size;};
}
#if __has_include("JSFXResources.h")
 #include "JSFXResources.h"
#else
 namespace jsfx_gfx_resources {
 static constexpr const EmbeddedResource* kEntries=nullptr;
 static constexpr size_t kEntryCount=0;
 static constexpr const char* kSourceDir="";
 }
#endif
namespace jsfx_gfx_resources {
inline std::string normalizeName(std::string name)
{
  std::replace(name.begin(),name.end(),'\\','/');
  while(name.compare(0,2,"./")==0)name.erase(0,2);
  if(name.compare(0,13,"../Resources/")==0)name.erase(0,13);
  else if(name.compare(0,10,"Resources/")==0)name.erase(0,10);
  std::vector<std::string> parts;size_t pos=0;
  while(pos<=name.size()) {
    size_t end=name.find('/',pos);if(end==std::string::npos)end=name.size();
    const auto p=name.substr(pos,end-pos);
    if(p==".."&&!parts.empty()&&parts.back()!="..")parts.pop_back();
    else if(!p.empty()&&p!=".")parts.push_back(p);
    pos=end+1;
  }
  std::string out;for(const auto& p:parts){if(!out.empty())out+='/';out+=p;}return out;
}
inline std::filesystem::path moduleDirectory()
{
#if defined(_WIN32)
  HMODULE module=nullptr;
  if(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      reinterpret_cast<LPCWSTR>(&moduleDirectory),&module)) {
    std::vector<wchar_t> path(32768);
    DWORD n=GetModuleFileNameW(module,path.data(),(DWORD)path.size());
    if(n && n<path.size())return std::filesystem::path(std::wstring(path.data(),n)).parent_path();
  }
#else
  Dl_info info{};
  if(dladdr(reinterpret_cast<void*>(&moduleDirectory),&info)&&info.dli_fname)
    return std::filesystem::u8path(info.dli_fname).parent_path();
#endif
  return {};
}
inline bool inflatePng(const std::vector<unsigned char>& input,size_t size,std::vector<unsigned char>& output)
{
  juce::MemoryInputStream memory(input.data(),input.size(),false);
  juce::GZIPDecompressorInputStream stream(&memory,false,juce::GZIPDecompressorInputStream::zlibFormat);
  output.resize(size);size_t done=0;
  while(done<size){const int n=stream.read(output.data()+done,(int)std::min<size_t>(size-done,1048576));if(n<=0)return false;done+=(size_t)n;}
  unsigned char extra=0;return stream.read(&extra,1)==0;
}
inline bool safeOtherImageDimensions(const unsigned char* p,size_t n)
{
  const auto valid=[](uint32_t w,uint32_t h){return w>0&&h>0&&w<=8192&&h<=8192&&(uint64_t)w*h<=16777216;};
  if(n>=10&&(!std::memcmp(p,"GIF87a",6)||!std::memcmp(p,"GIF89a",6)))return valid(p[6]|uint32_t(p[7])<<8,p[8]|uint32_t(p[9])<<8);
  if(n<4||p[0]!=255||p[1]!=216)return false;
  size_t i=2;
  while(i<n){if(p[i++]!=255)return false;while(i<n&&p[i]==255)++i;if(i>=n)return false;const int marker=p[i++];
    if(marker==217||marker==218)return false;
    if(marker==1||(marker>=208&&marker<=215))continue;
    if(i+2>n)return false;size_t length=(size_t)p[i]*256+p[i+1];if(length<2||length>n-i)return false;
    if(marker>=192&&marker<=207&&marker!=196&&marker!=200&&marker!=204){if(length<8)return false;return valid(uint32_t(p[i+5])*256+p[i+6],uint32_t(p[i+3])*256+p[i+4]);}
    i+=length;
  }
  return false;
}
inline juce::Image decodeImage(const void* data,size_t size)
{
  if(!data||!size||size>64u*1024u*1024u)return {};
  const auto* p=static_cast<const unsigned char*>(data);
  if(size>=8&&p[0]==137&&!std::memcmp(p+1,"PNG\r\n\032\n",7)) {
    PngPixels decoded;
    if(!decodePng(data,size,decoded,inflatePng))return {};
    juce::Image image(juce::Image::ARGB,decoded.width,decoded.height,true,juce::SoftwareImageType());
    juce::Image::BitmapData bits(image,juce::Image::BitmapData::writeOnly);
    if(bits.pixelStride!=4)return {};
    for(int y=0;y<decoded.height;++y)std::memcpy(bits.getLinePointer(y),decoded.argb.data()+(size_t)y*decoded.width,(size_t)decoded.width*4);
    return image;
  }
  if(!safeOtherImageDimensions(p,size))return {};
  auto decoded=juce::ImageFileFormat::loadFrom(data,size);
  if(!decoded.isValid()||decoded.getWidth()>8192||decoded.getHeight()>8192)return {};
  juce::Image image(juce::Image::ARGB,decoded.getWidth(),decoded.getHeight(),true,juce::SoftwareImageType());
  juce::Image::BitmapData bits(image,juce::Image::BitmapData::writeOnly);
  for(int y=0;y<image.getHeight();++y)for(int x=0;x<image.getWidth();++x){const uint32_t pixel=decoded.getPixelAt(x,y).getARGB();std::memcpy(bits.getLinePointer(y)+x*4,&pixel,4);}
  return image;
}
inline juce::Image loadImage(const juce::String& filename)
{
  try {
    const std::string name=filename.toRawUTF8();if(name.empty()||name.size()>32768)return {};
    const auto path=std::filesystem::u8path(name);
    const auto normalized=normalizeName(name);
    // Embedded assets are immutable; gfx_loadimg always decodes a fresh writable slot.
    if(!path.is_absolute())for(size_t i=0;i<kEntryCount;++i)
      if(normalized==kEntries[i].name)return decodeImage(kEntries[i].bytes,kEntries[i].size);
    std::vector<std::filesystem::path> candidates;
    if(path.is_absolute())candidates.push_back(path);
    else {
      const auto module=moduleDirectory();
      if(!module.empty()) {
        candidates.push_back(module/"Resources"/std::filesystem::u8path(normalized));
        candidates.push_back(module.parent_path()/"Resources"/std::filesystem::u8path(normalized));
        candidates.push_back(module.parent_path().parent_path()/"Resources"/std::filesystem::u8path(normalized));
      }
      if(*kSourceDir) {
        const auto source=std::filesystem::u8path(kSourceDir);
        candidates.push_back(source/path);
        candidates.push_back(source.parent_path()/"Resources"/std::filesystem::u8path(normalized));
      }
    }
    for(const auto& candidate:candidates) {
      std::error_code ec;const auto size=std::filesystem::file_size(candidate,ec);
      if(ec||!size||size>64u*1024u*1024u)continue;
      std::ifstream stream(candidate,std::ios::binary);if(!stream)continue;
      std::vector<unsigned char> bytes((size_t)size);
      if(!stream.read(reinterpret_cast<char*>(bytes.data()),(std::streamsize)size))continue;
      auto image=decodeImage(bytes.data(),bytes.size());if(image.isValid())return image;
    }
  } catch(const std::exception&) {}
  return {};
}
inline juce::Image cursorImage(const juce::String& name)
{
  auto raw=loadImage(name);
  if(!raw.isValid()||raw.getWidth()>256||raw.getHeight()>256)return {};
  juce::Image premultiplied(juce::Image::ARGB,raw.getWidth(),raw.getHeight(),true,juce::SoftwareImageType());
  juce::Image::BitmapData data(raw,juce::Image::BitmapData::readOnly);
  for(int y=0;y<raw.getHeight();++y)for(int x=0;x<raw.getWidth();++x){uint32_t p=0;std::memcpy(&p,data.getLinePointer(y)+x*4,4);premultiplied.setPixelAt(x,y,juce::Colour(p));}
  return premultiplied;
}
} // namespace jsfx_gfx_resources
