// SPDX-License-Identifier: Zlib
// Include after JSFXDSP.h and the drawing formatter. This context belongs to
// the processor, survives editor closure, and is shared by DSP and native GFX.
#pragma once
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#if DSPJSFX_NATIVE_GFX_LEGACY
namespace jsfx_native_gfx {
class Frame;
class Strings {
public:
    Frame* graphics=nullptr; // lifetime is the processor's graphics context
    std::mutex mutex;
    std::unordered_map<int64_t,std::string> values;
    std::unordered_set<int64_t> literals;
    static constexpr size_t lengthHint=65536;
    int64_t nextHandle=int64_t(1)<<48;
    static int64_t key(double v) noexcept {
        return std::isfinite(v) && v>=-9007199254740991. && v<=9007199254740991. ? (int64_t)(v+0.5) : -1;
    }
    static int integer(double v) noexcept { return jsfx_gfx::boundedGfxInt(v); }
    void reset() {
        std::lock_guard lock(mutex); values.clear(); literals.clear(); nextHandle=int64_t(1)<<48;
        for(int i=0;i<DSPJSFX_STRING_LITERALS_COUNT;++i) {
            const auto& s=DSPJSFX_STRING_LITERALS[i];
            values.emplace(s.handle,std::string((const char*)s.data,(size_t)s.length));
            literals.insert(s.handle);
        }
    }
    Strings(){reset();}
    std::string read(double handle) {
        std::lock_guard lock(mutex); return readUnlocked(handle);
    }
    void write(double handle,std::string text) {
        std::lock_guard lock(mutex);
        if(writable(handle)) values[key(handle)]=std::move(text);
    }
    double create(std::string text) {
        std::lock_guard lock(mutex); const auto handle=nextHandle++;
        values[handle]=std::move(text);return (double)handle;
    }
    bool writable(double handle) const {
        const auto k=key(handle);
        return k>=0 && !literals.count(k);
    }
    std::string readUnlocked(double handle) const {
        auto it=values.find(key(handle));return it!=values.end()?it->second:std::string();
    }
    double execute(int opcode,const double* a,int n);
};
inline Strings* stringContext(DSPJSFX_State* s) noexcept {
    return s?static_cast<Strings*>(s->nativeStrings):nullptr;
}
// Binary string types follow WDL/eel2/eel_strings.h (packed 'uc', 'US', etc).
inline int stringType(int type) {
    int flag=0;
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    flag=16;
#endif
    if(std::toupper((type>>8)&255)=='U')flag|=32;
    else if(type>255 && std::toupper(type&255)=='U'){flag|=32;type>>=8;}
    type&=255;
    if(std::isupper(type))flag^=16;else type+='A'-'a';
    switch(type){case 'F':return flag|4|64;case 'D':return flag|8|64;
        case 'S':return flag|2;case 'I':return flag|4;default:return flag|1;}
}
inline double stringNumber(int flag,const char* src) {
    union {char b[8];float f;double d;int32_t i;int16_t s;int8_t c;uint32_t u;uint16_t us;uint8_t uc;} v{};
    const int size=flag&15;
    if(flag&16)for(int i=0;i<size;++i)v.b[i]=src[size-1-i];else std::memcpy(v.b,src,(size_t)size);
    if(flag&64)return size==8?v.d:v.f;
    if(flag&32)return size==4?v.u:size==2?v.us:v.uc;
    return size==4?v.i:size==2?v.s:v.c;
}
inline void stringPutNumber(int flag,char* dest,double value) {
    union {char b[8];float f;double d;int32_t i;int16_t s;int8_t c;uint32_t u;uint16_t us;uint8_t uc;} v{};
    const int size=flag&15;
    if(flag&64){if(size==8)v.d=value;else v.f=(float)value;}
    else {
        // Integer conversion wraps without invoking C++ out-of-range casts.
        double raw=std::isfinite(value)?std::fmod(std::trunc(value),4294967296.):0;
        if(raw<0)raw+=4294967296.;const uint32_t u=(uint32_t)raw;
        if(size==4)v.u=u;else if(size==2)v.us=(uint16_t)u;else v.uc=(uint8_t)u;
    }
    if(flag&16)for(int i=0;i<size;++i)dest[i]=v.b[size-1-i];else std::memcpy(dest,v.b,(size_t)size);
}
inline double Strings::execute(int opcode,const double* a,int n) {
    std::lock_guard lock(mutex);
    if(n<1)return 0;
    const double dest=a[0];
    auto first=readUnlocked(dest);
    auto second=n>1?readUnlocked(a[1]):std::string();
    const auto set=[&](std::string text){if(writable(dest))values[key(dest)]=std::move(text);return dest;};
    switch(opcode) {
    case DSPJSFX_STRING_STRLEN:return (double)first.size();
    case DSPJSFX_STRING_STRCPY:return set(std::move(second));
    case DSPJSFX_STRING_STRNCPY:
        if(a[2]>0 || a[2]==0 && key(dest)==key(a[1]))second.resize(std::min(second.size(),(size_t)integer(a[2])));
        return set(std::move(second));
    case DSPJSFX_STRING_STRCAT:case DSPJSFX_STRING_STRNCAT:
        if(first.size()>lengthHint)return dest;
        if(opcode==DSPJSFX_STRING_STRNCAT && a[2]>0)second.resize(std::min(second.size(),(size_t)integer(a[2])));
        return set(first+second);
    case DSPJSFX_STRING_STRCMP:case DSPJSFX_STRING_STRICMP:
    case DSPJSFX_STRING_STRNCMP:case DSPJSFX_STRING_STRNICMP: {
        bool ci=opcode==DSPJSFX_STRING_STRICMP || opcode==DSPJSFX_STRING_STRNICMP;
        size_t limit=n>2 && a[2]>=0?(size_t)integer(a[2]):std::max(first.size(),second.size())+1;
        for(size_t i=0;i<limit;++i) {
            unsigned char x=i<first.size()?(unsigned char)first[i]:0,y=i<second.size()?(unsigned char)second[i]:0;
            if(ci){x=(unsigned char)std::tolower(x);y=(unsigned char)std::tolower(y);}
            if(x!=y)return (signed char)x<(signed char)y?-1:1;if(!x && i>=first.size() && i>=second.size())break;
        }
        return 0;
    }
    case DSPJSFX_STRING_STR_GETCHAR: {
        int offset=integer(a[1]);if(a[1]<0)offset+=(int)first.size();
        int flag=n>2?stringType(integer(a[2])):33;
        return offset>=0 && (size_t)offset+(flag&15)<=first.size()?stringNumber(flag,first.data()+offset):0;
    }
    case DSPJSFX_STRING_STR_SETCHAR: {
        int offset=integer(a[1]);if(a[1]<0)offset+=(int)first.size();
        int flag=n>3?stringType(integer(a[3])):33;
        if(offset<0 || (size_t)offset>first.size())return dest;
        if((size_t)offset==first.size()) {
            if(first.size()>lengthHint)return dest;
            first.resize(first.size()+(flag&15));
        }
        char bytes[8]{};stringPutNumber(flag,bytes,a[2]);
        std::memcpy(first.data()+offset,bytes,std::min((size_t)(flag&15),first.size()-(size_t)offset));
        return set(std::move(first));
    }
    case DSPJSFX_STRING_STR_SETLEN:
        first.resize((size_t)std::clamp(integer(a[1]),0,(int)lengthHint),' ');return set(std::move(first));
    case DSPJSFX_STRING_STRCPY_SUBSTR:case DSPJSFX_STRING_STRCPY_FROM: {
        int offset=integer(a[2]);if(offset<0)offset=std::max(0,(int)second.size()+offset);
        int length=std::max(0,(int)second.size()-offset);
        if(n>3)length=a[3]<0?std::max(0,length+integer(a[3])):std::min(length,integer(a[3]));
        return set((size_t)offset<second.size()?second.substr((size_t)offset,(size_t)length):std::string());
    }
    case DSPJSFX_STRING_STR_INSERT: {
        int offset=integer(a[2]);if(offset<0){second=second.substr(std::min(second.size(),(size_t)-offset));offset=0;}
        if(first.size()<=lengthHint)first.insert(std::min(first.size(),(size_t)offset),second);
        return set(std::move(first));
    }
    case DSPJSFX_STRING_STR_DELSUB: {
        int offset=integer(a[1]),length=integer(a[2]);if(offset<0){length+=offset;offset=0;}
        if(length>0 && (size_t)offset<first.size())first.erase((size_t)offset,(size_t)length);
        return set(std::move(first));
    }
    case DSPJSFX_STRING_SPRINTF: {
        auto result=jsfx_gfx::formatGfxPrintf(second.c_str(),n-2,[&](int i){return a[i+2];},
            [&](double handle){const auto it=values.find(key(handle));return it==values.end()?"":it->second.c_str();});
        if(result.size()>16383)result.resize(16383);return set(std::move(result));
    }
    default:return 0;
    }
}
} // namespace jsfx_native_gfx
extern "C" double jsfx_native_string_dispatch(DSPJSFX_State* state,int32_t opcode,const double* args,int32_t count) {
    auto* strings=jsfx_native_gfx::stringContext(state);
    return strings && args?strings->execute(opcode,args,count):0;
}
#endif
