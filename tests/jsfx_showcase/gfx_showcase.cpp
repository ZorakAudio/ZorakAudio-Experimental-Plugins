// SPDX-License-Identifier: Zlib
// Actual Abyss @init/@gfx, EEL engine and CPU LICE renderer. The fallback test
// double provides only JUCE allocation/host/font contracts, not JUCE validation.
#include "JSFXDSP.h"
#ifdef ZA_SHOWCASE_REAL_JUCE
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_graphics/juce_graphics.h>
#else
#include "juce_contract_stub.h"
#endif
#include <fstream>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "YSFXGfxInterpreter.h"
#include "JsfxGfxMemorySync.h"
#include "JsfxGfxFramePool.h"
#include "numeric_runtime.inc"
extern "C" void jsfx_ensure_mem(DSPJSFX_State* st,int64_t n){
 if(n<=0||!st->mem||n>st->memN)throw std::runtime_error("test memory limit");
 st->memUsed=std::max(st->memUsed,n);
}
static int indexOf(const char* name){for(int i=0;i<DSPJSFX_VARS_COUNT;++i)if(std::strcmp(name,DSPJSFX_VARS[i].name)==0)return i;throw std::runtime_error(std::string("missing var: ")+name);}
static void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main(int argc,char**argv)try{
 require(argc>1,"Pass expanded source path");
 std::ifstream f(argv[1]);std::ostringstream stream;stream<<f.rdbuf();const auto source=stream.str();
 require(!source.empty(),"empty source");
 DSPJSFX_State st{};std::vector<double> heap(8*1024*1024);st.mem=heap.data();st.memN=heap.size();st.srate=48000;st.samplesblock=512;
 const char* aliases[]={"diffusion","current_verb_decay","verb_mod_depth","current_verb_mod_rate","current_verb_lowpass","current_verb_highpass","shimmer","drop_mode","drops","nonlinearity","wet"};
 const double values[]={.6,.8,.1,.3,1,0,.2,0,0,0,.25};
 for(int i=0;i<11;++i){st.sliders[i]=values[i];st.vars[indexOf(aliases[i])]=values[i];}
 jsfx_init(&st);jsfx_slider(&st);
 const auto originalHeap=heap;
 size_t maxMaps=0,commands=0;int totalChanged=0,nonBlack=0;
 for(int opening=0;opening<2;++opening){
 jsfx_gfx::Interpreter interpreter(source.c_str());
 if(!interpreter.gfxCompiledOk())std::cerr<<interpreter.getLastError().toRawUTF8()<<'\n';
 require(interpreter.gfxCompiledOk(),"Abyss @gfx failed to compile");
 jsfx_gfx::Interpreter::Snapshot snap;snap.sliders=st.sliders;snap.slidersCount=64;snap.vars=st.vars;snap.varsCount=DSPJSFX_VARS_COUNT;snap.logicalMemN=heap.size();snap.srate=48000;snap.samplesblock=512;
 // Exactly the package's declared EXPLICIT/no-ranges policy, not AUTO mirroring.
 std::array<GfxMirrorRange,kMaxGfxMemSpans> ranges{};
 require(buildDirectionalGfxRanges(heap.size(),heap.size(),true,{},kGfxSyncToGfx,ranges)==0,"unexpected incoming memory");
 require(buildDirectionalGfxRanges(heap.size(),heap.size(),true,{},kGfxSyncFromGfx,ranges)==0,"unexpected writeback memory");
 jsfx_gfx::FramePool frames;
 int changedParticles=0;double lastParticle=0;
 // Resize and then recreate the editor; neither may alter native audio memory.
 for(int frame=0;frame<24;++frame){
   const int w=frame<12?640:920,h=frame<12?400:580;
   juce::Image image;require(frames.acquire(image,w,h),"no writable canvas");
   jsfx_gfx::GfxRenderSession renderer(image,[&](bool discard){if(!discard)jsfx_gfx::copyFramebufferHistory(image,frames.front());});
   auto binding=interpreter.bindRenderer(renderer);
   interpreter.renderFrame(w,h,snap);
   const auto stats=renderer.finish(interpreter.getCommands());commands=std::max(commands,interpreter.getCommands().size());
   require(stats.mainFramebufferChanged && stats.blits>=1,"Abyss did not composite an offscreen frame");
   maxMaps=std::max(maxMaps,(size_t)stats.surfaceMaps);
   double particle=0;interpreter.readMemRange((int64_t)st.vars[indexOf("particles")],&particle,1);
   if(frame&&particle!=lastParticle)++changedParticles;lastParticle=particle;
   nonBlack=0;for(int y=0;y<h;y+=5)for(int x=0;x<w;x+=5)if((image.getPixelAt(x,y).getARGB()&0xffffffu))++nonBlack;
   frames.publish(image);
 }
 require(changedParticles>=20,"private particles are being reset/frozen by snapshot synchronization");
 require(nonBlack>20,"particle image is blank");
 require(heap==originalHeap,"GFX changed native DSP memory");
 totalChanged+=changedParticles;
 }

 // Native JSFX drag/drop file path: the dropped pathname is written into an
 // EEL string slot, file_open() resolves that string, and file_mem() publishes
 // one contiguous dirty range instead of scalar writes. Keep this tiny/textual
 // so the contract-double build exercises the bridge without pretending to be
 // an audio decoder.
 {
   const auto filePath=std::filesystem::temp_directory_path()/"za_gfx_file_api_test.txt";
   {std::ofstream out(filePath);out<<"1 2 3\n";}
   const char* fileSource="@gfx 100 100\n"
     "gfx_getdropfile(0,14) ? ( h=file_open(14); h>=0 ? ( n=file_mem(h,100,3); file_close(h); ); gfx_getdropfile(-1); );\n";
   jsfx_gfx::Interpreter fileInterpreter(fileSource);
   require(fileInterpreter.gfxCompiledOk(),"file API @gfx failed to compile");
   fileInterpreter.addDroppedFile(juce::String::fromUTF8(filePath.string().c_str()));
   jsfx_gfx::Interpreter::Snapshot fileSnap;fileSnap.logicalMemN=256;
   fileInterpreter.renderFrame(100,100,fileSnap);
   double loaded[3]{};fileInterpreter.readMemRange(100,loaded,3);
   jsfx_gfx::MemRange forced[4]{};const int forcedN=fileInterpreter.copyForcedDirtyMemRanges(forced,4);
   jsfx_gfx::MemRange persistent[4]{};const int persistentN=fileInterpreter.copyPersistentFileMemRanges(persistent,4);
   std::error_code ec;std::filesystem::remove(filePath,ec);
   require(loaded[0]==1.0&&loaded[1]==2.0&&loaded[2]==3.0,"gfx_getdropfile/file_open/file_mem did not load data");
   require(forcedN==1&&persistentN==1&&forced[0].base==100&&forced[0].count==3,
           "file_mem did not report one bulk dirty span");
 }

 std::cout<<"PASS Abyss GFX: 48 frames, 2 sizes, 2 editor lifetimes; particle_updates="<<totalChanged<<" sampled_nonblack_pixels="<<nonBlack<<" max_commands="<<commands<<" max_surface_maps="<<maxMaps<<" memory_snapshot_bytes=0; native_file_drop=PASS\n";
#ifdef ZA_SHOWCASE_REAL_JUCE
 std::cout<<"Backend: real JUCE + supplied WDL/EEL + CPU LICE\n";
#else
 std::cout<<"Backend: JUCE contract double + supplied WDL/EEL + CPU LICE; NOT a JUCE/host build\n";
#endif
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
