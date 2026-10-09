#include "JsfxGfxMenus.h"
#include <thread>
#include "JitProcessor.h"
#define NOMINMAX
#include "JsfxGfxResources.h"
#include <iostream>
#include <cmath>
#include "../../tests/runtime/SyntheticBank.h"
#if JUCE_WINDOWS
#include <windows.h>
#else
namespace juce::detail { bool dispatchNextMessageOnSystemQueue(bool); }
namespace { namespace linux_test { void dispatch() { while(juce::detail::dispatchNextMessageOnSystemQueue(true)) {} } } }
#endif

using namespace juce;
namespace {
void require(bool condition, const String& text) { if (!condition) throw std::runtime_error(text.toStdString()); }
void fill(AudioBuffer<float>& buffer) { for (int i = 0; i < buffer.getNumSamples(); ++i) { buffer.setSample(0, i, 0.8f); buffer.setSample(1, i, -0.4f); } }
void checkGain(const AudioBuffer<float>& buffer, float gain) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        require(std::abs(buffer.getSample(0, i) - 0.8f * gain) < 2e-6f, "Incorrect left channel");
        require(std::abs(buffer.getSample(1, i) + 0.4f * gain) < 2e-6f, "Incorrect right channel");
    }
}
double runCode(JitProcessor& processor, const String& source, const String& mode, float expected, int count = 256, double timeoutMs = 45000) {
    auto before = processor.engine.activeRevision();
    const auto began = Time::getMillisecondCounterHiRes();
    processor.runSource(source, mode);
    AudioBuffer<float> buffer(2, count); MidiBuffer midi;
    while (Time::getMillisecondCounterHiRes() - began < timeoutMs) {
        processor.pollCompiledSchema();buffer.setSize(std::max(2,std::max(processor.getTotalNumInputChannels(),processor.getTotalNumOutputChannels())),count);buffer.clear();fill(buffer);processor.processBlock(buffer,midi);
        if (processor.engine.activeRevision() != before) { if(std::isfinite(expected)){try{checkGain(buffer, expected);}catch(const std::exception& e){throw std::runtime_error(std::string(e.what())+" for: "+source.toStdString()+" actual "+String(buffer.getSample(0,0)).toStdString());}} return (Time::getMillisecondCounterHiRes() - began) / 1000; }
        require(!processor.engine.status().startsWith("Run failed"), processor.engine.status());
        Thread::sleep(5);
    }
    throw std::runtime_error("Compile/swap timed out: " + processor.engine.status().toStdString());
}
}
#include "JitControlCheck.h"
#include "JitInterfaceCheck.h"
#include "JitExampleCheck.h"
#include "JitSampleGraphicsCheck.h"
int main(int argc,char** argv) {
#if JUCE_WINDOWS
    // Test failures belong in diagnostics, not modal dialogs over the DAW.
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
#endif
    ScopedJuceInitialiser_GUI init;
    try {
        if(argc>1 && String(argv[1])=="--examples"){checkExamples(argc>2 && String(argv[2])=="--cpp-frontend",argc>3?File(String(argv[3])):File{});return 0;}
        if(argc>1 && String(argv[1])=="--interface-unicode"){checkInterfaceUnicode(argc>2?File(String(argv[2])):File{});return 0;}
        if(argc>1 && String(argv[1])=="--control-defaults"){checkControls(argc>2 && String(argv[2])=="--cpp-frontend");return 0;}
        if(argc>1 && String(argv[1])=="--graphics-concurrency"){checkGraphicsConcurrency();return 0;}
        if(argc>2 && String(argv[1])=="--sample-gfx-stress"){checkSampleGraphics(File(String(argv[2])));return 0;}
        if(argc>3 && String(argv[1])=="--graphics-values"){
            JitProcessor p;if(argc>4 && String(argv[4])=="--cpp-frontend")p.setFrontend("cpp-frontend");p.prepareToPlay(48000,256);
            runCode(p,File(String(argv[2])).loadFileAsString(),"jsfx",std::numeric_limits<float>::quiet_NaN());
            require(p.engine.renderGraphics(100,100,0,0,0).isValid(),"Probe graphics executes");
            const auto queries=JSON::parse(File(String(argv[3])).loadFileAsString());DynamicObject::Ptr values=new DynamicObject;
            require(queries.isArray(),"Probe queries are an array");for(auto& name:*queries.getArray())values->setProperty(name.toString(),p.engine.inspectGraphicsVariable(name.toString()));
            std::cout<<JSON::toString(var(values.get()))<<'\n';return 0;
        }
        if(argc>2 && String(argv[1])=="--loaded-bank"){
            File sourceFile{String(argv[2])};require(sourceFile.existsAsFile(),"Loaded-bank source does not exist: "+sourceFile.getFullPathName());const bool corpus=sourceFile.getFileNameWithoutExtension()=="Corpus";
            auto folder=File::getSpecialLocation(File::tempDirectory).getNonexistentChildFile("za-runtime-bank","",false);
            struct Cleanup{File file;~Cleanup(){if(file.getParentDirectory()==File::getSpecialLocation(File::tempDirectory) && file.getFileName().startsWith("za-runtime-bank"))file.deleteRecursively();}} cleanup{folder};
            auto paths=makeRuntimeBank(folder);JitProcessor p;p.prepareToPlay(48000,4096);p.setSourcePath(sourceFile.getFullPathName());p.setNonRealtime(true);
            runCode(p,sourceFile.loadFileAsString(),"jsfx",std::numeric_limits<float>::quiet_NaN(),256,300000);p.engine.setFileSlot(0,paths);
            AudioBuffer<float> audio(std::max(2,std::max(p.getTotalNumInputChannels(),p.getTotalNumOutputChannels())),256);MidiBuffer midi;bool ready=false;auto started=Time::getMillisecondCounterHiRes();
            while(Time::getMillisecondCounterHiRes()-started<60000){audio.clear();midi.clear();p.processBlock(audio,midi);require(!p.engine.memoryFaulted(),"Loaded bank memory fault");
                ready=corpus?p.engine.inspectVariable("loaded_count")==3 && p.engine.inspectVariable("ready")>0 && p.engine.inspectVariable("xc_ready")>0 && p.engine.inspectVariable("pe_ready")>0 && p.engine.inspectVariable("pg_ready")>0 && p.engine.inspectVariable("pg_initializing")==0:p.engine.inspectVariable("sample_count")==3 && p.engine.inspectVariable("bank_loaded")>0 && p.engine.inspectVariable("rescan_pending")==0;
                if(ready)break;Thread::sleep(2);
            }
            if(!ready){for(auto name:{"loaded_count","sample_count","ready","build_phase","cg_stage","cg_disabled","pool_state","pool_loaded","grain_count","corpus_count","xc_ready","pe_ready","rescan_pending"})std::cerr<<name<<'='<<p.engine.inspectVariable(name)<<'\n';throw std::runtime_error("Loaded bank did not finish actual analysis");}
            double peak=0;for(int block=0;block<400;++block){audio.clear();midi.clear();if(block==0)midi.addEvent(MidiMessage::noteOn(1,60,(uint8)100),0);if(block==300)midi.addEvent(MidiMessage::noteOff(1,60),0);p.processBlock(audio,midi);require(!p.engine.memoryFaulted(),"Loaded playback memory fault");for(int c=0;c<p.getTotalNumOutputChannels();++c)for(int f=0;f<256;++f){double value=audio.getSample(c,f);require(std::isfinite(value),"Loaded playback finite");peak=std::max(peak,std::abs(value));}}
            require(peak>1e-6,"Loaded instrument MIDI produced no output");
            if(!corpus){p.setSliderValue(16,3);audio.clear();midi.clear();p.processBlock(audio,midi);require(p.engine.inspectVariable("corpus_count")>0,"Sample granular mode built its actual corpus");double granularPeak=0;
                for(int block=0;block<400;++block){audio.clear();midi.clear();if(block==0)midi.addEvent(MidiMessage::noteOn(1,60,(uint8)100),0);if(block==300)midi.addEvent(MidiMessage::noteOff(1,60),0);p.processBlock(audio,midi);for(int c=0;c<p.getTotalNumOutputChannels();++c)for(int f=0;f<256;++f){double value=audio.getSample(c,f);require(std::isfinite(value),"Loaded granular finite audio");granularPeak=std::max(granularPeak,std::abs(value));}}
                require(granularPeak>1e-6 && !p.engine.memoryFaulted(),"Loaded granular MIDI output");std::cout<<"GRANULAR PASS peak="<<granularPeak<<" corpus="<<p.engine.inspectVariable("corpus_count")<<"\n";}
            auto image=p.engine.renderGraphics(1000,700,0,0,0);require(image.isValid() && !p.engine.memoryFaulted(),"Loaded instrument GFX");MemoryBlock saved;p.getStateInformation(saved);
            std::cout<<"LOADED BANK PASS "<<sourceFile.getFileNameWithoutExtension()<<" files=3 peak="<<peak<<" savedBytes="<<saved.getSize()<<" grains="<<p.engine.inspectVariable(corpus?"grain_count":"corpus_count")<<"\n";return 0;
        }
        if(argc>1 && String(argv[1])=="--oversampling") {
            JitProcessor p;p.prepareToPlay(48000,4096);
            runCode(p,"slider1:g=.5<0,1,.01>Gain\nslider2:#label=\"text\"<string>Label\n@init\n#constant=\"const\";\n@slider\ngain=g*(srate/48000)*(strlen(#label)/4)*(strcmp(#constant,\"const\")==0);pdc_delay=128;\n@block\nwhile(midirecv(o,m,n,v))(midisend(o,m,n,v));\n@sample\nspl0*=gain;spl1*=gain;\n@gfx\ngfx_rect(0,0,10,10);","jsfx",.5f);
            p.engine.setStringSlider(1,"kept");
            for(int choice=0;choice<4;++choice)for(int count:{1,257,4096}) {
                p.oversamplingParameter->setValueNotifyingHost(float(choice)/3);
                AudioBuffer<float> audio(2,count);MidiBuffer midi;fill(audio);midi.addEvent(MidiMessage::noteOn(1,60,(uint8)100),std::min(17,count-1));p.processBlock(audio,midi);
                checkGain(audio,.5f*(1<<choice));require(p.engine.latencySamples.load()==128/(1<<choice),"Oversampling latency scales to host samples");require(p.engine.stringSlider(1)=="kept","String control and constants survive rate reset");
                bool note=false;for(auto event:midi)if(event.getMessage().isNoteOn())note|=event.samplePosition==std::min(17,count-1);require(note,"MIDI offsets round trip through engine rate");
                require(p.engine.renderGraphics(100,100,0,0,0).isValid(),"GFX survives engine rate reset");
            }
            p.setSliderValue(0,.25);const auto old=p.engine.activeRevision();p.prepareToPlay(44100,4096);auto started=Time::getMillisecondCounterHiRes();AudioBuffer<float> changedRate(2,256);MidiBuffer noMidi;
            while(p.engine.activeRevision()==old && Time::getMillisecondCounterHiRes()-started<15000){p.pollCompiledSchema();fill(changedRate);p.processBlock(changedRate,noMidi);require(!p.engine.status().startsWith("Run failed"),p.engine.status());Thread::sleep(5);}
            require(p.engine.activeRevision()!=old,"Host rate recompile activated");checkGain(changedRate,.25f*8*44100/48000);require(p.engine.stringSlider(1)=="kept","Host-rate prepare preserves string controls");p.prepareToPlay(48000,4096);
            runCode(p,"@faust block\nimport(\"stdfaust.lib\");process=spl0*(rdtable(16,fconstant(int fSamplingFreq,<math.h>),0)/48000),spl1*(rdtable(16,fconstant(int fSamplingFreq,<math.h>),0)/48000);","jsfx",8);
            for(int choice=0;choice<4;++choice){p.oversamplingParameter->setValueNotifyingHost(float(choice)/3);AudioBuffer<float> audio(2,257);MidiBuffer midi;fill(audio);p.processBlock(audio,midi);checkGain(audio,float(1<<choice));}
            MemoryBlock saved;p.getStateInformation(saved);JitProcessor restored;restored.prepareToPlay(48000,4096);restored.setStateInformation(saved.getData(),int(saved.getSize()));AudioBuffer<float> audio(2,256);MidiBuffer midi;auto start=Time::getMillisecondCounterHiRes();while(!restored.engine.activeRevision() && Time::getMillisecondCounterHiRes()-start<15000){restored.pollCompiledSchema();fill(audio);restored.processBlock(audio,midi);require(!restored.engine.status().startsWith("Run failed"),restored.engine.status());Thread::sleep(5);}require(restored.engine.activeRevision()>0,"Oversampled saved program activated");checkGain(audio,8);require(restored.oversamplingParameter->getIndex()==3,"Host oversampling state restored");
            std::cout<<"OVERSAMPLING PASS: 1/2/4/8x, variable blocks, rate initialization, MIDI, latency, strings, GFX, FAUST tables and state restore\n";return 0;
        }
        if(argc>1 && String(argv[1])=="--large-state") {
            JitProcessor p;p.prepareToPlay(48000,256);
            String code="options:maxmem=4096\n@init\n";
            for(int n=0;n<33000;++n)code+="large_"+String(n)+"="+String(n)+";\n";
            code+="job=defer(large_32999/65998;);gain=0;\n@block\ntask_finished(job) ? (gain=task_result(job);task_release(job););\n@sample\nspl0*=gain;spl1*=gain;";
            runCode(p,code,"jsfx",std::numeric_limits<float>::quiet_NaN(),256,300000);
            require((int)p.engine.compiledDescriptor().getProperty("metadata",{}).getProperty("var_cap",0)>32768,"Variable extent exceeds old PoC bound");
            AudioBuffer<float> audio(2,256);MidiBuffer midi;bool done=false;
            auto began=Time::getMillisecondCounterHiRes();while(Time::getMillisecondCounterHiRes()-began<5000){fill(audio);p.processBlock(audio,midi);if(std::abs(audio.getSample(0,0)-.4f)<1e-6){done=true;break;}Thread::sleep(5);}
            require(done,"Large variable extent and private task snapshot execute correctly");
            std::cout<<"LARGE STATE PASS: more than 32768 variables, actual DSP output, task-private snapshot\n";return 0;
        }
        if(argc>1 && String(argv[1])=="--source-resources") {
            auto root=File::getSpecialLocation(File::tempDirectory).getNonexistentChildFile("za-jit-resources","",false);auto src=root.getChildFile("src");auto deps=src.getChildFile("Dependencies");auto sibling=root.getChildFile("Resources");auto nested=src.getChildFile("Resources");require(deps.createDirectory().wasOk()&&sibling.createDirectory().wasOk()&&nested.createDirectory().wasOk(),"Fixture directories");
            deps.getChildFile("library.jsfx-inc").replaceWithText("@init\nfunction imported_gain()( .5; );\n");
            auto png=[](const File& file,Colour colour){Image image(Image::ARGB,8,8,true,SoftwareImageType());{Graphics g(image);g.fillAll(colour);}auto stream=file.createOutputStream();require(stream!=nullptr,"PNG fixture");PNGImageFormat format;require(format.writeImageToStream(image,*stream),"Write PNG");};
            png(src.getChildFile("direct.png"),Colours::red);png(sibling.getChildFile("sibling.png"),Colours::green);png(nested.getChildFile("nested.png"),Colours::blue);
            auto direct=jsfx_gfx_resources::loadImage("direct.png",src.getFullPathName().toRawUTF8(),false);require(direct.isValid() && direct.getPixelAt(2,2)==Colours::red,"Shared PNG decoder: "+String(direct.isValid()?direct.getPixelAt(2,2).toString():"invalid"));
            for(bool cpp:{false,true}) {
                JitProcessor p;p.setFrontend(cpp?"cpp-frontend":"python-reference");p.setSourcePath(src.getChildFile("editor.jsfx").getFullPathName());p.prepareToPlay(48000,256);
                for(auto name:{"direct.png","sibling.png","../Resources/sibling.png","Resources/nested.png"}) {
                    String code="import library.jsfx-inc\n@sample\nspl0*=imported_gain();spl1*=imported_gain();\n@gfx 64 64\ngfx_set(1,1,1,1);gfx_dest=-1;gfx_x=gfx_y=0;gfx_loadimg(1,\""+String(name)+"\");gfx_blit(1,1,0);";
                    runCode(p,code,"jsfx",.5f);auto image=p.engine.renderGraphics(64,64,0,0,0);auto colour=image.getPixelAt(2,2);require(image.isValid() && (String(name).contains("direct")?colour==Colours::red:String(name).contains("nested")?colour==Colours::blue:colour==Colours::green),"Image resource: "+String(name)+" pixel "+colour.toString());
                }
                runCode(p,"filename:0,sibling.png\nimport library.jsfx-inc\n@sample\nspl0*=imported_gain();spl1*=imported_gain();\n@gfx 64 64\ngfx_set(1,1,1,1);gfx_dest=-1;gfx_x=gfx_y=0;gfx_blit(0,1,0);","jsfx",.5f);
                require(p.engine.renderGraphics(64,64,0,0,0).getPixelAt(2,2)==Colours::green,"filename image lookup");
                auto originalRoot=p.sourcePath();auto newRoot=root.getChildFile("other/editor.jsfx").getFullPathName();
                p.setSourcePath(newRoot);p.runSource("import missing-dependency.jsfx-inc\n@sample\nspl0*=.5;spl1*=.5;","jsfx");
                {auto start=Time::getMillisecondCounterHiRes();while(!p.engine.status().startsWith("Run failed") && Time::getMillisecondCounterHiRes()-start<15000)Thread::sleep(5);require(p.engine.status().startsWith("Run failed"),"Invalid edit failed");}
                MemoryBlock preset;p.getStateInformation(preset);JitProcessor restored;restored.prepareToPlay(48000,256);restored.setStateInformation(preset.getData(),(int)preset.getSize());auto start=Time::getMillisecondCounterHiRes();AudioBuffer<float> audio(2,256);MidiBuffer midi;while(!restored.engine.activeRevision() && Time::getMillisecondCounterHiRes()-start<15000){restored.pollCompiledSchema();fill(audio);restored.processBlock(audio,midi);require(!restored.engine.status().startsWith("Run failed"),restored.engine.status());Thread::sleep(5);}require(restored.engine.activeRevision()!=0 && restored.sourcePath()==p.sourcePath(),"Source folder state restore");require(restored.engine.renderGraphics(64,64,0,0,0).getPixelAt(2,2)==Colours::green,"Restored image root");
                require(restored.engine.appliedSourcePath()==originalRoot && restored.sourcePath()==newRoot,"Failed draft retains separate applied dependency root");
            }
            require(root.getParentDirectory()==File::getSpecialLocation(File::tempDirectory) && root.getFileName().startsWith("za-jit-resources"),"Fixture cleanup scope");root.deleteRecursively();std::cout<<"SOURCE/RESOURCE PASS: both frontends, nested imports, direct and Resources images, filename slots, saved-state restore\n";return 0;
        }
        bool cpp=argc>1 && String(argv[1])=="--cpp-frontend";int sourceArg=cpp?2:1;
        const bool catalogMode=argc>sourceArg && String(argv[sourceArg])=="--catalog";if(catalogMode)++sourceArg;
        if(argc>sourceArg){
            File sourceFile{String(argv[sourceArg])};JitProcessor catalog;if(cpp)catalog.setFrontend("cpp-frontend");catalog.prepareToPlay(48000,4096);catalog.setSourcePath(sourceFile.getFullPathName());catalog.setNonRealtime(true);
            const auto compileSeconds=runCode(catalog,sourceFile.loadFileAsString(),"jsfx",std::numeric_limits<float>::quiet_NaN(),256,catalogMode?600000:120000);
            require(!catalog.engine.memoryFaulted(),"Catalog initialization memory fault");
            auto image=catalog.engine.renderGraphics(1000,700,0,0,0);int coloured=0;
            if(image.isValid()){for(int y=0;y<700;y+=5)for(int x=0;x<1000;x+=5)if(image.getPixelAt(x,y).getARGB()!=Colours::black.getARGB())++coloured;
                auto output=File::getCurrentWorkingDirectory().getChildFile(sourceFile.getFileNameWithoutExtension().replaceCharacter(' ','-')+"-JIT.png");if(auto stream=output.createOutputStream()){PNGImageFormat png;png.writeImageToStream(image,*stream);}}
            if(!catalogMode)require(image.isValid() && coloured>10,"Catalog GFX drew content");
            double peak=0;for(int repeat=0;repeat<4;++repeat)for(int count:{1,7,64,257,1024}){
                AudioBuffer<float> samples(std::max(2,std::max(catalog.getTotalNumInputChannels(),catalog.getTotalNumOutputChannels())),count);MidiBuffer messages;
                for(int c=0;c<samples.getNumChannels();++c)for(int f=0;f<count;++f)samples.setSample(c,f,repeat==0?0:float(((f*17+c*31)%251-125)/512.));
                if(repeat==1)messages.addEvent(MidiMessage::noteOn(1,60,(uint8)96),0);if(repeat==3)messages.addEvent(MidiMessage::noteOff(1,60),0);
                catalog.processBlock(samples,messages);require(!catalog.engine.memoryFaulted(),"Catalog DSP/GFX memory fault");
                for(int c=0;c<catalog.getTotalNumOutputChannels();++c)for(int f=0;f<count;++f){const auto value=samples.getSample(c,f);require(std::isfinite(value),"Catalog finite audio");peak=std::max(peak,std::abs(double(value)));}
            }
            MemoryBlock state;catalog.getStateInformation(state);require(state.getSize()>0,"Catalog saved state");std::cout<<"{\"passed\":true,\"compileSeconds\":"<<compileSeconds<<",\"visibleControls\":";int visible=0;for(auto& d:catalog.declarations)visible+=catalog.sliderVisible[(size_t)d.index0].load();std::cout<<visible<<",\"parameters\":"<<catalog.getParameters().size()<<",\"savedBytes\":"<<state.getSize()<<",\"gfxPixels\":"<<coloured<<",\"peak\":"<<peak<<",\"memoryFault\":false}\n";return 0;
        }
        {JitProcessor primed;primed.runSource("slider1:g=.25<0,1,.01>Gain\n@slider\ncolour=g*4;\n@gfx 100 100\ngfx_set(colour,0,0,1);gfx_rect(0,0,gfx_w,gfx_h);","jsfx");auto start=Time::getMillisecondCounterHiRes();while(!primed.engine.compiledDescriptor().isObject() && Time::getMillisecondCounterHiRes()-start<15000){require(!primed.engine.status().startsWith("Run failed"),primed.engine.status());Thread::sleep(5);}primed.pollCompiledSchema();require(primed.engine.activeRevision()==0 && primed.engine.renderGraphics(100,100,0,0,0).getPixelAt(20,20).getRed()>240,"@slider primes GFX before audio");}
        {JitProcessor original;original.prepareToPlay(48000,256);runCode(original,"slider1:g=.25<0,1,.01>Gain\n@init\nstartup=g;\n@slider\ncolour=startup*4;\n@sample\nspl0*=g;spl1*=g;\n@gfx 100 100\ngfx_set(colour,0,0,1);gfx_rect(0,0,gfx_w,gfx_h);","jsfx",.25f);
            original.setSliderValue(0,.2);MemoryBlock preset;original.getStateInformation(preset);JitProcessor restored;restored.setStateInformation(preset.getData(),(int)preset.getSize());
            auto start=Time::getMillisecondCounterHiRes();while(!restored.engine.compiledDescriptor().isObject() && Time::getMillisecondCounterHiRes()-start<15000){require(!restored.engine.status().startsWith("Run failed"),restored.engine.status());Thread::sleep(5);}restored.pollCompiledSchema();
            auto image=restored.engine.renderGraphics(100,100,0,0,0);require(restored.engine.activeRevision()==0 && image.isValid() && std::abs(image.getPixelAt(20,20).getRed()-204)<=2,"Restored host values prime @slider before paused GFX");}
        {JitProcessor paused;paused.runSource("@gfx 100 100\ngfx_set(1,0,0,1);gfx_rect(0,0,gfx_w,gfx_h);","jsfx");auto start=Time::getMillisecondCounterHiRes();while(!paused.engine.compiledDescriptor().isObject() && Time::getMillisecondCounterHiRes()-start<15000){require(!paused.engine.status().startsWith("Run failed"),paused.engine.status());Thread::sleep(5);}paused.pollCompiledSchema();auto preview=paused.engine.renderGraphics(100,100,0,0,0);require(paused.engine.activeRevision()==0 && preview.isValid() && preview.getPixelAt(20,20).getRed()>240,"GFX works before first audio callback");paused.runSource("@sample\nspl0*=;","jsfx");auto failedStart=Time::getMillisecondCounterHiRes();while(!paused.engine.status().startsWith("Run failed") && Time::getMillisecondCounterHiRes()-failedStart<10000)Thread::sleep(5);require(paused.engine.status().startsWith("Run failed"),"Paused invalid compile");AudioBuffer<float> pausedAudio(2,256);MidiBuffer pausedMidi;fill(pausedAudio);paused.processBlock(pausedAudio,pausedMidi);require(paused.engine.activeRevision()>0,"Previous paused program activates after a failed Run");}
        JitProcessor processor;if(cpp)processor.setFrontend("cpp-frontend"); processor.prepareToPlay(48000, 4096);
        AudioBuffer<float> buffer(2, 256); MidiBuffer midi; processor.pollCompiledSchema(); fill(buffer); processor.processBlock(buffer, midi); checkGain(buffer, 1);
        runCode(processor,"@init\n#out=\"\";gain=match(\"%{#out}s\",\"bar\")*(strcmp(#out,\"bar\")==0)*.5;\n@sample\nspl0*=gain;spl1*=gain;","jsfx",.5f);
        runCode(processor,"slider1:g=.5<0,1,.01>Gain\n@init\ng=.25;\n@sample\nspl0*=g;spl1*=g;","jsfx",.25f);
        for(int n=0;n<3;++n){fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,.25f);}processor.setSliderValue(0,.75);fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,.75f);
        runCode(processor,"slider1:g=.5<0,1,.01>Gain\n@block\n!written ? (slider1=.25;g=.25;written=1;);\n@sample\nspl0*=slider1;spl1*=g;","jsfx",.25f);
        const auto hostBeforeGraphics=processor.sliderValue(0);processor.engine.renderGraphics(100,100,0,0,0);require(processor.sliderValue(0)==hostBeforeGraphics,"Drawing GFX does not automate a DSP-authored slider");
        for(int n=0;n<3;++n){fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,.25f);}processor.setSliderValue(0,.75);fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,.75f);
        runCode(processor,"slider1:mask=0<0,2251799813685247,1>Mask\n@block\n!written ? (mask=1125899906842625;sliderchange(slider1);written=1;);\n@sample\nspl0*=(mask==1125899906842625);spl1*=(mask==1125899906842625);","jsfx",1);
        for(int n=0;n<4;++n){fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,1);}require(processor.engine.inspectVariable("mask")==1125899906842625.,"DSP slider notification retains exact large bit mask");
        runCode(processor,"slider1:mask=0<0,2251799813685247,1>Mask\n@slider\nnotifications+=1;\n@sample\nspl0*=(mask==1125899906842625);spl1*=(mask==1125899906842625);\n@gfx\nmouse_cap&1 ? (mask=1125899906842625;slider_automate(slider1););","jsfx",0);
        const double notificationCount=processor.engine.inspectVariable("notifications");processor.engine.renderGraphics(100,100,0,0,1);
        for(int n=0;n<4;++n){fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,1);}require(processor.engine.inspectVariable("notifications")>notificationCount,"Pending GFX slider notification executes @slider");require(processor.engine.inspectVariable("mask")==1125899906842625.,"GFX notification retains exact large bit mask before host acknowledgment");
        runCode(processor,"slider1:g=.1<.1,.5,.2{A,B,C}>Choice\n@sample\nspl0*=g;spl1*=g;","jsfx",.1f);processor.setSliderValue(0,.3);fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,.3f);require(processor.engine.controls[0].load()==double(processor.declarations[0].rangeStart)+double(processor.declarations[0].step),"Fractional choice retains declared double value");
        const auto outgoingRevision=processor.engine.activeRevision();runCode(processor,"slider1:g=.5<0,1,.01>Gain\n@sample\nspl0*=g;spl1*=g;","jsfx",.5f);
        const auto beforeNotification=processor.sliderValue(0);processor.engine.graphicsSliderChanged(outgoingRevision,0,.9,2);require(processor.sliderValue(0)==beforeNotification,"Outgoing GFX notification cannot alter new program controls");
        runCode(processor,"options:maxmem=4096\n@init\nfft(4090,32);\n@sample\nspl0=.8;spl1=-.4;","jsfx",0);
        runCode(processor,"@init\ngain=(tempo==120 && num_ch==2)*.5;\n@sample\nspl0*=gain;spl1*=gain;","jsfx",.5f);
        runCode(processor,"@init\nlastTempo=120;\n@block\nhost_transport_valid ? lastTempo=tempo;gain=lastTempo/300;\n@sample\nspl0*=gain;spl1*=gain;","jsfx",.4f);
        {AudioPlayHead::PositionInfo position;position.setBpm(150);position.setIsPlaying(true);position.setTimeInSamples(48000);fill(buffer);processor.engine.process(buffer,&midi,&position);checkGain(buffer,.5f);fill(buffer);processor.engine.process(buffer,&midi,nullptr);checkGain(buffer,.5f);}
        runCode(processor,"@init\ngain=.5;\n@block\nza_sleep_ready=1;\n@sample\nspl0*=gain;spl1*=gain;\n@gfx\nmouse_cap&1 ? gain=.25;","jsfx",.5f);
        {for(int n=0;n<4;++n){buffer.clear();processor.processBlock(buffer,midi);}require(processor.engine.idleSleeping.load(),"Cooperative sleep after explicit readiness and silent output");processor.engine.idleOverride.store(4);for(int n=0;n<4;++n){buffer.clear();processor.processBlock(buffer,midi);require(!processor.engine.idleSleeping.load(),"Production Never override prevents sleep");}processor.engine.idleOverride.store(0);buffer.clear();processor.processBlock(buffer,midi);require(processor.engine.idleSleeping.load(),"Auto restores production cooperative readiness");processor.engine.renderGraphics(100,100,0,0,1);fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,.25f);require(!processor.engine.idleSleeping.load(),"Graphics event and new input wake");processor.setNonRealtime(true);for(int n=0;n<4;++n){buffer.clear();processor.processBlock(buffer,midi);require(!processor.engine.idleSleeping.load(),"Offline rendering stays awake");}processor.setNonRealtime(false);}
        runCode(processor,"@init\na=1;b=2;x=0;y=atomic_set(a,3);z=atomic_add(a,a);old=atomic_setifequal(a,6,9);result=atomic_exch(a,b);gain=(atomic_get(a)==2 && b==9 && result==2 &&old==6 && y==3 &&z==6)*.5;\n@sample\nspl0*=gain;spl1*=gain;\n@gfx\nmouse_cap&1 ? atomic_set(gain,.25);", "jsfx",.5f);
        processor.engine.renderGraphics(100,100,0,0,1);fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,.25f);
        runCode(processor,"slider1:g=0.5<0,1,0.01>-Hidden gain\n@sample\nspl0*=g;spl1*=g;\n@gfx\ngfx_set(.1,.2,.3,1);gfx_rect(0,0,gfx_w,gfx_h);", "jsfx",.5f);
        JitEngine::GraphicsInput hidden;processor.engine.renderGraphics(400,300,0,0,0,0,0,0,&hidden);require(!(hidden.visible[0]&1),"Declared hidden slider remains hidden");
        runCode(processor,"slider1:g=0.5<0,1,0.01>-Hidden gain\n@sample\nspl0*=g;spl1*=g;\n@gfx\nslider_show(slider1,mouse_cap&1);", "jsfx",.5f);processor.engine.renderGraphics(100,100,0,0,1,0,0,0,&hidden);require(hidden.visible[0]&1,"Explicit slider_show reveals a declared-hidden control");
        runCode(processor,"@init\nvalue=.5;\n@sample\nspl0*=value;spl1*=value;\n@gfx\nmouse_cap&1 ? value=.25;\n@serialize\nfile_var(0,value);", "jsfx",.5f);
        processor.engine.renderGraphics(100,100,0,0,1);MemoryBlock serializedPreset;processor.getStateInformation(serializedPreset);{JitProcessor restored;restored.prepareToPlay(48000,256);restored.setStateInformation(serializedPreset.getData(),(int)serializedPreset.getSize());auto start=Time::getMillisecondCounterHiRes();while(!restored.engine.activeRevision() && Time::getMillisecondCounterHiRes()-start<15000){restored.pollCompiledSchema();fill(buffer);restored.processBlock(buffer,midi);Thread::sleep(5);}require(restored.engine.activeRevision()!=0,"Serialized program activated");require(restored.frontend()==processor.frontend() && restored.engine.appliedFrontend()==processor.engine.appliedFrontend(),"Compiler choice restored");fill(buffer);restored.processBlock(buffer,midi);checkGain(buffer,.25f);}
        runCode(processor,"@init\n#remember=\"init\";\n@sample\nspl0*=strlen(#remember)/5;spl1*=strlen(#remember)/5;\n@gfx\nmouse_cap&1 ? #remember=\"1234567\";\n@serialize\nfile_string(0,#remember);", "jsfx",.8f);processor.engine.renderGraphics(100,100,0,0,1);{MemoryBlock state;processor.getStateInformation(state);JitProcessor restored;restored.prepareToPlay(48000,256);restored.setStateInformation(state.getData(),(int)state.getSize());auto began=Time::getMillisecondCounterHiRes();while(!restored.engine.activeRevision() && Time::getMillisecondCounterHiRes()-began<15000){restored.pollCompiledSchema();fill(buffer);restored.processBlock(buffer,midi);require(!restored.engine.status().startsWith("Run failed"),restored.engine.status());Thread::sleep(5);}fill(buffer);restored.processBlock(buffer,midi);try{checkGain(buffer,1.4f);}catch(const std::exception& e){throw std::runtime_error(std::string(e.what())+" serialization string actual "+String(buffer.getSample(0,0)).toStdString());}}
        AudioProcessor::TrackProperties track;track.name="track";processor.updateTrackProperties(track);runCode(processor,"@init\npdc_delay=123;\n@block\ntrack_name(#name);gain=strlen(#name)/5;\n@sample\nspl0*=gain;spl1*=gain;","jsfx",1);require(processor.engine.latencySamples.load()==123,"Custom latency forwarded");
        runCode(processor,"@init\ntask=defer_reduce(i,16,SUM,0,i+1);gain=0;\n@block\ntask_finished(task) ? gain=task_result(task)/136;\n@sample\nspl0*=gain;spl1*=gain;", "jsfx",std::numeric_limits<float>::quiet_NaN());
        {auto start=Time::getMillisecondCounterHiRes();bool done=false;while(Time::getMillisecondCounterHiRes()-start<3000){fill(buffer);processor.processBlock(buffer,midi);if(std::abs(buffer.getSample(0,0)-.8f)<1e-6){done=true;break;}Thread::sleep(5);}require(done,"Native deferred reduction completed");}
        runCode(processor,"options:maxmem=4096\n@init\n0[0]=7;gain=0;phase=0;\n@block\nphase==0 ? (arena=task_arena_clone(__memtop(),0,0);arena>0 ? phase=1;);phase==1 && task_arena_status(arena)==4 ? (task_arena_preserve(arena,0,1);job=defer_arena(arena,0,0[0]=100;0[1]=2;gain=.5;);job>0 ? phase=2;);phase==2 && task_finished(job) ? (adopt=task_arena_adopt(arena,gain);adopt==1 ? (gain*=(0[0]==7 && 0[1]==2);task_release(job);phase=3;););\n@sample\nspl0*=gain;spl1*=gain;\n@gfx\ngfx_rect(0,0,0[0],0[1]);","jsfx",std::numeric_limits<float>::quiet_NaN());
        {auto start=Time::getMillisecondCounterHiRes();bool done=false;while(Time::getMillisecondCounterHiRes()-start<5000){processor.engine.renderGraphics(100,100,0,0,0);fill(buffer);processor.processBlock(buffer,midi);if(std::abs(buffer.getSample(0,0)-.4f)<1e-6){done=true;break;}Thread::sleep(5);}require(done,"Native heap snapshot, preserved controls, adoption and graphics rebinding completed");}
        runCode(processor,"@init\ngain=0;\n@block\nwhile(midirecv(offset,msg,note,velocity))(gain=velocity/127;midisend(offset,msg,note,velocity));\n@sample\nspl0*=gain;spl1*=gain;", "jsfx",0);
        midi.addEvent(MidiMessage::noteOn(1,60,(uint8)127),17);fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,1);require(midi.getNumEvents()==1 && (*midi.begin()).samplePosition==17,"MIDI receive/send offset preserved");midi.clear();
        {AudioPlayHead::PositionInfo playing;playing.setIsPlaying(true);playing.setTimeInSamples(48000);midi.addEvent(MidiMessage::noteOn(1,61,(uint8)127),23);fill(buffer);processor.engine.process(buffer,&midi,&playing);midi.clear();AudioPlayHead::PositionInfo stopped;stopped.setIsPlaying(false);stopped.setTimeInSamples(48256);fill(buffer);processor.engine.process(buffer,&midi,&stopped);bool off=false;for(auto e:midi)off|=e.getMessage().isNoteOff();require(off,"Production MIDI cleanup sends note ends at transport stop");midi.clear();}
        {JitProcessor notes;notes.prepareToPlay(48000,256);runCode(notes,"@block\nwhile(midirecv(o,m,n,v))(midisend(o,m,n,v));\n@sample\nspl0*=1;spl1*=1;","jsfx",1);AudioBuffer<float> audio(2,256);MidiBuffer events;events.addEvent(MidiMessage::noteOn(1,62,(uint8)127),0);fill(audio);notes.processBlock(audio,events);events.clear();notes.factory();notes.pollCompiledSchema();fill(audio);notes.processBlock(audio,events);bool ended=false;for(auto e:events)ended|=e.getMessage().isNoteOff() && e.getMessage().getNoteNumber()==62;require(ended,"Default ends notes emitted by the outgoing program");}
        {JitProcessor notes;if(cpp)notes.setFrontend("cpp-frontend");notes.prepareToPlay(48000,256);
         const String source="@block\n!sent ? (midisend(0,$x90,62,100);sent=1;);\n@sample\nspl0*=1;spl1*=1;";
         runCode(notes,source,"jsfx",1);const auto old=notes.engine.activeRevision();notes.runSource("desc:Replacement\n"+source,"jsfx");AudioBuffer<float> audio(2,256);MidiBuffer events;auto started=Time::getMillisecondCounterHiRes();
         while(notes.engine.activeRevision()==old && Time::getMillisecondCounterHiRes()-started<15000){notes.pollCompiledSchema();events.clear();fill(audio);notes.processBlock(audio,events);Thread::sleep(5);}
         require(notes.engine.activeRevision()!=old,"MIDI replacement activated");
         auto assertHeld=[&](const char* reason){bool held=false,ended=false;for(auto event:events){auto m=event.getMessage();if(m.isNoteOff() && m.getNoteNumber()==62){held=false;ended=true;}if(m.isNoteOn() && m.getNoteNumber()==62)held=true;}require(ended && held,reason);};
         assertHeld("Retiring note-off precedes replacement note-on at the same timestamp");notes.oversamplingParameter->setValueNotifyingHost(1.f/3);events.clear();fill(audio);notes.processBlock(audio,events);assertHeld("Oversampling reset releases old notes before reinitialized note-ons");}
        runCode(processor,"slider1:#label=\"initial\"<string>Label\n@sample\nspl0*=strlen(#label)/7;spl1*=strlen(#label)/7;", "jsfx",1);
        require(processor.engine.stringSlider(0)=="initial","String slider default");processor.engine.setStringSlider(0,"abcdefg");fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,1);
        {MemoryBlock preset;processor.getStateInformation(preset);JitProcessor restored;restored.prepareToPlay(48000,256);restored.setStateInformation(preset.getData(),(int)preset.getSize());auto began=Time::getMillisecondCounterHiRes();while(!restored.engine.activeRevision() && Time::getMillisecondCounterHiRes()-began<15000){restored.pollCompiledSchema();fill(buffer);restored.processBlock(buffer,midi);Thread::sleep(5);}require(restored.engine.stringSlider(0)=="abcdefg","String control state round trip");}
        {auto wav=File::getSpecialLocation(File::tempDirectory).getChildFile("za-jit-synthetic-"+Uuid().toString()+".wav");
         struct Remove {File file;~Remove(){file.deleteFile();}} remove{wav};WavAudioFormat format;auto stream=wav.createOutputStream();auto writer=std::unique_ptr<AudioFormatWriter>(format.createWriterFor(stream.get(),48000,2,24,{},0));require(writer!=nullptr,"Synthetic WAV writer");stream.release();AudioBuffer<float> source(2,64);for(int c=0;c<2;++c)for(int f=0;f<64;++f)source.setSample(c,f,.25f);require(writer->writeFromAudioSampleBuffer(source,0,64),"Synthetic WAV write");writer.reset();
         auto path=wav.getFullPathName().replaceCharacter('\\','/');
         // The production @gfx service opens a path directly. The DSP service
         // reads filename slots from a cache populated by the loader thread.
         runCode(processor,"@init\ngain=0;\n@sample\nspl0*=gain;spl1*=gain;\n@gfx\nh=file_open(\""+path+"\");file_riff(h,nch,sr);file_mem(h,0,128);file_close(h);gain=0[0]*2;","jsfx",0);
         processor.engine.renderGraphics(100,100,0,0,0);fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,.5f);
         runCode(processor,"filename:0,"+path+"\n@init\ngain=0;\n@block\nh=file_open(0);h>=0 ? (file_riff(h,nch,sr);file_mem(h,0,128);file_close(h);gain=(nch==2 && sr==48000)*0[0]*2;);\n@sample\nspl0*=gain;spl1*=gain;","jsfx",std::numeric_limits<float>::quiet_NaN());
         {auto began=Time::getMillisecondCounterHiRes();bool loaded=false;while(Time::getMillisecondCounterHiRes()-began<10000){fill(buffer);processor.processBlock(buffer,midi);if(std::abs(buffer.getSample(0,0)-.4f)<1e-6){loaded=true;break;}Thread::sleep(10);}require(loaded,"Production cached file service reads synthetic WAV");}
         runCode(processor,"filename:0,#FILE:test,Synthetic input\n@init\npool=sample_pool_from_slot(0,\"main\");sample_pool_commit(pool);\n@block\nsample_pool_adopt(pool);gain=sample_read(pool,sample_get(pool,0),0,0)*2;\n@sample\nspl0*=gain;spl1*=gain;","jsfx",std::numeric_limits<float>::quiet_NaN());StringArray paths;paths.add(wav.getFullPathName());processor.engine.setFileSlot(0,paths);auto began=Time::getMillisecondCounterHiRes();bool loaded=false;while(Time::getMillisecondCounterHiRes()-began<10000){fill(buffer);processor.processBlock(buffer,midi);if(std::abs(buffer.getSample(0,0)-.4f)<1e-6){loaded=true;break;}Thread::sleep(10);}require(loaded,"Background sample pool reads synthetic WAV");MemoryBlock savedPool;processor.getStateInformation(savedPool);require(JSON::parse(savedPool.toString()).getProperty("serialized",{}).getProperty("fileSlots",{}).isObject(),"File selection stored in state");
        }
        runCode(processor,"@init\ngmem_attach(\"jit-check-shared\");gmem[0]=.5;\n@sample\nspl0*=gmem[0];spl1*=gmem[0];", "jsfx",.5f);
        runCode(processor,"@init\n0[0]=1;fft(0,16);fft_permute(0,16);fft_ipermute(0,16);ifft(0,16);gain=0[0]/16;\n@sample\nspl0*=gain;spl1*=gain;", "jsfx",1);
        auto jsfx = runCode(processor, "@sample\nspl0*=0.5;spl1*=0.5;", "jsfx", 0.5f);
        require(processor.engine.compiledDescriptor().getProperty("frontend","").toString()==(cpp?"cpp-frontend":"python-reference"),"Selected frontend consumed by compiler");
        auto mixed = runCode(processor, "slider1:0.5<0,1,0.01>Gain\n@block\ngain=slider1;\n@faust block\nprocess=spl0*gain,spl1*gain;", "jsfx", 0.5f);
        processor.parameters[0]->setValueNotifyingHost(0.25f);
        processor.pollCompiledSchema(); fill(buffer); processor.processBlock(buffer, midi); checkGain(buffer, 0.25f);
        runCode(processor, "@sample\nspl0*=0.5;spl1*=0.5;\n@faust block\nprocess=spl0*0.5,spl1*0.5;", "jsfx", 0.25f);
        runCode(processor, "@faust block\nprocess=spl0*0.5,spl1*0.5;\n@block\ngain=abs(spl0);\n@sample\nspl0*=gain;spl1*=gain;", "jsfx", 0.2f);
        runCode(processor,"@init\nx=.5;\n@sample\nx=.25;\n@faust sample\nprocess=spl0*x,spl1*x;", "jsfx",.25f);
        runCode(processor,"@sample\nx=spl0;\n@faust block\nprocess=x*.5,spl1*.5;", "jsfx",.5f);
        runCode(processor,"@faust block\nimport(\"stdfaust.lib\");process=spl0*rdtable(16,1,0),spl1*rdtable(16,1,0);", "jsfx",1);
        {bool hasTable=false;auto stages=processor.engine.compiledDescriptor().getProperty("metadata",{}).getProperty("faust_stages",{});if(auto* list=stages.getArray())for(auto& stage:*list)if(auto* sizes=stage.getProperty("table_sizes",{}).getArray())hasTable|=!sizes->isEmpty();require(hasTable,"Table fixture retained a generated table");
         String tableSource="@faust block\nimport(\"stdfaust.lib\");process=spl0*(rdtable(16,ma.SR,0)/48000),spl1*(rdtable(16,ma.SR,0)/48000);";JitProcessor other;other.prepareToPlay(24000,256);runCode(other,tableSource,"jsfx",.5f);runCode(processor,tableSource,"jsfx",1);for(int i=0;i<10;++i){fill(buffer);other.processBlock(buffer,midi);checkGain(buffer,.5f);fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,1);}}
        auto faust = runCode(processor, "process=_,_:*(0.25),*(0.25);", "faust", 0.25f, 1024);
        runCode(processor,"process=_,_:*(hslider(\"Gain\",0.5,0,1,0.01)),*(hslider(\"Gain\",0.5,0,1,0.01));","faust",0.5f);
        require(processor.getParameters().size()==2 && processor.parameters[0]->getName(80)=="Gain","Faust generated control");
        processor.setSliderValue(0,0.25);fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,0.25f);
        const String aliases="slider1:drive=6<0,30,0.01>Drive\nslider2:sigma=0.45<0.05,2,0.001>Sigma\nslider3:bias=0<-1,1,0.0001>Bias\nslider4:mix=1<0,1,0.001>Mix\nslider5:outdb=0<-24,24,0.01>Output (dB)\nslider6:os=2<1,3,1{1x,2x,4x}>Oversample\nslider7:dcblock=1<0,1,1{Off,On}>DC Blocker\nslider8:autogain=1<0,1,1{Off,On}>Auto Gain (RMS)\nslider9:xover=180<20,4000,1>Split (Hz)\n@sample\nspl0*=mix*0.5;spl1*=mix*0.5;\n@gfx 400 300\ngfx_set(0.1,0.2,0.3,1);gfx_rect(0,0,gfx_w,gfx_h);gfx_set(0,1,0,1);gfx_rect(20,20,100,30);gfx_x=20;gfx_y=80;gfx_drawstr(\"Native compiled GFX\");mouse_cap&1 ? (mix=0.25;slider_automate(mix));slider_show(slider9,mouse_cap&1);\n";
        runCode(processor,aliases,"jsfx",0.5f);
        require(processor.getParameters().size()==10,"Nine generated controls");
        require(processor.parameters[5]->getName(80)=="Oversample","Choice label");
        require(std::abs(processor.sliderValue(1)-0.45)<1e-6 && processor.sliderValue(5)==2,"Declared defaults");
        auto gfx=processor.engine.renderGraphics(400,300,0,0,0);
        require(gfx.isValid() && gfx.getPixelAt(40,30).getGreen()>240,"Native GFX rectangle pixels");
        JitEngine::GraphicsInput visibleInput;processor.engine.renderGraphics(400,300,0,0,0,0,0,0,&visibleInput);require((visibleInput.visible[0]&(UINT64_C(1)<<8))==0,"slider_show hides control");
        processor.engine.renderGraphics(400,300,50,30,1,0,0,0,&visibleInput);require((visibleInput.visible[0]&(UINT64_C(1)<<8))!=0,"slider_show reveals control");
        processor.engine.renderGraphics(400,300,50,30,1);
        fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,0.125f);
        require(std::abs(processor.sliderValue(3)-0.25)<1e-6,"GFX alias automation");
        runCode(processor,"@sample\nspl0*=0.25;spl1*=0.25;","jsfx",0.25f);
        auto active = processor.engine.activeRevision();
        processor.runSource("@sample\nspl0*=;", "jsfx");
        auto began = Time::getMillisecondCounterHiRes();
        while (!processor.engine.status().startsWith("Run failed") && Time::getMillisecondCounterHiRes() - began < 15000) Thread::sleep(5);
        require(processor.engine.status().startsWith("Run failed"), "Invalid source wasn't rejected");
        require(processor.engine.activeRevision() == active, "Compile failure replaced active code");
        processor.pollCompiledSchema(); fill(buffer); processor.processBlock(buffer, midi); checkGain(buffer, 0.25f);
        MemoryBlock saved; processor.getStateInformation(saved);
        processor.factory(); processor.pollCompiledSchema(); fill(buffer); processor.processBlock(buffer, midi); checkGain(buffer, 1);
        require(processor.engine.activeRevision() == 0, "Default did not restore factory");
        processor.setStateInformation(saved.getData(), static_cast<int>(saved.getSize()));
        began = Time::getMillisecondCounterHiRes();
        while (processor.engine.activeRevision() == 0 && Time::getMillisecondCounterHiRes() - began < 30000) { processor.pollCompiledSchema(); fill(buffer); processor.processBlock(buffer, midi); Thread::sleep(5); }
        processor.pollCompiledSchema(); fill(buffer); processor.processBlock(buffer, midi); checkGain(buffer, 0.25f);
        // A stateful full-block filter, checked against an independent biquad.
        auto previous = processor.engine.activeRevision();
        processor.runSource("import(\"stdfaust.lib\");\nprocess=fi.lowpass(2,1200),fi.lowpass(2,1200);", "faust");
        began = Time::getMillisecondCounterHiRes();
        while (processor.engine.activeRevision() == previous && Time::getMillisecondCounterHiRes() - began < 30000) {
            processor.pollCompiledSchema(); AudioBuffer<float> empty(2, 0); processor.processBlock(empty, midi); Thread::sleep(5);
            require(!processor.engine.status().startsWith("Run failed"), processor.engine.status());
        }
        require(processor.engine.activeRevision() != previous, "Stateful filter activation");
        double k = std::tan(3.141592653589793 * 1200 / 48000), norm = 1 / (1 + std::sqrt(2.0) * k + k * k);
        double b0 = k * k * norm, b1 = 2 * b0, b2 = b0, a1 = 2 * (k * k - 1) * norm, a2 = (1 - std::sqrt(2.0) * k + k * k) * norm;
        double x1 = 0, x2 = 0, y1 = 0, y2 = 0, maximumError = 0;
        int position = 0;
        for (int count : {1, 17, 256, 1024, 4096}) {
            AudioBuffer<float> impulse(2, count); impulse.clear();
            if (position == 0) { impulse.setSample(0, 0, 1); impulse.setSample(1, 0, -1); }
            processor.processBlock(impulse, midi);
            for (int frame = 0; frame < count; ++frame, ++position) {
                double input = position == 0 ? 1 : 0;
                double expected = b0 * input + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                x2 = x1; x1 = input; y2 = y1; y1 = expected;
                maximumError = std::max(maximumError, std::abs(impulse.getSample(0, frame) - expected));
                require(std::abs(impulse.getSample(1, frame) + expected) < 2e-6, "Filter right impulse");
            }
        }
        require(maximumError < 2e-6, "Stateful Faust filter differs from independent biquad");
        runCode(processor,"@init\nchosen=1;\n@sample\nspl0*=chosen;spl1*=chosen;\n@gfx\nmouse_cap&1 ? chosen=gfx_showmenu(\"First|Second\");", "jsfx",1.0f);
        JsfxGfxMenuBridge menu;JitEngine::GraphicsInput menuInput;menuInput.menu=&menu;
        std::thread menuWorker([&]{processor.engine.renderGraphics(400,300,10,10,1,0,0,0,&menuInput);});
        String menuDescription;int mx=0,my=0;auto menuStart=Time::getMillisecondCounterHiRes();bool opened=false;
        while(Time::getMillisecondCounterHiRes()-menuStart<3000 && !(opened=menu.takePendingOpen(menuDescription,mx,my)))Thread::sleep(5);
        if(opened)menu.completeMenu(2);else menu.close();menuWorker.join();

        menu.close();require(opened && menuDescription=="First|Second","Compiled native modal GFX menu request");
        fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,2.0f);
        {JsfxGfxMenuBridge saveMenu;JitEngine::GraphicsInput saveInput;saveInput.menu=&saveMenu;std::thread worker([&]{processor.engine.renderGraphics(400,300,10,10,1,0,0,0,&saveInput);});auto start=Time::getMillisecondCounterHiRes();String text;int x,y;bool opened=false;while(Time::getMillisecondCounterHiRes()-start<3000 && !(opened=saveMenu.takePendingOpen(text,x,y)))Thread::sleep(5);if(!opened){saveMenu.close();worker.join();throw std::runtime_error("Snapshot menu opened");}MemoryBlock preset;processor.getStateInformation(preset);worker.join();require(preset.getSize()>0,"Saving state cancels a modal GFX menu without message-thread deadlock");}
        runCode(processor,"@init\ngain=1;\n@sample\nspl0*=gain;spl1*=gain;\n@gfx\ngfx_getchar()==65 ? gain=.25;mouse_wheel>0 ? gain=.5;", "jsfx",1.0f);
        processor.engine.renderGraphics(400,300,0,0,0,65);fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,.25f);
        processor.engine.renderGraphics(400,300,0,0,0,0,0,120);fill(buffer);processor.processBlock(buffer,midi);checkGain(buffer,.5f);
        runCode(processor,aliases,"jsfx",0.5f);
        std::unique_ptr<AudioProcessorEditor> editor(processor.createEditor());
        require(editor->getWidth() == 1150 && editor->getHeight() == 800, "Editor construction");
        editor->setSize(900,600);require(editor->getWidth()==900 && editor->getHeight()==600,"Editor resizing");
        CodeEditorComponent* sourceEditor=nullptr;for(int i=0;i<editor->getNumChildComponents();++i)if(auto* c=dynamic_cast<CodeEditorComponent*>(editor->getChildComponent(i)))sourceEditor=c;
        require(sourceEditor!=nullptr,"Source editor");require(!sourceEditor->isVisible(),"Default plugin view hides source");for(int i=0;i<editor->getNumChildComponents();++i)if(auto* button=dynamic_cast<TextButton*>(editor->getChildComponent(i)))if(button->getButtonText()=="Edit")button->onClick();require(sourceEditor->isVisible(),"Edit button reveals source");
        const auto original=processor.draft();
        require(sourceEditor->keyPressed(KeyPress('A',ModifierKeys::ctrlModifier,0)),"Ctrl+A");
        require(sourceEditor->keyPressed(KeyPress('C',ModifierKeys::ctrlModifier,0)),"Ctrl+C");
        require(SystemClipboard::getTextFromClipboard()==original,"Clipboard source text");
        for(int n=0;n<10 && SystemClipboard::getTextFromClipboard()!="replacement";++n){SystemClipboard::copyTextToClipboard("replacement");Thread::sleep(5);}require(SystemClipboard::getTextFromClipboard()=="replacement","Clipboard replacement write");sourceEditor->keyPressed(KeyPress('V',ModifierKeys::ctrlModifier,0));
        require(processor.draft()=="replacement","Ctrl+V actual: "+processor.draft()+" clipboard: "+SystemClipboard::getTextFromClipboard());sourceEditor->keyPressed(KeyPress('Z',ModifierKeys::ctrlModifier,0));
        require(processor.draft()==original,"Ctrl+Z");editor->setSize(1150,800);
        for(int i=0;i<editor->getNumChildComponents();++i)if(auto* button=dynamic_cast<TextButton*>(editor->getChildComponent(i)))if(button->getButtonText()=="Show plugin")button->onClick();require(!sourceEditor->isVisible(),"Plugin view restored");
        Thread::sleep(150);
        auto snapshot = editor->createComponentSnapshot(editor->getLocalBounds());
        auto imageFile = File::getCurrentWorkingDirectory().getChildFile("JITEditor-preview.png");
        if (auto stream = imageFile.createOutputStream()) {
            stream->setPosition(0); stream->truncate();
            PNGImageFormat format; format.writeImageToStream(snapshot, *stream);
        }
        editor.reset();
        processor.runSource("@sample\nspl0*=0.1;spl1*=0.1;", "jsfx"); processor.factory();
        Thread::sleep(500); processor.pollCompiledSchema(); fill(buffer); processor.processBlock(buffer, midi); checkGain(buffer, 1);
        runCode(processor, "@sample\nspl0*=0.5;spl1*=0.5;", "jsfx", 0.5f);
        const int iterations = 20000;
        began = Time::getMillisecondCounterHiRes();
        for (int i = 0; i < iterations; ++i) { fill(buffer); processor.processBlock(buffer, midi); }
        double jsfxMs = Time::getMillisecondCounterHiRes() - began;
        runCode(processor, "process=_,_:*(0.5),*(0.5);", "faust", 0.5f);
        began = Time::getMillisecondCounterHiRes();
        for (int i = 0; i < iterations; ++i) { fill(buffer); processor.processBlock(buffer, midi); }
        double faustMs = Time::getMillisecondCounterHiRes() - began;
        std::cout << "{\"passed\":true,\"jsfxCompileSeconds\":" << jsfx << ",\"mixedCompileSeconds\":" << mixed
                  << ",\"faustCompileSeconds\":" << faust << ",\"filterMaxError\":" << maximumError << ",\"blocks\":" << iterations << ",\"blockSize\":256,\"jsfxMs\":" << jsfxMs << ",\"faustMs\":" << faustMs << "}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
