#pragma once
#include "JitExamples.h"
namespace {
void checkExamples(bool cpp,const juce::File& output){
    juce::Array<juce::var> results;
    for(auto rate:{44100,48000})for(auto& example:jitExamples){
        JitProcessor p;if(cpp)p.setFrontend("cpp-frontend");p.prepareToPlay(rate,4096);
        runCode(p,juce::String::fromUTF8(example.source),example.mode,std::numeric_limits<float>::quiet_NaN());
        require(p.getTotalNumInputChannels()==2 && p.getTotalNumOutputChannels()==2,"Example stereo pin inference");
        auto metadata=p.engine.compiledDescriptor().getProperty("metadata",{});
        if(juce::String(example.name).startsWith("Hybrid")){
            const auto stages=metadata.getProperty("faust_stages",{});int count=0;
            if(auto* list=stages.getArray())for(auto& stage:*list)if(stage.getProperty("kind",{}).toString()=="faust"){++count;require((bool)stage.getProperty("block_mode",false) && !(bool)stage.getProperty("fused",true),"Hybrid explicitly computes a non-fused full block");}
            require(count==1,"Hybrid has one Faust DSP stage");
        }
        juce::AudioBuffer<float> audio(2,256);juce::MidiBuffer midi;double peak=0;int position=0;
        auto block=[&](int count){audio.setSize(2,count);for(int f=0;f<count;++f,++position){auto x=.18f*std::sin(float(position)*.047f);audio.setSample(0,f,x);audio.setSample(1,f,.5f*x);}p.processBlock(audio,midi);
            for(int f=0;f<count;++f)for(int c=0;c<2;++c){auto v=audio.getSample(c,f);require(std::isfinite(v) && std::abs(v)<10,"Finite bounded example audio");peak=std::max(peak,(double)std::abs(v));}
        };
        for(int n=0;n<120;++n)block(256);for(int count:{1,17,256,1024,4096})block(count);
        require(peak>1e-4 && !p.engine.memoryFaulted(),"Example processes audible audio with variable block sizes");
        if(juce::String(example.name).contains("linked stereo compressor"))for(int f=0;f<audio.getNumSamples();++f)require(std::abs(audio.getSample(1,f)-.5f*audio.getSample(0,f))<1e-6,"Linked compressor preserves stereo ratio");
        if(juce::String(example.name).contains("stereo utility")){
            p.setSliderValue(1,0);block(256);for(int f=0;f<256;++f)require(audio.getSample(0,f)==audio.getSample(1,f),"JSFX width zero is mono");
        }
        int controls=(int)p.declarations.size();
        if(juce::String(example.name).startsWith("Hybrid")){
            const bool studio=juce::String(example.name).contains("studio");const int w=studio?860:540,h=studio?500:180;
            auto image=p.engine.renderGraphics(w,h,0,0,0);require(image.isValid(),"Hybrid interactive graphics render");
            const auto before=p.sliderValue(0);const double x=studio?20+.8*((860.-80)/3):20+.8*(540-40);const double y=studio?115:72;
            ParameterEvents changes;p.parameters[0]->addListener(&changes);p.engine.renderGraphics(w,h,x,y,1);p.engine.renderGraphics(w,h,x,y,0);pumpEditor();
            require(p.sliderValue(0)>before && !changes.events.empty(),"EEL drag changes actual automatable parameter");p.parameters[0]->removeListener(&changes);
            block(256); // Host/GFX writes are consumed at an audio callback boundary.
            if(studio){
                require(controls==12 && p.engine.inspectVariable("input_peak")>0 && p.engine.inspectVariable("output_peak")>0 && std::isfinite(p.engine.inspectVariable("gain_reduction")),"Faust exports real meters without additional audio pins");
                for(int slot=0;slot<11;++slot){const int column=slot<4?0:slot<8?1:2,row=slot<4?slot:slot<8?slot-4:slot-8;const double colWidth=(860.-80)/3;
                    const double dragX=20+column*(colWidth+20)+.6*colWidth,dragY=82+row*64+33;
                    p.engine.renderGraphics(w,h,dragX,dragY,1);p.engine.renderGraphics(w,h,dragX,dragY,0);auto& declaration=p.declarations[(size_t)slot];require(p.sliderValue(slot)>=declaration.min && p.sliderValue(slot)<=declaration.max,"Every hybrid drag updates a bounded parameter");
                    block(256);
                }
                const double col=(860.-80)/3,x3=60+2*col;p.engine.renderGraphics(w,h,x3+20,305,1);p.engine.renderGraphics(w,h,x3+20,305,0);require(p.sliderValue(11)==1,"EEL bypass button reaches DSP parameter");
                block(256);p.setSliderValue(10,0); // After UI gesture publication, use unity output for the dry-path comparison.
                for(int n=0;n<120;++n)block(256);
                for(int f=0;f<256;++f){const float expected=.18f*std::sin(float(position-256+f)*.047f);require(std::abs(audio.getSample(0,f)-expected)<1e-5,"Hybrid bypass settles to dry audio at unity output: actual="+juce::String(audio.getSample(0,f),8)+" expected="+juce::String(expected,8)+" hostOutput="+juce::String(p.sliderValue(10))+" dspOutput="+juce::String(p.engine.inspectVariable("output_db"))+" wet="+juce::String(p.engine.inspectVariable("channel_wet"))+" bypass="+juce::String(p.engine.inspectVariable("bypass")));}
            }
            if(output.isDirectory() && rate==48000){auto screenshot=p.engine.renderGraphics(w,h,0,0,0);auto file=output.getChildFile(studio?"studio-channel.png":"hybrid-gain.png");auto stream=file.createOutputStream();require(stream && juce::PNGImageFormat().writeImageToStream(screenshot,*stream),"Hybrid example screenshot");}
        }
        juce::MemoryBlock saved;p.getStateInformation(saved);require(saved.getSize()>juce::String::fromUTF8(example.source).length(),"Full example program stored in host preset state");
        if(juce::String(example.name).contains("studio")){JitProcessor restored;restored.prepareToPlay(rate,4096);restored.setStateInformation(saved.getData(),(int)saved.getSize());waitForRun(restored,0);
            require(restored.engine.appliedSource()==juce::String::fromUTF8(example.source) && restored.declarations.size()==12,"Host preset restores full complex program without original source file");
            for(int slot=0;slot<12;++slot)require(std::abs(restored.sliderValue(slot)-p.sliderValue(slot))<1e-5,"Complex preset restores all edited controls");
        }
        juce::DynamicObject::Ptr row=new juce::DynamicObject;row->setProperty("example",example.name);row->setProperty("sampleRate",rate);row->setProperty("controls",controls);row->setProperty("peak",peak);row->setProperty("passed",true);results.add(juce::var(row.get()));
    }
    juce::DynamicObject::Ptr summary=new juce::DynamicObject;summary->setProperty("passed",true);summary->setProperty("frontend",cpp?"cpp-frontend":"python-reference");summary->setProperty("cases",results);
    std::cout<<juce::JSON::toString(juce::var(summary.get()))<<'\n';
}
}
