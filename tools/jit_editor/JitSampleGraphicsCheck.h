#pragma once
namespace {
void checkSampleGraphics(const juce::File& sourceFile){
    using namespace juce;
    require(sourceFile.existsAsFile(),"Sample source exists");
    auto folder=File::getSpecialLocation(File::tempDirectory).getNonexistentChildFile("za-runtime-bank","",false);
    struct Cleanup {File file;~Cleanup(){if(file.getParentDirectory()==File::getSpecialLocation(File::tempDirectory) && file.getFileName().startsWith("za-runtime-bank"))file.deleteRecursively();}} cleanup{folder};
    auto paths=makeRuntimeBank(folder,61);
    JitProcessor p;p.prepareToPlay(48000,4096);p.setSourcePath(sourceFile.getFullPathName());p.setNonRealtime(true);
    runCode(p,sourceFile.loadFileAsString(),"jsfx",std::numeric_limits<float>::quiet_NaN(),256,300000);
    p.engine.setFileSlot(0,paths);
    AudioBuffer<float> audio(2,256);MidiBuffer midi;bool ready=false;const auto started=Time::getMillisecondCounterHiRes();
    while(Time::getMillisecondCounterHiRes()-started<60000){
        audio.clear();midi.clear();p.processBlock(audio,midi);
        require(!p.engine.memoryFaulted(),"Sample loading memory fault");
        if(p.engine.inspectVariable("sample_count")==61 && p.engine.inspectVariable("bank_loaded")>0 && p.engine.inspectVariable("rescan_pending")==0){ready=true;break;}
        Thread::sleep(2);
    }
    require(ready,"All 61 synthetic samples loaded and analyzed");
    p.setSliderValue(31,6);p.setSliderValue(32,-4);p.setSliderValue(33,9);
    audio.clear();p.processBlock(audio,midi);
    Image reference;for(int i=0;i<5;++i)reference=p.engine.renderGraphics(1600,900,0,0,0).createCopy();
    const double mx=p.engine.inspectGraphicsVariable("map_x"),my=p.engine.inspectGraphicsVariable("map_y"),mw=p.engine.inspectGraphicsVariable("map_w"),mh=p.engine.inspectGraphicsVariable("map_h");
    const int gx=roundToInt(p.engine.inspectGraphicsVariable("proc_graph_x")),gy=roundToInt(p.engine.inspectGraphicsVariable("proc_graph_y")),gw=roundToInt(p.engine.inspectGraphicsVariable("proc_graph_w")),gh=roundToInt(p.engine.inspectGraphicsVariable("proc_graph_h"));
    require(mx>=0 && mw>61 && gh>10 && gx+gw<=1600,"Sample chart/EQ layout is valid");
    // gfx_rect uses floored coordinates. At this zoom a bar is only two
    // pixels wide, so rounded logical centers can land in the one-pixel gap.
    auto barCount=[&](const Image& image){int count=0;for(int i=0;i<61;++i){const int x=(int)std::floor(mx+i*mw/61+1);auto pixel=image.getPixelAt(x,roundToInt(my+mh-8));count+=pixel.getRed()>50 && pixel.getGreen()>40;}return count;};
    const auto evidenceFolder=File::getCurrentWorkingDirectory().getChildFile("build/runtime-unification");
    if(auto stream=evidenceFolder.getChildFile("sample-gfx-reference.png").createOutputStream()){stream->truncate();PNGImageFormat().writeImageToStream(reference,*stream);}
    std::cout<<"REFERENCE: map="<<mx<<','<<my<<','<<mw<<','<<mh<<" bars="<<barCount(reference)<<" sampleCount="<<p.engine.inspectGraphicsVariable("sample_count")<<" EQ="<<gx<<','<<gy<<','<<gw<<','<<gh<<std::endl;
    require(barCount(reference)==61,"Serial reference contains all sample bars");
    auto orange=[](Colour pixel){return pixel.getRed()>220 && pixel.getGreen()>145 && pixel.getGreen()<205 && pixel.getBlue()<100;};
    std::vector<Point<int>> curve;
    for(int x=gx+12;x<gx+gw-12;++x)for(int y=gy+15;y<gy+gh-15;++y)if(orange(reference.getPixelAt(x,y)))curve.push_back({x,y});
    require(curve.size()>100,"Serial reference contains the non-flat EQ response");
    std::atomic<int> blocks{0};std::atomic<bool> finite{true};std::atomic<double> peak{0};
    int missingBars=0,missingEqFrames=0,minimumBars=61;Image finalFrame;
    {
        std::jthread worker([&](std::stop_token stop){
            AudioBuffer<float> buffer(2,256);MidiBuffer notes;
            while(!stop.stop_requested()){
                const int block=blocks.load();buffer.clear();notes.clear();
                if(block%16==0){notes.addEvent(MidiMessage::allNotesOff(1),0);notes.addEvent(MidiMessage::noteOn(1,48+(block/16)%24,(uint8)100),1);}
                p.processBlock(buffer,notes);
                double maxValue=peak.load();for(int c=0;c<2;++c)for(int i=0;i<256;++i){const auto value=buffer.getSample(c,i);if(!std::isfinite(value))finite.store(false);maxValue=std::max(maxValue,(double)std::abs(value));}peak.store(maxValue);blocks.fetch_add(1);
            }
        });
        for(int frame=0;frame<200;++frame){
            finalFrame=p.engine.renderGraphics(1600,900,0,0,0).createCopy();const int bars=barCount(finalFrame);minimumBars=std::min(minimumBars,bars);missingBars+=bars!=61;
            int missing=0;for(auto point:curve){bool found=false;for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)found|=orange(finalFrame.getPixelAt(point.x+dx,point.y+dy));missing+=!found;}
            missingEqFrames+=missing>int(curve.size()/100);
        }
    }
    std::cout<<"SAMPLE GFX STRESS: files=61 frames=200 audioBlocks="<<blocks.load()<<" noteRetriggers="<<(blocks.load()+15)/16<<" badBankFrames="<<missingBars<<" minimumBars="<<minimumBars<<" badEqFrames="<<missingEqFrames<<" curvePixels="<<curve.size()<<" peak="<<peak.load()<<"\n";
    require(finite.load() && peak.load()>1e-6 && !p.engine.memoryFaulted(),"Sample concurrent MIDI audio remains finite and audible");
    require(blocks.load()>32 && missingBars==0 && missingEqFrames==0,"Concurrent MIDI must preserve all sample bars and the unchanged EQ response");
    auto evidence=evidenceFolder.getChildFile("sample-gfx-fixed.png");
    if(auto stream=evidence.createOutputStream()){stream->truncate();PNGImageFormat().writeImageToStream(finalFrame,*stream);}
}
}
