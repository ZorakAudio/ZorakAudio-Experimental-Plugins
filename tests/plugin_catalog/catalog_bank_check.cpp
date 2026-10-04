// SPDX-License-Identifier: Zlib
// Loaded production services, including convolution and live TextureXY input.
#define main catalogLifecycleMain
#include "catalog_editor_check.cpp"
#undef main
static juce::MouseEvent event(juce::Component& c, juce::Point<float> point) {
    const auto now=juce::Time::getCurrentTime();
    return {juce::Desktop::getInstance().getMainMouseSource(),point,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),1,0,0,0,0,&c,&c,now,point,now,1,false};
}
int main(int argc,char** argv) try {
    require(argc==3,"Pass bank output directory and slug");juce::ScopedJuceInitialiser_GUI gui;
    juce::File directory(argv[1]);directory.createDirectory();std::string slug(argv[2]);
    std::vector<std::string> bank;makeBank(directory.getChildFile("bank"),bank);
    if(slug=="PsychoConvolver") {
        juce::File f(bank[0]);auto out=f.createOutputStream();require(out!=nullptr,"IR open failed");out->setPosition(0);out->truncate();
        juce::WavAudioFormat wav;std::unique_ptr<juce::AudioFormatWriter>w(wav.createWriterFor(out.get(),48000,2,24,{},0));require(w!=nullptr,"IR encoder failed");out.release();
        juce::AudioBuffer<float>b(2,3072);uint32_t seed=27;
        for(int c=0;c<2;++c)for(int i=0;i<3072;++i){seed=seed*1664525u+1013904223u;float noise=float((seed>>8)/8388607.5-1);b.setSample(c,i,i==0?.4f:float(.012*noise*std::exp(-i/420.)));}
        require(w->writeFromAudioSampleBuffer(b,0,3072),"IR write failed");
    }
    std::unique_ptr<juce::AudioProcessor>p(createPluginFilter());Transport transport;p->setPlayHead(&transport);p->prepareToPlay(48000,512);
    for(auto* parameter:p->getParameters())if(auto* ranged=dynamic_cast<juce::RangedAudioParameter*>(parameter)){
        auto name=parameter->getName(200);if(name.containsIgnoreCase("Output Mode"))ranged->setValueNotifyingHost(ranged->convertTo0to1(1));
        if(slug=="PsychoConvolver" && name.containsIgnoreCase("Wet"))ranged->setValueNotifyingHost(1);
    }
    std::unique_ptr<juce::AudioProcessorEditor>e(p->createEditorIfNeeded());require(e!=nullptr,"Editor missing");e->addToDesktop(juce::ComponentPeer::windowIsTemporary);e->setVisible(true);pump(100);
    std::atomic<bool>stop{false},bad{false};std::atomic<uint64_t>blocks{0};std::atomic<float>peak{0};
    std::thread audio([&]{uint64_t position=0;while(!stop){juce::AudioBuffer<float>b(std::max({2,p->getTotalNumInputChannels(),p->getTotalNumOutputChannels()}),256);for(int c=0;c<b.getNumChannels();++c)for(int i=0;i<256;++i)b.setSample(c,i,slug=="TextureXY"?0:float(.2*std::sin((position+i)*.021+c*.1)));juce::MidiBuffer m;p->processBlock(b,m);for(int c=0;c<b.getNumChannels();++c)for(int i=0;i<256;++i){float v=b.getSample(c,i);if(!std::isfinite(v))bad=true;updatePeak(peak,std::abs(v));}position+=256;transport.samples=position;++blocks;std::this_thread::sleep_for(std::chrono::milliseconds(4));}});Join join{stop,audio};
    std::vector<const char*>paths;for(auto& s:bank)paths.push_back(s.c_str());za_sample_gfx_load_bank(p.get(),paths.data(),int(paths.size()));
    // Snapshots carry only GFX-published fields; private DSP flags are not valid readiness oracles.
    const char* loaded=slug=="PsychoConvolver"?"raw_loaded":slug=="TextureXY"?"tex_frames":"tex_loaded";
    const char* ready=slug=="PsychoConvolver"?"ir_ready":slug=="TexturePM"?"tex_loaded":"cand_count";
    try{until([&]{return read(p.get(),loaded)>0 && read(p.get(),ready)>0;},"Loaded DSP service not ready",1200);}
    catch(...){save(*e,directory,"not-ready.png");for(auto* name:{"raw_loaded","ir_ready","tex_loaded","tex_frames","wf_ready","cand_count","analysis_ready"})std::cerr<<name<<'='<<read(p.get(),name)<<'\n';throw;}
    peak=0;int pathPixels=-1,strayPixels=-1,waveformRows=-1;
    if(slug=="TextureXY"){
        auto* view=canvas(*e);require(view!=nullptr,"TextureXY canvas missing");view->mouseDown(event(*view,{130,140}));pump(90);
        for(int i=0;i<12;++i){view->mouseDrag(event(*view,{float(130+i*12),float(140+i*5)}));pump(90);}
        // point_count is UI-owned (FROM_GFX), so a DSP->GFX snapshot reports zero.
        // Verify the actual renderer's path and reject the stale-zero fan observed
        // when DSP publications overwrite the UI-owned gesture arrays.
        auto image=view->createComponentSnapshot(view->getLocalBounds());pathPixels=strayPixels=0;
        for(int y=100;y<std::min(580,image.getHeight());++y)for(int x=32;x<std::min(600,image.getWidth());++x){auto colour=image.getPixelAt(x,y);if(colour.getGreen()>120 && colour.getRed()<90 && colour.getBlue()>70){++pathPixels;if(x<110 || x>290 || y<115 || y>220)++strayPixels;}}
        save(*e,directory,"drawing.png");view->mouseUp(event(*view,{262,195}));peak=0;pump(300);
        std::cerr<<"TextureXY path pixels="<<pathPixels<<" stray="<<strayPixels<<" post-release peak="<<peak<<'\n';
        require(pathPixels>50 && strayPixels<50,"TextureXY path corrupted by stale heap data");require(peak>1e-6,"TextureXY released gesture produced no audio");
    }else pump(1200);
    require(peak>1e-6,"Loaded wet path produced no audio");require(!bad,"Loaded audio was not finite");if(slug=="PsychoConvolver")require(peak<4,"Broadband IR produced excessive gain");
    if(slug=="Contour" || slug=="Texture") {
        auto* view=canvas(*e);require(view!=nullptr,"Waveform canvas missing");auto image=view->createComponentSnapshot(view->getLocalBounds());waveformRows=0;
        for(int y=100;y<std::min(235,image.getHeight());++y){bool row=false;for(int x=32;x<image.getWidth()-32;++x){auto c=image.getPixelAt(x,y);if(c.getBlue()>150 && c.getGreen()>140 && c.getRed()>100){row=true;break;}}waveformRows+=row;}
        double value=0;size_t copied=0,capacity=0;int64_t logical=0;za_sample_gfx_snapshot(p.get(),"wf_ready",&value,&copied,&capacity,&logical);
        std::cerr<<"Waveform rows="<<waveformRows<<" copied="<<copied<<" logical="<<logical<<" ready="<<value<<'\n';
        if(waveformRows<=20){save(*e,directory,"flat-waveform.png");throw std::runtime_error("Loaded waveform did not render");}
    }
    save(*e,directory,"loaded.png");stop=true;audio.join();e->setVisible(false);p->editorBeingDeleted(e.get());e.reset();p->releaseResources();p->setPlayHead(nullptr);
    std::cout<<"{\"worker_complete\":true,\"loaded\":true,\"ready\":true,\"loaded_variable\":\""<<loaded<<"\",\"ready_variable\":\""<<ready<<"\",\"audio_blocks\":"<<blocks<<",\"peak\":"<<peak<<",\"finite_audio\":true,\"ui_path_pixels\":"<<pathPixels<<",\"ui_path_stray_pixels\":"<<strayPixels<<",\"waveform_rows\":"<<waveformRows<<",\"fixture_files\":3}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
