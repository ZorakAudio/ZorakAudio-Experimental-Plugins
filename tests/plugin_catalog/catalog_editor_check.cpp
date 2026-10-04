// SPDX-License-Identifier: Zlib
// Production processor/editor lifecycle; never substitutes a graphics renderer.
#include <juce_audio_utils/juce_audio_utils.h>
#include <atomic>
#include <iostream>
#include <thread>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
#if CATALOG_JSFX
extern "C" bool za_sample_gfx_snapshot(juce::AudioProcessor*, const char*, double*, size_t*, size_t*, int64_t*);
extern "C" uint64_t za_sample_gfx_frames();
extern "C" bool za_sample_gfx_native();
extern "C" bool za_sample_gfx_legacy();
extern "C" void za_sample_gfx_load_bank(juce::AudioProcessor*, const char* const*, int);
#endif
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
static void pump(int ms) { juce::MessageManager::getInstance()->runDispatchLoopUntil(ms); }
static juce::Component* canvas(juce::Component& c) {
    if (c.getName() == "EEL JSFX graphics" || c.getName() == "Native JSFX graphics") return &c;
    for (auto* child : c.getChildren()) if (auto* found = canvas(*child)) return found;
    return nullptr;
}
static void save(juce::Component& c, const juce::File& d, const char* name) {
    auto out = d.getChildFile(name).createOutputStream(); require(out != nullptr, "Screenshot open failed");
    out->setPosition(0); out->truncate(); juce::PNGImageFormat png;
    require(png.writeImageToStream(c.createComponentSnapshot(c.getLocalBounds()), *out), "Screenshot encode failed");
}
static double read(juce::AudioProcessor* p, const char* name) {
#if CATALOG_JSFX
    double v = 0; size_t copied = 0, capacity = 0; int64_t logical = 0;
    return za_sample_gfx_snapshot(p, name, &v, &copied, &capacity, &logical) ? v : -1;
#else
    return -1;
#endif
}
static void until(auto condition, const char* message, int attempts = 240) {
    for (int i = 0; i < attempts && !condition(); ++i) pump(25);
    require(condition(), message);
}
static void makeBank(const juce::File& d, std::vector<std::string>& paths) {
    d.createDirectory();
    for (int item = 0; item < 3; ++item) {
        auto file = d.getChildFile("fixture-" + juce::String(item) + ".wav");
        auto out = file.createOutputStream(); require(out != nullptr, "WAV open failed"); out->setPosition(0); out->truncate();
        juce::WavAudioFormat wav; std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(out.get(), 48000, 2, 24, {}, 0));
        require(writer != nullptr, "WAV encoder failed"); out.release(); juce::AudioBuffer<float> b(2, 48000);
        for (int c = 0; c < 2; ++c) for (int i = 0; i < b.getNumSamples(); ++i) {
            double t = i / 48000.; b.setSample(c, i, float(.3 * std::min(1., t / .003) * std::exp(-t * (3 + item)) *
                std::sin(2 * juce::MathConstants<double>::pi * ((170 + item * 110) * t + 80 * t * t) + c * .03)));
        }
        require(writer->writeFromAudioSampleBuffer(b, 0, b.getNumSamples()), "WAV write failed"); paths.push_back(file.getFullPathName().toStdString());
    }
}
struct Transport : juce::AudioPlayHead {
    std::atomic<int64_t> samples{0};
    juce::Optional<PositionInfo> getPosition() const override {
        PositionInfo p; p.setTimeInSamples(samples.load()); p.setTimeInSeconds(samples.load() / 48000.);
        p.setPpqPosition(samples.load() / 24000.); p.setBpm(120); p.setIsPlaying(true); p.setTimeSignature(TimeSignature{4,4}); return p;
    }
};
struct Join {
    std::atomic<bool>& stop; std::thread& thread;
    ~Join() { stop = true; if (thread.joinable()) thread.join(); }
};
static void updatePeak(std::atomic<float>& peak, float v) { auto old = peak.load(); while (v > old && !peak.compare_exchange_weak(old, v)) {} }
int main(int argc, char** argv) try {
    require(argc == 4, "Pass output directory, plugin slug, gfx/no-gfx");
    juce::ScopedJuceInitialiser_GUI gui; juce::File directory(argv[1]); directory.createDirectory(); std::string slug(argv[2]);
    const bool gfx = std::string(argv[3]) == "gfx";
    std::unique_ptr<juce::AudioProcessor> p(createPluginFilter()), peer;
    Transport transport; p->setPlayHead(&transport); p->prepareToPlay(48000,512);
    std::unique_ptr<juce::AudioProcessorEditor> peerEditor;
    if (slug.starts_with("IPCProbe")) {
        peer.reset(createPluginFilter()); peer->setPlayHead(&transport); peer->prepareToPlay(48000,512);
        auto* role = dynamic_cast<juce::RangedAudioParameter*>(peer->getParameters()[0]); require(role != nullptr, "IPC role parameter missing");
        role->setValueNotifyingHost(role->convertTo0to1(1));
        peerEditor.reset(peer->createEditorIfNeeded()); require(peerEditor != nullptr, "IPC peer editor missing");
        peerEditor->addToDesktop(juce::ComponentPeer::windowIsTemporary); peerEditor->setVisible(true);
    }
    std::atomic<bool> stop{false}, bad{false}, silent{false}; std::atomic<int> noteRequest{0};
    std::atomic<uint64_t> blocks{0}; std::atomic<float> peak{0}, peerPeak{0};
    std::thread audio([&] {
        uint64_t position = 0; const int sizes[] = {64,128,256,511,17};
        while (!stop) {
            int n = sizes[blocks.load() % 5]; juce::AudioBuffer<float> b(std::max({2,p->getTotalNumInputChannels(),p->getTotalNumOutputChannels()}),n);
            for (int c = 0; c < b.getNumChannels(); ++c) for (int i = 0; i < n; ++i) b.setSample(c,i,silent ? 0 : float(.08 * std::sin((position+i)*.021+c*.1)));
            juce::MidiBuffer m; int request = noteRequest.exchange(0);
            if (request > 0 || (!silent && blocks % 100 == 0)) m.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);
            if (request < 0 || (!silent && blocks % 100 == 50)) m.addEvent(juce::MidiMessage::noteOff(1,60),0);
            p->processBlock(b,m);
            for (int c = 0; c < b.getNumChannels(); ++c) for (int i = 0; i < n; ++i) { float v=b.getSample(c,i); if(!std::isfinite(v))bad=true; updatePeak(peak,std::abs(v)); }
            if (peer) {
                juce::AudioBuffer<float> pb(std::max({2,peer->getTotalNumInputChannels(),peer->getTotalNumOutputChannels()}),n); pb.clear(); juce::MidiBuffer pm; peer->processBlock(pb,pm);
                for(int c=0;c<pb.getNumChannels();++c)for(int i=0;i<n;++i){float v=pb.getSample(c,i);if(!std::isfinite(v))bad=true;updatePeak(peerPeak,std::abs(v));}
            }
            ++blocks; position += n; transport.samples=position; std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
    }); Join join{stop,audio};
    int gfxLifetimes=0, bankFiles=-1; double grains=-1, received=-1;
    for (int lifetime=0;lifetime<2;++lifetime) {
        std::unique_ptr<juce::AudioProcessorEditor> e(p->createEditorIfNeeded()); require(e != nullptr,"Editor missing");
        e->addToDesktop(juce::ComponentPeer::windowIsTemporary); e->setVisible(true); pump(100);
#if CATALOG_JSFX
        require(!za_sample_gfx_native() && !za_sample_gfx_legacy(), "Non-Joep default mode changed");
        if (gfx) { require(canvas(*e)!=nullptr,"GFX canvas missing"); auto before=za_sample_gfx_frames(); until([&]{return za_sample_gfx_frames()>before;},"GFX worker made no progress"); ++gfxLifetimes; }
        if (!lifetime && (slug=="Sample" || slug=="Corpus")) {
            std::vector<std::string> bank; makeBank(directory.getChildFile("bank"),bank); std::vector<const char*> paths; for(auto& s:bank)paths.push_back(s.c_str());
            silent=true; za_sample_gfx_load_bank(p.get(),paths.data(),int(paths.size())); bankFiles=int(paths.size());
            if(slug=="Sample")until([&]{return read(p.get(),"sample_count")==3 && read(p.get(),"bank_loaded")==1;},"Sample bank not decoded",1200);
            else { until([&]{return read(p.get(),"grain_count")>0 && read(p.get(),"ready")==1;},"Corpus grains not ready",1200);
                until([&]{return read(p.get(),"pe_ready")==1 && read(p.get(),"pg_ready")==1 && read(p.get(),"playable_grains")>0 && read(p.get(),"pg_initializing")==0;},"Corpus analysis not ready",1200); grains=read(p.get(),"grain_count"); }
            peak=0; noteRequest=1; pump(600); require(peak>1e-6,"Bank produced no MIDI-triggered audio");
        }
        if(!lifetime && peer) { until([&]{return read(peer.get(),"rx_count")>0 && peerPeak>1e-6;},"IPC data/audio not received"); received=read(peer.get(),"rx_count"); }
#endif
        auto beforeBlocks=blocks.load(); pump(180); require(blocks>beforeBlocks,"Audio stopped during GUI work");
        save(*e,directory,lifetime?"reopened.png":"open.png");
        if (!lifetime) {
            for(int i=0;i<std::min(3,p->getParameters().size());++i) { auto* parameter=p->getParameters()[i]; float v=parameter->getValue(); parameter->beginChangeGesture(); parameter->setValueNotifyingHost(std::min(1.f,v+.035f)); parameter->endChangeGesture(); pump(60); parameter->setValueNotifyingHost(v); }
            e->setSize(e->getWidth()+120,e->getHeight()+90);pump(150);save(*e,directory,"resized.png");
        }
        e->setVisible(false); p->editorBeingDeleted(e.get()); e.reset(); pump(80);
    }
    if(peerEditor){peerEditor->setVisible(false);peer->editorBeingDeleted(peerEditor.get());peerEditor.reset();}
    stop=true;audio.join(); require(!bad && blocks>=20,"Audio failed finite/progress gate");
    juce::MemoryBlock state;p->getStateInformation(state);require(state.getSize()>0,"Empty host state");
    if(!p->getParameters().isEmpty()){auto* parameter=p->getParameters()[0];float v=parameter->getValue();parameter->setValueNotifyingHost(v<.5f?.8f:.2f);p->setStateInformation(state.getData(),int(state.getSize()));require(std::abs(parameter->getValue()-v)<1e-5f,"Parameter state restoration failed");}
    p->releaseResources();p->prepareToPlay(96000,512);
    for(int k=0;k<12;++k){juce::AudioBuffer<float>b(std::max({2,p->getTotalNumInputChannels(),p->getTotalNumOutputChannels()}),128);b.clear();juce::MidiBuffer m;p->processBlock(b,m);for(int c=0;c<b.getNumChannels();++c)for(int i=0;i<128;++i)require(std::isfinite(b.getSample(c,i)),"96kHz audio not finite");}
    p->releaseResources();if(peer)peer->releaseResources();p->setPlayHead(nullptr);
    std::cout<<"{\"worker_complete\":true,\"audio_blocks\":"<<blocks<<",\"finite_audio\":true,\"peak\":"<<peak<<",\"editor_lifetimes\":2,\"gfx_lifetimes\":"<<gfxLifetimes<<",\"host_parameter_state\":true,\"sample_rates\":[48000,96000],\"bank_files\":"<<bankFiles<<",\"grain_count\":"<<grains<<",\"ipc_received\":"<<received<<",\"ipc_peer_peak\":"<<peerPeak<<"}\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
