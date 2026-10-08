#pragma once
#include "JitEngine.h"
#include "JitHostInterface.h"
#include "JsfxSliderDeclarations.h"

class JitProcessor final : public juce::AudioProcessor, public JitHostInterface, private juce::AsyncUpdater
{
public:
    JitProcessor();
    ~JitProcessor() override;
    const juce::String getName() const override { return "ZorakAudio JIT Editor PoC"; }
    void prepareToPlay(double rate,int block) override;
    void releaseResources() override {engine.resetRequested.store(true);}
    void reset() override {engine.resetRequested.store(true);}
    bool isBusesLayoutSupported(const BusesLayout& layout) const override {
        return layout.getMainInputChannelSet().size() <= 64 && layout.getMainOutputChannelSet().size() <= 64;
    }
    void updateTrackProperties(const TrackProperties& properties)override{engine.setTrackName(properties.name.value_or(juce::String{}));}
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return true; }
    double getTailLengthSeconds() const override { return engine.tailSeconds.load(); }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;
    void runSource(const juce::String&, const juce::String&);
    void setFrontend(const juce::String&);
    juce::String frontend() const;
    void factory();
    void setSourcePath(const juce::String&);
    juce::String sourcePath()const;
    void setSourceFile(const juce::String&);
    juce::String sourceFile()const;
    juce::String draft() const;
    juce::String language() const;
    void updateDraft(const juce::String&, const juce::String&);
    JitEngine engine;
    std::array<juce::RangedAudioParameter*, 256> parameters {};
    juce::AudioParameterChoice* oversamplingParameter=nullptr;
    std::vector<JsfxSliderDecl> declarations;
    std::array<std::atomic<bool>,256> sliderVisible{};
    std::atomic<uint64_t> schemaRevision {0};
    void pollCompiledSchema();
    bool commitHostConfiguration() override;
    void endGfxGestures();
    void setSliderValue(int slot, double value);
    double sliderValue(int slot) const;
    bool resetSliderValue(int slot);
private:
    void handleAsyncUpdate() override {if(getLatencySamples()!=engine.latencySamples.load())setLatencySamples(engine.latencySamples.load());pollCompiledSchema();for(int i=0;i<256;++i)if(gfxPending[(size_t)i].exchange(false)){const auto events=gfxEvents[(size_t)i].exchange(0);if(gfxRevision[(size_t)i].load()==configured.load())applyGfxSlider(i,gfxValues[(size_t)i].load(),events);}}
    void applyGfxSlider(int,double,int);
    std::array<bool,256> gfxGestures{};
    std::array<std::atomic<int>,256> gfxEvents{};
    std::array<std::atomic<bool>,256> gfxPending{};
    std::array<std::atomic<double>,256> gfxValues{};
    std::array<std::atomic<uint64_t>,256> gfxRevision{};
    std::atomic<uint64_t> configured {0};
    juce::var pendingSchema;
    std::array<double,256> restoredValues {};
    bool restoring = false, restoredNormalized = false;
    std::array<int,256> slotIndex{};
    mutable juce::CriticalSection sourceLock;
    juce::String source = "desc:JIT Editor stereo gain\nslider1:0.5<0,1,0.01>Gain\n@sample\nspl0 *= slider1;\nspl1 *= slider1;\n";
    juce::String mode = "jsfx", origin, savedSourceFile, selectedFrontend="python-reference";
};
