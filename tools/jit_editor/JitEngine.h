#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>

namespace jsfx_gfx {struct AsyncMenuPort;}
class JitEngine final : private juce::Thread
{
public:
    explicit JitEngine(juce::File runtime = {});
    ~JitEngine() override;
    bool prepare(double sampleRate,int maximumBlock,const juce::var& currentControls = {});
    uint64_t submit(const juce::String& source, const juce::String& mode, const juce::String& sourcePath = {},const juce::var& serialized = {},const juce::String& frontend = "python-reference",const juce::var& initialControls = {},bool normalizedControls = false);
    void restoreFactory();
    void setTrackName(const juce::String&);
    juce::AudioProcessor* hostProcessor=nullptr; // Set once by the owning processor before Run.
    std::atomic<int> latencySamples{0};
    std::atomic<double> tailSeconds{0};
    std::atomic<bool> idleSleeping{false},graphicsVisible{false};
    std::atomic<int> oversamplingChoice{0};
    std::atomic<int> idleOverride{0}; // Production Auto/Input/Event/Free/Always/Cooperative modes.
    std::atomic<bool> resetRequested{false};
    void setFileSlot(int,const juce::StringArray&);
    void setStringSlider(int,const juce::String&);
    juce::String stringSlider(int);
    void process(juce::AudioBuffer<float>&,juce::MidiBuffer* = nullptr,const juce::AudioPlayHead::PositionInfo* = nullptr,bool offline = false);
    juce::String status() const;
    juce::String appliedSource() const;
    juce::String appliedSourcePath() const;
    juce::String appliedMode() const;
    juce::String appliedFrontend() const;
    uint64_t activeRevision() const { return activeId.load(); }
    std::array<std::atomic<double>, 256> controls;
    std::function<void()> compiledCallback;
    juce::var compiledDescriptor() const;
    void allowAdoption();
    void shutdown();
    juce::var serializeState(bool includeGuest = true);
    bool memoryFaulted();
    double inspectVariable(const juce::String&);
    double inspectGraphicsVariable(const juce::String&);
    void refreshSliderVisibility();
    struct GraphicsInput{jsfx_gfx::AsyncMenuPort* menu=nullptr;std::vector<int> keysDown;std::vector<juce::String> drops;bool focused=true;int cursor=32512;std::array<uint64_t,4> visible{~UINT64_C(0),~UINT64_C(0),~UINT64_C(0),~UINT64_C(0)};};
    juce::Image renderGraphics(int width, int height, double mouseX, double mouseY, int mouseCap, int key = 0, double wheelX = 0, double wheelY = 0, GraphicsInput* input = nullptr);
    std::function<void(uint64_t,int,double,int)> graphicsSliderChanged;
    std::function<void(uint64_t,const std::array<uint64_t,4>&)> sliderVisibilityChanged;
private:
    struct Program;
    struct Request { juce::String source, mode, sourcePath; uint64_t id = 0, epoch = 0; double rate = 48000; int capacity = 4096;juce::var serialized;juce::String frontend="python-reference";int factor=1;juce::var initialControls;bool normalizedControls=false; };
    void run() override;
    void retire(Program*);
    void collectRetired();
    void setStatus(juce::String);
    std::unique_ptr<Program> compile(const Request&);
    juce::File runtime;
    mutable juce::CriticalSection messageLock;
    juce::CriticalSection graphicsLock;
    std::mutex menuLock;
    jsfx_gfx::AsyncMenuPort* renderingMenu=nullptr;
    Request request;
    juce::String message = "Factory: stereo passthrough. Write code and press Run.";
    juce::String hostTrackName;
    juce::String applied, appliedOrigin, appliedLanguage = "jsfx", appliedCompiler = "python-reference";
    std::atomic<uint64_t> revision {0}, epoch {0}, activeId {0}, permittedRevision {0};
    std::atomic<double> rate {48000};
    std::atomic<int> capacity {4096};
    std::atomic<bool> factoryRequested {false};
    std::atomic<Program*> pending {nullptr}, retired {nullptr};
    // Compiler candidates remain private until the host primes controls and pins.
    std::atomic<Program*> adoptable {nullptr};
    std::atomic<Program*> displayed {nullptr}, graphicsReader {nullptr};
    std::atomic<bool> adoptionEnabled {true};
    juce::var readyDescriptor;
    Program* active = nullptr; // exclusively owned by audio thread while processing
};
