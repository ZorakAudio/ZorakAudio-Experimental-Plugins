#include "JitDeclarations.h"
#include "JitEngine.h"
#define NOMINMAX
#include "JitPlatform.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>

#include "JitBridge.h"
#include "JsfxCompiledProgram.h"
#include "JsfxStateVariables.h"
#include "JsfxStateReset.h"
#include "JsfxHeapMemory.h"
#include "JsfxProcessingSafety.h"
#include "JsfxGraphicsVariables.h"
#include "JsfxHostEnvironment.h"
#include "JsfxAudioHost.h"
#include "JsfxSliderDeclarations.h"
#include "YSFXGfxInterpreter.h"
#include "JsfxParameterState.h"
#include "JsfxGfxResources.h"
extern "C" void jsfx_gfx_aot(DSPJSFX_State*) {} // Frame::run is replaced by the instance JIT entrypoint.
#include "NativeGfxPrototype.h"
#include "WDL/fft.h"
#include "JsfxNumericBuiltins.h"
#include "JsfxLegacyAtomics.h"
static void noteTrackedJsfxMemUsed(DSPJSFX_State* st,int64_t used){if(st)st->memUsed=std::max(st->memUsed,used);}
#include "JsfxMidiBuiltins.h"
#include "JsfxMidiHost.h"
#include "JsfxIdleRuntime.h"
#include "DspJsfxRuntime.h"
#include "JsfxTasks.h"
#include "JsfxFaustEngine.h"
extern "C" void jsfx_faust_process(DSPJSFX_State* state,const float* const* in,float* const* out,int32_t channels,int32_t count) {za::jsfx::processFaustContext<za::jsfx::FaustEngine>(state,in,out,channels,count);}
#include "JsfxFileRuntime.h"
static za::jsfx::FileRuntime* jitFileOwner(DSPJSFX_State* state){auto* frame=jsfx_native_gfx::owner(state);return frame?static_cast<za::jsfx::FileRuntime*>(frame->dynamicPoolOwner):nullptr;}
#define JSFX_SAMPLE_POOL_OWNER(state) jitFileOwner(state)
#include "JsfxSamplePoolBuiltins.h"
#undef JSFX_SAMPLE_POOL_OWNER
#define JSFX_FILE_RUNTIME_OWNER(state) jitFileOwner(state)
#include "JsfxFileBuiltins.h"
#undef JSFX_FILE_RUNTIME_OWNER

#include "JsfxMemoryBuiltins.h"

#include "JsfxSliderBuiltins.h"
extern "C" double jsfx_file_string(DSPJSFX_State* st,double handle,double dest){double* args[]={&handle,&dest};return jsfx_native_gfx_dispatch(st,DSPJSFX_FILE_STRING,args,2);}

namespace {
using namespace juce;
[[noreturn]] void fail(const String& text) { throw std::runtime_error(text.toStdString()); }
var property(const var& v, const char* name) { return v.getProperty(name, {}); }
int integer(const var& v, const char* name) { return static_cast<int>(property(v, name)); }
std::vector<var> array(const var& value) {
    std::vector<var> result;
    if (auto* items = value.getArray()) for (auto& item : *items) result.push_back(item);
    return result;
}
struct LinkElement { uint8_t kind; const char* value; size_t length; };
struct SymbolAddress { const char* name; uint64_t address; };
struct LLVMApi {
    jit_platform::Module module = nullptr;
    void* (*create)(void*, bool, bool, char**) = nullptr;
    void (*dispose)(void*) = nullptr;
    void* (*link)(void*, const char*, LinkElement*, size_t, SymbolAddress*, size_t, SymbolAddress*, size_t, char**) = nullptr;
    bool (*release)(void*, char**) = nullptr;
    void (*freeString)(char*) = nullptr;
    template<class T> T symbol(const char* name) {
        auto result = reinterpret_cast<T>(jit_platform::symbol(module, name));
        if (!result) fail(String("Bundled LLVM missing symbol: ") + name);
        return result;
    }
    explicit LLVMApi(const File& runtime) {
        module = jit_platform::loadLLVM(runtime);
        create = symbol<decltype(create)>("LLVMPY_CreateLLJITCompiler");
        dispose = symbol<decltype(dispose)>("LLVMPY_LLJITDispose");
        link = symbol<decltype(link)>("LLVMPY_LLJIT_Link");
        release = symbol<decltype(release)>("LLVMPY_LLJIT_Dylib_Tracker_Dispose");
        freeString = symbol<decltype(freeString)>("LLVMPY_DisposeString");
        symbol<void (*)()>("LLVMPY_InitializeNativeTarget")();
        symbol<void (*)()>("LLVMPY_InitializeNativeAsmPrinter")();
    }
    String error(char* text) {
        String result = text ? String::fromUTF8(text) : "Unknown LLVM error";
        if (text) freeString(text);
        return result;
    }
};
std::mutex llvmMutex;
std::map<String, std::unique_ptr<LLVMApi>> llvmLibraries;
LLVMApi& llvmFor(const File& runtime) {
    auto key = runtime.getFullPathName();
    auto& value = llvmLibraries[key];
    if (!value) value = std::make_unique<LLVMApi>(runtime);
    return *value;
}
uint64_t mathAddress(const String& name) {
    if(name=="memcpy")return reinterpret_cast<uint64_t>(&std::memcpy);
    if(name=="memmove")return reinterpret_cast<uint64_t>(&std::memmove);
    if(name=="memset")return reinterpret_cast<uint64_t>(&std::memset);
    if(name=="malloc")return reinterpret_cast<uint64_t>(&std::malloc);
    if(name=="calloc")return reinterpret_cast<uint64_t>(&std::calloc);
    if(name=="free")return reinterpret_cast<uint64_t>(&std::free);
#define UNARY(x) if (name == #x) return reinterpret_cast<uint64_t>(static_cast<double (*)(double)>(&std::x));
    UNARY(sin) UNARY(cos) UNARY(tan) UNARY(asin) UNARY(acos) UNARY(atan)
    UNARY(sinh) UNARY(cosh) UNARY(tanh) UNARY(exp) UNARY(exp2) UNARY(log)
    UNARY(log2) UNARY(log10) UNARY(sqrt) UNARY(floor) UNARY(ceil) UNARY(round)
    UNARY(trunc) UNARY(fabs)
#undef UNARY
#define BINARY(x) if (name == #x) return reinterpret_cast<uint64_t>(static_cast<double (*)(double, double)>(&std::x));
    BINARY(atan2) BINARY(pow) BINARY(fmod) BINARY(remainder) BINARY(copysign) BINARY(fmin) BINARY(fmax)
#undef BINARY
#define JSFX_RUNTIME_EXPORT(x) if(name==#x)return reinterpret_cast<uint64_t>(&x);
#include "JsfxRuntimeExports.inc"
#undef JSFX_RUNTIME_EXPORT
    fail("Unsupported native import: " + name);
}

double eelStore(double value) {
    uint64_t bits; std::memcpy(&bits, &value, 8);
    auto exponent = (bits >> 52) & 2047;
    return exponent == 0 || exponent == 2047 ? 0.0 : value;
}
}

struct JitEngine::Program : za::jsfx::MidiHostRuntime,za::jsfx::IdleRuntime,za::jsfx::AudioHostRuntime,za::jsfx::ParameterState {
    bool offline=false;
    bool idleIsNonRealtime()const noexcept override{return offline;}
    std::atomic<bool> pendingWake{false};
    using Section = za::jsfx::CompiledProgram<DSPJSFX_State>::Section;
    using Bulk = void (*)(void*, double*, int, int, int, int, double**);
    using Compute = void (*)(void*, int, double**, double**);
    struct FaustStorage {std::vector<za::jsfx::Zone> zones;std::vector<za::jsfx::Binding> signals,exports;std::vector<int> tables;};
    struct Slider { int slot; double minimum,maximum,step;std::vector<int> aliases;bool isChoice=false,reversed=false;double rangeStart=0; };
    struct StringControl {int slot;double handle;std::string text;};
    std::vector<StringControl> stringControls;
    za::jsfx::RuntimeMidiNoteTracker pendingOutputNoteEnds;
    std::atomic<bool> fileRateReload{false};
    int factor=1;
    LLVMApi* api = nullptr;
    void* jit = nullptr;
    void* tracker = nullptr;
    Request request;
    var descriptor;
    za::jsfx::StateVariables variables;
    std::mutex atomicMutex,lifecycleMutex;
    DSPJSFX_State state{};
    std::unique_ptr<jsfx_native_gfx::Frame> graphics;
    za::jsfx::HostTransportTracker transportTracker;
    za::jsfx::HostVariableBindings hostVariableBindings;
    std::unique_ptr<za::jsfx::DspJsfxRuntime> hostRuntime;
    std::unique_ptr<jsfx_tasks::Runtime> tasks;std::unique_ptr<za::jsfx::FileRuntime> pools;
    Image canvas;
    int channelCount = 2;
    std::vector<FaustStorage> faustStorage;std::unique_ptr<za::jsfx::FaustEngine> faust;
    std::vector<Slider> sliders;
    jsfx_gfx::SliderMask graphicsTouch;
    za::jsfx::GraphicsVariables graphicsVariables;
    bool hasGraphics=false;
    std::array<uint64_t,4> reportedVisibility{};bool reportedVisibilityValid=false;
    void reportVisibility(const std::function<void(uint64_t,const std::array<uint64_t,4>&)>& callback){
        std::array<uint64_t,4> visible;for(int i=0;i<4;++i)visible[(size_t)i]=std::atomic_ref<uint64_t>(state.sliderVisibleMask[i]).load();
        if(callback && (!reportedVisibilityValid || visible!=reportedVisibility)){callback(request.id,visible);reportedVisibility=visible;reportedVisibilityValid=true;}
    }
    za::jsfx::CompiledProgram<DSPJSFX_State> compiled;
    int splOffset = 0, sliderOffset = 0, varsOffset = 0, rateOffset = 0, countOffset = 0;
    int capacity = 0;
    int sizeOffset = 0;
    Program* next = nullptr;
    double read(int offset) const { double value; value=jsfxCellLoad(reinterpret_cast<const double*>(reinterpret_cast<const char*>(&state)+offset)); return value; }
    void write(int offset, double value) { jsfxCellStore(reinterpret_cast<double*>(reinterpret_cast<char*>(&state)+offset),value); }
    double readVar(int index) const{return jsfxCellLoad(&state.vars[index]);}
    void writeVar(int index,double value){jsfxCellStore(&state.vars[index],value);}
    double get(const char* name)const{const int i=graphics->findIndex(name);if(i<0)return 0;const int alias=graphics->aliasFor(i);return alias>=0?double(state.sliders[alias]):readVar(i);}
    void set(int i,double value){if(i<0)return;const int alias=graphics->aliasFor(i);if(alias>=0)state.sliders[alias]=value;else writeVar(i,value);}
    void set(const char* name,double value){set(graphics->findIndex(name),value);}
    void applyGraphicsVariables(){graphicsVariables.apply(state,[this](int slot,double value){state.sliders[slot]=value;retainScriptSlider(slot,value);for(auto& item:sliders)if(item.slot==slot)for(int index:item.aliases)writeVar(index,value);});}
    bool applyFactor(int requested,const std::array<std::atomic<double>,256>& controls) {
        if(requested==factor)return false;
        std::unique_lock lock(lifecycleMutex,std::try_to_lock);
        if(!lock.owns_lock())return false;
        auto* st=&state;
        const auto extent=za::jsfx::variableCount(*st);
        pendingOutputNoteEnds=midiOutputNoteTracker;
        for(auto& item:stringControls)item.text=midiRuntime.nativeStrings.read(item.handle);
        if(tasks)tasks->reset();
        za::jsfx::resetGuestStatePreservingHeap(*st,variables,(size_t)extent);
        lastSlidersValid=false;internalSliderPendingMask.clear();factor=requested;st->srate=request.rate*factor;st->currentSampleRate=st->srate;
        graphicsTouch.clear();
        st->hostOwner=graphics.get();st->atomicContext=&atomicMutex;st->taskContext=tasks.get();st->faustContext=faust.get();
        hostRuntime->reset();prepareMidiRuntime(*st,capacity);transportTracker.reset();
        midiRuntime.nativeStrings.graphics=graphics.get();
        for(auto& item:stringControls){midiRuntime.nativeStrings.write(item.handle,item.text);st->sliders[item.slot]=item.handle;}
        for(auto& item:sliders){const double value=controls[(size_t)item.slot].load();st->sliders[item.slot]=value;lastSliders[(size_t)item.slot]=value;}
        lastSlidersValid=true;
        st->sliderVisibilityInit=1;for(auto& word:st->sliderVisibleMask)word=~UINT64_C(0);
        for(auto& item:array(property(descriptor,"sliders")))if((bool)property(item,"hidden"))st->sliderVisibleMask[integer(item,"slot")/64]&=~(UINT64_C(1)<<(integer(item,"slot")%64));
        za::jsfx::initialiseHostDefaults(channelCount,hostVariableBindings,[this](int index,double value){set(index,value);});
        graphics->resetDrawing();graphics->bindLegacy(*st,graphics->frameWidth,graphics->frameHeight);
        for(auto* name:{"gfx_r","gfx_g","gfx_b","gfx_a","gfx_a2"})set(name,1);
        set("gfx_dest",-1);set("gfx_texth",8);
        updateSmartIdleSampleRate(double(st->srate));resetSmartIdleRuntime();
        if(faust)faust->reset(double(st->srate));
        compiled.initialiseAndPrime(*st,[this,st]{for(auto& item:sliders)for(int index:item.aliases)writeVar(index,st->sliders[item.slot]);});
        graphics->ownedVariables.bind(graphics->state,(size_t)st->varsN);graphicsVariables.seed(*st,graphics->state);
        st->pendingNoteCleanup=1;pools->cacheTargetRate.store(factor>1?request.rate*factor:0);fileRateReload.store(true);pendingWake.store(true);
        return true;
    }
    ~Program() {
        // Destruction and ORC unloading happen only on worker/destructor threads.
        tasks.reset();hostRuntime.reset();
        pools.reset();
        za::jsfx::releaseHeap(state);
        faust.reset();
        std::lock_guard<std::mutex> lock(llvmMutex);
        if (tracker) { char* error = nullptr; api->release(tracker, &error); if (error) api->freeString(error); }
        if (jit) api->dispose(jit);
    }
    bool syncControls(const std::array<std::atomic<double>,256>& controls) {
        bool changed=false,scriptChange=false;auto& native=state;
        for(auto& item:sliders){
            const auto bit=UINT64_C(1)<<(item.slot%64);const int word=item.slot/64;
            const uint64_t pending=std::atomic_ref<uint64_t>(native.pendingSliderChangeMask[word]).load() |
                std::atomic_ref<uint64_t>(native.pendingSliderAutomateMask[word]).load() |
                std::atomic_ref<uint64_t>(native.pendingSliderAutomateEndMask[word]).load();
            // GFX may have notified a script write before the host's float
            // parameter callback arrives. Retain the original double first.
            if(pending&bit){retainScriptSlider(item.slot,read(sliderOffset+item.slot*8));scriptChange=true;}
            const double host=controls[(size_t)item.slot].load(std::memory_order_relaxed);
            const double value=resolveHostSlider(item.slot,item,host,host);
            changed|=applyHostSlider(native,item.slot,value,[&](double v){for(int index:item.aliases)writeVar(index,v);});
        }
        lastSlidersValid=true;if(changed || scriptChange)compiled.parametersChanged(native);return changed || scriptChange;
    }
    IdleBlock process(AudioBuffer<float>& buffer, const std::array<std::atomic<double>, 256>& controls,bool wake,const Topology& topology) {
        const int count = safeBlockSize(buffer.getNumSamples(),factor);
        if(count>capacity){state.memoryFault=1;buffer.clear();return {SmartIdleMode::AlwaysAwake,false,true};}
        write(countOffset, count);
        std::memcpy(reinterpret_cast<char*>(&state) + sizeOffset, &count, 4);
        const bool filesPromoted=pools && pools->promotePendingFileLoads();
        const bool changed=syncControls(controls);
        const bool eventWake=pendingWake.exchange(false);
        const bool poolAdoption=pools->hasDeferredSamplePoolAdoptionPending();
        auto decision=beginIdleBlock(buffer.getNumSamples(),hostInPtrs.data(),topology.inputs,wake || filesPromoted || changed || eventWake || poolAdoption);
        if(decision.skip)buffer.clear();else compiled.processAudio(state,inPtrs.data(),outPtrs.data(),topology.channels,count);
        return decision;
    }
};

JitEngine::JitEngine(File location) : Thread("JIT Editor compiler"), runtime(location.isDirectory() ? location : jit_platform::findRuntime(reinterpret_cast<const void*>(&eelStore))) {
    for (auto& control : controls) control.store(0.5f);
    startThread();
}
void JitEngine::shutdown() { signalThreadShouldExit(); notify(); stopThread(-1); }
JitEngine::~JitEngine() {
    shutdown();
    delete pending.exchange(nullptr);
    delete adoptable.exchange(nullptr);
    delete active;
    collectRetired();
}
void JitEngine::setStatus(String value) { const ScopedLock lock(messageLock); message = std::move(value); }
String JitEngine::status() const {
    const ScopedLock lock(messageLock);
    if (message.startsWith("Compiled successfully") && activeId.load() == revision.load())
        return (appliedLanguage=="faust" ? String("Running FAUST program.") : appliedCompiler=="cpp-frontend" ? String("Running with C++ frontend + production LLVM compiler.") : String("Running with standard compiler.")) + " Run resets DSP history; Default restores passthrough.";
    if(message.startsWith("Compiled successfully") && adoptionEnabled.load())return "Ready; GFX is active. Audio starts on the next processing block.";
    if (message.startsWith("Factory requested") && !factoryRequested.load()) return "Factory: stereo passthrough.";
    return message;
}
void JitEngine::setTrackName(const String& name){{const ScopedLock lock(messageLock);hostTrackName=name;}const ScopedLock lock(graphicsLock);Program* p=nullptr;do{p=displayed.load();graphicsReader.store(p);}while(p!=displayed.load());if(p && p->hostRuntime)p->hostRuntime->setHostTrackName(name.toStdString());graphicsReader.store(nullptr);}
void JitEngine::setStringSlider(int slot,const String& text){if(slot<0||slot>=256)return;const ScopedLock lock(graphicsLock);Program* p=nullptr;do{p=displayed.load();graphicsReader.store(p);}while(p!=displayed.load());if(p && p->graphics){auto h=p->read(p->sliderOffset+slot*8);p->midiRuntime.nativeStrings.write(h,text.toStdString());p->pendingWake.store(true);}graphicsReader.store(nullptr);}
String JitEngine::stringSlider(int slot){if(slot<0||slot>=256)return {};const ScopedLock lock(graphicsLock);Program* p=nullptr;do{p=displayed.load();graphicsReader.store(p);}while(p!=displayed.load());String value;if(p && p->graphics)value=String::fromUTF8(p->midiRuntime.nativeStrings.read(p->read(p->sliderOffset+slot*8)).c_str());graphicsReader.store(nullptr);return value;}
void JitEngine::setFileSlot(int slot,const StringArray& paths){const ScopedLock lock(graphicsLock);Program* p=nullptr;do{p=displayed.load();graphicsReader.store(p);}while(p!=displayed.load());if(p && p->pools)p->pools->setSlot(slot,paths);graphicsReader.store(nullptr);}
String JitEngine::appliedSource() const { const ScopedLock lock(messageLock); return applied; }
String JitEngine::appliedSourcePath() const { const ScopedLock lock(messageLock); return appliedOrigin; }
String JitEngine::appliedMode() const { const ScopedLock lock(messageLock); return appliedLanguage; }
String JitEngine::appliedFrontend() const { const ScopedLock lock(messageLock); return appliedCompiler; }
bool JitEngine::prepare(double sampleRate,int maximumBlock,const var& currentControls) {
    if(rate.load()==sampleRate && capacity.load()==std::max(4096,maximumBlock))return false;
    rate.store(sampleRate); capacity.store(std::max(4096, maximumBlock)); epoch.fetch_add(1);
    auto code = appliedSource();
    if(code.isNotEmpty()){submit(code,appliedMode(),appliedSourcePath(),serializeState(false),appliedFrontend(),currentControls);return true;}
    return false;
}
uint64_t JitEngine::submit(const String& source, const String& mode,const String& sourcePath,const var& serialized,const String& frontend,const var& initialControls,bool normalizedControls) {
    const auto id = revision.fetch_add(1) + 1;
    { const ScopedLock lock(messageLock); request = {source, mode, sourcePath, id, epoch.load(), rate.load(), capacity.load(),serialized,frontend,za::jsfx::AudioHostRuntime::factorFromChoice(oversamplingChoice.load()),initialControls,normalizedControls}; message = "Compiling... previous program continues."; }
    notify();
    return id;
}
void JitEngine::restoreFactory() {
    auto id=revision.fetch_add(1)+1; adoptionEnabled.store(false);
    {const ScopedLock lock(messageLock);readyDescriptor=JSON::parse("{\"metadata\":{\"io_channels\":{\"inputs\":2,\"outputs\":2}}}");readyDescriptor.getDynamicObject()->setProperty("revision",(int64)id);}
    factoryRequested.store(true);
    { const ScopedLock lock(messageLock); applied.clear(); appliedOrigin.clear(); appliedLanguage="jsfx"; message = "Factory requested: stereo passthrough at next audio block."; }
    if(compiledCallback)compiledCallback();
    notify();
}
void JitEngine::retire(Program* value) {
    if (!value) return;
    auto* head = retired.load(std::memory_order_relaxed);
    do { value->next = head; } while (!retired.compare_exchange_weak(head, value, std::memory_order_release, std::memory_order_relaxed));
}
void JitEngine::collectRetired() {
    auto* value = retired.exchange(nullptr, std::memory_order_acquire);
    while (value) { auto* next = value->next; if(graphicsReader.load()==value) retire(value); else delete value; value = next; }
}
void JitEngine::process(AudioBuffer<float>& buffer,MidiBuffer* midi,const AudioPlayHead::PositionInfo* position,bool offline) {
    // Keep the outgoing instance alive through this block's final MIDI output.
    // The new program clears its input buffer, so cleanup must be appended after
    // processing, using the same production note tracker as transport cleanup.
    struct Retirement {
        JitEngine& owner;MidiBuffer* midi;Program* value=nullptr;
        ~Retirement(){if(value){if(midi)value->midiOutputNoteTracker.prependUnsafeNoteEndsToMidiBuffer(*midi,0,false);value->midiOutputNoteTracker.clear();owner.retire(value);}}
    } retirement{*this,midi};
    if (adoptionEnabled.load() && factoryRequested.exchange(false)) { displayed.store(nullptr); retirement.value=active; active = nullptr; activeId.store(0); }
    if (adoptionEnabled.load()) if (auto* candidate = adoptable.exchange(nullptr)) {
        if (candidate->request.id == permittedRevision.load() && candidate->request.epoch == epoch.load()) {
            retirement.value=active; active = candidate; displayed.store(active); activeId.store(candidate->request.id);
        } else if(candidate->request.epoch==epoch.load() && candidate->request.id>permittedRevision.load()){
            // A compiler publication can race the initial enabled read. Keep
            // the initialized candidate until the host acknowledges its schema.
            Program* empty=nullptr;if(!adoptable.compare_exchange_strong(empty,candidate)){
                auto* expected=candidate;displayed.compare_exchange_strong(expected,nullptr);retire(candidate);
            }
        } else {auto* expected=candidate;displayed.compare_exchange_strong(expected,nullptr);retire(candidate);}
    }
    if(!active){latencySamples.store(0);tailSeconds.store(0);idleSleeping.store(false);}
    if (active && active->request.epoch == epoch.load()){
        auto* st=&active->state;
        za::jsfx::SilenceOnMemoryFault silenceOnMemoryFault{*st,buffer,midi};
        if(active->applyFactor(za::jsfx::AudioHostRuntime::factorFromChoice(oversamplingChoice.load()),controls))notify();
        active->applyGraphicsVariables();
        if(resetRequested.exchange(false)){active->hostRuntime->reset();active->midiRuntime.beginBlock();active->resetSmartIdleRuntime();st->pendingNoteCleanup=1;st->midiInCount=st->midiInReadIndex=st->midiOutCount=0;}
        const int factor=active->factor,engineSamples=za::jsfx::AudioHostRuntime::safeBlockSize(buffer.getNumSamples(),factor);
        auto io=property(property(active->descriptor,"metadata"),"io_channels");
        if(!hostProcessor)fail("Audio host port is not bound");
        const auto topology=active->prepareHostAudio(*hostProcessor,buffer,integer(io,"inputs"),integer(io,"outputs"),factor);
        auto transport=za::jsfx::collectTransport(position);
        auto set=[&](int index,double value){active->set(index,value);};
        const bool transportJump=za::jsfx::syncHostTransport(transport,active->transportTracker,buffer.getNumSamples(),active->request.rate,topology.channels,active->hostVariableBindings,set);
        active->offline=offline;
        active->smartIdleUserOverrideMode.store(static_cast<int>(za::jsfx::IdleRuntime::storedSmartIdleModeToEnum(idleOverride.load())));
        za::jsfx::syncGfxActivity(graphicsVisible.load(),offline,active->hostVariableBindings,set);
        tailSeconds.store(active->smartIdleTailLengthSeconds);
        const bool cleanup=active->observeMidiTransport(transport.observation.valid,transport.observation.playing);
        const bool emergency=st->pendingNoteCleanup!=0;
        const bool hadIncomingMidi=midi && !midi->isEmpty();
        MidiBuffer noMidi;
        active->importMidiToState(*st,midi?*midi:noMidi,buffer.getNumSamples(),engineSamples,factor,emergency || (cleanup && active->midiInputNoteTracker.hasAnyUnsafeState()),emergency);
        if(midi){
            midi->clear();
            if(emergency || (cleanup && active->midiOutputNoteTracker.hasAnyUnsafeState()))active->appendTrackedOutputMidiCleanupToBuffer(*midi,0,emergency);
        }
        st->pendingNoteCleanup=0;
        active->hostRuntime->beginBlock(*st,engineSamples);
        za::jsfx::IdleRuntime::IdleBlock idleBlock{za::jsfx::IdleRuntime::SmartIdleMode::AlwaysAwake};
        {za::jsfx::FileRuntime::ScopedSamplePoolReads reads(*active->pools);idleBlock=active->process(buffer,controls,transportJump || hadIncomingMidi || emergency || active->hostRuntime->hasReadyMessages() || active->hostRuntime->hasPendingForThisInstance(),topology);}
        active->hostRuntime->endBlock(*st);
        if(factor>1)active->downsampleOversampledOutputToHost(topology.outputs,buffer.getNumSamples(),engineSamples,factor);
        latencySamples.store(za::jsfx::latencyFromGuest(active->get("pdc_delay"),factor));
        const bool midiActivity=midi && active->flushMidiFromState(*st,*midi,factor,buffer.getNumSamples());
        if(midi){active->pendingOutputNoteEnds.prependUnsafeNoteEndsToMidiBuffer(*midi,0,false);active->pendingOutputNoteEnds.clear();}

        auto* state=&active->state;std::array<uint64_t,4> change{},automate{},end{};
        for(int i=0;i<4;++i){change[i]=std::atomic_ref<uint64_t>(state->pendingSliderChangeMask[i]).exchange(0);automate[i]=std::atomic_ref<uint64_t>(state->pendingSliderAutomateMask[i]).exchange(0);end[i]=std::atomic_ref<uint64_t>(state->pendingSliderAutomateEndMask[i]).exchange(0);}
        for(auto& d:active->sliders){auto bit=UINT64_C(1)<<(d.slot%64);int word=d.slot/64;int events=(change[word]&bit?1:0)|(automate[word]&bit?2:0)|(end[word]&bit?4:0);if(events){const double value=active->read(active->sliderOffset+d.slot*8);active->retainScriptSlider(d.slot,value);if(graphicsSliderChanged)graphicsSliderChanged(active->request.id,d.slot,value,events);}}
        bool sliderActivity=false;for(int word=0;word<4;++word)sliderActivity|=change[word] || automate[word] || end[word];
        active->finishIdleBlock(idleBlock,active->hostOutPtrs.data(),topology.outputs,buffer.getNumSamples(),engineSamples,midiActivity,sliderActivity);
        idleSleeping.store(active->smartIdleRuntime.sleeping);
        if(active->hasGraphics)active->graphicsVariables.publish(*st);
        active->reportVisibility(sliderVisibilityChanged);

    }else if(active){buffer.clear();if(midi)midi->clear();}
}

std::unique_ptr<JitEngine::Program> JitEngine::compile(const Request& jobRequest) {
    auto folder = File::getSpecialLocation(File::tempDirectory).getChildFile("ZorakAudio-JIT-Editor").getChildFile(Uuid().toString());
    if (!folder.createDirectory()) fail("Cannot create private compilation folder");
    struct FolderGuard {
        File value;
        ~FolderGuard() {
            auto root = File::getSpecialLocation(File::tempDirectory).getChildFile("ZorakAudio-JIT-Editor");
            if (value.isAChildOf(root)) value.deleteRecursively();
        }
    } folderGuard {folder};
    DynamicObject::Ptr requestObject = new DynamicObject;
    requestObject->setProperty("source", jobRequest.source); requestObject->setProperty("mode", jobRequest.mode);requestObject->setProperty("sourcePath",jobRequest.sourcePath);
    requestObject->setProperty("frontend",jobRequest.frontend);
    if (!folder.getChildFile("request.json").replaceWithText(JSON::toString(var(requestObject.get())))) fail("Cannot save compilation request");
    auto ir = jit_platform::runHelper(runtime, folder, [&] { return threadShouldExit() || revision.load() != jobRequest.id; });
    auto descriptor = JSON::parse(folder.getChildFile("program.json"));
    Array<var> declarations;
    for(auto& item:jitDeclarations(jobRequest.source,descriptor)) {
        DynamicObject::Ptr d=new DynamicObject;
        d->setProperty("slot",item.index0);d->setProperty("minimum",item.rangeStart);d->setProperty("maximum",item.rangeEnd);
        d->setProperty("step",item.step);d->setProperty("default",item.def);d->setProperty("label",item.name);
        d->setProperty("hidden",item.hidden);d->setProperty("isString",item.isString);d->setProperty("stringDefault",item.stringDefault);d->setProperty("varName",item.varName);d->setProperty("shape",(int)item.shape);d->setProperty("shapeModifier",item.shapeModifier);
        Array<var> choices;for(auto& name:item.choices)choices.add(name);d->setProperty("choices",choices);
        declarations.add(var(d.get()));
    }
    descriptor.getDynamicObject()->setProperty("sliders",declarations);
    descriptor.getDynamicObject()->setProperty("revision",(int64)jobRequest.id);
    if (integer(descriptor, "version") != 1 || ir.isEmpty()) fail("Invalid compiler response");
    auto program = std::make_unique<Program>();
    program->request = jobRequest; program->descriptor = descriptor; program->capacity = za::jsfx::AudioHostRuntime::safeBlockSize(jobRequest.capacity,8);program->factor=jobRequest.factor;
    program->hasGraphics=(bool)property(property(property(descriptor,"metadata"),"sections_present"),"gfx");
    // ABI17 has a fixed typed host state; variable extent is bound separately.
    program->channelCount=std::max(0, integer(property(property(descriptor,"metadata"),"io_channels"),"process"));
    auto offsets = property(descriptor, "offsets");
    program->splOffset = integer(offsets, "spl"); program->sliderOffset = integer(offsets, "sliders");
    program->varsOffset = integer(offsets, "vars"); program->rateOffset = integer(offsets, "srate"); program->countOffset = integer(offsets, "samplesblock");
    program->sizeOffset = integer(offsets, "currentBlockSize");
    if(integer(descriptor,"stateAbi")!=DSPJSFX_RUNTIME_STATE_ABI || integer(descriptor,"stateSize")!=sizeof(DSPJSFX_State) || program->varsOffset!=offsetof(DSPJSFX_State,vars) || program->sliderOffset!=offsetof(DSPJSFX_State,sliders) || program->sizeOffset!=offsetof(DSPJSFX_State,currentBlockSize))fail("Compiler/runtime state ABI mismatch; compiler and plugin must come from the same build");
    for (auto& definition : array(property(descriptor, "sliders"))) {
        if((bool)property(definition,"isString"))continue;
        Program::Slider control {integer(definition, "slot"), static_cast<double>(property(definition, "minimum")),
            static_cast<double>(property(definition, "maximum")), static_cast<double>(property(definition, "step")), {}};
        control.isChoice=!array(property(definition,"choices")).empty();control.reversed=control.minimum>control.maximum;control.rangeStart=control.minimum;
        auto aliases = property(property(descriptor, "metadata"), "slider_aliases");
        if (auto* object = aliases.getDynamicObject()) for (auto& item : object->getProperties())
            if (static_cast<int>(item.value) == control.slot)
                control.aliases.push_back(static_cast<int>(property(property(property(descriptor, "metadata"), "vars"), item.name.toString().toRawUTF8())));
        program->sliders.push_back(std::move(control));
    }
    std::map<String, uint64_t> addresses;
    {
        std::lock_guard<std::mutex> lock(llvmMutex);
        program->api = &llvmFor(runtime);
        char* error = nullptr;
        program->jit = program->api->create(nullptr, false, false, &error);
        if (!program->jit) fail(program->api->error(error));
        std::vector<std::string> importedNames, exportedNames;
        for (auto& value : array(property(descriptor, "imports"))) importedNames.push_back(value.toString().toStdString());
        for (auto& value : array(property(descriptor, "exports"))) exportedNames.push_back(value.toString().toStdString());
        // Code generation can lower LLVM intrinsics into CRT calls after the IR import scan.
        for(const char* name:{"memcpy","memmove","memset","malloc","calloc","free"})if(std::find(importedNames.begin(),importedNames.end(),name)==importedNames.end())importedNames.emplace_back(name);
        std::vector<SymbolAddress> imports, exports;
        for (auto& name : importedNames) imports.push_back({name.c_str(), mathAddress(name)});
        for (auto& name : exportedNames) exports.push_back({name.c_str(), 0});
        auto text = ir.toStdString();
        LinkElement element {0, text.c_str(), text.size()};
        program->tracker = program->api->link(program->jit, "editor", &element, 1, imports.data(), imports.size(), exports.data(), exports.size(), &error);
        if (!program->tracker) fail(program->api->error(error));
        for (auto& symbol : exports) addresses[String(symbol.name)] = symbol.address;
    }
    auto address = [&](const String& name) -> uint64_t {
        auto found = addresses.find(name);
        if (found == addresses.end() || found->second == 0) fail("Missing native entrypoint: " + name);
        return found->second;
    };
    program->compiled.init = reinterpret_cast<Program::Section>(address("jsfx_init"));
    program->compiled.process=reinterpret_cast<za::jsfx::CompiledProgram<DSPJSFX_State>::Process>(address("jsfx_process_block"));
    program->compiled.slider = reinterpret_cast<Program::Section>(address("jsfx_slider"));program->compiled.serialize=reinterpret_cast<Program::Section>(address("jsfx_serialize"));
    auto stages = array(property(property(descriptor, "metadata"), "faust_stages"));
    if (stages.empty()) {
        program->compiled.block = reinterpret_cast<Program::Section>(address("jsfx_block"));
        program->compiled.sample = reinterpret_cast<Program::Section>(address("jsfx_sample"));
    }
    program->write(program->rateOffset, jobRequest.rate*jobRequest.factor);
    program->write(integer(offsets, "currentSampleRate"), jobRequest.rate*jobRequest.factor);
    for (auto& definition : array(property(descriptor, "sliders")))
        program->write(program->sliderOffset + integer(definition, "slot") * 8, static_cast<double>(property(definition, "default")));
    for(auto& d:jitDeclarations(jobRequest.source,descriptor))if(!d.isString){
        const double value=d.isChoice?std::llround((double(d.def)-double(d.rangeStart))/(d.reversed?-double(d.step):double(d.step))):double(d.def);
        program->write(program->sliderOffset+d.index0*8,za::jsfx::hostSliderValue(d,value));
    }
    if(auto* values=jobRequest.initialControls.getArray())for(auto& d:jitDeclarations(jobRequest.source,descriptor)) {
        if(d.isString || d.index0>=values->size())continue;
        double value=(double)(*values)[d.index0];
        if(jobRequest.normalizedControls){
            if(d.isChoice)value=std::llround(value*(d.choices.size()-1));
            else if(d.shape==JsfxSliderDecl::Shape::Sqr)value=curveFrom01_sqr(float(value),d.rangeStart,d.rangeEnd,d.shapeModifier>0?d.shapeModifier:2);
            else if(d.shape==JsfxSliderDecl::Shape::Log)value=curveFrom01_log(float(value),d.rangeStart,d.rangeEnd,d.shapeModifier);
            else value=d.rangeStart+value*(d.rangeEnd-d.rangeStart);
        }else if(d.isChoice)value=(value-double(d.rangeStart))/(d.reversed?-double(d.step):double(d.step));
        program->write(program->sliderOffset+d.index0*8,za::jsfx::hostSliderValue(d,value));
    }
    auto* native=&program->state;
    int64_t slots=(int64)property(property(descriptor,"metadata"),"memtop_slots");if(slots<1)fail("Invalid compiler heap extent");
    if(!za::jsfx::allocateHeap(*native,slots))fail("Unable to allocate JSFX heap");
    program->graphics=std::make_unique<jsfx_native_gfx::Frame>();
    auto& frame=*program->graphics;
    program->pools=std::make_unique<za::jsfx::FileRuntime>();program->pools->cacheTargetRate.store(jobRequest.factor>1?jobRequest.rate*jobRequest.factor:0);program->pools->externalWake=[this,p=program.get()]{p->pendingWake.store(true);notify();};frame.dynamicPoolOwner=program->pools.get();
    frame.dynamicFileDispatch=[p=program.get()](int op,double** args,int count)->double{
        if(!count||!args[0])return std::numeric_limits<double>::quiet_NaN();
        auto* state=&p->state;
        using S=za::jsfx::Serialization;std::optional<double> result;
        double handle=jsfxCellLoad(args[0]);
        if(op==DSPJSFX_FILE_AVAIL)result=p->pools->serialization.dispatch(*state,S::Avail,handle);
        else if(op==DSPJSFX_FILE_VAR && count>=2)result=p->pools->serialization.dispatch(*state,S::Var,handle,args[1]);
        else if(op==DSPJSFX_FILE_STRING && count>=2)result=p->pools->serialization.dispatch(*state,S::String,handle,args[1]);
        else if(op==DSPJSFX_FILE_MEM && count>=3)result=p->pools->serialization.dispatch(*state,S::Mem,handle,nullptr,jsfxCellLoad(args[1]),jsfxCellLoad(args[2]));
        return result.value_or(std::numeric_limits<double>::quiet_NaN());
    };
    auto vars=property(property(descriptor,"metadata"),"vars");
    program->variables.bind(*native,(size_t)std::max(1,integer(property(descriptor,"metadata"),"var_cap")));
    frame.ownedVariables.bind(frame.state,(size_t)native->varsN);frame.dynamicIsolatedGlobals=true;
    std::vector<uint8_t> gfxDirections((size_t)native->varsN,0);auto flags=property(property(descriptor,"metadata"),"editor_gfx_var_flags");
    if(auto* object=vars.getDynamicObject())for(auto& value:object->getProperties()){const int index=(int)value.value;if(index>=0 && index<native->varsN)gfxDirections[(size_t)index]=(uint8_t)(int)flags.getProperty(value.name,0);}
    program->graphicsVariables.configure(std::move(gfxDirections));
    frame.dynamicAliases.resize((size_t)native->varsN,-1);
    if(auto* object=vars.getDynamicObject())for(auto& value:object->getProperties())frame.dynamicIndices[value.name.toString().toStdString()]=(int)value.value;
    program->hostVariableBindings=za::jsfx::HostVariableBindings([&frame](const char* name){return frame.findIndex(name);});
    auto alias=property(property(descriptor,"metadata"),"slider_aliases");
    if(auto* object=property(property(descriptor,"metadata"),"named_strings").getDynamicObject())
        for(auto& value:object->getProperties())frame.dynamicNamedStrings[value.name.toString().toStdString()]=(int64)value.value;
    if(auto* object=alias.getDynamicObject())for(auto& value:object->getProperties()) {auto it=frame.dynamicIndices.find(value.name.toString().toStdString());if(it!=frame.dynamicIndices.end())frame.dynamicAliases[(size_t)it->second]=(int)value.value;}
    native->hostOwner=&frame;
    program->prepareMidiRuntime(*native,program->capacity);
    program->midiRuntime.nativeStrings.graphics=&frame;
    program->hostRuntime=std::make_unique<za::jsfx::DspJsfxRuntime>();program->hostRuntime->attachToState(native);{const ScopedLock lock(messageLock);program->hostRuntime->setHostTrackName(hostTrackName.toStdString());}
    if((bool)property(property(descriptor,"metadata"),"has_tasks")){const auto metadata=property(descriptor,"metadata");const bool workers=metadata.hasProperty("has_task_workers")?(bool)property(metadata,"has_task_workers"):true;program->tasks=std::make_unique<jsfx_tasks::Runtime>((int)native->varsN,workers);native->taskContext=program->tasks.get();program->pools->taskHeap.state=native;program->pools->taskHeap.capacity=slots;program->pools->taskHeap.lifecycleMutex=&program->lifecycleMutex;program->pools->taskHeap.rebind=[p=program.get()](DSPJSFX_State* state){p->graphics->bindLegacy(*state,p->graphics->frameWidth,p->graphics->frameHeight);};za::jsfx::TaskRuntimeHooks<&jitFileOwner>::bind(*program->tasks);}
    native->sliderVisibilityInit=1;for(auto& word:native->sliderVisibleMask)word=~UINT64_C(0);
    for(auto& d:jitDeclarations(jobRequest.source,descriptor))if(d.hidden)native->sliderVisibleMask[d.index0/64]&=~(UINT64_C(1)<<(d.index0%64));
    native->atomicContext=&program->atomicMutex;
    frame.state.atomicContext=&program->atomicMutex;
    frame.state.nativeStrings=&program->midiRuntime.nativeStrings; frame.state.sharedState=&frame.state;
    std::unordered_map<int64_t,std::string> literalValues;
    for(auto& literal:array(property(property(descriptor,"metadata"),"string_literals"))) {
        auto handle=(int64)property(literal,"handle");auto text=property(literal,"text").toString().toStdString();
        literalValues[handle]=text;
    }
    program->midiRuntime.nativeStrings.configureLiterals(std::move(literalValues));
    for(auto* name:{"gfx_r","gfx_g","gfx_b","gfx_a","gfx_a2"})program->set(name,1);
    program->set("gfx_dest",-1);program->set("gfx_texth",8);
    auto expandedSource=property(descriptor,"expandedSource").toString();frame.configureResources(expandedSource.toRawUTF8());
    auto capabilities=property(property(descriptor,"metadata"),"sample_pool");program->pools->setFileRuntimeCapabilities((bool)property(capabilities,"uses_sample_pool"),(bool)property(capabilities,"uses_legacy_file_io"));program->pools->initialiseFileSlots(parseJsfxFilenameDecls(expandedSource.toRawUTF8()));
    auto assetRoot=jobRequest.sourcePath.isNotEmpty()?File(jobRequest.sourcePath).getParentDirectory():File::getCurrentWorkingDirectory();
    program->pools->sourceDirectory=assetRoot;
    frame.imageLoader=[assetRoot](const String& name){auto root=assetRoot.getFullPathName();return jsfx_gfx_resources::loadImage(name,root.toRawUTF8(),false);};
    frame.fileResolver=[assetRoot,pools=program->pools.get()](int slot,const String& name){std::vector<File> files;auto paths=pools->slotPaths(slot);for(auto& path:paths)files.emplace_back(path);if(files.empty())files.push_back(File::isAbsolutePath(name)?File(name):assetRoot.getChildFile(name));return files;};
    for(auto& declaration:parseJsfxFilenameDecls(expandedSource.toRawUTF8()))if(declaration.defaultPath.isNotEmpty()){StringArray paths;auto f=File::isAbsolutePath(declaration.defaultPath)?File(declaration.defaultPath):assetRoot.getChildFile(declaration.defaultPath);paths.add(f.getFullPathName());program->pools->setSlot(declaration.index0,paths);}
    program->compiled.gfx=reinterpret_cast<Program::Section>(address("jsfx_gfx_aot"));
    for(auto& d:jitDeclarations(jobRequest.source,descriptor))if(d.isString){auto named=property(property(descriptor,"metadata"),"named_strings");double h=(double)named.getProperty(d.varName.toLowerCase(),0);if(!h)h=program->midiRuntime.nativeStrings.create(d.stringDefault.toStdString());else program->midiRuntime.nativeStrings.write(h,d.stringDefault.toStdString());program->write(program->sliderOffset+d.index0*8,h);auto it=frame.dynamicIndices.find(d.varName.toLowerCase().toStdString());if(it!=frame.dynamicIndices.end())program->writeVar(it->second,h);program->stringControls.push_back({d.index0,h,d.stringDefault.toStdString()});}
    za::jsfx::initialiseHostDefaults(program->channelCount,program->hostVariableBindings,[p=program.get()](int index,double value){p->set(index,value);});
    auto io=property(property(descriptor,"metadata"),"io_channels");auto mi=property(property(descriptor,"metadata"),"midi");
    program->initialiseIdle(*native,expandedSource.toRawUTF8(),{integer(io,"inputs"),integer(io,"outputs"),(bool)property(mi,"accepts_midi_input"),(bool)property(mi,"produces_midi_output"),program->pools->fileSlotCount()>0},[p=program.get()](const char* name){return p->graphics->findIndex(name);});
    if(!stages.empty()){
        program->faustStorage.resize(stages.size());std::vector<za::jsfx::Stage> plan;plan.reserve(stages.size());
        auto binding=[&](const var& v){za::jsfx::Binding b;auto values=array(v);if(values.size()>=2){b.kind=(int)values[0];b.index=(int)values[1];if(b.kind==1)if(auto* object=alias.getDynamicObject())for(auto& item:object->getProperties())if((int)item.value==b.index){auto it=frame.dynamicIndices.find(item.name.toString().toStdString());if(it!=frame.dynamicIndices.end())b.alias=it->second;}}return b;};
        for(size_t i=0;i<stages.size();++i){auto desc=stages[i];auto& storage=program->faustStorage[i];za::jsfx::Stage stage;auto name=property(desc,"name").toString();auto kind=property(desc,"kind").toString();stage.kind=kind=="faust"?2:kind=="sample"?1:0;stage.island=integer(desc,"island");stage.fused=(bool)property(desc,"fused");
            if(stage.kind!=2){stage.eel=reinterpret_cast<void(*)(DSPJSFX_State*)>(address(name));if(property(desc,"bulk_name").isString())stage.bulk=reinterpret_cast<void(*)(DSPJSFX_State*,double*,int,int,int,int,double**)>(address(property(desc,"bulk_name").toString()));}
            else{
                stage.size=integer(desc,"size");stage.inputs=integer(desc,"inputs");stage.outputs=integer(desc,"outputs");stage.audioOutputs=integer(desc,"audio_outputs");stage.captureStage=integer(desc,"capture_stage");stage.condition=binding(property(desc,"condition_binding"));
                stage.classInit=reinterpret_cast<void(*)(int)>(address("classInit"+name));stage.constants=reinterpret_cast<void(*)(void*,int)>(address("instanceConstants"+name));stage.clear=reinterpret_cast<void(*)(void*)>(address("instanceClear"+name));stage.allocate=reinterpret_cast<void(*)(void*)>(address("allocate"+name));stage.destroy=reinterpret_cast<void(*)(void*)>(address("destroy"+name));stage.compute=reinterpret_cast<void(*)(void*,int,double**,double**)>(address("compute"+name));stage.bindTables=reinterpret_cast<void(*)(void**)>(address("bindTables"+name));stage.seedTables=reinterpret_cast<void(*)(void**)>(address("seedTables"+name));
                for(auto& zone:array(property(desc,"zones")))storage.zones.push_back({integer(zone,"offset"),(double)property(zone,"default"),binding(property(zone,"binding"))});
                for(auto& v:array(property(desc,"signal_bindings")))storage.signals.push_back(binding(v));for(auto& v:array(property(desc,"export_bindings")))storage.exports.push_back(binding(v));for(auto& v:array(property(desc,"table_sizes")))storage.tables.push_back((int)v);
                stage.zones=storage.zones.data();stage.zoneCount=(int)storage.zones.size();stage.signals=storage.signals.data();stage.signalCount=(int)storage.signals.size();stage.exports=storage.exports.data();stage.exportCount=(int)storage.exports.size();stage.tableSizes=storage.tables.data();stage.tableCount=(int)storage.tables.size();
            }plan.push_back(stage);
        }
        program->faust=std::make_unique<za::jsfx::FaustEngine>(std::move(plan),program->channelCount,integer(property(descriptor,"metadata"),"faust_quantum"),program->compiled.slider);program->faust->prepare(jobRequest.rate*jobRequest.factor,program->capacity,jobRequest.rate);native->faustContext=program->faust.get();
    }
    auto savedState=jobRequest.serialized;
    if(savedState.isObject()){
        if(auto* object=property(savedState,"strings").getDynamicObject())for(auto& item:object->getProperties()){int slot=item.name.toString().getIntValue();if(slot>=0 && slot<256)program->midiRuntime.nativeStrings.write(program->read(program->sliderOffset+slot*8),item.value.toString().toStdString());}
        if(auto* object=property(savedState,"fileSlots").getDynamicObject())for(auto& item:object->getProperties()){StringArray paths;for(auto& value:array(item.value))paths.add(value.toString());program->pools->setSlot(item.name.toString().getIntValue(),paths);}
        savedState=property(savedState,"cells");
    }
    for(auto& definition:program->sliders){program->lastSliders[(size_t)definition.slot]=native->sliders[definition.slot];for(int index:definition.aliases)program->writeVar(index,native->sliders[definition.slot]);}
    program->lastSlidersValid=true;
    program->compiled.initialiseAndPrime(*native,[&]{for(auto& definition:program->sliders)for(int index:definition.aliases)program->writeVar(index,native->sliders[definition.slot]);});
    if(auto* saved=savedState.getArray()){for(auto& v:*saved)program->pools->serialization.cells.push_back((double)v);program->pools->serialization.reading=true;program->pools->serialization.active=true;program->compiled.saveOrRestore(program->state);program->pools->serialization.active=false;}
    program->graphicsVariables.seed(*native,frame.state);
    return program;
}
void JitEngine::run() {
    uint64_t handled = 0;
    while (!threadShouldExit()) {
        collectRetired();
        // File selection/resampling work uses the loader thread. Never take
        // its queue mutex from the audio callback that changes engine rate.
        {const ScopedLock lock(graphicsLock);if(auto* p=displayed.load();p && p->fileRateReload.exchange(false))p->pools->requestRateReload();}
        Request work;
        { const ScopedLock lock(messageLock); work = request; }
        if (work.id > handled && work.id == revision.load()) {
            handled = work.id;
            try {
                auto value = compile(work);
                const ScopedLock lock(messageLock);
                if (revision.load() == work.id && epoch.load() == work.epoch && !threadShouldExit()) {
                    adoptionEnabled.store(false);
                    readyDescriptor=value->descriptor;
                    auto* abandoned=pending.exchange(value.release());
                    if(abandoned){auto* expected=abandoned;displayed.compare_exchange_strong(expected,nullptr);retire(abandoned);}
                    applied = work.source; appliedOrigin=work.sourcePath; appliedLanguage = work.mode;appliedCompiler=work.frontend;
                    message = "Compiled successfully; waiting for host configuration.";
                    if(compiledCallback) compiledCallback();
                }
            } catch (const std::exception& error) {
                if (revision.load() == work.id) setStatus("Run failed; previous program retained.\n" + String::fromUTF8(error.what()));
            }
        }
        wait(500); // Run wakes immediately; idle instances need not poll at 50 Hz.
    }
    collectRetired();
}

void JitEngine::allowAdoption(){
    const ScopedLock lock(messageLock);
    const auto permitted=(uint64_t)(int64)readyDescriptor.getProperty("revision",0);
    auto* value=pending.exchange(nullptr);
    if(value && value->request.id!=permitted){retire(value);value=nullptr;}
    // The host has just installed this request's defaults/restored controls.
    // Acknowledge that initial host snapshot without erasing writes made by
    // @init/@slider/@serialize, exactly as the production prepare path does.
    if(value){for(auto& slider:value->sliders)value->lastSliders[(size_t)slider.slot]=controls[(size_t)slider.slot].load();value->lastSlidersValid=true;}
    permittedRevision.store(permitted);displayed.store(value);
    retire(adoptable.exchange(value));
    adoptionEnabled.store(true);
}
var JitEngine::serializeState(bool includeGuest){
    // Saving from the host message thread must not wait for a menu whose
    // dismissal needs that same thread. Cancel under the menu lifetime lock
    // until the worker yields the graphics state; also handles late menu opens.
    while(!graphicsLock.tryEnter()){{std::lock_guard lock(menuLock);if(renderingMenu)renderingMenu->cancelAll();}Thread::sleep(1);}
    struct Unlock{CriticalSection& lock;~Unlock(){lock.exit();}} unlock{graphicsLock};Program* program=nullptr;do{program=displayed.load();graphicsReader.store(program);}while(program!=displayed.load());
    struct Release {std::atomic<Program*>& pin;~Release(){pin.store(nullptr);}} release{graphicsReader};
    if(!program)return var();
    const std::lock_guard lifecycle(program->lifecycleMutex);
    DynamicObject::Ptr data=new DynamicObject;
    program->applyGraphicsVariables();if(program->hasGraphics)program->graphicsVariables.publish(program->state);
    if(includeGuest){
        program->pools->serialization.cells.clear();program->pools->serialization.cursor=0;program->pools->serialization.reading=false;program->pools->serialization.active=true;
        try {if(program->compiled.serialize)program->compiled.saveOrRestore(program->state);}catch(...){program->pools->serialization.active=false;throw;}program->pools->serialization.active=false;
        Array<var> cells;for(auto value:program->pools->serialization.cells)cells.add(value);data->setProperty("cells",cells);
    }
    DynamicObject::Ptr strings=new DynamicObject;for(auto& d:jitDeclarations(program->request.source,program->descriptor))if(d.isString)strings->setProperty(String(d.index0),String::fromUTF8(program->midiRuntime.nativeStrings.read(program->read(program->sliderOffset+d.index0*8)).c_str()));data->setProperty("strings",var(strings.get()));
    DynamicObject::Ptr slots=new DynamicObject;for(int i=0;i<program->pools->fileSlotCount();++i){Array<var> paths;for(auto& path:program->pools->slotPaths(i))paths.add(path);if(!paths.isEmpty())slots->setProperty(String(i),paths);}data->setProperty("fileSlots",var(slots.get()));
    DynamicObject::Ptr origin=new DynamicObject;origin->setProperty("source",program->request.source);origin->setProperty("mode",program->request.mode);origin->setProperty("sourcePath",program->request.sourcePath);origin->setProperty("frontend",program->request.frontend);data->setProperty("program",var(origin.get()));
    return var(data.get());
}
var JitEngine::compiledDescriptor() const { const ScopedLock lock(messageLock); return readyDescriptor; }
bool JitEngine::memoryFaulted(){
    const ScopedLock lock(graphicsLock);Program* p=nullptr;
    do{p=displayed.load();graphicsReader.store(p);}while(p!=displayed.load());
    const bool result=p && p->state.memoryFault!=0;
    graphicsReader.store(nullptr);return result;
}
double JitEngine::inspectVariable(const String& name){
    const ScopedLock lock(graphicsLock);Program* p=nullptr;
    do{p=displayed.load();graphicsReader.store(p);}while(p!=displayed.load());
    double result=std::numeric_limits<double>::quiet_NaN();
    if(p){const auto key=name.toLowerCase();auto it=p->graphics->dynamicIndices.find(key.toStdString());if(it!=p->graphics->dynamicIndices.end())result=p->get(key.toRawUTF8());}
    graphicsReader.store(nullptr);return result;
}
double JitEngine::inspectGraphicsVariable(const String& name){
    const ScopedLock lock(graphicsLock);Program* p=nullptr;
    do{p=displayed.load();graphicsReader.store(p);}while(p!=displayed.load());
    const double result=p?p->graphics->get(name.toLowerCase().toRawUTF8()):std::numeric_limits<double>::quiet_NaN();
    graphicsReader.store(nullptr);return result;
}
void JitEngine::refreshSliderVisibility(){
    const ScopedLock lock(graphicsLock);Program* p=nullptr;
    do{p=displayed.load();graphicsReader.store(p);}while(p!=displayed.load());
    if(p && sliderVisibilityChanged){std::array<uint64_t,4> visible;for(int i=0;i<4;++i)visible[(size_t)i]=std::atomic_ref<uint64_t>(p->state.sliderVisibleMask[i]).load();sliderVisibilityChanged(p->request.id,visible);}
    graphicsReader.store(nullptr);
}
Image JitEngine::renderGraphics(int width,int height,double x,double y,int cap,int key,double wheelX,double wheelY,GraphicsInput* input) {
    const ScopedLock renderLock(graphicsLock);
    Program* program=nullptr;
    do { program=displayed.load();graphicsReader.store(program); } while(program!=displayed.load());
    struct Release {std::atomic<Program*>& pin;~Release(){pin.store(nullptr);}} release{graphicsReader};
    if(!program || !program->graphics || !program->compiled.gfx) return {};
    const std::lock_guard lifecycle(program->lifecycleMutex);
    if(cap || key || wheelX || wheelY || (input && !input->drops.empty()))program->pendingWake.store(true);
    {std::lock_guard lock(menuLock);renderingMenu=input?input->menu:nullptr;}
    struct ClearMenu{std::mutex& lock;jsfx_gfx::AsyncMenuPort*& port;~ClearMenu(){std::lock_guard guard(lock);port=nullptr;}} clearMenu{menuLock,renderingMenu};
    auto& frame=*program->graphics;
    if(!program->canvas.isValid() || program->canvas.getWidth()!=width || program->canvas.getHeight()!=height)
        program->canvas=Image(Image::ARGB,std::max(1,width),std::max(1,height),true);
    program->graphicsVariables.begin(program->state,frame.state);frame.begin(program->state,width,height);
    frame.set("mouse_x",x);frame.set("mouse_y",y);frame.set("mouse_cap",cap);frame.set("mouse_wheel",frame.get("mouse_wheel")+wheelY);frame.set("mouse_hwheel",frame.get("mouse_hwheel")+wheelX);
    frame.focused=input ? input->focused : true;frame.visible=true;if(key)frame.keys.push_back(key);
    if(input){frame.menuPort=input->menu;frame.keysDown.clear();for(auto code:input->keysDown)frame.keysDown.insert(code);if(!input->drops.empty())frame.pendingDroppedFileBatches.push_back(input->drops);}

    jsfx_gfx::GfxRenderSession renderer(program->canvas);frame.renderPort=&renderer;
    std::array<double,256> before{},after{};std::array<const Program::Slider*,256> definitions{};for(auto& def:program->sliders){before[(size_t)def.slot]=frame.state.sliders[def.slot];definitions[(size_t)def.slot]=&def;}
    za::jsfx::DspJsfxSamplePool::ReadBatch batch;program->compiled.render(frame.state);renderer.finish(frame.commands);frame.renderPort=nullptr;frame.menuPort=nullptr;
    program->graphicsVariables.finish(frame.state);
    if(frame.state.memoryFault)program->state.memoryFault=frame.state.memoryFault;
    if(input){input->cursor=frame.cursorResource;for(int i=0;i<4;++i)input->visible[(size_t)i]=std::atomic_ref<uint64_t>(frame.state.sliderVisibleMask[i]).load();}
    if(displayed.load()!=program)return {};
    for(auto& def:program->sliders)after[(size_t)def.slot]=frame.state.sliders[def.slot];
    const auto changes=za::jsfx::actualSliderChanges(frame.changeMask,before.data(),after.data(),[&](int i){return definitions[(size_t)i]!=nullptr;},[&](int i,double a,double b){return za::jsfx::sliderValuesEquivalent(*definitions[(size_t)i],a,b);});
    auto automate=frame.automateMask,end=frame.automateEndMask;
    za::jsfx::bracketGfxGesture(changes,automate,end,program->graphicsTouch,(cap&(1|2|64))!=0);
    for(auto& def:program->sliders)if(changes.test(def.slot) || automate.test(def.slot) || end.test(def.slot))program->graphicsVariables.queueSlider(def.slot,after[(size_t)def.slot]);
    // Native GFX records notifications in its frame. Publish those through
    // the production audio-side mailbox before host float conversion; this
    // preserves the exact script value and schedules the next @slider pass.
    mergeSliderMaskWords(program->state.pendingSliderChangeMask,changes | automate | end);
    mergeSliderMaskWords(program->state.pendingSliderAutomateMask,automate);
    mergeSliderMaskWords(program->state.pendingSliderAutomateEndMask,end);
    if(graphicsSliderChanged)for(auto& def:program->sliders) {
        auto value=after[(size_t)def.slot];
        int events=(changes.test(def.slot)?1:0)|(automate.test(def.slot)?2:0)|(end.test(def.slot)?4:0);
        if(events) graphicsSliderChanged(program->request.id,def.slot,value,events);
    }
    return program->canvas;
}
