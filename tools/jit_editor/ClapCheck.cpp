// Load the packaged module through the public CLAP ABI, with no JUCE linkage.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "TestModule.h"
#include <clap/ext/timer-support.h>
#include <clap/clap.h>
#include <clap/ext/state.h>
#include <clap/ext/params.h>
#include <clap/ext/thread-check.h>
#include <clap/ext/audio-ports.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

static bool audioThread = false;
static const clap_host_thread_check_t threads {[](const clap_host_t*) { return !audioThread; }, [](const clap_host_t*) { return audioThread; }};
static bool restartPending=false,callbackPending=false;
static int paramRescans=0,portRescans=0;
static const clap_host_params_t hostParams{[](const clap_host_t*,uint32_t flags){if(flags&CLAP_PARAM_RESCAN_ALL)++paramRescans;},[](const clap_host_t*,clap_id,uint32_t){},[](const clap_host_t*){}};
static const clap_host_audio_ports_t hostPorts{[](const clap_host_t*,uint32_t){return true;},[](const clap_host_t*,uint32_t){++portRescans;}};
static const clap_plugin_t* timerPlugin = nullptr;
static bool timerRegistered = false;
static clap_id timerId = 0;
static const clap_host_timer_support_t timers {
    [](const clap_host_t*,uint32_t,clap_id* id){*id=1;timerId=1;timerRegistered=true;return true;},
    [](const clap_host_t*,clap_id){timerRegistered=false;return true;}
};
static const void* extension(const clap_host_t*, const char* id) { if(std::strcmp(id,CLAP_EXT_TIMER_SUPPORT)==0)return &timers;if(std::strcmp(id,CLAP_EXT_PARAMS)==0)return &hostParams;if(std::strcmp(id,CLAP_EXT_AUDIO_PORTS)==0)return &hostPorts;return std::strcmp(id, CLAP_EXT_THREAD_CHECK) == 0 ? &threads : nullptr; }
static void request(const clap_host_t*) {}
static void restart(const clap_host_t*){restartPending=true;}
static void callback(const clap_host_t*){callbackPending=true;}
static void pump(){pumpTestMessages();if(timerRegistered && timerPlugin){auto* timer=static_cast<const clap_plugin_timer_support_t*>(timerPlugin->get_extension(timerPlugin,CLAP_EXT_TIMER_SUPPORT));if(timer)timer->on_timer(timerPlugin,timerId);}}
static uint32_t noEvents(const clap_input_events_t*) { return 0; }
static const clap_event_header_t* noEvent(const clap_input_events_t*, uint32_t) { return nullptr; }
static bool acceptEvent(const clap_output_events_t*, const clap_event_header_t*) { return true; }
static void require(bool ok, const std::string& text) { if (!ok) throw std::runtime_error(text); }
struct Stream { std::string text; size_t position = 0; };
static int64_t read(const clap_istream_t* stream, void* destination, uint64_t count) {
    auto& value = *static_cast<Stream*>(stream->ctx);
    auto size = std::min(static_cast<size_t>(count), value.text.size() - value.position);
    std::memcpy(destination, value.text.data() + value.position, size); value.position += size; return static_cast<int64_t>(size);
}
static int64_t write(const clap_ostream_t* stream, const void* source, uint64_t count) {
    static_cast<Stream*>(stream->ctx)->text.append(static_cast<const char*>(source), static_cast<size_t>(count)); return static_cast<int64_t>(count);
}
static std::string quote(const std::string& text) {
    std::string result = "\"";
    for (char ch : text) { if (ch == '\n') result += "\\n"; else { if (ch == '"' || ch == '\\') result += '\\'; result += ch; } }
    return result + '"';
}
int main(int argc, char** argv) try {
    require(argc == 2 || argc == 3, "Pass packaged CLAP path");
    const auto path=testArgument(1,argv);
    auto module = loadTestModule(path.c_str()); require(module != nullptr, "Load packaged CLAP");
    auto* entry = reinterpret_cast<const clap_plugin_entry_t*>(testSymbol(module, "clap_entry"));
    require(entry && entry->init(path.c_str()), "CLAP entry init");
    auto* factory = static_cast<const clap_plugin_factory_t*>(entry->get_factory(CLAP_PLUGIN_FACTORY_ID));
    require(factory && factory->get_plugin_count(factory) == 1, "CLAP factory");
    auto* descriptor = factory->get_plugin_descriptor(factory, 0);
    clap_host_t host {CLAP_VERSION, nullptr, "JIT Editor package test", "ZorakAudio", "", "0", extension, restart, request, callback};
    auto* plugin = factory->create_plugin(factory, &host, descriptor->id); require(plugin && plugin->init(plugin), "CLAP plugin init");
    timerPlugin = plugin;
    auto* state = static_cast<const clap_plugin_state_t*>(plugin->get_extension(plugin, CLAP_EXT_STATE)); require(state, "CLAP state");
    auto* params = static_cast<const clap_plugin_params_t*>(plugin->get_extension(plugin, CLAP_EXT_PARAMS)); require(params && params->count(plugin) == 0, "No artificial fixed controls");
    require(plugin->activate(plugin, 48000, 1, 1024), "CLAP activate");
    audioThread = true; require(plugin->start_processing(plugin), "CLAP start"); audioThread = false;
    auto* ports=static_cast<const clap_plugin_audio_ports_t*>(plugin->get_extension(plugin,CLAP_EXT_AUDIO_PORTS));require(ports,"Audio ports");
    float extra[62][1024]{};
    float left[1024], right[1024], outL[1024] {}, outR[1024] {};
    float* ins[64] {left, right}; float* outs[64] {outL, outR};for(int i=2;i<64;++i){ins[i]=extra[i-2];outs[i]=extra[i-2];}
    clap_audio_buffer_t input {}, output {}; input.data32 = ins; input.channel_count = 2; output.data32 = outs; output.channel_count = 2;
    clap_input_events_t inEvents {nullptr, noEvents, noEvent}; clap_output_events_t outEvents {nullptr, acceptEvent};
    clap_process_t process {}; process.frames_count = 1024; process.audio_inputs = &input; process.audio_inputs_count = 1;
    process.audio_outputs = &output; process.audio_outputs_count = 1; process.in_events = &inEvents; process.out_events = &outEvents;
    auto processBlock = [&] {
        pump();
        if(restartPending){restartPending=false;audioThread=true;plugin->stop_processing(plugin);audioThread=false;plugin->deactivate(plugin);
            // Reactivate before servicing the callback to exercise the rescan fallback.
            callbackPending=false;
            process.audio_inputs_count=ports->count(plugin,true);process.audio_outputs_count=ports->count(plugin,false);
            clap_audio_port_info_t info{};if(process.audio_inputs_count){require(ports->get(plugin,0,true,&info),"Input port info");input.channel_count=info.channel_count;}
            if(process.audio_outputs_count){require(ports->get(plugin,0,false,&info),"Output port info");output.channel_count=info.channel_count;}
            require(plugin->activate(plugin,48000,1,1024),"Restart activate");audioThread=true;require(plugin->start_processing(plugin),"Restart processing");audioThread=false;
        }
        if(callbackPending){callbackPending=false;plugin->on_main_thread(plugin);}
        for (int i = 0; i < 1024; ++i) { left[i] = .8f; right[i] = -.4f; }
        audioThread = true; auto result = plugin->process(plugin, &process); audioThread = false;
        require(result != CLAP_PROCESS_ERROR, "CLAP process"); process.steady_time += 1024; };
    auto check = [&](float gain) { for (int i = 0; i < 1024; ++i)
        if (std::abs(outL[i] - .8f * gain) > 2e-6f || (output.channel_count>1 && std::abs(outR[i] + .4f * gain) > 2e-6f)) return false; return true; };
    auto save = [&] { Stream stream; clap_ostream_t out {&stream, write}; require(state->save(plugin, &out), "CLAP save"); return stream.text; };
    auto load = [&](const std::string& text) { Stream stream {text, 0}; clap_istream_t in {&stream, read}; require(state->load(plugin, &in), "CLAP load"); };
    auto run = [&](const std::string& source, const std::string& mode, float gain) {
        load("{\"appliedFrontend\":\"cpp-frontend\",\"frontend\":\"cpp-frontend\",\"version\":1,\"appliedSource\":" + quote(source) + ",\"appliedMode\":" + quote(mode) + "}");
        auto began = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - began < std::chrono::seconds(45)) {
            processBlock(); if (check(gain)) return;
            auto info = save(); require(info.find("Run failed") == std::string::npos, info);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        throw std::runtime_error("Packaged Run timed out: " + save());
    };
    processBlock(); require(check(1), "Initial factory passthrough");
    run("slider1:gain=0.5<0,1,0.01>Gain\n@sample\nspl0*=gain;spl1*=gain;", "jsfx", .5f);
    require(params->count(plugin)==2,"Generated host parameter count");clap_param_info_t pi{};require(params->get_info(plugin,0,&pi) && std::string(pi.name)=="Gain","Generated host label");
    run("@sample\nspl0*=0.25;spl1*=0.25;spl2*=0.25;spl3*=0.25;","jsfx",.25f);
    require(input.channel_count==4 && output.channel_count==4,"Inferred four-channel pins");
    run("in_pin:none\n@sample\nspl0=0.4;spl1=-0.2;","jsfx",.5f);require(process.audio_inputs_count==0 && output.channel_count==2,"Generator has no inputs");
    run("@block\ngain=0.25;\n@faust block\nprocess=spl0*gain,spl1*gain;", "jsfx", .25f);
    run("process=_,_:*(0.125),*(0.125);", "faust", .125f);
    run("g=hslider(\"Gain\",.5,0,1,.01); b=hslider(\"Binary\",0,0,1,1); n=nentry(\"Entry\",0,0,1,1); e=checkbox(\"Enable\"); t=button(\"Trigger\"); process=*(g+b+n+e+t),*(g+b+n+e+t);","faust",.5f);
    require(params->count(plugin)==6,"Faust binary/continuous host controls generated");
    for(uint32_t i=0;i<5;++i){clap_param_info_t info{};require(params->get_info(plugin,i,&info),"Faust parameter info");const bool stepped=(info.flags&CLAP_PARAM_IS_STEPPED)!=0;require(stepped==(std::string(info.name)!="Gain"),"Binary controls stepped; continuous gain retains its full range");}
    run("process=_*0.25;","faust",.25f);require(input.channel_count==1 && output.channel_count==1,"Pure Faust mono inference");
    run("process=_,_:*(0.125),*(0.125);","faust",.125f);
    auto saved = save();
    load("{\"appliedFrontend\":\"cpp-frontend\",\"frontend\":\"cpp-frontend\",\"version\":1,\"appliedSource\":\"@sample\\nspl0*=;\",\"appliedMode\":\"jsfx\"}");
    auto began = std::chrono::steady_clock::now();
    while (save().find("Run failed") == std::string::npos && std::chrono::steady_clock::now() - began < std::chrono::seconds(15)) {
        processBlock(); require(check(.125f), "Failed compile changed audio"); std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    require(save().find("Run failed") != std::string::npos, "Invalid code rejection");
    load("{\"appliedFrontend\":\"cpp-frontend\",\"frontend\":\"cpp-frontend\",\"version\":1,\"appliedSource\":\"\"}"); for(int i=0;i<100 && !check(1);++i){processBlock();std::this_thread::sleep_for(std::chrono::milliseconds(5));}require(check(1), "Default restored passthrough");
    load(saved); began = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - began < std::chrono::seconds(45)) {
        processBlock(); if (check(.125f)) break; std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    require(check(.125f), "Saved state recompiled");
    require(paramRescans>=5 && portRescans>=5,"Host rescans performed");
    audioThread = true; plugin->stop_processing(plugin); audioThread = false;
    plugin->deactivate(plugin); plugin->destroy(plugin); timerPlugin=nullptr; entry->deinit(); closeTestModule(module);
    std::cout << "PACKAGED CLAP PASS: factory, JSFX Run, mixed full-block Faust Run, pure Faust Run, binary/continuous Faust parameters, failed compile retains audio, Default, saved-state recompile, dynamic parameters, inferred 4-channel and zero-input pins, host restart/rescan\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
