// Exercise the shared host-injection contract without a JUCE link dependency.
#include <algorithm>
#include <cassert>
#include <cstring>
#include <map>
#include <string>
#define JUCE_MAJOR_VERSION 6
namespace juce {
template<class T>T jmax(T a,T b){return std::max(a,b);}
template<class T>T jlimit(T lo,T hi,T value){return std::clamp(value,lo,hi);}
struct AudioPlayHead {
    struct CurrentPositionInfo {bool isPlaying=false,isRecording=false;long long timeInSamples=0;double timeInSeconds=0,ppqPosition=0,bpm=120;};
    bool getCurrentPosition(CurrentPositionInfo&){return false;}
};
}
#include "JsfxHostEnvironment.h"
int main(){
    using namespace za::jsfx;
    std::map<std::string,int> variables;
    for(size_t i=0;i<hostVariableNames.size();++i)variables[hostVariableNames[i]]=int(i)*3;
    variables.erase("beat_position"); // Missing variables are legal.
    int lookups=0;
    auto resolve=[&](const char* name){++lookups;auto i=variables.find(name);return i==variables.end()?-1:i->second;};
    HostVariableBindings bindings(resolve);
    HostVariableBindings copied(bindings);
    assert(copied.indices==bindings.indices);
    std::map<int,double> oldValues,newValues;
    auto named=[&](const char* name,double value){int i=resolve(name);if(i>=0)oldValues[i]=value;};
    auto indexed=[&](int i,double value){if(i>=0)newValues[i]=value;};
    initialiseHostDefaults(2,named);initialiseHostDefaults(2,bindings,indexed);
    HostTransportTracker oldTracker,newTracker;
    for(int n=0;n<1000;++n){
        TransportFrame frame;auto& o=frame.observation;
        o.valid=n%7!=0;o.playing=n%3!=0;o.recording=n%5==0;
        o.hasSamples=n%2==0;o.hasSeconds=n%4!=0;o.samples=n*64+(n%19==0?1000:0);o.seconds=n*.01;
        frame.haveBeats=n%4==0;frame.beats=n*.1;frame.haveBpm=n%6==0;frame.bpm=90+n%80;
        bool oldJump=syncHostTransport(frame,oldTracker,64,48000,2,named);
        const int before=lookups;
        bool newJump=syncHostTransport(frame,newTracker,64,48000,2,bindings,indexed);
        syncGfxActivity(n%2==0,n%5==0,bindings,indexed);
        assert(lookups==before); // No audio-thread lookup, including misses.
        syncGfxActivity(n%2==0,n%5==0,named);
        assert(oldJump==newJump && oldValues==newValues);
    }
    // A replacement JIT program gets its own indices, not the previous schema.
    HostVariableBindings replacement([](const char* name){return std::strcmp(name,"tempo")==0?81:-1;});
    std::map<int,double> replacementValues;
    initialiseHostDefaults(2,replacement,[&](int i,double v){if(i>=0)replacementValues[i]=v;});
    assert(replacementValues.size()==1 && replacementValues[81]==120);
}
