#include "ZAEventExtractor.h"
#include <iostream>
#include <fstream>
#include <random>
#include <stdexcept>
using namespace za::fileimport::extractor;
struct Fixture {
    std::string name; double rate; std::vector<float> left, right; std::vector<std::pair<double,double>> truth;
    Fixture(std::string n,double sr,double seconds):name(n),rate(sr),left(size_t(sr*seconds)),right(left.size()){}
    void noise(float amplitude){std::mt19937 gen(512);std::uniform_real_distribution<float>d(-amplitude,amplitude);for(size_t i=0;i<left.size();++i)left[i]=right[i]=d(gen);}
    void hit(double at,double duration,double gain=0.7,bool slow=false,bool inverse=false){
        truth.push_back({at,at+duration});std::mt19937 gen(int(at*10000)+4);std::uniform_real_distribution<double>d(-1,1);
        for(int i=0;i<int(duration*rate);++i){double t=i/rate,env=std::pow(std::max(0.0,1-t/duration),2);if(slow)env*=std::min(1.0,t/.08);const double v=gain*env*(.5*std::sin(2*3.141592653589793*155*t)+.5*d(gen));size_t n=size_t(at*rate)+i;if(n<left.size()){left[n]+=float(v);right[n]+=float(inverse?-v:v);}}
    }
    AudioView audio(const float** pointers){pointers[0]=left.data();pointers[1]=right.data();return{pointers,2,int(left.size()),rate};}
};
int main(int argc,char**argv)try{
    std::vector<Fixture> tests;
    for(double sr:{22050.,44100.,48000.,96000.}){
        Fixture f("separate-decays-"+std::to_string(int(sr)),sr,3);f.hit(.25,.28);f.hit(.95,.22,.24);f.hit(1.65,.6,.5);tests.push_back(std::move(f));
        Fixture g("adjacent-hits-"+std::to_string(int(sr)),sr,2);g.hit(.2,.23);g.hit(.46,.23);g.hit(.72,.23);g.hit(.98,.23);tests.push_back(std::move(g));
    }
    Fixture silence("digital-silence",48000,2);tests.push_back(std::move(silence));
    Fixture noisy("quiet-events-on-background",48000,5);noisy.noise(.0005f);noisy.hit(.8,.3,.03);noisy.hit(2.1,.4,.1);noisy.hit(3.6,.3,.025);tests.push_back(std::move(noisy));
    Fixture anti("anti-phase-stereo",48000,3);anti.hit(.3,.4,.7,false,true);anti.hit(1.3,.4,.5,false,true);tests.push_back(std::move(anti));
    Fixture slow("soft-gestures",48000,4);slow.hit(.5,.8,.4,true);slow.hit(2.1,.9,.3,true);tests.push_back(std::move(slow));
    Fixture ticks("spurious-short-fragments",48000,3);ticks.hit(.3,.3);ticks.hit(1.2,.3);ticks.left[92000]=ticks.right[92000]=.7f;tests.push_back(std::move(ticks));
    Fixture steady("sustained-low-tone",48000,4);steady.truth.push_back({.4,3.2});for(int i=19200;i<153600;++i)steady.left[i]=steady.right[i]=float(.25*std::sin(2*3.141592653589793*60*i/48000));tests.push_back(std::move(steady));
    Fixture edges("source-start-and-end",48000,2);edges.hit(0,.3);edges.hit(1.65,.35);tests.push_back(std::move(edges));
    Fixture brief("short-legitimate-events",48000,3);brief.hit(.2,.065);brief.hit(.8,.07);brief.hit(1.4,.06);tests.push_back(std::move(brief));
    Fixture bed("constant-background-only",48000,3);bed.noise(.003f);tests.push_back(std::move(bed));
    Fixture texture("weak-tail-rebounds",48000,2);texture.hit(.2,.3);texture.hit(.41,.055,.05);texture.hit(.65,.035,.025);texture.truth={{.2,.685}};tests.push_back(std::move(texture));
    int failed=0;double oldError=0,newError=0;int oldCountError=0,newCountError=0;int total=0;
    std::ofstream report(argc>1?argv[1]:"detector-results.csv");report<<"fixture,truth,original,adaptive,original_boundary_error_ms,adaptive_boundary_error_ms\n";
    for(auto&f:tests){const float* p[2];auto audio=f.audio(p);auto a=analyse(audio);Settings s;auto old=detect(a,s),now=detectAdaptive(audio,a,s);
        const auto error=[&](const Detection&v){double result=0;for(const auto&t:f.truth){double nearest=1e9;for(const auto&r:v.regions)nearest=std::min(nearest,std::abs(r.start/f.rate-t.first)+std::abs(r.end/f.rate-t.second));result+=std::min(2.0,nearest)*1000;}return result;};
        const double e0=error(old),e1=error(now);oldError+=e0;newError+=e1;oldCountError+=std::abs(int(old.regions.size())-int(f.truth.size()));newCountError+=std::abs(int(now.regions.size())-int(f.truth.size()));total+=int(f.truth.size());
        bool ok=now.regions.size()==f.truth.size();int end=0;for(const auto&r:now.regions){ok&=r.start>=end&&r.end>r.start&&r.end<=audio.samples;end=r.end;}
        if(!f.truth.empty())ok&=e1/f.truth.size()<110;
        if(!ok)++failed;std::cout<<(ok?"PASS ":"FAIL ")<<f.name<<" truth="<<f.truth.size()<<" old="<<old.regions.size()<<" new="<<now.regions.size()<<" error(ms)="<<e0<<" / "<<e1<<'\n';
        if(!ok)for(const auto&r:now.regions)std::cout<<"  "<<r.start/f.rate<<" - "<<r.end/f.rate<<'\n';
        report<<f.name<<','<<f.truth.size()<<','<<old.regions.size()<<','<<now.regions.size()<<','<<e0<<','<<e1<<'\n';
        if(!detectAdaptive(audio,a,s,[]{return true;}).regions.empty())throw std::runtime_error("Cancellation failed");
        auto again=detectAdaptive(audio,a,s);if(again.regions.size()!=now.regions.size())throw std::runtime_error("Nondeterministic detector");
        for(size_t i=0;i<again.regions.size();++i)if(again.regions[i].start!=now.regions[i].start||again.regions[i].end!=now.regions[i].end)throw std::runtime_error("Nondeterministic boundaries");
    }
    // v3 retains default v2 boundaries; stronger grouping changes only high settings.
    for(auto&f:tests){const float*p[2];auto audio=f.audio(p);auto a=analyse(audio);auto prior=detectAdaptive(audio,a,{}),now=detectAdaptive(audio,a,{}, {},3);if(prior.regions.size()!=now.regions.size())throw std::runtime_error("v3 changed balanced event count");for(size_t i=0;i<prior.regions.size();++i)if(prior.regions[i].start!=now.regions[i].start||prior.regions[i].end!=now.regions[i].end)throw std::runtime_error("v3 changed balanced boundaries");}
    {const float*p[2];auto audio=tests[1].audio(p);auto a=analyse(audio);Settings grouped;grouped.gesture=1;grouped.quietMs=160;auto now=detectAdaptive(audio,a,grouped,{},3);if(now.regions.size()!=1)throw std::runtime_error("Full gesture grouping did not combine close sub-actions");}
    {Fixture f("late-background-between-gestures",48000,3);f.noise(.0003f);std::fill(f.left.begin(),f.left.begin()+4800,0.f);std::fill(f.right.begin(),f.right.begin()+4800,0.f);f.hit(.4,.35,.5);f.hit(1.5,.35,.4);const float*p[2];auto audio=f.audio(p);auto a=analyse(audio);Settings grouped;grouped.gesture=1;grouped.quietMs=160;grouped.minimumMs=80;auto now=detectAdaptive(audio,a,grouped,{},3);if(now.regions.size()!=2)throw std::runtime_error("Relative quiet dips failed to separate complete gestures");for(const auto&r:now.regions)if(r.end-r.start>audio.rate*.6)throw std::runtime_error("Whole gesture swallowed adjacent activity");}
    // Wider parameter and cancellation/bounds coverage, without assuming every
    // sensitivity/gesture setting should produce the default event count.
    for(auto&f:tests){const float*p[2];auto audio=f.audio(p);auto a=analyse(audio);for(float sens:{0.f,.5f,1.f})for(float gesture:{0.f,1.f})for(float tail:{0.f,1.f}){Settings s;s.sensitivity=sens;s.gesture=gesture;s.tails=tail;auto result=detectAdaptive(audio,a,s);int end=0;for(const auto&r:result.regions){if(r.start<end||r.end<=r.start||r.end>audio.samples)throw std::runtime_error("Parameter sweep produced invalid boundaries");end=r.end;}}}
    const float*p[2];auto audio=tests[0].audio(p);auto a=analyse(audio);Settings dry,wet;dry.tails=0;wet.tails=1;auto d=detectAdaptive(audio,a,dry),w=detectAdaptive(audio,a,wet);if(d.regions.size()!=w.regions.size()||d.regions.empty()||d.regions[0].end>w.regions[0].end)throw std::runtime_error("Tail preservation moved the wrong way");
    std::cout<<"Boundary sum ("<<total<<" events): "<<oldError<<" -> "<<newError<<" ms; count errors "<<oldCountError<<" -> "<<newCountError<<"; failed="<<failed<<"; parameter sweep passed\n";return failed?1:0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
