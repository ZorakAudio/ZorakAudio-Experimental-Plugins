#pragma once
#include <array>

// Same wire format and duration quantization as Sampler MIDI Hub v1.0.1.
static void corpusDurationPacket(juce::MidiBuffer& midi,int id,double duration,int boundary,int attack,int release,int at,int action=0,int target=0){
  int ticks=int(std::floor(duration*64000+.5));
  std::array<int,20> data{1,action,id/128,id%128,target/128,target%128,60,action==7?0:100,boundary,1,0,64,64,1,ticks/16384,(ticks/128)%128,ticks%128,attack,release,0};
  auto cc=[&](int controller,int value){midi.addEvent(juce::MidiMessage::controllerEvent(1,controller,value),at);};
  int sum=0;cc(118,83);for(int n=0;n<20;++n){cc(119,data[n]);sum=(sum+(n+1)*data[n])%128;}cc(119,sum);cc(118,84);
}

template<class Read> void corpusDurationCheck(juce::AudioProcessor* p,juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,const std::string& path){
  std::ofstream csv(path);csv<<"relative,boundary,release_ms,rendered_samples,release_nonzero_samples,peak,grains\n";
  auto check=[](bool good,const std::string& reason){if(!good)throw std::runtime_error("Duration check: "+reason);};
  auto closeEnough=[&](double a,double b){check(std::abs(a-b)<1.01,"expected "+std::to_string(b)+", got "+std::to_string(a));};
  auto heap=[&](const char* symbol,int n){std::vector<double> x(n);check(za_native_gfx_heap_copy(p,symbol,x.data(),n)==n,"heap snapshot");return x;};
  int command=1;const int blockSize=audio.getNumSamples();constexpr double rate=48000;constexpr int onset=17;
  auto process=[&](){p->processBlock(audio,midi);};
  auto stop=[&](){za_test_gfx_button(p,24,(++command%16)*1048576+14);for(int n=0;n<8;++n){audio.clear();midi.clear();process();}};
  for(auto item:std::vector<std::pair<int,double>>{{12,1},{8,0},{73,0},{25,800},{26,5},{27,100},{28,.75},{29,180},{70,.4},{71,-.35},{72,-.6},{49,1}})za_test_gfx_button(p,item.first,item.second);
  for(int relative:{0,1})for(int boundary:{0,1,2}){
    stop();za_test_gfx_button(p,75,relative);
    double release=(relative?.9:.18)*rate,total=4*rate,releaseStart=onset+(boundary==1?total-release:total),end=onset+total+(boundary==1?0:release);
    double eventClock=value("clock")+onset,peak=0;int releaseNonzero=0;std::vector<float> reference;
    const int count=int(std::ceil((end+blockSize*2)/blockSize));
    for(int block=0;block<count;++block){audio.clear();midi.clear();int start=block*blockSize;
      if(!block)corpusDurationPacket(midi,1,4,boundary,127,127,onset);
      int endAt=int(onset+total)-start;if(endAt>=0 && endAt<blockSize)corpusDurationPacket(midi,1,0,boundary,127,127,endAt,7);
      process();
      if(!block){auto e=heap("event_base",64);closeEnough(e[5],eventClock);closeEnough(e[6]-e[5],relative?1200:240);closeEnough(e[7]-e[6],relative?24000:4800);closeEnough(e[9]-e[5],releaseStart-onset);closeEnough(e[10]-e[5],end-onset);closeEnough(e[19],release);check(e[57]==relative,"birth-time timing mode");}
      for(int i=0;i<blockSize;++i)for(int c=0;c<2;++c){double x=audio.getSample(c,i);check(std::isfinite(x),"nonfinite audio");peak=std::max(peak,std::abs(x));reference.push_back(float(x));
        if(start+i>=releaseStart && start+i<end && std::abs(x)>1e-8)++releaseNonzero;
        if(start+i>=std::ceil(end))check(x==0,"audio overran envelope end");}
    }
    check(peak>1e-8 && releaseNonzero>16,"no audible release");check(value("record_done")>0 && value("record_overflow")==0,"event not exactly recorded");
    auto recorded=heap("record_env",64);closeEnough(recorded[10]-recorded[5],end-onset);closeEnough(recorded[19],release);
    csv<<relative<<','<<boundary<<','<<release/rate*1000<<','<<reference.size()/2<<','<<releaseNonzero<<','<<peak<<','<<value("record_n")<<'\n';
    std::cout<<"DURATION relative="<<relative<<" boundary="<<boundary<<" release_ms="<<release/rate*1000<<" nonzero_release="<<releaseNonzero<<std::endl;
    if(relative && boundary==1){
      za_test_gfx_button(p,75,0);za_test_gfx_button(p,29,5);za_test_gfx_button(p,72,.8);za_test_gfx_button(p,24,(++command%16)*1048576+4);
      size_t at=0;double error=0;constexpr size_t padding=(144-onset)*2;
      for(int b=0;b<count+1;++b){audio.clear();midi.clear();process();for(int i=0;i<blockSize;++i)for(int c=0;c<2;++c){double expected=at<padding || at-padding>=reference.size()?0:reference[at-padding];error=std::max(error,std::abs(double(audio.getSample(c,i))-expected));++at;}}
      check(error<1e-6,"REPEAT changed captured duration envelope, error="+std::to_string(error));std::cout<<"DURATION REPEAT changed timing/release/curve waveform error="<<error<<std::endl;
      za_test_gfx_button(p,29,180);za_test_gfx_button(p,72,-.6);
    }
  }
  // A real same-family Continue must adopt a longer relative release while
  // leaving the onset/selection alive. Its on-time End may not chop Tail.
  for(int boundary:{0,1,2}){
    stop();za_test_gfx_button(p,75,1);double startClock=value("clock"),handoff=onset+38400,total=handoff+192000,release=43200;
    double end=total+(boundary==1?0:release);bool transferred=false;int tailAudio=0;
    for(int b=0;b<int(std::ceil((end+blockSize)/blockSize));++b){audio.clear();midi.clear();int start=b*blockSize;
      if(!b)corpusDurationPacket(midi,1,.8,boundary,127,127,onset);
      int at=int(handoff)-start;if(at>=0 && at<blockSize){corpusDurationPacket(midi,1,0,boundary,127,127,at,7);corpusDurationPacket(midi,2,4,boundary,127,127,at,6,1);}
      int finish=int(total)-start;if(finish>=0 && finish<blockSize)corpusDurationPacket(midi,2,0,boundary,127,127,finish,7);
      process();if(at>=0 && at<blockSize){auto e=heap("event_base",64);check(e[4]==2,"Continue did not retain family");closeEnough(e[19],release);closeEnough(e[10]-startClock,end);transferred=true;}
      for(int i=0;i<blockSize;++i){if(start+i>=total && start+i<end && std::abs(audio.getSample(0,i))>1e-8)++tailAudio;if(start+i>=end)check(audio.getSample(0,i)==0 && audio.getSample(1,i)==0,"Continue overrun");}
    }
    check(transferred && (boundary==1 || tailAudio>16),"Continue/tail was cut short");std::cout<<"DURATION Continue boundary="<<boundary<<" full_release_ms=900 tail_audio="<<tailAudio<<std::endl;
  }
  stop();za_test_gfx_button(p,75,1);za_test_gfx_button(p,12,0);
  uint32_t dsp=0,gfx=0;za_native_gfx_faults(p,&dsp,&gfx);check(!dsp && !gfx && value("output_faults")==0,"runtime fault");
  std::cout<<"DURATION actual processor packet/audio, exact end, Continue and REPEAT passed; runtime faults="<<dsp<<','<<gfx<<std::endl;
}
