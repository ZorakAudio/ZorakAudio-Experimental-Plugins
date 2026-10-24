#pragma once
#include <numeric>

// Read-only diagnostic of the released selection algorithm. The source and
// runtime are unchanged; the host compares actual choices with cached evidence.
template<class Read> void corpusFollowDiagnostic(juce::AudioProcessor* p,juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,const std::string& path){
 auto check=[](bool ok,const char* why){if(!ok)throw std::runtime_error(why);};
 auto heap=[&](const char* name,int n){std::vector<double> x(n);check(za_native_gfx_heap_copy(p,name,x.data(),n)==n,"Follow diagnostic heap read failed");return x;};
 auto process=[&](){audio.clear();midi.clear();p->processBlock(audio,midi);};
 const int grainCount=int(value("grain_count"));auto grains=heap("grain_base",grainCount*64);
 auto feature=[&](int id,int d){return grains[id*64+(d<5?16+d:51+d-5)];};
 auto eligible=[&](int id){return id>=0 && id<grainCount && grains[id*64+6]>.00001 && grains[id*64+9]<.5;};
 std::vector<int> valid;for(int id=0;id<grainCount;++id)if(eligible(id))valid.push_back(id);check(!valid.empty(),"No eligible diagnostic grains");
 std::ofstream corpus(path+".corpus.csv");corpus<<"id,start_s,duration_s,motion,brightness,eligible,quiet\n";
 for(int id=0;id<grainCount;++id)corpus<<id<<','<<grains[id*64+2]/grains[id*64+4]<<','<<grains[id*64+3]/grains[id*64+4]<<','<<feature(id,3)<<','<<feature(id,1)<<','<<eligible(id)<<','<<grains[id*64+60]<<'\n';corpus.close();
 int highMotion=0,highBright=0,jointHigh=0;for(int id:valid){bool m=feature(id,3)>=.9,b=feature(id,1)>=.9;highMotion+=m;highBright+=b;jointHigh+=m&&b;}
 std::cout<<"FOLLOW coverage eligible="<<valid.size()<<" motion>=90%="<<highMotion<<" brightness>=90%="<<highBright<<" both>=90%="<<jointHigh<<std::endl;
 struct Rule{int descriptor,mode;double start,end;};struct Case{const char* name;std::vector<Rule> rules;};
 std::vector<Case> cases{{"motion_floor",{{3,2,1,1}}},{"brightness_floor",{{1,2,1,1}}},{"both_floors",{{3,2,1,1},{1,2,1,1}}},{"both_targets",{{3,0,1,1},{1,0,1,1}}},{"motion_floor_brightness_target50",{{3,2,1,1},{1,0,.5,.5}}},{"motion_floor_brightness_ramp",{{3,2,1,1},{1,0,1,0}}}};
 for(auto [id,x]:std::vector<std::pair<int,float>>{{10,0},{12,1},{25,4000},{8,0},{45,0},{76,0},{82,0},{84,0},{85,-1},{14,0},{15,0},{50,0}})corpusSet(p,id,x);
 constexpr double tolerance=.1;std::ofstream rows(path+".choices.csv"),summary(path+".summary.csv");
 rows<<"case,variation,influence,hierarchy,seed,ordinal,time_s,grain,motion,brightness,error1,error2,global_best_error1,global_best_error2,candidate_best_error1,candidate_best_error2,global_joint_matches,candidate_joint_matches,global_improves_both,candidate_improves_both,candidates\n";
 summary<<"case,variation,influence,hierarchy,choices,mean_error1,mean_error2,mean_motion,mean_brightness,violates_rule1,violates_rule2,joint_misses,joint_misses_with_global_match,joint_misses_with_candidate_match,dominated_global,dominated_candidate\n";
 for(const auto& test:cases)for(float variation:{.35f,0.f})for(float influence:{1.f,2.f}){
  for(int l=0;l<12;++l)corpusSet(p,86+l*13,0);
  for(int l=0;l<int(test.rules.size());++l){auto r=test.rules[l];corpusSet(p,86+l*13,float(3072+r.descriptor+r.mode*64));corpusSet(p,87+l*13,1);
   for(int j=0;j<11;++j){int packed=0;for(int k=0;k<3;++k){int ix=j*3+k;if(ix<32){int q=int(std::round((r.start+(r.end-r.start)*ix/31)*126));packed+=q<<(k*7);}}corpusSet(p,88+l*13+j,float(packed));}}
  corpusSet(p,2,variation);corpusSet(p,81,influence);corpusSet(p,56,5);process();
  int choices=0,miss1=0,miss2=0,jointMiss=0,jointMissGlobal=0,jointMissCandidate=0,domGlobal=0,domCandidate=0;double errors[2]{},motion=0,bright=0;
  for(int seed:{12345,23579,71237}){corpusSet(p,11,float(seed));process();int previous=0;
   for(int block=0;block<800;++block){audio.clear();midi.clear();if(!block)midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);p->processBlock(audio,midi);
    int count=int(value("record_n"));if(count<=previous)continue;
    auto records=heap("record_base",count*20);int selected=int(records[(count-1)*20]);double at=records[(count-1)*20+2]/48000.;double u=std::clamp(at/4.,0.,1.);
    auto error=[&](int id,int lane){if(lane>=int(test.rules.size()))return 0.;auto r=test.rules[lane];double requested=r.start+(r.end-r.start)*u;double actual=feature(id,r.descriptor);return r.mode==0?std::abs(actual-requested):r.mode==1?std::max(0.,actual-requested):std::max(0.,requested-actual);};
    double got[2]{error(selected,0),error(selected,1)},gb[2]{1,1},cb[2]{1,1};int globalJoint=0,candidateJoint=0;bool globalDominates=false,candidateDominates=false;
    auto inspect=[&](int id,double* best,int& joint,bool& dominates){if(!eligible(id))return;double a=error(id,0),b=error(id,1);best[0]=std::min(best[0],a);best[1]=std::min(best[1],b);joint+=a<=tolerance && b<=tolerance;if(test.rules.size()==2 && a<=got[0]-.05 && b<=got[1]-.05)dominates=true;};
    for(int id:valid)inspect(id,gb,globalJoint,globalDominates);
    int nc=int(value("candidate_n"));auto candidates=heap("candidates",nc*2);for(int j=0;j<nc;++j)inspect(int(candidates[j*2]),cb,candidateJoint,candidateDominates);
    ++choices;errors[0]+=got[0];errors[1]+=got[1];motion+=feature(selected,3);bright+=feature(selected,1);miss1+=got[0]>tolerance;miss2+=got[1]>tolerance;
    bool missed=got[0]>tolerance || got[1]>tolerance;jointMiss+=missed;jointMissGlobal+=missed&&globalJoint>0;jointMissCandidate+=missed&&candidateJoint>0;domGlobal+=globalDominates;domCandidate+=candidateDominates;
    rows<<test.name<<','<<variation<<','<<influence<<",5,"<<seed<<','<<count<<','<<at<<','<<selected<<','<<feature(selected,3)<<','<<feature(selected,1)<<','<<got[0]<<','<<got[1]<<','<<gb[0]<<','<<gb[1]<<','<<cb[0]<<','<<cb[1]<<','<<globalJoint<<','<<candidateJoint<<','<<globalDominates<<','<<candidateDominates<<','<<nc<<'\n';previous=count;
   }
  }
  check(choices>0,"No generated diagnostic choices");summary<<test.name<<','<<variation<<','<<influence<<",5,"<<choices<<','<<errors[0]/choices<<','<<errors[1]/choices<<','<<motion/choices<<','<<bright/choices<<','<<miss1<<','<<miss2<<','<<jointMiss<<','<<jointMissGlobal<<','<<jointMissCandidate<<','<<domGlobal<<','<<domCandidate<<'\n';
  std::cout<<"FOLLOW case="<<test.name<<" variation="<<variation<<" influence="<<influence<<" n="<<choices<<" mean_error="<<errors[0]/choices<<','<<errors[1]/choices<<" missed="<<jointMiss<<" despite_global="<<jointMissGlobal<<" despite_candidate="<<jointMissCandidate<<" better_both_candidate="<<domCandidate<<std::endl;
 }
 uint32_t dsp=0,gfx=0;za_native_gfx_faults(p,&dsp,&gfx);check(!dsp&&!gfx && value("output_faults")==0,"Follow diagnostic runtime fault");std::cout<<"FOLLOW diagnostic finished faults="<<dsp<<','<<gfx<<std::endl;
}
