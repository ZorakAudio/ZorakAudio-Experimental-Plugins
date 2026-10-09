#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#include "JsfxTasks.h"
#include <cassert>
#include <iostream>
#include <string>
extern "C" void jsfx_ensure_mem(DSPJSFX_State *, int64_t) { assert(false); }
static int index(const char *name) {
  for (const auto &v : DSPJSFX_VARS)
    if (std::string(v.name) == name)
      return v.index;
  assert(false);
  return 0;
}
static double value(DSPJSFX_State &s, const char *name) {
  return double(s.vars[index(name)]);
}
static double api(DSPJSFX_State &s, int op, double handle, double arg = 0) {
  double args[] = {handle, arg};
  return jsfx_task_api(&s, op, args, op == 5 ? 2 : 1);
}
static void wait(DSPJSFX_State &s, double handle) {
  for (int i = 0; i < 10000; ++i) {
    int status = int(api(s, 0, handle));
    if (status == 3)
      return;
    assert(status != 4 && status != 5 && status != -1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  assert(false && "task timeout");
}
static double cancellable(DSPJSFX_State *s, double, const double *) {
  while (!jsfx_task_api(s, 50, nullptr, 0))
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  return 42;
}
static double leaf(DSPJSFX_State *, double, const double *) { return 42; }
static double invalidChild(DSPJSFX_State *s, double, const double *) {
  jsfx_task_submit(s, leaf, nullptr, 0, 0, 5000, 0, 0);
  return 7;
}
static double parent(DSPJSFX_State *s, double, const double *) {
  jsfx_task_submit(s, cancellable, nullptr, 0, 0, 1, 0, 0);
  return 7;
}
static double arenaFirst(DSPJSFX_State* s,double,const double*) {
  assert(jsfx_tasks::Runtime::privateHeap(s));
  s->mem[0]=double(s->mem[0])+double(s->sliders[0]);s->vars[0]=91;
  return 1;
}
static double arenaNext(DSPJSFX_State* s,double,const double*) {s->mem[1]=double(s->mem[0])*2;return 1;}
static double retryApi(DSPJSFX_State& s,int op,std::initializer_list<double> args) {
  double result;
  do {result=jsfx_task_api(&s,op,args.begin(),int(args.size()));if(result==-2) std::this_thread::yield();}while(result==-2);
  return result;
}
static void arenaChecks(DSPJSFX_State& s) {
  DSPJSFX_Cell memory[64]{};s.mem=memory;s.memN=64;s.sliders[0]=3;memory[0]=7;
  const double a=retryApi(s,20,{64,0,0});assert(a>0);
  assert(retryApi(s,20,{64,0,0})==-1);
  while(retryApi(s,21,{a})!=3) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  assert(retryApi(s,23,{a})==-4); // cannot seal a partially copied snapshot
  assert(retryApi(s,22,{a,1,8})==-4); // contiguous prefix required
  assert(retryApi(s,22,{a,0,64})==64);
  assert(retryApi(s,23,{a})==1);
  s.sliders[0]=100;memory[0]=20;
  double first=-2,next=-2;
  while(first==-2) first=jsfx_task_submit_arena(&s,arenaFirst,nullptr,0,a,0);
  assert(first>0);
  double bad=-2;while(bad==-2) bad=jsfx_task_submit_arena(&s,arenaNext,nullptr,0,a,0);
  assert(bad==-4); // every writer must depend on the previous writer
  while(next==-2) next=jsfx_task_submit_arena(&s,arenaNext,nullptr,0,a,first);
  wait(s,next);
  assert(double(memory[0])==20); // worker never wrote live heap
  assert(retryApi(s,24,{a,0,2})==2);
  assert(double(memory[0])==10 && double(memory[1])==20);
  assert(retryApi(s,25,{a,0})==91);
  assert(retryApi(s,29,{a,0})==1 && double(s.vars[0])==91);
  assert(retryApi(s,26,{a})==1);
  assert(retryApi(s,24,{a,0,2})==-4);
  retryApi(s,4,{first});retryApi(s,4,{next});
  s.mem=nullptr;s.memN=0;
}
static void heapSwapChecks(DSPJSFX_State& s) {
  auto* runtime=static_cast<jsfx_tasks::Runtime*>(s.taskContext);
  s.mem=static_cast<DSPJSFX_Cell*>(std::calloc(64,sizeof(DSPJSFX_Cell)));s.memN=64;s.sliders[0]=3;s.mem[0]=7;
  auto* original=s.mem;
  static std::mutex lifecycle;
  runtime->setHeapSwapHooks([](DSPJSFX_State* x){return x->mem && x->memN==64;},[](DSPJSFX_State* x){assert(x->mem);},[](DSPJSFX_State*){return lifecycle.try_lock();},[](DSPJSFX_State*){lifecycle.unlock();});
#if DSPJSFX_NATIVE_GFX_LEGACY
  double a=-1;while(a==-1 || a==-2) a=retryApi(s,30,{64,0,0});
  assert(a>0);
  while(retryApi(s,21,{a})!=4) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  assert(retryApi(s,31,{a,0,1})==1);
  assert(retryApi(s,31,{a,0,262145})==-3);
  double first=-2,next=-2;
  while(first==-2)first=jsfx_task_submit_arena(&s,arenaFirst,nullptr,0,a,0);
  while(next==-2)next=jsfx_task_submit_arena(&s,arenaNext,nullptr,0,a,first);
  wait(s,next);s.mem[0]=99;
  lifecycle.lock();const double adoptionArgs[]={a,0};
  assert(runtime->arenaApi(&s,32,adoptionArgs,2)==-2 && s.mem==original);
  lifecycle.unlock();
  assert(retryApi(s,32,{a,0})==1);
  assert(s.mem!=original && double(s.mem[0])==99 && double(s.mem[1])==20 && double(s.vars[0])==91);
  assert(retryApi(s,32,{a,0})==-4); // consumed exactly once
  retryApi(s,4,{first});retryApi(s,4,{next});
  double cancelledArena=-1;
  while(cancelledArena==-1 || cancelledArena==-2)cancelledArena=retryApi(s,30,{64,0,0});
  while(retryApi(s,21,{cancelledArena})!=4)std::this_thread::sleep_for(std::chrono::milliseconds(1));
  double job=-2;while(job==-2)job=jsfx_task_submit_arena(&s,cancellable,nullptr,0,cancelledArena,0);
  auto* live=s.mem;assert(retryApi(s,26,{cancelledArena})==1);
  for(int n=0;n<10000 && api(s,0,job)!=4;++n)std::this_thread::sleep_for(std::chrono::milliseconds(1));
  assert(api(s,0,job)==4 && retryApi(s,32,{cancelledArena,0})==-4 && s.mem==live);
  retryApi(s,4,{job});
#else
  assert(retryApi(s,30,{64,0,0})==-3); // fixed atomic heap required
#endif
  std::free(s.mem);s.mem=nullptr;s.memN=0;
}
int main() {
#ifdef JSFX_TASKS_TESTING
  // API-only programs have no worker threads.
  {jsfx_tasks::Runtime noWorkers(DSPJSFX_VARS_COUNT,false);
   std::this_thread::sleep_for(std::chrono::milliseconds(20));
   assert(noWorkers.testWorkerPasses()==0 && noWorkers.testParkedWorkers()==0);}
#endif
  auto runtime = std::make_unique<jsfx_tasks::Runtime>();
#ifdef JSFX_TASKS_TESTING
  for(int n=0;n<1000 && runtime->testParkedWorkers()!=2;++n)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  assert(runtime->testParkedWorkers()==2);
  const auto idlePasses=runtime->testWorkerPasses();
  std::this_thread::sleep_for(std::chrono::milliseconds(40));
  assert(runtime->testWorkerPasses()==idlePasses); // No idle timer wakeups.
  for(int n=0;n<20;++n){jsfx_tasks::Runtime idle;std::this_thread::sleep_for(std::chrono::milliseconds(1));}
#endif
  assert(!runtime->hasOutstandingWorkForIdle());
  DSPJSFX_State state{};
  za::jsfx::StateVariables variables;variables.bind(state,DSPJSFX_VARS_COUNT);
  state.taskContext = runtime.get();
  state.srate = 48000;
  arenaChecks(state);
  // A finished but unreleased result is still a polling obligation.
  double idleJob=0;while(idleJob<=0)idleJob=jsfx_task_submit(&state,leaf,nullptr,0,0,1,0,0);
  assert(runtime->hasOutstandingWorkForIdle());wait(state,idleJob);
  assert(runtime->hasOutstandingWorkForIdle());
  while(api(state,4,idleJob)==-2){};
  assert(!runtime->hasOutstandingWorkForIdle());
  heapSwapChecks(state);
  jsfx_init(&state);
  wait(state, value(state, "joined"));
  for (const char *name : {"product_task", "minimum_task", "maximum_task",
                           "any_task", "all_task", "empty_task", "local_task"})
    wait(state, value(state, name));
  assert(api(state, 2, value(state, "sum_task")) == 4950);
  assert(api(state, 2, value(state, "after_task")) == 9900);
  assert(api(state, 2, value(state, "nested_task")) == 7);
  assert(api(state, 2, value(state, "product_task")) == 24);
  assert(api(state, 2, value(state, "minimum_task")) == -2);
  assert(api(state, 2, value(state, "maximum_task")) == 1);
  assert(api(state, 2, value(state, "any_task")) == 1);
  assert(api(state, 2, value(state, "all_task")) == 1);
  assert(api(state, 2, value(state, "empty_task")) == 17);
  assert(api(state, 2, value(state, "local_task")) == 11);
  assert(api(state, 5, value(state, "mapped"), 99) ==
         297); // captured gain, not live 99
  jsfx_block(&state);
  assert(value(state, "adopted") == 9900);
  double job = 0;
  while (job <= 0)
    job = jsfx_task_submit(&state, cancellable, nullptr, 0, 0, 1, 0, 0);
  assert(api(state, 3, job) == 1 || api(state, 3, job) == -2);
  while (api(state, 0, job) != 4) {
    api(state, 3, job);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  double stale = job;
  runtime->reset();
  assert(api(state, 0, stale) == -1 || api(state, 0, stale) == 0);
  double bad = 0;
  while (bad <= 0)
    bad = jsfx_task_submit(&state, invalidChild, nullptr, 0, 0, 1, 0, 0);
  for (int n = 0; n < 1000 && api(state, 0, bad) != 5; ++n)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  assert(api(state, 0, bad) ==
         5); // A rejected child must fail structured completion.
  double tree = 0;
  while (tree <= 0)
    tree = jsfx_task_submit(&state, parent, nullptr, 0, 0, 1, 0, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  assert(api(state, 1, tree) == 0);
  while (api(state, 3, tree) == -2) {
  }
  for (int n = 0; n < 1000 && api(state, 0, tree) != 4; ++n)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (api(state, 0, tree) != 4)
    std::cerr << "Cancelled tree status: " << api(state, 0, tree) << '\n';
  assert(api(state, 0, tree) == 4);
  runtime->reset();
  // Exercise cancellation both before a parent starts and during child
  // creation.
  for (int attempt = 0; attempt < 20; ++attempt) {
    double raced = 0;
    while (raced <= 0)
      raced = jsfx_task_submit(&state, parent, nullptr, 0, 0, 1, 0, 0);
    if (attempt % 2)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    while (api(state, 3, raced) == -2) {
    }
    for (int n = 0; n < 1000 && api(state, 0, raced) != 4; ++n)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    assert(api(state, 0, raced) == 4);
    while (api(state, 4, raced) == -2) {
    }
  }
  // Generation reset invalidates old handles; task slots become reusable.
  for (int n = 0; n < 100; ++n) {
    double t = 0;
    while (t <= 0) {
      t = jsfx_task_submit(&state, leaf, nullptr, 0, 0, 1, 0, 0);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    wait(state, t);
    assert(api(state, 2, t) == 42);
    while (api(state, 4, t) == -2) {
    }
  }
  // Releasing a sealed input while work runs retains storage until quiescence.
  double create[] = {4};
  double b = 0;
  while (b <= 0)
    b = jsfx_task_api(&state, 10, create, 1);
  double seal[] = {b};
  while (jsfx_task_api(&state, 13, seal, 1) == -2) {
  }
  double running = 0;
  while (running <= 0)
    running = jsfx_task_submit(&state, cancellable, nullptr, 0, 0, 1, 0, 0);
  while (jsfx_task_api(&state, 14, seal, 1) == -2) {
  }
  // Destruction cancels and joins even an active callback.
  runtime.reset();
  std::cout << "Structured task integration passed\n";
}
