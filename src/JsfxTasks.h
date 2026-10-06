// SPDX-License-Identifier: Zlib
// Include after the generated JSFXDSP.h. No JUCE dependency.
#pragma once
#if DSPJSFX_HAS_TASKS
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>

namespace jsfx_tasks {
using Callback = double (*)(DSPJSFX_State *, double, const double *);
// Fixed per-instance budgets. Construction allocates; submission never does.
constexpr int Slots = 32, Iterations = 4096, Captures = 64, Buffers = 8;
constexpr int BufferCells = 65536;
enum Status {
  Invalid = -1,
  Busy = 0,
  Pending = 1,
  Running = 2,
  Succeeded = 3,
  Cancelled = 4,
  Failed = 5
};
static_assert(std::atomic<double>::is_always_lock_free &&
                  std::atomic<uint64_t>::is_always_lock_free,
              "Task result polling requires lock-free 64-bit atomics");
class Runtime {
  struct Job {
    bool arenaJob = false;
    Job *owner =
        nullptr; // Retained by the child's outstanding parent reference.
    DSPJSFX_State state{};
    std::array<double, Captures> captures{};
    std::array<std::atomic<double>, Iterations> values{};
    std::array<uint64_t, Slots> dependencies{};
    std::atomic<uint64_t> id{0}, epoch{0};
    uint64_t parent = 0;
    Callback callback = nullptr;
    std::atomic<int> status{Invalid}, count{0};
    int next = 0, completed = 0, executing = 0;
    int children = 0, refs = 0, mode = 0;
    std::atomic<bool> released{false};
    bool bodyDone = false;
    std::atomic<bool> cancel{false};
    std::atomic<bool> failed{false};
    double identity = 0;
    std::atomic<double> result{0};
  };
  struct Buffer {
    std::array<std::atomic<double>, BufferCells> data{};
    std::atomic<uint64_t> id{0};
    std::atomic<uint64_t> epoch{0};
    std::atomic<int> size{0};
    std::atomic<bool> sealed{false}, released{false};
  };
  struct HeapDeleter {void operator()(DSPJSFX_Cell* p) const noexcept {std::free(p);}};
  using Heap=std::unique_ptr<DSPJSFX_Cell[],HeapDeleter>;
  // One private heap per instance. Allocation/reclamation happen on workers.
  struct Arena {
    DSPJSFX_State state{};
    Heap memory;
    std::shared_ptr<void> lease;
    std::atomic<uint64_t> id{0}, epoch{0};
    std::atomic<int> status{0}; // 1 queued, 2 allocating, 3 copying, 4 sealed, 5 failed
    std::atomic<bool> released{false};
    std::array<std::atomic<double>, 4> progress{};
    int64_t size=0, copied=0;
    DSPJSFX_Cell* input=nullptr;
    bool clone=false;
    std::array<std::array<int64_t,2>,16> preserve{};
    int preserveCount=0;int64_t preserveCells=0;
    int refs=0;
    uint64_t lastWriter=0;
    double pool=0, generation=0;
  } arena;
  using AcquireLease = std::shared_ptr<void> (*)(DSPJSFX_State*,double,double);
  using ExecutePinned = double (*)(const std::shared_ptr<void>&,Callback,DSPJSFX_State*,const double*);
  using ValidateLease=bool (*)(const std::shared_ptr<void>&);
  ValidateLease validateLease=nullptr;
  using ValidateHeap=bool (*)(DSPJSFX_State*);
  using RebindHeap=void (*)(DSPJSFX_State*);
  ValidateHeap validateHeap=nullptr;
  RebindHeap rebindHeap=nullptr;
  AcquireLease acquireLease=nullptr;
  ExecutePinned executePinned=nullptr;
  std::unique_ptr<std::array<Job, Slots>> jobs{new std::array<Job, Slots>};
  std::unique_ptr<std::array<Buffer, Buffers>> buffers{
      new std::array<Buffer, Buffers>};
  std::mutex mutex;
  std::array<std::thread, 2> threads;
  std::atomic<bool> stopping{false};
  uint64_t serial = 0;
  std::atomic<uint64_t> epoch{1};
  int active = 0, cursor = 0;
  inline static thread_local Runtime *current = nullptr;
  inline static thread_local Job *currentJob = nullptr;
  static bool terminal(int s) { return s >= Succeeded; }
  static uint64_t handle(double v) {
    return std::isfinite(v) && v > 0 && v < 9007199254740992. &&
                   std::floor(v) == v
               ? (uint64_t)v
               : 0;
  }
  Job *find(uint64_t id) {
    if (!id)
      return nullptr;
    for (auto &j : *jobs)
      if (j.id == id)
        return &j;
    return nullptr;
  }
  Buffer *buffer(uint64_t id) {
    if (!id)
      return nullptr;
    for (auto &b : *buffers)
      if (b.id == id)
        return &b;
    return nullptr;
  }
  bool descendant(Job &j, uint64_t id) {
    Job *node = &j;
    for (int n = 0; n < Slots && node; ++n) {
      if (node->id == id)
        return true;
      node = find(node->parent);
    }
    return false;
  }
  void collect() {
    for (auto &j : *jobs)
      if (j.id && terminal(j.status) && j.released && !j.refs)
        j.id = 0;
    if (!active)
      for (auto &b : *buffers)
        if (b.released)
          b.id = 0;
  }
  void finish(Job &j) {
    if (terminal(j.status) || !j.bodyDone || j.executing || j.children)
      return;
    const int finalStatus = j.failed.load()   ? Failed
                            : j.cancel.load() ? Cancelled
                                              : Succeeded;
    if (finalStatus == Succeeded && j.mode >= 2) {
      double v = j.identity;
      for (int i = 0; i < j.count; ++i) {
        double x = j.values[i];
        switch (j.mode) {
        case 2:
          v += x;
          break;
        case 3:
          v *= x;
          break;
        case 4:
          v = std::isnan(v) || std::isnan(x)
                  ? std::numeric_limits<double>::quiet_NaN()
                  : std::min(v, x);
          break;
        case 5:
          v = std::isnan(v) || std::isnan(x)
                  ? std::numeric_limits<double>::quiet_NaN()
                  : std::max(v, x);
          break;
        case 6:
          v = (v != 0 || x != 0) ? 1 : 0;
          break;
        case 7:
          v = (v != 0 && x != 0) ? 1 : 0;
          break;
        }
      }
      j.result = v;
    }
    if (j.arenaJob) { --arena.refs; if (finalStatus!=Succeeded) arena.status=5; }
    j.status.store(finalStatus);
    --active;
    for (auto &child : *jobs)
      if (child.id && child.parent == j.id) {
        --child.refs;
        child.released = true;
      }
    for (auto id : j.dependencies)
      if (auto *d = find(id))
        --d->refs;
    if (auto *p = find(j.parent)) {
      --p->children;
      --p->refs;
      if (j.status != Succeeded) {
        p->cancel.store(true);
        if (j.status == Failed)
          p->failed.store(true);
      }
      finish(*p);
    }
  }
  bool ready(Job &j) {
    for (auto id : j.dependencies)
      if (id) {
        auto *d = find(id);
        if (!d || d->status == Cancelled || d->status == Failed) {
          j.cancel.store(true);
          if (!d || d->status == Failed)
            j.failed.store(true);
          return false;
        }
        if (d->status != Succeeded)
          return false;
      }
    return true;
  }
  bool cancelled(Job &job) const {
    for (auto *j = &job; j; j = j->owner)
      if (j->cancel.load())
        return true;
    return stopping.load();
  }
  double reject(double code) {
    if (current == this && currentJob) {
      if (!cancelled(*currentJob))
        currentJob->failed.store(true);
      currentJob->cancel.store(true);
    }
    return code;
  }
  void run() {
    while (!stopping.load()) {
      bool allocate=false, reclaim=false;
      Heap retiredMemory;
      std::shared_ptr<void> retiredLease;
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (arena.id && arena.released && !arena.refs && arena.status!=2 && arena.status!=6) {
          retiredMemory=std::move(arena.memory);retiredLease=std::move(arena.lease);
          arena.status=6; reclaim=true;
        } else if (arena.id && arena.status==1) {arena.status=2; allocate=true;}
      }
      if (reclaim) {retiredMemory.reset();retiredLease.reset();std::lock_guard<std::mutex> lock(mutex);arena.id=0;arena.status=0;}
      if (allocate) {
        Heap memory;
        std::shared_ptr<void> lease;
        bool ok=true;
        try {
          if (arena.pool!=0) {
            lease=acquireLease ? acquireLease(&arena.state,arena.pool,arena.generation) : nullptr;
            if (!lease) ok=false;
          }
          if (ok) {
            memory.reset(static_cast<DSPJSFX_Cell*>(std::calloc(size_t(arena.size),sizeof(DSPJSFX_Cell))));
            if(!memory) throw std::bad_alloc();
            for (int64_t i=0;i<arena.size;++i) {
              if((i&16383)==0 && (arena.released || stopping)) {ok=false;break;}
              memory[i]=arena.clone ? double(arena.input[i]) : 0.;
            }
          }
        } catch (...) {ok=false;}
        std::lock_guard<std::mutex> lock(mutex);
        arena.memory=std::move(memory);arena.lease=std::move(lease);
        arena.state.mem=arena.memory.get();arena.state.memN=arena.size;
        arena.copied=ok && arena.clone ? arena.size : 0;
        arena.status=ok && !arena.released ? (arena.clone ? 4 : 3) : 5;
        continue;
      }
      Job *selected = nullptr;
      int first = 0, last = 0;
      {
        std::lock_guard<std::mutex> lock(mutex);
        for (int k = 0; k < Slots; ++k) {
          auto &j = (*jobs)[(cursor + k) % Slots];
          if (!j.id || terminal(j.status))
            continue;
          if (cancelled(j)) {
            j.cancel.store(true);
            j.bodyDone = true;
            finish(j);
            continue;
          }
          if (!ready(j))
            continue;
          if (j.count == 0) {
            j.bodyDone = true;
            finish(j);
            continue;
          }
          if (j.next >= j.count)
            continue;
          selected = &j;
          first = j.next;
          last = std::min(j.count.load(), first + 32);
          j.next = last;
          ++j.executing;
          j.status = Running;
          cursor = (cursor + k + 1) % Slots;
          break;
        }
      }
      if (!selected) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      auto &j = *selected;
      current = this;
      currentJob = &j;
      for (int i = first; i < last && !j.cancel.load() && !stopping.load();
           ++i) {
        DSPJSFX_State local{};
        for (int v = 0; v < DSPJSFX_VARS_COUNT; ++v)
          local.vars[v] = double(j.state.vars[v]);
        local.srate = j.state.srate;
        local.samplesblock = j.state.samplesblock;
        local.taskContext = this;
        try {
          if (j.arenaJob) {
            j.values[i]=executePinned && arena.lease
                ? executePinned(arena.lease,j.callback,&arena.state,j.captures.data())
                : j.callback(&arena.state,0,j.captures.data());
            if (arena.state.memoryFault) {j.failed=true;j.cancel=true;}
          } else j.values[i] = j.callback ? j.callback(&local, double(i), j.captures.data()) : 0;
        } catch (...) {
          j.failed.store(true);
          j.cancel.store(true);
        }
      }
      currentJob = nullptr;
      current = nullptr;
      {
        std::lock_guard<std::mutex> lock(mutex);
        --j.executing;
        j.completed += last - first;
        if (j.completed == j.count || j.cancel.load()) {
          j.bodyDone = true;
          if (j.mode == 0 && j.count)
            j.result = j.values[0].load();
        }
        finish(j);
        collect();
      }
    }
  }

public:
  void setHeapSwapHooks(ValidateHeap validate,RebindHeap rebind) {validateHeap=validate;rebindHeap=rebind;}
  void setArenaHooks(AcquireLease acquire, ExecutePinned execute, ValidateLease validate) { acquireLease=acquire;executePinned=execute;validateLease=validate; }
  static bool privateHeap(const DSPJSFX_State* state) {
    return current && currentJob && currentJob->arenaJob && state==&current->arena.state;
  }
  static bool sampleReadAllowed(DSPJSFX_State* state,double pool) {
    if(!privateHeap(state)) return true;
    const bool allowed=current->arena.lease && pool==current->arena.pool;
    if(!allowed) {state->memoryFault=1;currentJob->failed=true;currentJob->cancel=true;}
    return allowed;
  }
  Runtime() {
    // Explicitly initialize atomic storage for C++17 too, and commit its pages
    // before an audio callback can submit the first task.
    for (auto &job : *jobs)
      for (auto &value : job.values)
        value.store(0);
    for (auto &buffer : *buffers)
      for (auto &value : buffer.data)
        value.store(0);
    try {
      for (auto &t : threads)
        t = std::thread([this] { run(); });
    } catch (...) {
      stopping.store(true);
      for (auto &t : threads)
        if (t.joinable())
          t.join();
      throw;
    }
  }
  ~Runtime() {
    stopping.store(true);
    for (auto &j : *jobs)
      j.cancel.store(true);
    for (auto &t : threads)
      if (t.joinable())
        t.join();
  }
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
  void reset() {
    const auto generation = epoch.fetch_add(1) + 1;
    if (arena.id && arena.epoch<generation) arena.released=true;
    for (auto &j : *jobs)
      if (j.id.load() && j.epoch.load() < generation) {
        j.cancel.store(true);
        j.released.store(true);
      }
    for (auto &b : *buffers)
      if (b.id.load() && b.epoch.load() < generation)
        b.released.store(true);
  }
  double checkpoint() const {
    if (current != this || !currentJob)
      return 0;
    if (cancelled(*currentJob)) {
      currentJob->cancel.store(true);
      return 1;
    }
    return 0;
  }
  double submit(DSPJSFX_State *source, Callback callback,
                const double *captures, int n, double dependency, double count,
                int mode, double identity, double arenaHandle=0) {
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    if (current == this)
      lock.lock();
    else if (!lock.try_lock())
      return -2;
    if (n < 0 || n > Captures || !std::isfinite(count) || count < 0 ||
        count > Iterations || std::floor(count) != count)
      return reject(-3);
    collect();
    const bool inArena=arenaHandle!=0;
    if (inArena && (current==this || arena.id!=handle(arenaHandle) || arena.released || arena.epoch!=epoch || arena.status!=4 || handle(dependency)!=arena.lastWriter)) return reject(-4);
    Job *p = current == this ? currentJob : nullptr;
    if (p && (p->cancel.load() || p->epoch != epoch))
      return reject(-4);
    auto *d = dependency ? find(handle(dependency)) : nullptr;
    if (dependency && (!d || d->released || d->epoch != epoch ||
                       d->status == Cancelled || d->status == Failed))
      return reject(-4);
    Job *j = nullptr;
    for (auto &candidate : *jobs)
      if (!candidate.id) {
        j = &candidate;
        break;
      }
    if (!j) {
      if (p) {
        p->failed.store(true);
        p->cancel.store(true);
      }
      return -1;
    }
    const auto id = ++serial;
    j->arenaJob=inArena;
    if (inArena) {++arena.refs;arena.lastWriter=id;}
    j->epoch = epoch.load();
    j->parent = p ? p->id.load() : 0;
    j->owner = p;
    j->callback = callback;
    j->count = int(count);
    j->next = j->completed = j->executing = j->children = 0;
    j->refs = p ? 1 : 0;
    j->mode = mode;
    j->identity = identity;
    j->result = identity;
    j->released = false;
    j->bodyDone = false;
    j->cancel.store(false);
    j->failed.store(false);
    j->status = Pending;
    j->dependencies.fill(0);
    if (d) {
      j->dependencies[0] = d->id;
      ++d->refs;
    }
    if (p) {
      ++p->children;
      ++p->refs;
    }
    for (int i = 0; i < n; ++i)
      j->captures[i] = captures[i];
#if DSPJSFX_NATIVE_GFX_LEGACY
    if (source->sharedState)
      source = static_cast<DSPJSFX_State *>(source->sharedState);
#endif
    for (int i = 0; i < DSPJSFX_VARS_COUNT; ++i)
      j->state.vars[i] = double(source->vars[i]);
    j->state.srate = source->srate;
    j->state.samplesblock = source->samplesblock;
    ++active;
    j->id.store(id);
    if (j->epoch.load() != epoch.load()) {
      j->cancel.store(true);
      j->released.store(true);
      return -4;
    }
    return double(id);
  }
  double arenaApi(DSPJSFX_State* source,int op,const double* a,int n) {
#if DSPJSFX_NATIVE_GFX_LEGACY
    if(source && source->sharedState) source=static_cast<DSPJSFX_State*>(source->sharedState);
#endif
    if (op==27) {
      if (!currentJob || !currentJob->arenaJob || n!=4) return reject(-3);
      for(int i=0;i<4;++i) arena.progress[i]=a[i];return 1;
    }
    std::unique_lock<std::mutex> lock(mutex,std::try_to_lock);
    if (!lock.owns_lock()) return -2;
    if (current==this) return reject(-3);
    if (op==20 || op==30) {
      if(op==30) {
#if DSPJSFX_NATIVE_GFX_LEGACY
        if(!validateHeap || !source || !validateHeap(source)) return -3;
#else
        return -3;
#endif
      }
      if(n!=3 || !std::isfinite(a[1]) || !std::isfinite(a[2]) || a[1]<0 || a[2]<0 || a[1]>=9007199254740992. || a[2]>=9007199254740992. || std::floor(a[1])!=a[1] || std::floor(a[2])!=a[2] || !std::isfinite(a[0]) || a[0]<=0 || a[0]>25165824 || std::floor(a[0])!=a[0] || !source || a[0]!=source->memN) return -3;
      if(arena.id) return -1;
      arena.state={};
      for(int i=0;i<DSPJSFX_VARS_COUNT;++i) arena.state.vars[i]=double(source->vars[i]);
      for(int i=0;i<DSPJSFX_MAX_SLIDERS;++i) arena.state.sliders[i]=double(source->sliders[i]);
      arena.state.srate=double(source->srate);arena.state.samplesblock=double(source->samplesblock);
      arena.state.hostOwner=source->hostOwner;arena.state.taskContext=this;
      arena.input=source->mem;arena.clone=op==30;arena.preserveCount=0;arena.preserveCells=0;
      arena.size=int64_t(a[0]);arena.copied=0;arena.refs=0;arena.lastWriter=0;
      arena.pool=a[1];arena.generation=a[2];arena.released=false;arena.epoch=epoch.load();
      for(auto& v:arena.progress) v=0;
      arena.status=1;arena.id=++serial;return double(arena.id.load());
    }
    if(!n || arena.id!=handle(a[0]) || arena.released || arena.epoch!=epoch) return -4;
    if(op==21) return arena.status;
    if(op==28) {
      if(n!=2 || a[1]<0 || a[1]>=4 || std::floor(a[1])!=a[1]) return -3;
      return arena.progress[int(a[1])].load();
    }
    if(op==26) {
      arena.released=true;
      for(auto& j:*jobs) if(j.id && j.arenaJob && !terminal(j.status)) j.cancel=true;
      return 1;
    }
    if(op==23) {
      if(arena.status!=3 || arena.copied!=arena.size) return -4;
      arena.status=4;return 1;
    }
    if ((op==24 || op==25 || op==29 || op==32) && arena.pool!=0 && (!validateLease || !validateLease(arena.lease))) {arena.status=5;return -4;}
    if(op==31) {
      if(n!=3 || arena.preserveCount==16 || !std::isfinite(a[1]) || !std::isfinite(a[2]) || a[1]<0 || a[2]<0 || std::floor(a[1])!=a[1] || std::floor(a[2])!=a[2] || a[1]+a[2]>arena.size || arena.preserveCells+a[2]>262144) return -3;
      arena.preserve[arena.preserveCount++]={int64_t(a[1]),int64_t(a[2])};arena.preserveCells+=int64_t(a[2]);return 1;
    }
    if(op==32) {
#if DSPJSFX_NATIVE_GFX_LEGACY
      if(arena.status!=4 || arena.refs || !arena.clone || !source || source->mem!=arena.input || source->memN!=arena.size || !validateHeap || !validateHeap(source) || n<1 || n>4097) return -4;
      for(int i=1;i<n;++i) if(!std::isfinite(a[i]) || a[i]<0 || a[i]>=DSPJSFX_VARS_COUNT || std::floor(a[i])!=a[i]) return -3;
      for(int r=0;r<arena.preserveCount;++r) {
        const auto first=arena.preserve[r][0],end=first+arena.preserve[r][1];
        for(int64_t i=first;i<end;++i) arena.memory[i]=double(source->mem[i]);
      }
      if(arena.pool!=0 && (!validateLease || !validateLease(arena.lease))) {arena.status=5;return -4;}
      auto* replacement=arena.memory.release();
      arena.memory.reset(source->mem); // Old heap is retained; never freed here.
      source->mem=replacement;
      for(int i=1;i<n;++i) source->vars[int(a[i])]=double(arena.state.vars[int(a[i])]);
      arena.state.mem=arena.memory.get();
      if(rebindHeap) rebindHeap(source);
      arena.released=true; // Worker reclaims the old heap and sample lease.
      return 1;
#else
      return -3;
#endif
    }
    if(op==29) {
      if(arena.status!=4 || arena.refs || !source || n<1 || n>4097) return -4;
      for(int i=1;i<n;++i) if(!std::isfinite(a[i]) || a[i]<0 || a[i]>=DSPJSFX_VARS_COUNT || std::floor(a[i])!=a[i]) return -3;
      for(int i=1;i<n;++i) source->vars[int(a[i])]=double(arena.state.vars[int(a[i])]);
      return 1;
    }
    if(op==25) {
      if(arena.status!=4 || arena.refs || !std::isfinite(a[1]) || a[1]<0 || a[1]>=DSPJSFX_VARS_COUNT || std::floor(a[1])!=a[1]) return -4;
      return double(arena.state.vars[int(a[1])]);
    }
    if(op==22 || op==24) {
      if(n!=3 || !std::isfinite(a[1]) || !std::isfinite(a[2]) || a[1]<0 || a[2]<0 || a[2]>16384 || a[1]+a[2]>arena.size || std::floor(a[1])!=a[1] || std::floor(a[2])!=a[2] || !source || source->memN<arena.size) return -3;
      if(op==22 && (arena.status!=3 || int64_t(a[1])!=arena.copied) || op==24 && (arena.status!=4 || arena.refs)) return -4;
      for(int64_t i=int64_t(a[1]);i<int64_t(a[1]+a[2]);++i) {
        if(op==22) arena.memory[i]=double(source->mem[i]);else source->mem[i]=double(arena.memory[i]);
      }
      if(op==22) arena.copied+=int64_t(a[2]);return a[2];
    }
    return -3;
  }
  double api(int op, const double *a, int n, DSPJSFX_State* source=nullptr) {
    if(op>=20 && op<=32) return arenaApi(source,op,a,n);
    if (op == 50)
      return checkpoint();
    if (op == 12) {
      const auto id = n == 2 ? handle(a[0]) : 0;
      auto *b = buffer(id);
      if (!b || b->epoch.load() != epoch.load() || !b->sealed.load() ||
          (b->released.load() && current != this) || !std::isfinite(a[1]) ||
          a[1] < 0 || a[1] >= b->size.load() || std::floor(a[1]) != a[1])
        return reject(-4);
      const auto result = b->data[int(a[1])].load();
      return b->id.load() == id && b->epoch.load() == epoch.load() ? result : reject(-4);
    }
    if (op == 0 || op == 1 || op == 2 || op == 5) {
      const auto id = n ? handle(a[0]) : 0;
      auto *job = find(id);
      if (!job)
        return op == 0 ? double(Invalid) : op == 1 ? 0 : reject(-4);
      const auto generation = epoch.load();
      if (job->epoch.load() != generation ||
          (job->released.load() && current != this))
        return op == 0 ? double(Invalid) : op == 1 ? 0 : reject(-4);
      const int status = job->status.load();
      double result = 0;
      if (op == 0)
        result = status;
      else if (op == 1)
        result = terminal(status) ? 1 : 0;
      else if (status == Succeeded) {
        if (op == 2)
          result = job->result.load();
        else if (n == 2 && std::isfinite(a[1]) && a[1] >= 0 &&
                 a[1] < job->count.load() && std::floor(a[1]) == a[1])
          result = job->values[int(a[1])].load();
        else
          return reject(-3);
      } else
        return reject(-4);
      return job->id.load() == id && job->epoch.load() == generation &&
                     epoch.load() == generation
                 ? result
             : op == 0 ? double(Invalid)
                       : op == 1 ? 0 : reject(-4);
    }
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    if (current == this)
      lock.lock();
    else if (!lock.try_lock())
      return op == 0 || op == 1 ? Busy : -2;
    Job *j = n ? find(handle(a[0])) : nullptr;
    if (j && (j->epoch != epoch || j->released && current != this))
      j = nullptr;
    switch (op) {
    case 3:
      if (!j)
        return 0;
      for (auto &child : *jobs)
        if (child.id && descendant(child, j->id))
          child.cancel.store(true);
      return 1;
    case 4:
      if (!j)
        return 0;
      j->released = true;
      collect();
      return 1;
    case 6: { // A join has no body; dependencies retain their handles.
      if (n < 1 || n > Slots)
        return reject(-3);
      std::array<Job *, Slots> ds{};
      for (int i = 0; i < n; ++i) {
        ds[i] = find(handle(a[i]));
        if (!ds[i] || ds[i]->epoch != epoch || ds[i]->released)
          return reject(-4);
      }
      collect();
      Job *target = nullptr;
      for (auto &c : *jobs)
        if (!c.id) {
          target = &c;
          break;
        }
      if (!target)
        return reject(-1);
      auto *parent = current == this ? currentJob : nullptr;
      const auto joinId = ++serial;
      target->arenaJob=false;
      target->epoch = epoch.load();
      target->parent = parent ? parent->id.load() : 0;
      target->owner = parent;
      target->count = 0;
      target->next = target->completed = target->executing = target->children =
          target->refs = 0;
      target->refs = parent ? 1 : 0;
      target->mode = 0;
      target->callback = nullptr;
      target->result = 0;
      target->released = false;
      target->bodyDone = false;
      target->cancel.store(false);
      target->failed.store(false);
      target->status = Pending;
      target->dependencies.fill(0);
      if (parent) {
        ++parent->children;
        ++parent->refs;
      }
      for (int i = 0; i < n; ++i) {
        target->dependencies[i] = ds[i]->id.load();
        ++ds[i]->refs;
      }
      ++active;
      target->id.store(joinId);
      if (target->epoch.load() != epoch.load()) {
        target->cancel.store(true);
        target->released.store(true);
        return reject(-4);
      }
      return double(joinId);
    }
    case 10: {
      if (current == this || n != 1 || !std::isfinite(a[0]) || a[0] < 0 ||
          a[0] > BufferCells || std::floor(a[0]) != a[0])
        return -3;
      collect();
      for (auto &b : *buffers)
        if (!b.id) {
          const auto id = ++serial;
          b.epoch.store(epoch.load());
          b.size = int(a[0]);
          b.sealed.store(false);
          b.released.store(false);
          b.id.store(id);
          if (b.epoch.load() != epoch.load()) {
            b.released.store(true);
            return -4;
          }
          return double(id);
        }
      return -1;
    }
    case 11:
    case 12:
    case 13:
    case 14: {
      auto *b = n ? buffer(handle(a[0])) : nullptr;
      if (!b || b->epoch.load() != epoch.load() ||
          b->released && current != this)
        return -4;
      if (op == 13) {
        if (current == this)
          return -3;
        b->sealed = true;
        return 1;
      }
      if (op == 14) {
        if (current == this)
          return -3;
        b->released = true;
        collect();
        return 1;
      }
      if (n < 2 || !std::isfinite(a[1]) || a[1] < 0 || a[1] >= b->size ||
          std::floor(a[1]) != a[1])
        return -3;
      if (op == 11) {
        if (current == this || b->sealed || n != 3)
          return -3;
        b->data[int(a[1])] = a[2];
        return 1;
      }
      return b->sealed ? b->data[int(a[1])].load() : -4;
    }
    }
    return -3;
  }
};
} // namespace jsfx_tasks
extern "C" double jsfx_task_submit(DSPJSFX_State *st, jsfx_tasks::Callback fn,
                                   const double *captures, int n,
                                   double dependency, double count, int mode,
                                   double identity) {
  return st && st->taskContext
             ? static_cast<jsfx_tasks::Runtime *>(st->taskContext)
                   ->submit(st, fn, captures, n, dependency, count, mode,
                            identity)
             : -4;
}
extern "C" double jsfx_task_submit_arena(DSPJSFX_State* st, jsfx_tasks::Callback fn,const double* captures,int n,double arena,double dependency) {
  return st && st->taskContext ? static_cast<jsfx_tasks::Runtime*>(st->taskContext)->submit(st,fn,captures,n,dependency,1,0,0,arena) : -4;
}
extern "C" double jsfx_task_api(DSPJSFX_State *st, int op, const double *args,
                                int n) {
  return st && st->taskContext
             ? static_cast<jsfx_tasks::Runtime *>(st->taskContext)
                   ->api(op, args, n, st)
             : -4;
}
#endif
