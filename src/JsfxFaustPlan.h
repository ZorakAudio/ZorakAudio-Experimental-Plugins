// SPDX-License-Identifier: Zlib
#pragma once
// One stage contract for compiled AOT metadata and runtime JIT metadata.
namespace za::jsfx {
struct Binding {int kind=-1,index=0,alias=-1;};
struct Zone {int offset;double initial;Binding source;};
struct Stage {int kind=0,island=-1;bool fused=false;void(*eel)(DSPJSFX_State*)=nullptr;int size=0,inputs=0,outputs=0,audioOutputs=0;void(*classInit)(int)=nullptr;void(*constants)(void*,int)=nullptr;void(*clear)(void*)=nullptr;void(*allocate)(void*)=nullptr;void(*destroy)(void*)=nullptr;void(*compute)(void*,int,double**,double**)=nullptr;const Zone* zones=nullptr;int zoneCount=0;const Binding* signals=nullptr;int signalCount=0;const Binding* exports=nullptr;int exportCount=0;const int* tableSizes=nullptr;int tableCount=0;void(*bindTables)(void**)=nullptr;void(*seedTables)(void**)=nullptr;int captureStage=-1;Binding condition{};void(*bulk)(DSPJSFX_State*,double*,int,int,int,int,double**)=nullptr;};
}
