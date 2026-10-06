#include "DspJsfxSamplePoolStorage.h"
#include <cassert>
#include <iostream>
struct Generation {uint64_t publicationRequestId=0,decodedBytes=1;int value=0;};
int main(){
  using Storage=za::jsfx::SamplePoolStorage<Generation>;
  Storage storage;auto first=std::make_shared<Generation>();first->value=1;
  std::weak_ptr<Generation> old=first;storage.requestStarted(1);assert(storage.publish(first,1));
  std::unique_ptr<Storage::ReaderScope> lease;
  {Storage::ReadBatch batch;Storage::ReaderScope read(storage);assert(read.generation->value==1);
   lease=std::make_unique<Storage::ReaderScope>(storage,true);}
  storage.requestStarted(2);assert(storage.requestedId()!=lease->generation->publicationRequestId);
  auto second=std::make_shared<Generation>();second->value=2;assert(storage.publish(second,2));first.reset();
  storage.reclaimRetired();assert(!old.expired());
  {Storage::ReadBatch batch;Storage::ReaderScope current(storage);assert(current.generation->value==2);
   assert(batch.bindPinned(storage,lease->generation));Storage::ReaderScope pinned(storage);assert(pinned.generation->value==1);}
  lease.reset();storage.reclaimRetired();assert(old.expired());
  std::cout<<"Independent generation lease, pending-request invalidation, binding and reclamation passed\n";
}
