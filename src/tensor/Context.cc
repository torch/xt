#include "Context.h"
#include "TH.h"
#ifdef XT_HAS_CUDA
#include "THC.h"
#endif

namespace xt {

thread_local Context defaultContext;

Context::Context()
  : generator_(nullptr), thcstate_(nullptr)
{
}

std::shared_ptr<THGenerator> Context::generator(std::shared_ptr<THGenerator> gen)
{
  if(gen) {
    generator_ = gen;
  }
  if(!generator_) { // init on demand
    generator_ = std::shared_ptr<THGenerator>(THGenerator_new(), THGenerator_free);
  }
  return generator_;
}

std::shared_ptr<THCState> Context::thcstate(std::shared_ptr<THCState> state)
{
  if(state) {
    thcstate_ = state;
  }
#ifdef XT_HAS_CUDA
  if(!thcstate_) { // init on demand
    thcstate_ = std::shared_ptr<THCState>(THCState_alloc(), THCState_free);
    // /* Enable the caching allocator unless THC_CACHING_ALLOCATOR=0 */
    // char* thc_caching_allocator = getenv("THC_CACHING_ALLOCATOR");
    // if (!thc_caching_allocator || strcmp(thc_caching_allocator, "0") != 0) {
    //   THCState_setDeviceAllocator(state, THCCachingAllocator_get());
    //   state->cudaHostAllocator = &THCCachingHostAllocator;
    // }
    THCudaInit(thcstate_.get());
  }
#endif
  return thcstate_;
}

bool Context::hasGPU()
{
#ifdef XT_HAS_CUDA
  return true;
#else
  return false;
#endif
}

Context::~Context()
{
}

}
