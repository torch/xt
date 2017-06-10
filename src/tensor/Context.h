#ifndef XT_CONTEXT_H
#define XT_CONTEXT_H

#include <thread>

struct THGenerator;
struct THCState;

namespace xt {

class Context
{
public:
  Context();
  std::shared_ptr<THGenerator> generator(std::shared_ptr<THGenerator> gen = nullptr);
  std::shared_ptr<THCState> thcstate(std::shared_ptr<THCState> state = nullptr);
  bool hasGPU(); // return true if compiled with GPU support
  ~Context();
private:
  std::shared_ptr<THGenerator> generator_;
  std::shared_ptr<THCState> thcstate_;
};

extern thread_local Context defaultContext;

}

#endif
