#ifndef XT_DISPATCH_H
#define XT_DISPATCH_H

#include "Tensor.h"
#include <array>
#include <utility>

namespace xt {

// Tensor version
template<typename F, typename TensorT, typename... Args>
auto dispatch(TensorT&& t, Args&&... args) -> typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, TensorT&&, Args&&...)>::type
{
  using ReturnT = typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, TensorT&&, Args&&...)>::type;
  using FunctionT = std::function<ReturnT (F&, TensorT&&, Args&&...)>;
  F functor;

  if(t.device() == kCPU) {
    static std::array<FunctionT, 7> dyn = {{
        &F::template cpu<uint8_t>,
        &F::template cpu<int8_t>,
        &F::template cpu<int16_t>,
        &F::template cpu<int32_t>,
        &F::template cpu<int64_t>,
        &F::template cpu<float>,
        &F::template cpu<double>,
      }};
    return dyn.at(t.type())(functor, std::forward<TensorT>(t), std::forward<Args>(args)...);
  } else if(t.device() == kGPU) {
    static std::array<FunctionT, 7> dyn = {{
        &F::template gpu<uint8_t>,
        &F::template gpu<int8_t>,
        &F::template gpu<int16_t>,
        &F::template gpu<int32_t>,
        &F::template gpu<int64_t>,
        &F::template gpu<float>,
        &F::template gpu<double>,
      }};
    return dyn.at(t.type())(functor, std::forward<TensorT>(t), std::forward<Args>(args)...);
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

// Context, Tensor version
template<typename F, typename TensorT, typename... Args>
auto dispatch(Context& ctx, TensorT&& t, Args&&... args) -> typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, Context&, TensorT&&, Args&&...)>::type
{
  using ReturnT = typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, Context&, TensorT&&, Args&&...)>::type;
  using FunctionT = std::function<ReturnT (F&, Context&, TensorT&&, Args&&...)>;
  F functor;

  if(t.device() == kCPU) {
    static std::array<FunctionT, 7> dyn = {{
        &F::template cpu<uint8_t>,
        &F::template cpu<int8_t>,
        &F::template cpu<int16_t>,
        &F::template cpu<int32_t>,
        &F::template cpu<int64_t>,
        &F::template cpu<float>,
        &F::template cpu<double>,
      }};
    return dyn.at(t.type())(functor, ctx, std::forward<TensorT>(t), std::forward<Args>(args)...);
  } else if(t.device() == kGPU) {
    static std::array<FunctionT, 7> dyn = {{
        &F::template gpu<uint8_t>,
        &F::template gpu<int8_t>,
        &F::template gpu<int16_t>,
        &F::template gpu<int32_t>,
        &F::template gpu<int64_t>,
        &F::template gpu<float>,
        &F::template gpu<double>,
      }};
    return dyn.at(t.type())(functor, ctx, std::forward<TensorT>(t), std::forward<Args>(args)...);
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

// type/device version
template<typename F, typename... Args>
auto dispatch(TensorType ttype, TensorDevice tdev, Args&&... args) -> typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, Args&&...)>::type
{
  using ReturnT = typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, Args&&...)>::type;
  using FunctionT = std::function<ReturnT (F&, Args&&...)>;
  F functor;

  if(tdev == kCPU) {
    static std::array<FunctionT, 7> dyn = {{
        &F::template cpu<uint8_t>,
        &F::template cpu<int8_t>,
        &F::template cpu<int16_t>,
        &F::template cpu<int32_t>,
        &F::template cpu<int64_t>,
        &F::template cpu<float>,
        &F::template cpu<double>,
      }};
    return dyn.at(ttype)(functor, std::forward<Args>(args)...);
  } else if(tdev == kGPU) {
    static std::array<FunctionT, 7> dyn = {{
        &F::template gpu<uint8_t>,
        &F::template gpu<int8_t>,
        &F::template gpu<int16_t>,
        &F::template gpu<int32_t>,
        &F::template gpu<int64_t>,
        &F::template gpu<float>,
        &F::template gpu<double>,
      }};
    return dyn.at(ttype)(functor, std::forward<Args>(args)...);
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

}

#endif
