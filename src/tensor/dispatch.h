#ifndef XT_DISPATCH_H
#define XT_DISPATCH_H

#include "Tensor.h"
#include <array>

namespace xt {

// Tensor version
template<class F, class ... T>
auto dispatch(Tensor& t, T&... args) -> typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, Tensor&, T&...)>::type
{
  using ReturnType = typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, Tensor&, T&...)>::type;
  if(t.device() == kCPU) {
    static std::array<std::function<ReturnType (F&, Tensor&, T&...)>, 7> dyn = {{
        &F::template cpu<uint8_t>,
        &F::template cpu<int8_t>,
        &F::template cpu<int16_t>,
        &F::template cpu<int32_t>,
        &F::template cpu<int64_t>,
        &F::template cpu<float>,
        &F::template cpu<double>,
      }};
    F functor;
    return dyn.at(t.type())(functor, t, args...);
  } else if(t.device() == kGPU) {
    static std::array<std::function<ReturnType (F&, Tensor&, T&...)>, 7> dyn = {{
        &F::template gpu<uint8_t>,
        &F::template gpu<int8_t>,
        &F::template gpu<int16_t>,
        &F::template gpu<int32_t>,
        &F::template gpu<int64_t>,
        &F::template gpu<float>,
        &F::template gpu<double>,
      }};
    F functor;
    return dyn.at(t.type())(functor, t, args...);
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

// Context, Tensor version
template<class F, class ... T>
auto dispatch(Context& ctx, Tensor& t, T&... args) -> typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, Context&, Tensor&, T&...)>::type
{
  using ReturnType = typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, Context&, Tensor&, T&...)>::type;
  if(t.device() == kCPU) {
    static std::array<std::function<ReturnType (F&, Context&, Tensor&, T&...)>, 7> dyn = {{
        &F::template cpu<uint8_t>,
        &F::template cpu<int8_t>,
        &F::template cpu<int16_t>,
        &F::template cpu<int32_t>,
        &F::template cpu<int64_t>,
        &F::template cpu<float>,
        &F::template cpu<double>,
      }};
    F functor;
    return dyn.at(t.type())(functor, t, args...);
  } else if(t.device() == kGPU) {
    static std::array<std::function<ReturnType (F&, Context&, Tensor&, T&...)>, 7> dyn = {{
        &F::template gpu<uint8_t>,
        &F::template gpu<int8_t>,
        &F::template gpu<int16_t>,
        &F::template gpu<int32_t>,
        &F::template gpu<int64_t>,
        &F::template gpu<float>,
        &F::template gpu<double>,
      }};
    F functor;
    return dyn.at(t.type())(functor, ctx, t, args...);
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

// const Tensor version
template<class F, class ... T>
auto dispatch(const Tensor& t, T&... args) -> typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, const Tensor&, T&...)>::type
{
  using ReturnType = typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, const Tensor&, T&...)>::type;
  if(t.device() == kCPU) {
    static std::array<std::function<ReturnType (F&, const Tensor&, T&...)>, 7> dyn = {{
        &F::template cpu<uint8_t>,
        &F::template cpu<int8_t>,
        &F::template cpu<int16_t>,
        &F::template cpu<int32_t>,
        &F::template cpu<int64_t>,
        &F::template cpu<float>,
        &F::template cpu<double>,
      }};
    F functor;
    return dyn.at(t.type())(functor, t, args...);
  } else if(t.device() == kGPU) {
    static std::array<std::function<ReturnType (F&, const Tensor&, T&...)>, 7> dyn = {{
        &F::template gpu<uint8_t>,
        &F::template gpu<int8_t>,
        &F::template gpu<int16_t>,
        &F::template gpu<int32_t>,
        &F::template gpu<int64_t>,
        &F::template gpu<float>,
        &F::template gpu<double>,
      }};
    F functor;
    return dyn.at(t.type())(functor, t, args...);
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

// Context, const Tensor version
template<class F, class ... T>
auto dispatch(Context& ctx, const Tensor& t, T&... args) -> typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, Context&, const Tensor&, T&...)>::type
{
  using ReturnType = typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, Context&, const Tensor&, T&...)>::type;
  if(t.device() == kCPU) {
    static std::array<std::function<ReturnType (F&, Context&, const Tensor&, T&...)>, 7> dyn = {{
        &F::template cpu<uint8_t>,
        &F::template cpu<int8_t>,
        &F::template cpu<int16_t>,
        &F::template cpu<int32_t>,
        &F::template cpu<int64_t>,
        &F::template cpu<float>,
        &F::template cpu<double>,
      }};
    F functor;
    return dyn.at(t.type())(functor, ctx, t, args...);
  } else if(t.device() == kGPU) {
    static std::array<std::function<ReturnType (F&, Context&, const Tensor&, T&...)>, 7> dyn = {{
        &F::template gpu<uint8_t>,
        &F::template gpu<int8_t>,
        &F::template gpu<int16_t>,
        &F::template gpu<int32_t>,
        &F::template gpu<int64_t>,
        &F::template gpu<float>,
        &F::template gpu<double>,
      }};
    F functor;
    return dyn.at(t.type())(functor, ctx, t, args...);
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

// type/device version
template<class F, class ... T>
auto dispatch(TensorType ttype, TensorDevice tdev, T&... args) -> typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, T&...)>::type
{
  using ReturnType = typename std::result_of<decltype(&F::template cpu<int64_t>)(F&, T&...)>::type;
  if(tdev == kCPU) {
    static std::array<std::function<ReturnType (F&, T&...)>, 7> dyn = {{
        &F::template cpu<uint8_t>,
        &F::template cpu<int8_t>,
        &F::template cpu<int16_t>,
        &F::template cpu<int32_t>,
        &F::template cpu<int64_t>,
        &F::template cpu<float>,
        &F::template cpu<double>,
      }};
    F functor;
    return dyn.at(ttype)(functor, args...);
  } else if(tdev == kGPU) {
    static std::array<std::function<ReturnType (F&, T&...)>, 7> dyn = {{
        &F::template gpu<uint8_t>,
        &F::template gpu<int8_t>,
        &F::template gpu<int16_t>,
        &F::template gpu<int32_t>,
        &F::template gpu<int64_t>,
        &F::template gpu<float>,
        &F::template gpu<double>,
      }};
    F functor;
    return dyn.at(ttype)(functor, args...);
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

}

#endif
