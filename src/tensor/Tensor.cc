#include "Tensor.h"
#include "TensorTH.h"
#include <array>
#include "dispatch.h"
#include "TH.h"
#ifdef XT_HAS_CUDA
#include "THC.h"
#endif
#undef THTensor

namespace xt {

static THCState* thcstate()
{
  return defaultContext.thcstate().get();
}

int64_t operator "" _i64(unsigned long long int x)
{
  return (int64_t)x;
}

template<> TensorType Tensor::type<uint8_t>()
{
  return kUInt8;
}

template<> TensorType Tensor::type<int8_t>()
{
  return kInt8;
}

template<> TensorType Tensor::type<int16_t>()
{
  return kInt16;
}

template<> TensorType Tensor::type<int32_t>()
{
  return kInt32;
}

template<> TensorType Tensor::type<int64_t>()
{
  return kInt64;
}

template<> TensorType Tensor::type<float>()
{
  return kFloat;
}

template<> TensorType Tensor::type<double>()
{
  return kDouble;
}

Tensor::~Tensor()
{
  clear();
}

template<> void Tensor::value(uint8_t value, TensorDevice device)
{
  resize({}, kUInt8, device);
  if(device == kCPU) {
    THByteTensor_set1d((THByteTensor*)th_tensor_, 0, value);
#ifdef XT_HAS_CUDA
  } else if(device == kGPU) {
    THCudaByteTensor_set1d(thcstate(), (THCudaByteTensor*)th_tensor_, 0, value);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template<> void Tensor::value(int8_t value, TensorDevice device)
{
  resize({}, kInt8, device);
  if(device == kCPU) {
    THCharTensor_set1d((THCharTensor*)th_tensor_, 0, value);
#ifdef XT_HAS_CUDA
  } else if(device == kGPU) {
    THCudaCharTensor_set1d(defaultContext.thcstate().get(), (THCudaCharTensor*)th_tensor_, 0, value);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template<> void Tensor::value(int16_t value, TensorDevice device)
{
  resize({}, kInt16, device);
  if(device == kCPU) {
    THShortTensor_set1d((THShortTensor*)th_tensor_, 0, value);
#ifdef XT_HAS_CUDA
  } else if(device == kGPU) {
    THCudaShortTensor_set1d(defaultContext.thcstate().get(), (THCudaShortTensor*)th_tensor_, 0, value);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template<> void Tensor::value(int32_t value, TensorDevice device)
{
  resize({}, kInt32, device);
  if(device == kCPU) {
    THIntTensor_set1d((THIntTensor*)th_tensor_, 0, value);
#ifdef XT_HAS_CUDA
  } else if(device == kGPU) {
    THCudaIntTensor_set1d(defaultContext.thcstate().get(), (THCudaIntTensor*)th_tensor_, 0, value);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template<> void Tensor::value(int64_t value, TensorDevice device)
{
  resize({}, kInt64, device);
  if(device == kCPU) {
    THLongTensor_set1d((THLongTensor*)th_tensor_, 0, value);
#ifdef XT_HAS_CUDA
  } else if(device == kGPU) {
    THCudaLongTensor_set1d(defaultContext.thcstate().get(), (THCudaLongTensor*)th_tensor_, 0, value);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template<> void Tensor::value(float value, TensorDevice device)
{
  resize({}, kFloat, device);
  if(device == kCPU) {
    THFloatTensor_set1d((THFloatTensor*)th_tensor_, 0, value);
#ifdef XT_HAS_CUDA
  } else if(device == kGPU) {
    THCudaTensor_set1d(defaultContext.thcstate().get(), (THCudaTensor*)th_tensor_, 0, value);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template<> void Tensor::value(double value, TensorDevice device)
{
  resize({}, kDouble, device);
  if(device == kCPU) {
    THDoubleTensor_set1d((THDoubleTensor*)th_tensor_, 0, value);
#ifdef XT_HAS_CUDA
  } else if(device == kGPU) {
    THCudaDoubleTensor_set1d(defaultContext.thcstate().get(), (THCudaDoubleTensor*)th_tensor_, 0, value);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

Tensor& Tensor::tovalue()
{
  if((dim() != 1) || (size(0) != 1)) {
    throw std::invalid_argument("tensor is not a 1-dim 1-size tensor");
  }
  isValue_ = true;
  return *this;
}

template<typename T> Tensor::Tensor(T v)
  : type_(kDouble), device_(kUnknown), isValue_(false), th_tensor_(nullptr)
{
  value(v);
}

template Tensor::Tensor(uint8_t);
template Tensor::Tensor(int8_t);
template Tensor::Tensor(int16_t);
template Tensor::Tensor(int32_t);
template Tensor::Tensor(int64_t);
template Tensor::Tensor(float);
template Tensor::Tensor(double);

Tensor::Tensor(const Tensor& o)
  : type_(kDouble), device_(kUnknown), isValue_(false), th_tensor_(nullptr)
{
  *this = o;
}

Tensor::Tensor(Tensor&& o)
  : type_(o.type_), device_(o.device_), isValue_(o.isValue_), th_tensor_(o.th_tensor_)
{
  o.th_tensor_ = nullptr;
  o.device_ = kUnknown;
}

Tensor& Tensor::operator=(const Tensor& o) &
{
  clear();
  type_ = o.type_;
  device_ = o.device_;
  isValue_ = o.isValue_;
  if(device_ != kUnknown) {
    o.retain();
    th_tensor_ = o.th_tensor_;
  }
  return *this;
}

Tensor& Tensor::operator=(const Tensor& o) &&
{
  copy_(*this, o);
  return *this;
}

Tensor& Tensor::operator=(Tensor&& o) &
{
  clear();
  type_ = o.type_;
  device_ = o.device_;
  isValue_ = o.isValue_;
  th_tensor_ = o.th_tensor_;
  if(device_ != kUnknown) {
    o.th_tensor_ = nullptr;
    o.device_ = kUnknown;
  }
  return *this;
}

Tensor::Tensor()
  : type_(kDouble), device_(kUnknown), isValue_(false), th_tensor_(nullptr)
{
}

Tensor::Tensor(TensorType type, TensorDevice device)
  : type_(kDouble), device_(kUnknown), isValue_(false), th_tensor_(nullptr)
{
  resize(type, device);
}

Tensor::Tensor(const std::vector<int64_t>& sizes, TensorType type, TensorDevice device)
  : type_(kDouble), device_(kUnknown), isValue_(false), th_tensor_(nullptr)
{
  resize(sizes, type, device);
}

Tensor::Tensor(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, TensorType type, TensorDevice device)
  : type_(kDouble), device_(kUnknown), isValue_(false), th_tensor_(nullptr)
{
  resize(sizes, strides, type, device);
}

TensorType Tensor::type() const
{
  return type_;
}

TensorDevice Tensor::device() const
{
  return device_;
}

template<> THByteTensor* Tensor::THTensor<THByteTensor>() const
{
  if(device_ == kCPU) {
    return (THByteTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("uint8_t tensor expected");
  }
}
template<> THCharTensor* Tensor::THTensor<THCharTensor>() const
{
  if(device_ == kCPU) {
    return (THCharTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("int8_t tensor expected");
  }
}
template<> THShortTensor* Tensor::THTensor<THShortTensor>() const
{
  if(device_ == kCPU) {
    return (THShortTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("int16_t tensor expected");
  }
}
template<> THIntTensor* Tensor::THTensor<THIntTensor>() const
{
  if(device_ == kCPU) {
    return (THIntTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("int32_t tensor expected");
  }
}
template<> THLongTensor* Tensor::THTensor<THLongTensor>() const
{
  if(device_ == kCPU) {
    return (THLongTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("int64_t tensor expected");
  }
}
template<> THFloatTensor* Tensor::THTensor<THFloatTensor>() const
{
  if(device_ == kCPU) {
    return (THFloatTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("float tensor expected");
  }
}
template<> THDoubleTensor* Tensor::THTensor<THDoubleTensor>() const
{
  if(device_ == kCPU) {
    return (THDoubleTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("double tensor expected");
  }
}
#ifdef XT_HAS_CUDA
template<> THCudaByteTensor* Tensor::THTensor<THCudaByteTensor>() const
{
  if(device_ == kGPU) {
    return (THCudaByteTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("uint8_t cuda tensor expected");
  }
}
template<> THCudaCharTensor* Tensor::THTensor<THCudaCharTensor>() const
{
  if(device_ == kGPU) {
    return (THCudaCharTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("int8_t cuda tensor expected");
  }
}
template<> THCudaShortTensor* Tensor::THTensor<THCudaShortTensor>() const
{
  if(device_ == kGPU) {
    return (THCudaShortTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("int16_t cuda tensor expected");
  }
}
template<> THCudaIntTensor* Tensor::THTensor<THCudaIntTensor>() const
{
  if(device_ == kGPU) {
    return (THCudaIntTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("int32_t cuda tensor expected");
  }
}
template<> THCudaLongTensor* Tensor::THTensor<THCudaLongTensor>() const
{
  if(device_ == kGPU) {
    return (THCudaLongTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("int64_t cuda tensor expected");
  }
}
template<> THCudaTensor* Tensor::THTensor<THCudaTensor>() const
{
  if(device_ == kGPU) {
    return (THCudaTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("float cuda tensor expected");
  }
}
template<> THCudaDoubleTensor* Tensor::THTensor<THCudaDoubleTensor>() const
{
  if(device_ == kGPU) {
    return (THCudaDoubleTensor*)(th_tensor_);
  } else {
    throw std::invalid_argument("double cuda tensor expected");
  }
}
#endif

int64_t Tensor::dim() const
{
  if(device_ == kUnknown) {
    return -1;
  } else if(isValue_) {
    return 0;
  } else if(device_ == kCPU) {
    static std::array<std::function<int64_t (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) {int64_t dim = THByteTensor_nDimension(t.THTensor<THByteTensor>()); return (dim == 0 ? -1 : dim);},
        [](const Tensor& t) {int64_t dim = THCharTensor_nDimension(t.THTensor<THCharTensor>()); return (dim == 0 ? -1 : dim);},
        [](const Tensor& t) {int64_t dim = THShortTensor_nDimension(t.THTensor<THShortTensor>()); return (dim == 0 ? -1 : dim);},
        [](const Tensor& t) {int64_t dim = THIntTensor_nDimension(t.THTensor<THIntTensor>()); return (dim == 0 ? -1 : dim);},
        [](const Tensor& t) {int64_t dim = THLongTensor_nDimension(t.THTensor<THLongTensor>()); return (dim == 0 ? -1 : dim);},
        [](const Tensor& t) {int64_t dim = THFloatTensor_nDimension(t.THTensor<THFloatTensor>()); return (dim == 0 ? -1 : dim);},
        [](const Tensor& t) {int64_t dim = THDoubleTensor_nDimension(t.THTensor<THDoubleTensor>()); return (dim == 0 ? -1 : dim);}
      }};
    return dyn.at(type_)(*this);
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    static std::array<std::function<int64_t (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) {int64_t dim = THCudaByteTensor_nDimension(thcstate(), t.THTensor<THCudaByteTensor>()); return (dim == 0 ? -1 : dim);},
        [](const Tensor& t) {int64_t dim = THCudaCharTensor_nDimension(thcstate(), t.THTensor<THCudaCharTensor>()); return (dim == 0 ? -1 : dim);},
        [](const Tensor& t) {int64_t dim = THCudaShortTensor_nDimension(thcstate(), t.THTensor<THCudaShortTensor>()); return (dim == 0 ? -1 : dim);},
        [](const Tensor& t) {int64_t dim = THCudaIntTensor_nDimension(thcstate(), t.THTensor<THCudaIntTensor>()); return (dim == 0 ? -1 : dim);},
        [](const Tensor& t) {int64_t dim = THCudaLongTensor_nDimension(thcstate(), t.THTensor<THCudaLongTensor>()); return (dim == 0 ? -1 : dim);},
        [](const Tensor& t) {int64_t dim = THCudaTensor_nDimension(thcstate(), t.THTensor<THCudaTensor>()); return (dim == 0 ? -1 : dim);},
        [](const Tensor& t) {int64_t dim = THCudaDoubleTensor_nDimension(thcstate(), t.THTensor<THCudaDoubleTensor>()); return (dim == 0 ? -1 : dim);}
      }};
    return dyn.at(type_)(*this);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

int64_t Tensor::offset() const
{
  if(device_ == kUnknown) {
    return 0;
  } else if(device_ == kCPU) {
    static std::array<std::function<int64_t (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) {return THByteTensor_storageOffset(t.THTensor<THByteTensor>());},
        [](const Tensor& t) {return THCharTensor_storageOffset(t.THTensor<THCharTensor>());},
        [](const Tensor& t) {return THShortTensor_storageOffset(t.THTensor<THShortTensor>());},
        [](const Tensor& t) {return THIntTensor_storageOffset(t.THTensor<THIntTensor>());},
        [](const Tensor& t) {return THLongTensor_storageOffset(t.THTensor<THLongTensor>());},
        [](const Tensor& t) {return THFloatTensor_storageOffset(t.THTensor<THFloatTensor>());},
        [](const Tensor& t) {return THDoubleTensor_storageOffset(t.THTensor<THDoubleTensor>());}
      }};
    return dyn.at(type_)(*this);
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    static std::array<std::function<int64_t (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) {return THCudaByteTensor_storageOffset(thcstate(), t.THTensor<THCudaByteTensor>());},
        [](const Tensor& t) {return THCudaCharTensor_storageOffset(thcstate(), t.THTensor<THCudaCharTensor>());},
        [](const Tensor& t) {return THCudaShortTensor_storageOffset(thcstate(), t.THTensor<THCudaShortTensor>());},
        [](const Tensor& t) {return THCudaIntTensor_storageOffset(thcstate(), t.THTensor<THCudaIntTensor>());},
        [](const Tensor& t) {return THCudaLongTensor_storageOffset(thcstate(), t.THTensor<THCudaLongTensor>());},
        [](const Tensor& t) {return THCudaTensor_storageOffset(thcstate(), t.THTensor<THCudaTensor>());},
        [](const Tensor& t) {return THCudaDoubleTensor_storageOffset(thcstate(), t.THTensor<THCudaDoubleTensor>());}
      }};
    return dyn.at(type_)(*this);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

int64_t Tensor::size(int64_t idx) const
{
  if((device_ == kUnknown) || idx < 0 || idx >= this->dim() || isValue_) {
    throw std::out_of_range("invalid dimension");
  } else if(device_ == kCPU) {
    static std::array<std::function<int64_t (const Tensor&, int64_t idx)>, 7> dyn = {{
        [](const Tensor& t, int64_t dim) {return THByteTensor_size(t.THTensor<THByteTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCharTensor_size(t.THTensor<THCharTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THShortTensor_size(t.THTensor<THShortTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THIntTensor_size(t.THTensor<THIntTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THLongTensor_size(t.THTensor<THLongTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THFloatTensor_size(t.THTensor<THFloatTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THDoubleTensor_size(t.THTensor<THDoubleTensor>(), dim);}
      }};
    return dyn.at(type_)(*this, idx);
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    static std::array<std::function<int64_t (const Tensor&, int64_t idx)>, 7> dyn = {{
        [](const Tensor& t, int64_t dim) {return THCudaByteTensor_size(thcstate(), t.THTensor<THCudaByteTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCudaCharTensor_size(thcstate(), t.THTensor<THCudaCharTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCudaShortTensor_size(thcstate(), t.THTensor<THCudaShortTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCudaIntTensor_size(thcstate(), t.THTensor<THCudaIntTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCudaLongTensor_size(thcstate(), t.THTensor<THCudaLongTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCudaTensor_size(thcstate(), t.THTensor<THCudaTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCudaDoubleTensor_size(thcstate(), t.THTensor<THCudaDoubleTensor>(), dim);}
      }};
    return dyn.at(type_)(*this, idx);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

template <typename FS, typename FD, typename T> std::vector<int64_t> tpl_size(FS size, FD nDimension, T* t) {
  std::vector<int64_t> dims(nDimension(t));
  for(uint64_t i = 0; i < dims.size(); i++) {
    dims[i] = size(t, i);
  }
  return dims;
}

template <typename FS, typename FD, typename T> std::vector<int64_t> tpl_size_cuda(FS size, FD nDimension, T* t) {
  std::vector<int64_t> dims(nDimension(thcstate(), t));
  for(uint64_t i = 0; i < dims.size(); i++) {
    dims[i] = size(thcstate(), t, i);
  }
  return dims;
}

std::vector<int64_t> Tensor::size() const
{
  if(isValue_) {
    return std::vector<int64_t>{};
  } else if(device_ == kCPU) {
    static std::array<std::function<std::vector<int64_t> (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) { return tpl_size(THByteTensor_size, THByteTensor_nDimension, t.THTensor<THByteTensor>()); },
        [](const Tensor& t) { return tpl_size(THCharTensor_size, THCharTensor_nDimension, t.THTensor<THCharTensor>()); },
        [](const Tensor& t) { return tpl_size(THShortTensor_size, THShortTensor_nDimension, t.THTensor<THShortTensor>()); },
        [](const Tensor& t) { return tpl_size(THIntTensor_size, THIntTensor_nDimension, t.THTensor<THIntTensor>()); },
        [](const Tensor& t) { return tpl_size(THLongTensor_size, THLongTensor_nDimension, t.THTensor<THLongTensor>()); },
        [](const Tensor& t) { return tpl_size(THFloatTensor_size, THFloatTensor_nDimension, t.THTensor<THFloatTensor>()); },
        [](const Tensor& t) { return tpl_size(THDoubleTensor_size, THDoubleTensor_nDimension, t.THTensor<THDoubleTensor>()); },
      }};
    return dyn.at(type_)(*this);
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    static std::array<std::function<std::vector<int64_t> (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) { return tpl_size_cuda(THCudaByteTensor_size, THCudaByteTensor_nDimension, t.THTensor<THCudaByteTensor>()); },
        [](const Tensor& t) { return tpl_size_cuda(THCudaCharTensor_size, THCudaCharTensor_nDimension, t.THTensor<THCudaCharTensor>()); },
        [](const Tensor& t) { return tpl_size_cuda(THCudaShortTensor_size, THCudaShortTensor_nDimension, t.THTensor<THCudaShortTensor>()); },
        [](const Tensor& t) { return tpl_size_cuda(THCudaIntTensor_size, THCudaIntTensor_nDimension, t.THTensor<THCudaIntTensor>()); },
        [](const Tensor& t) { return tpl_size_cuda(THCudaLongTensor_size, THCudaLongTensor_nDimension, t.THTensor<THCudaLongTensor>()); },
        [](const Tensor& t) { return tpl_size_cuda(THCudaTensor_size, THCudaTensor_nDimension, t.THTensor<THCudaTensor>()); },
        [](const Tensor& t) { return tpl_size_cuda(THCudaDoubleTensor_size, THCudaDoubleTensor_nDimension, t.THTensor<THCudaDoubleTensor>()); },
      }};
    return dyn.at(type_)(*this);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

template <typename FS, typename FD, typename T> std::vector<int64_t> tpl_stride(FS stride, FD nDimension, T* t) {
  std::vector<int64_t> dims(nDimension(t));
  for(uint64_t i = 0; i < dims.size(); i++) {
    dims[i] = stride(t, i);
  }
  return dims;
}

template <typename FS, typename FD, typename T> std::vector<int64_t> tpl_stride_cuda(FS stride, FD nDimension, T* t) {
  std::vector<int64_t> dims(nDimension(thcstate(), t));
  for(uint64_t i = 0; i < dims.size(); i++) {
    dims[i] = stride(thcstate(), t, i);
  }
  return dims;
}

std::vector<int64_t> Tensor::stride() const
{
  if(isValue_) {
    return std::vector<int64_t>{};
  } else if(device_ == kCPU) {
    static std::array<std::function<std::vector<int64_t> (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) { return tpl_stride(THByteTensor_stride, THByteTensor_nDimension, t.THTensor<THByteTensor>()); },
        [](const Tensor& t) { return tpl_stride(THCharTensor_stride, THCharTensor_nDimension, t.THTensor<THCharTensor>()); },
        [](const Tensor& t) { return tpl_stride(THShortTensor_stride, THShortTensor_nDimension, t.THTensor<THShortTensor>()); },
        [](const Tensor& t) { return tpl_stride(THIntTensor_stride, THIntTensor_nDimension, t.THTensor<THIntTensor>()); },
        [](const Tensor& t) { return tpl_stride(THLongTensor_stride, THLongTensor_nDimension, t.THTensor<THLongTensor>()); },
        [](const Tensor& t) { return tpl_stride(THFloatTensor_stride, THFloatTensor_nDimension, t.THTensor<THFloatTensor>()); },
        [](const Tensor& t) { return tpl_stride(THDoubleTensor_stride, THDoubleTensor_nDimension, t.THTensor<THDoubleTensor>()); },
      }};
    return dyn.at(type_)(*this);
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    static std::array<std::function<std::vector<int64_t> (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) { return tpl_stride_cuda(THCudaByteTensor_stride, THCudaByteTensor_nDimension, t.THTensor<THCudaByteTensor>()); },
        [](const Tensor& t) { return tpl_stride_cuda(THCudaCharTensor_stride, THCudaCharTensor_nDimension, t.THTensor<THCudaCharTensor>()); },
        [](const Tensor& t) { return tpl_stride_cuda(THCudaShortTensor_stride, THCudaShortTensor_nDimension, t.THTensor<THCudaShortTensor>()); },
        [](const Tensor& t) { return tpl_stride_cuda(THCudaIntTensor_stride, THCudaIntTensor_nDimension, t.THTensor<THCudaIntTensor>()); },
        [](const Tensor& t) { return tpl_stride_cuda(THCudaLongTensor_stride, THCudaLongTensor_nDimension, t.THTensor<THCudaLongTensor>()); },
        [](const Tensor& t) { return tpl_stride_cuda(THCudaTensor_stride, THCudaTensor_nDimension, t.THTensor<THCudaTensor>()); },
        [](const Tensor& t) { return tpl_stride_cuda(THCudaDoubleTensor_stride, THCudaDoubleTensor_nDimension, t.THTensor<THCudaDoubleTensor>()); },
      }};
    return dyn.at(type_)(*this);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

int64_t Tensor::stride(int64_t idx) const
{
  if((device_ == kUnknown) || idx < 0 || idx >= this->dim() || isValue_) {
    throw std::out_of_range("invalid dimension");
  } else if(device_ == kCPU) {
    static std::array<std::function<int64_t (const Tensor&, int64_t idx)>, 7> dyn = {{
        [](const Tensor& t, int64_t dim) {return THByteTensor_stride(t.THTensor<THByteTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCharTensor_stride(t.THTensor<THCharTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THShortTensor_stride(t.THTensor<THShortTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THIntTensor_stride(t.THTensor<THIntTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THLongTensor_stride(t.THTensor<THLongTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THFloatTensor_stride(t.THTensor<THFloatTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THDoubleTensor_stride(t.THTensor<THDoubleTensor>(), dim);}
      }};
    return dyn.at(type_)(*this, idx);
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    static std::array<std::function<int64_t (const Tensor&, int64_t idx)>, 7> dyn = {{
        [](const Tensor& t, int64_t dim) {return THCudaByteTensor_stride(thcstate(), t.THTensor<THCudaByteTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCudaCharTensor_stride(thcstate(), t.THTensor<THCudaCharTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCudaShortTensor_stride(thcstate(), t.THTensor<THCudaShortTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCudaIntTensor_stride(thcstate(), t.THTensor<THCudaIntTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCudaLongTensor_stride(thcstate(), t.THTensor<THCudaLongTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCudaTensor_stride(thcstate(), t.THTensor<THCudaTensor>(), dim);},
        [](const Tensor& t, int64_t dim) {return THCudaDoubleTensor_stride(thcstate(), t.THTensor<THCudaDoubleTensor>(), dim);}
      }};
    return dyn.at(type_)(*this, idx);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

int64_t Tensor::elemSize() const
{
  static std::array<std::function<int64_t ()>, 7> dyn = {{
      []() {return sizeof(uint8_t);},
      []() {return sizeof(int8_t);},
      []() {return sizeof(int16_t);},
      []() {return sizeof(int32_t);},
      []() {return sizeof(uint64_t);},
      []() {return sizeof(float);},
      []() {return sizeof(double);}
    }};
  return dyn.at(type_)();
}

Tensor& Tensor::resizeAs(const Tensor &o, bool wtype)
{
  resize(o.size(), (wtype ? o.type() : type_), (wtype ? o.device() : device_));
  return *this;
}

Tensor& Tensor::clear()
{
  if((device_ != kUnknown) && th_tensor_) {
    release();
  }
  type_ = kDouble;
  device_ = kUnknown;
  isValue_ = false;
  th_tensor_ = nullptr;
  return *this;
}

Tensor& Tensor::empty()
{
  auto type = type_;
  auto device = device_;
  clear();
  resize(type, device);
  return *this;
}

Tensor& Tensor::resize(TensorType type, TensorDevice device)
{
  clear();
  if(device == kCPU) {
    static std::array<std::function<void (Tensor&)>, 7> dyn = {{
        [](Tensor& t) {t.th_tensor_ = THByteTensor_new();},
        [](Tensor& t) {t.th_tensor_ = THCharTensor_new();},
        [](Tensor& t) {t.th_tensor_ = THShortTensor_new();},
        [](Tensor& t) {t.th_tensor_ = THIntTensor_new();},
        [](Tensor& t) {t.th_tensor_ = THLongTensor_new();},
        [](Tensor& t) {t.th_tensor_ = THFloatTensor_new();},
        [](Tensor& t) {t.th_tensor_ = THDoubleTensor_new();}
      }};
    dyn.at(type)(*this);
#ifdef XT_HAS_CUDA
  } else if(device == kGPU) {
    static std::array<std::function<void (Tensor&)>, 7> dyn = {{
        [](Tensor& t) {t.th_tensor_ = THCudaByteTensor_new(thcstate());},
        [](Tensor& t) {t.th_tensor_ = THCudaCharTensor_new(thcstate());},
        [](Tensor& t) {t.th_tensor_ = THCudaShortTensor_new(thcstate());},
        [](Tensor& t) {t.th_tensor_ = THCudaIntTensor_new(thcstate());},
        [](Tensor& t) {t.th_tensor_ = THCudaLongTensor_new(thcstate());},
        [](Tensor& t) {t.th_tensor_ = THCudaTensor_new(thcstate());},
        [](Tensor& t) {t.th_tensor_ = THCudaDoubleTensor_new(thcstate());}
      }};
    dyn.at(type)(*this);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
  type_ = type;
  device_ = device;
  return *this;
}

Tensor& Tensor::resize(const std::vector<int64_t>& sizes)
{
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for(int64_t i = sizes.size()-1; i >= 0; i--) {
    if(sizes[i] <= 0) {
      throw std::invalid_argument("sizes must be positive numbers");
    }
    strides[i] = stride;
    stride *= sizes[i];
  }
  resize(sizes, strides);
  return *this;
}

Tensor& Tensor::resize(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides)
{
  if(sizes.size() != strides.size()) {
    throw std::invalid_argument("sizes and strides size mismatch");
  }
  int64_t dim = sizes.size();
  isValue_ = (dim == 0 ? true : false);
  auto sizes_s = std::shared_ptr<THLongStorage>(THLongStorage_newWithSize(isValue_ ? 1 : dim), THLongStorage_free);
  auto strides_s = std::shared_ptr<THLongStorage>(THLongStorage_newWithSize(isValue_ ? 1 : dim), THLongStorage_free);
  if(isValue_) {
    sizes_s->data[0] = 1;
    strides_s->data[0] = 1;
  } else {
    for(uint64_t i = 0; i < sizes.size(); i++) {
      sizes_s->data[i] = sizes[i];
    }
    for(uint64_t i = 0; i < strides.size(); i++) {
      strides_s->data[i] = strides[i];
    }
  }
  if(device_ == kCPU) {
    static std::array<std::function<void (Tensor&, THLongStorage*, THLongStorage*)>, 7> dyn = {{
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THByteTensor_resize(t.THTensor<THByteTensor>(), sizes, strides);},
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THCharTensor_resize(t.THTensor<THCharTensor>(), sizes, strides);},
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THShortTensor_resize(t.THTensor<THShortTensor>(), sizes, strides);},
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THIntTensor_resize(t.THTensor<THIntTensor>(), sizes, strides);},
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THLongTensor_resize(t.THTensor<THLongTensor>(), sizes, strides);},
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THFloatTensor_resize(t.THTensor<THFloatTensor>(), sizes, strides);},
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THDoubleTensor_resize(t.THTensor<THDoubleTensor>(), sizes, strides);}
      }};
    dyn.at(type_)(*this, sizes_s.get(), strides_s.get());
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    static std::array<std::function<void (Tensor&, THLongStorage*, THLongStorage*)>, 7> dyn = {{
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THCudaByteTensor_resize(thcstate(), t.THTensor<THCudaByteTensor>(), sizes, strides);},
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THCudaCharTensor_resize(thcstate(), t.THTensor<THCudaCharTensor>(), sizes, strides);},
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THCudaShortTensor_resize(thcstate(), t.THTensor<THCudaShortTensor>(), sizes, strides);},
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THCudaIntTensor_resize(thcstate(), t.THTensor<THCudaIntTensor>(), sizes, strides);},
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THCudaLongTensor_resize(thcstate(), t.THTensor<THCudaLongTensor>(), sizes, strides);},
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THCudaTensor_resize(thcstate(), t.THTensor<THCudaTensor>(), sizes, strides);},
        [](Tensor& t, THLongStorage *sizes, THLongStorage *strides) {THCudaDoubleTensor_resize(thcstate(), t.THTensor<THCudaDoubleTensor>(), sizes, strides);}
      }};
    dyn.at(type_)(*this, sizes_s.get(), strides_s.get());
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
  return *this;
}

Tensor& Tensor::resize(const std::vector<int64_t>& sizes, TensorType type, TensorDevice device)
{
  resize(type, device);
  resize(sizes);
  return *this;
}

Tensor& Tensor::resize(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, TensorType type, TensorDevice device)
{
  resize(type, device);
  resize(sizes, strides);
  return *this;
}

template<> uint8_t* Tensor::data() const
{
  if(type_ != kUInt8) {
    throw std::invalid_argument("uint8_t tensor expected");
  }
  if(device_ == kCPU) {
    return THByteTensor_data(THTensor<THByteTensor>());
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    return THCudaByteTensor_data(thcstate(), THTensor<THCudaByteTensor>());
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template<> int8_t* Tensor::data() const
{
  if(type_ != kInt8) {
    throw std::invalid_argument("int8_t tensor expected");
  }
  if(device_ == kCPU) {
    return (int8_t*)THCharTensor_data(THTensor<THCharTensor>());
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    return (int8_t*)THCudaCharTensor_data(thcstate(), THTensor<THCudaCharTensor>());
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template<> int16_t* Tensor::data() const
{
  if(type_ != kInt16) {
    throw std::invalid_argument("int16_t tensor expected");
  }
  if(device_ == kCPU) {
    return THShortTensor_data(THTensor<THShortTensor>());
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    return THCudaShortTensor_data(thcstate(), THTensor<THCudaShortTensor>());
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template<> int32_t* Tensor::data() const
{
  if(type_ != kInt32) {
    throw std::invalid_argument("int32_t tensor expected");
  }
  if(device_ == kCPU) {
    return THIntTensor_data(THTensor<THIntTensor>());
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    return THCudaIntTensor_data(thcstate(), THTensor<THCudaIntTensor>());
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template<> int64_t* Tensor::data() const
{
  if(type_ != kInt64) {
    throw std::invalid_argument("int64_t tensor expected");
  }
  if(device_ == kCPU) {
    return (int64_t*)THLongTensor_data(THTensor<THLongTensor>());
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    return (int64_t*)THCudaLongTensor_data(thcstate(), THTensor<THCudaLongTensor>());
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template<> float* Tensor::data() const
{
  if(type_ != kFloat) {
    throw std::invalid_argument("float tensor expected");
  }
  if(device_ == kCPU) {
    return THFloatTensor_data(THTensor<THFloatTensor>());
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    return THCudaTensor_data(thcstate(), THTensor<THCudaTensor>());
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template<> double* Tensor::data() const
{
  if(type_ != kDouble) {
    throw std::invalid_argument("double tensor expected");
  }
  if(device_ == kCPU) {
    return THDoubleTensor_data(THTensor<THDoubleTensor>());
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    return THCudaDoubleTensor_data(thcstate(), THTensor<THCudaDoubleTensor>());
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

template<typename T> T Tensor::value() const
{
  if(!isValue_) {
    throw std::invalid_argument("value expected");
  }
  if(device_ == kCPU) {
    static std::array<std::function<T (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) {return (T)THByteTensor_get1d(t.THTensor<THByteTensor>(), 0); },
        [](const Tensor& t) {return (T)THCharTensor_get1d(t.THTensor<THCharTensor>(), 0); },
        [](const Tensor& t) {return (T)THShortTensor_get1d(t.THTensor<THShortTensor>(), 0); },
        [](const Tensor& t) {return (T)THIntTensor_get1d(t.THTensor<THIntTensor>(), 0); },
        [](const Tensor& t) {return (T)THLongTensor_get1d(t.THTensor<THLongTensor>(), 0); },
        [](const Tensor& t) {return (T)THFloatTensor_get1d(t.THTensor<THFloatTensor>(), 0); },
        [](const Tensor& t) {return (T)THDoubleTensor_get1d(t.THTensor<THDoubleTensor>(), 0); }
      }};
    return dyn.at(type_)(*this);
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    static std::array<std::function<T (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) {return (T)THCudaByteTensor_get1d(thcstate(), t.THTensor<THCudaByteTensor>(), 0); },
        [](const Tensor& t) {return (T)THCudaCharTensor_get1d(thcstate(), t.THTensor<THCudaCharTensor>(), 0); },
        [](const Tensor& t) {return (T)THCudaShortTensor_get1d(thcstate(), t.THTensor<THCudaShortTensor>(), 0); },
        [](const Tensor& t) {return (T)THCudaIntTensor_get1d(thcstate(), t.THTensor<THCudaIntTensor>(), 0); },
        [](const Tensor& t) {return (T)THCudaLongTensor_get1d(thcstate(), t.THTensor<THCudaLongTensor>(), 0); },
        [](const Tensor& t) {return (T)THCudaTensor_get1d(thcstate(), t.THTensor<THCudaTensor>(), 0); },
        [](const Tensor& t) {return (T)THCudaDoubleTensor_get1d(thcstate(), t.THTensor<THCudaDoubleTensor>(), 0); }
      }};
    return dyn.at(type_)(*this);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}
template uint8_t Tensor::value() const;
template int8_t Tensor::value() const;
template int16_t Tensor::value() const;
template int32_t Tensor::value() const;
template int64_t Tensor::value() const;
template float Tensor::value() const;
template double Tensor::value() const;

void Tensor::retain() const
{
  if(device_ == kCPU) {
    static std::array<std::function<void (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) {THByteTensor_retain((THByteTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCharTensor_retain((THCharTensor*)t.th_tensor_);},
        [](const Tensor& t) {THShortTensor_retain((THShortTensor*)t.th_tensor_);},
        [](const Tensor& t) {THIntTensor_retain((THIntTensor*)t.th_tensor_);},
        [](const Tensor& t) {THLongTensor_retain((THLongTensor*)t.th_tensor_);},
        [](const Tensor& t) {THFloatTensor_retain((THFloatTensor*)t.th_tensor_);},
      [](const Tensor& t) {THDoubleTensor_retain((THDoubleTensor*)t.th_tensor_);}
      }};
    dyn.at(type_)(*this);
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    static std::array<std::function<void (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) {THCudaByteTensor_retain(thcstate(), (THCudaByteTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCudaCharTensor_retain(thcstate(), (THCudaCharTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCudaShortTensor_retain(thcstate(), (THCudaShortTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCudaIntTensor_retain(thcstate(), (THCudaIntTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCudaLongTensor_retain(thcstate(), (THCudaLongTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCudaTensor_retain(thcstate(), (THCudaTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCudaDoubleTensor_retain(thcstate(), (THCudaDoubleTensor*)t.th_tensor_);}
      }};
    dyn.at(type_)(*this);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

void Tensor::release() const
{
  if(device_ == kCPU) {
    static std::array<std::function<void (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) {THByteTensor_free((THByteTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCharTensor_free((THCharTensor*)t.th_tensor_);},
        [](const Tensor& t) {THShortTensor_free((THShortTensor*)t.th_tensor_);},
        [](const Tensor& t) {THIntTensor_free((THIntTensor*)t.th_tensor_);},
        [](const Tensor& t) {THLongTensor_free((THLongTensor*)t.th_tensor_);},
        [](const Tensor& t) {THFloatTensor_free((THFloatTensor*)t.th_tensor_);},
      [](const Tensor& t) {THDoubleTensor_free((THDoubleTensor*)t.th_tensor_);}
      }};
    dyn.at(type_)(*this);
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    static std::array<std::function<void (const Tensor&)>, 7> dyn = {{
        [](const Tensor& t) {THCudaByteTensor_free(thcstate(), (THCudaByteTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCudaCharTensor_free(thcstate(), (THCudaCharTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCudaShortTensor_free(thcstate(), (THCudaShortTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCudaIntTensor_free(thcstate(), (THCudaIntTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCudaLongTensor_free(thcstate(), (THCudaLongTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCudaTensor_free(thcstate(), (THCudaTensor*)t.th_tensor_);},
        [](const Tensor& t) {THCudaDoubleTensor_free(thcstate(), (THCudaDoubleTensor*)t.th_tensor_);}
      }};
    dyn.at(type_)(*this);
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

template<typename T> Tensor Tensor::cast(TensorDevice device) const
{
  Tensor tmp(type<T>(), (device == kUnknown ? device_ : device));
  tmp.resizeAs(*this);
  copy_(tmp, *this);
  return tmp;
}

template Tensor Tensor::cast<uint8_t>(TensorDevice) const;
template Tensor Tensor::cast<int8_t>(TensorDevice) const;
template Tensor Tensor::cast<int16_t>(TensorDevice) const;
template Tensor Tensor::cast<int32_t>(TensorDevice) const;
template Tensor Tensor::cast<int64_t>(TensorDevice) const;
template Tensor Tensor::cast<float>(TensorDevice) const;
template Tensor Tensor::cast<double>(TensorDevice) const;

template<> std::string Tensor::typedesc<uint8_t>()
{
  return "uint8";
}
template<> std::string Tensor::typedesc<int8_t>()
{
  return "int8";
}
template<> std::string Tensor::typedesc<int16_t>()
{
  return "int16";
}
template<> std::string Tensor::typedesc<int32_t>()
{
  return "int32";
}
template<> std::string Tensor::typedesc<int64_t>()
{
  return "int64";
}
template<> std::string Tensor::typedesc<float>()
{
  return "float";
}
template<> std::string Tensor::typedesc<double>()
{
  return "double";
}

std::string Tensor::typedesc() const
{
  static std::array<std::function<std::string ()>, 7> dyn = {{
      &Tensor::typedesc<uint8_t>,
      &Tensor::typedesc<int8_t>,
      &Tensor::typedesc<int16_t>,
      &Tensor::typedesc<int32_t>,
      &Tensor::typedesc<int64_t>,
      &Tensor::typedesc<float>,
      &Tensor::typedesc<double>
    }};
  if(device_ == kUnknown) {
    return "unknown";
  } else if(device_ == kCPU || device_ == kGPU) {
    return dyn.at(type_)();
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

std::string Tensor::devicedesc() const
{
  if(device_ == kUnknown) {
    return "unknown";
  } else if(device_ == kCPU) {
    return "cpu";
#ifdef XT_HAS_CUDA
  } else if(device_ == kGPU) {
    return "gpu";
#endif
  } else {
    throw std::invalid_argument("unsupported device");
  }
}

} // namespace xt
