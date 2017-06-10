#ifndef XT_TENSOR_H
#define XT_TENSOR_H

#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>

// for now, we rely on TH for mem alloc
#include "Context.h"

namespace xt {

int64_t operator "" _i64(unsigned long long int x);

enum TensorDevice {
  kUnknown,
  kCPU,
  kGPU,
};

enum TensorType {
  kUInt8,
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kFloat,
  kDouble
};

class Tensor {
public:
  Tensor(); /* not allocated, no type */
  Tensor(const Tensor& o); // shallow copy
  Tensor(Tensor&& o);
  Tensor& operator=(const Tensor& o) &; // as copy constructor (shallow copy)
  Tensor& operator=(const Tensor& o) &&; // (deep copy)
  Tensor& operator=(Tensor&& o) &;
  Tensor(TensorType type, TensorDevice device = kCPU); /* TH struct allocated, not the data */
  Tensor(const std::vector<int64_t>& sizes, TensorType type, TensorDevice device = kCPU); /* full allocated */
  Tensor(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, TensorType type, TensorDevice device = kCPU); /* full allocated */
  template<typename T> Tensor(T value); /* creates a 0-dim tensor with given value */
  //  Tensor(Tensor &o, int64_t offset, std::vector<int64_t> sizes, std::vector<int64_t> strides); /* view */
  int64_t dim() const;
  int64_t offset() const; /* no notion of storage */
  int64_t size(int64_t dim) const;
  std::vector<int64_t> size() const;
  int64_t stride(int64_t dim) const;
  std::vector<int64_t> stride() const;
  int64_t elemSize() const;
  template<typename T> void value(T value, TensorDevice device=kCPU); // resize to value (0-dim tensor)
  Tensor& tovalue(); // convert a 1-dim tensor of size 1 into a value
  Tensor& resize(TensorType type, TensorDevice device = kCPU); // empty (kGPU or kCPU)
  Tensor& resize(const std::vector<int64_t>& sizes);
  Tensor& resize(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides);
  Tensor& resize(const std::vector<int64_t>& sizes, TensorType type, TensorDevice device = kCPU);
  Tensor& resize(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, TensorType type, TensorDevice device = kCPU);
  Tensor& resizeAs(const Tensor& o, bool wtype=false); // wtype = true: use same type/device than o
  TensorDevice device() const;
  TensorType type() const;
  std::string typedesc() const;
  std::string devicedesc() const;
  Tensor& empty(); /* keep type */
  Tensor& clear(); /* clear everything (unknown tensor) */
  template<typename T> static TensorType type();
  template<typename T> static std::string typedesc();
  template<typename T> T* data() const;
  template<typename T> T* data(std::vector<int64_t> dims) const;
  template<typename T> T value() const; /* only for 0-dim tensors */
  template<typename T> Tensor cast(TensorDevice device=kUnknown) const; // by default use same device than source
  std::ostream& print(std::ostream& stream, int64_t linesize=80) const;

  // operators
  Tensor operator[](int64_t dim);
  Tensor operator[](const Tensor &rhs); // rhs must be a value
  Tensor& operator++(int);
  Tensor& operator+=(const Tensor& rhs);
  Tensor& operator/=(const Tensor& rhs);
  friend bool operator==(const Tensor& lhs, const Tensor& rhs);
  friend bool operator!=(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator*(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator/(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator-(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator+(const Tensor& lhs, const Tensor& rhs);
  friend std::ostream& operator<<(std::ostream& stream, const Tensor& self);

  // return a view from a THTensor
  // assume the view is valid while the tensor is alive
  template<typename T> T* THTensor() const;
  ~Tensor();

private:
  void retain() const;
  void release() const;
  TensorType type_;
  TensorDevice device_;

  /* do not rely on this */
  /* highly subject to change */
  /* for now we rely on TH */
  bool isValue_;
  void* th_tensor_;
};

} // namespace xt

#endif
