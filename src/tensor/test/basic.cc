#include "xttensor.h"
#include <iostream>
#include <chrono>

using namespace xt;

struct sum_op_ref
{
  template<typename T> Tensor cpu(Tensor& x)
  {
    if(!isContiguous(x)) {
      throw std::invalid_argument("contiguous tensor expected");
    }
    T* x_p = x.data<T>();
    int64_t size = numel(x);
    T sum = 0;
    for(int64_t i = 0; i < size; i++) {
      sum += x_p[i];
    }
    return sum;
  };


  template<typename T> Tensor gpu(Tensor& x)
  {
    throw std::invalid_argument("device not supported");
  };
};

struct sum_op_const_ref
{
  template<typename T> Tensor cpu(const Tensor& x)
  {
    if(!isContiguous(x)) {
      throw std::invalid_argument("contiguous tensor expected");
    }
    T* x_p = x.data<T>();
    int64_t size = numel(x);
    T sum = 0;
    for(int64_t i = 0; i < size; i++) {
      sum += x_p[i];
    }
    return sum;
  };


  template<typename T> Tensor gpu(const Tensor& x)
  {
    throw std::invalid_argument("device not supported");
  };
};

struct sum_op_rvalue_ref
{
  template<typename T> Tensor cpu(Tensor&& x)
  {
    if(!isContiguous(x)) {
      throw std::invalid_argument("contiguous tensor expected");
    }
    T* x_p = x.data<T>();
    int64_t size = numel(x);
    T sum = 0;
    for(int64_t i = 0; i < size; i++) {
      sum += x_p[i];
    }
    return sum;
  };


  template<typename T> Tensor gpu(const Tensor&& x)
  {
    throw std::invalid_argument("device not supported");
  };
};

static void test(TensorDevice device)
{
  {
    std::cout << "resize:" << std::endl;
    Tensor a(kFloat, device);
    a.resize({3,4});
    std::cout << numel(a) << std::endl;
    a.resize({5, 7});
    std::cout << numel(a) << std::endl;
  }

  {
    std::cout << "ones and dot:" << std::endl;
    Tensor b = ones({3, 4}, kFloat, device);
    std::cout << b << std::endl;
    std::cout << numel(b) << std::endl;
    std::cout << dot(b, b) << std::endl;
  }

  {
    std::cout << "rand:" << std::endl;
    for(auto i = 0; i < 10; i++) {
      Tensor a = rand({3,4}, i % 2 == 0 ? kFloat : kDouble, device);
      std::cout << a << std::endl;
    }
  }

  {
    std::cout << "sort:" << std::endl;
    Tensor b = rand({3, 4}, kFloat, device);
    std::cout << b << std::endl;
    auto z = sort(b, 1);
    std::cout << std::get<0>(z) << std::endl;
    std::cout << std::get<1>(z) << std::endl;
  }

  if(device != kGPU)
  {
    std::cout << "randperm:" << std::endl;
    Tensor b = randperm(15, kFloat, device);
    std::cout << b << std::endl;
    Tensor rv, ri;
    std::tie(rv, ri) = sort(b, (int64_t)0);
    std::cout << rv << std::endl;
    std::cout << ri << std::endl;
  }

  {
    std::cout << "context: " << std::hex << (int64_t)&defaultContext << std::endl;
  }

  {
    std::cout << "add:" << std::endl;
    Tensor a = rand({3, 4}, kFloat, device);
    Tensor b = rand({3, 4}, kFloat, device);
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    Tensor c = add(a, add(a, b));
    std::cout << c << std::endl;
    Tensor d(3.f);
    std::cout << d << std::endl;
    std::cout << add(c, d) << std::endl;
  }

  {
    std::cout << "loads of adds:" << std::endl;
    auto begin = std::chrono::high_resolution_clock::now();
    Tensor d = ones({3, 4}, kFloat, device);
    Tensor r = zeros({3,4}, kFloat, device);
    for(auto i = 0; i < 100000; i++) {
      add_(r, r, d);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::dec << "   " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;
    std::cout << "   norm: " << norm(r).value<double>() << std::endl;
  }

  {
    std::cout << "loads of adds (with copy):" << std::endl;
    auto begin = std::chrono::high_resolution_clock::now();
    Tensor d = ones({3, 4}, kFloat, device);
    Tensor r = zeros({3,4}, kFloat, device);
    for(auto i = 0; i < 100000; i++) {
      r = add(r, d);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::dec << "   " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;
    std::cout << "   norm: " << norm(r).value<double>() << std::endl;
  }


  {
    std::cout << "isContiguous:" << std::endl;
    Tensor a = rand({3, 4}, kFloat, device);
    std::cout << isContiguous(a) << std::endl;
  }

  {
    std::cout << "mm:" << std::endl;
    Tensor a = rand({3, 4}, kFloat, device);
    Tensor b = rand({4}, kFloat, device);
    Tensor c = mv(a, b);
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
  }

  {
    std::cout << "squeeze:" << std::endl;
    Tensor a = rand({2, 1}, kFloat, device);
    std::cout << a << std::endl;
    Tensor b = squeeze(a);
    std::cout << b << std::endl;
    a = rand({1}, kFloat, device);
    std::cout << a << std::endl;
    b = squeeze(a);
    std::cout << b << std::endl;
  }

  {
    std::cout << "copy:" << std::endl;
    Tensor a = zeros({4, 3}, kFloat, device);
    std::cout << a << std::endl;
    Tensor e = rand({3, 4}, kDouble, device);
    std::cout << e << std::endl;
    copy_(a, e);
    std::cout << a << std::endl;
  }

  {
    std::cout << "abs(value):" << std::endl;
    std::cout << xt::abs(-3);
  }

  {
    std::cout << "eq (value):" << std::endl;
    Tensor a = Tensor(10.f);
    std::cout << (a == 11_i64) << " -- should be 0" << std::endl;
    std::cout << (a == 10_i64) << " -- should be 1" << std::endl;
    std::cout << (a == 10.) << " -- should be 1" << std::endl;
  }

  {
    std::cout << "adding a value with different type:" << std::endl;
    Tensor a = rand({4, 3}, kFloat, device);
    std::cout << a << std::endl;
    std::cout << add(a, 1) << std::endl;
  }

  {
    std::cout << "select:" << std::endl;
    Tensor a = rand({3, 7}, kFloat, device);
    std::cout << a << std::endl;
    std::cout << select(a, 1, 3) << std::endl;
    std::cout << select(select(a, 1, 3), 0, 2) << std::endl;
  }

  if(device == kCPU)
  {
    std::cout << "manual sum (ref dispatch):" << std::endl;
    Tensor a = rand({3, 7}, kFloat, device);
    std::cout << a << std::endl;
    std::cout << dispatch<sum_op_ref>(a) << " == " << sum(a) << std::endl;
  }

  if(device == kCPU)
  {
    std::cout << "manual sum (const ref dispatch):" << std::endl;
    const Tensor a = rand({3, 7}, kFloat, device);
    std::cout << a << std::endl;
    std::cout << dispatch<sum_op_const_ref>(a) << " == " << sum(a) << std::endl;
  }

  if(device == kCPU)
  {
    std::cout << "manual sum (rvalue ref dispatch):" << std::endl;
    Tensor a = rand({3, 7}, kFloat, device);
    std::cout << a << std::endl;
    std::cout << dispatch<sum_op_rvalue_ref>(std::move(a)) << " == " << sum(a) << std::endl;
  }

  {
    std::cout << "operators:" << std::endl;
    Tensor a = rand({3, 7}, kFloat, device);
    std::cout << a << std::endl;
    for(auto i = 0; i < 3; i++) {
      for(auto j = 0; j < 7; j++) {
        a[i][j] = a[i][j] + 3.14;
      }
    }
    std::cout << a << std::endl;
  }
}

int main()
{
  std::cout << "=========================== CPU ===========================" << std::endl;
  test(kCPU);
  if(defaultContext.hasGPU()) {
    std::cout << "=========================== GPU ===========================" << std::endl;
    test(kGPU);
  }
  return 0;
}
