# XT: a simple C++11 wrapper for torch TH/THC

The wrapper respects the semantics of Torch (in terms of default
arguments), except minor details due to differences between C++ in Lua in
the way default arguments are handled. The wrapper is a hacked version of
[cwrap](https://github.com/torch/cwrap) which spit out C++11 instead of
Lua/C API.

Tensor types are resolved dynamically, such that the API is generic and
does not include templates.

See the _generated_ `TensorTH.h` file to see the provided API. Excerpt:
```c++
std::tuple<Tensor, Tensor> sort(const Tensor& ccarg3);
enum TensorOrder {kAscend, kDescend};
std::tuple<Tensor, Tensor> sort(const Tensor& ccarg3, TensorOrder ccarg5);
std::tuple<Tensor, Tensor> sort(const Tensor& ccarg3, int64_t ccarg4);
std::tuple<Tensor, Tensor> sort(const Tensor& ccarg3, int64_t ccarg4, TensorOrder ccarg5);
Tensor band(const Tensor& ccarg2, const Tensor& ccarg3);
Tensor bhistc(const Tensor& ccarg2);
Tensor bhistc(const Tensor& ccarg2, int64_t ccarg3);
Tensor bhistc(const Tensor& ccarg2, int64_t ccarg3, double ccarg4, double ccarg5);
Tensor bmm(const Tensor& ccarg5, const Tensor& ccarg6);
Tensor bor(const Tensor& ccarg2, const Tensor& ccarg3);
Tensor bxor(const Tensor& ccarg2, const Tensor& ccarg3);
Tensor cat(const Tensor& ccarg2, const Tensor& ccarg3);
Tensor cat(const Tensor& ccarg2, const Tensor& ccarg3, int64_t ccarg4);
Tensor cband(const Tensor& ccarg2, const Tensor& ccarg3);
Tensor cbor(const Tensor& ccarg2, const Tensor& ccarg3);
Tensor cbxor(const Tensor& ccarg2, const Tensor& ccarg3);
```

Inplace operations are also provided, and suffixed by `_`:
```c++
Void unfold_(Tensor& ccarg1, const Tensor& ccarg2, int64_t ccarg3, int64_t ccarg4, int64_t ccarg5);
void uniform_(Tensor& ccarg1);
void uniform_(Tensor& ccarg1, Context &ccarg2);
void uniform_(Tensor& ccarg1, Context &ccarg2, const Tensor& ccarg3, const Tensor& ccarg4);
void uniform_(Tensor& ccarg1, const Tensor& ccarg3, const Tensor& ccarg4);
void var_(Tensor& ccarg1, const Tensor& ccarg2, int64_t ccarg3);
void var_(Tensor& ccarg1, const Tensor& ccarg2, int64_t ccarg3, bool ccarg4);
void xcorr2_(Tensor& ccarg1, const Tensor& ccarg4, const Tensor& ccarg5);
void xcorr2_(Tensor& ccarg1, const Tensor& ccarg4, const Tensor& ccarg5, const char ccarg8);
void xcorr3_(Tensor& ccarg1, const Tensor& ccarg4, const Tensor& ccarg5);
void xcorr3_(Tensor& ccarg1, const Tensor& ccarg4, const Tensor& ccarg5, const char ccarg9);
void zero_(Tensor& ccarg1);
void zeros_(Tensor& ccarg1, std::vector<int64_t> ccarg2);
void copy_(Tensor& d, const Tensor& s);
```

### Installation

TH/THC are provided (as git subtrees), so the repo is standalone. You will need a C++11 compiler.
```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/where/you/want # specify your dest directory
make install
```

_Note: Lua is included in the source dir, only because generations scripts rely on Lua._


### Example usage

Here is a simple example; again, the syntax follows Torch semantics.

```c++
using namespace xt; // assumed in the following

Tensor d = ones({3, 4}, kFloat);
Tensor r = zeros({3,4}, kFloat);
for(auto i = 0; i < 100000; i++) {
  r = add(r, d);
}
```

Want this running on the GPU?
```c++
using namespace xt; // assumed in the following

Tensor d = ones({3, 4}, kFloat, kGPU);
Tensor r = zeros({3,4}, kFloat, kGPU);
for(auto i = 0; i < 100000; i++) {
  r = add(r, d);
}
```

Operators are supported:
```c++
Tensor a = rand({3, 7}, kFloat, device);
std::cout << a << std::endl; // ostream support
for(auto i = 0; i < 3; i++) {
  for(auto j = 0; j < 7; j++) {
    a[i][j] = a[i][j] + 3.14; // various operators
  }
}
std::cout << a << std::endl;
```

See more in [sample files](src/tensor/test).

### Creating your kernel

It is easy to create new kernels, thanks to the `dispatch<>()` templated function. Example:
```c++
struct sum_op // a simple sum kernel (for CPU only)
{
  template<typename T> Tensor cpu(Tensor& x) // dispatch handles variable arguments for you
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

Tensor a = rand({3, 7}, kFloat);
std::cout << a << std::endl;
std::cout << dispatch<sum_op>(a) << " == " << sum(a) << std::endl;
```
