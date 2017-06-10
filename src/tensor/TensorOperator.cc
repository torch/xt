#include "Tensor.h"
#include "TensorTH.h"
#include "dispatch.h"

namespace xt {

Tensor& Tensor::operator+=(const Tensor& rhs)
{
  add_(*this, *this, rhs);
  return *this;
}

Tensor& Tensor::operator/=(const Tensor& rhs)
{
  if(rhs.dim() == 0)
    div_(*this, *this, rhs);
  else
    cdiv_(*this, *this, rhs);
  return *this;
}

Tensor Tensor::operator[](int64_t d)
{
  return select(*this, 0, d);
}

bool operator!=(const Tensor& lhs, const int64_t rhs)
{
  return !(lhs == rhs);
}

bool operator!=(const Tensor& lhs, const double rhs)
{
  return !(lhs == rhs);
}

bool operator==(const Tensor& lhs, const Tensor& rhs)
{
  return equal(lhs, rhs);
}

bool operator!=(const Tensor& lhs, const Tensor& rhs)
{
  return !(lhs == rhs);
}

Tensor Tensor::operator[](const Tensor &rhs)
{
  return (*this)[rhs.value<int64_t>()];
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs)
{
  return add(lhs, rhs);
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs)
{
  return add(lhs, -1, rhs);
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs)
{
  if(lhs.dim() == 0 || rhs.dim() == 0) {
    return mul(lhs, rhs);
  } else if(lhs.dim() == 2 && rhs.dim() == 2) {
    return mm(lhs, rhs);
  } else if(lhs.dim() == 2 && rhs.dim() == 1) {
    return mv(lhs, rhs);
  } else {
    throw std::invalid_argument("mul: unsupported dimensions");
  }
}

Tensor& Tensor::operator++(int)
{
  add_(*this, *this, Tensor(1));
  return *this;
}

Tensor operator/(const Tensor& lhs, const Tensor& rhs)
{
  return div(lhs, rhs);
}

std::ostream& operator<<(std::ostream& stream, const Tensor& self)
{
  return self.print(stream);
}

}
