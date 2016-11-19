#ifndef VEC_HEADER
#define VEC_HEADER

#include "Helpers.h"

template <typename T>
struct Vec3
{
  T x, y, z;

  __host__ __device__ Vec3<T>& operator=(const Vec3<T>& other)
  {
    this->x = other.x;
    this->y = other.y;
    this->z = other.z;
    return *this;
  }

  __host__ __device__ Vec3<T>& operator+=(const Vec3<T>& other)
  {
    this->x += other.x;
    this->y += other.y;
    this->z += other.z;
    return *this;
  }

  __host__ __device__ Vec3<T>& operator-=(const Vec3<T>& other)
  {
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;
    return *this;
  }

  __host__ __device__ Vec3<T> operator+(Vec3<T> lhs, const Vec3<T>& rhs)
  {
    lhs += rhs;
    return lhs;
  }

  __host__ __device__ Vec3<T> operator-(Vec3<T> lhs, const Vec3<T>& rhs)
  {
    lhs -= rhs;
    return rhs;
  }
};

#endif