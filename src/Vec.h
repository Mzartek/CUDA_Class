#ifndef VEC_HEADER
#define VEC_HEADER

#include "Helpers.h"

struct Vec3
{
  float x, y, z;

  __host__ __device__ Vec3() : x(0), y(0), z(0) {}
  __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

  __host__ __device__ void Normalize()
  {
    float length = sqrtf(x * x + y * y + z * z);
    x /= length;
    y /= length;
    z /= length;
  }

  __host__ __device__ Vec3& operator=(const Vec3& other)
  {
    this->x = other.x;
    this->y = other.y;
    this->z = other.z;
    return *this;
  }

  __host__ __device__ Vec3& operator=(float value)
  {
    this->x = value;
    this->y = value;
    this->z = value;
    return *this;
  }

  __host__ __device__ Vec3& operator+=(const Vec3& other)
  {
    this->x += other.x;
    this->y += other.y;
    this->z += other.z;
    return *this;
  }

  __host__ __device__ Vec3& operator+=(float value)
  {
    this->x += value;
    this->y += value;
    this->z += value;
    return *this;
  }

  __host__ __device__ Vec3& operator-=(const Vec3& other)
  {
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;
    return *this;
  }

  __host__ __device__ Vec3& operator-=(float value)
  {
    this->x -= value;
    this->y -= value;
    this->z -= value;
    return *this;
  }

  __host__ __device__ Vec3& operator*=(const Vec3& other)
  {
    this->x *= other.x;
    this->y *= other.y;
    this->z *= other.z;
    return *this;
  }

  __host__ __device__ Vec3& operator*=(float value)
  {
    this->x *= value;
    this->y *= value;
    this->z *= value;
    return *this;
  }

  __host__ __device__ Vec3& operator/=(const Vec3& other)
  {
    this->x /= other.x;
    this->y /= other.y;
    this->z /= other.z;
    return *this;
  }

  __host__ __device__ Vec3& operator/=(float value)
  {
    this->x /= value;
    this->y /= value;
    this->z /= value;
    return *this;
  }
};

__host__ __device__ inline Vec3 operator+(Vec3 lhs, const Vec3& rhs)
{
  lhs += rhs;
  return lhs;
}

__host__ __device__ inline Vec3 operator+(Vec3 lhs, float value)
{
  lhs += value;
  return lhs;
}

__host__ __device__ inline Vec3 operator-(Vec3 lhs, const Vec3& rhs)
{
  lhs -= rhs;
  return rhs;
}

__host__ __device__ inline Vec3 operator-(Vec3 lhs, float value)
{
  lhs -= value;
  return lhs;
}

__host__ __device__ inline Vec3 operator*(Vec3 lhs, const Vec3& rhs)
{
  lhs *= rhs;
  return rhs;
}

__host__ __device__ inline Vec3 operator*(Vec3 lhs, float value)
{
  lhs *= value;
  return lhs;
}

__host__ __device__ inline Vec3 operator/(Vec3 lhs, const Vec3& rhs)
{
  lhs /= rhs;
  return rhs;
}

__host__ __device__ inline Vec3 operator/(Vec3 lhs, float value)
{
  lhs /= value;
  return lhs;
}

#endif