#include "Helpers.h"

#include "Particle.h"

__host__ void rainParticlesMoveCPU_execute(Particle *src, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    src[i].Move();
  }
}

__host__ void rainParticlesCreateCPU_execute(Particle *src, size_t size)
{
  Vec3 direction = { 0, -1, 0 };
  for (size_t i = 0; i < size; ++i)
  {
    Vec3 initPosition = 
    {
      0.0f,
      100.0f,
      0.0f
    };
    src[i].Initialize(static_cast<float>(rand() % 100), 100.0f, 5.0f, initPosition, direction, 5.0f);
  }
}

__global__ void rainParticlesMoveGPU_execute(Particle *src, size_t size)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) src[index].Move();
}

__global__ void rainParticlesCreateGPU_execute(Particle *src, size_t size) 
{
  Vec3 direction = { 0, -1, 0 };
  Vec3 initPosition =
  {
    0.0f,
    100.0f,
    0.0f
  };
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) src[index].Initialize(0.0f, 100.0f, 5.0f, initPosition, direction, 5.0f);
}

int main_project(int argc, char **argv)
{
  return 0;
}
