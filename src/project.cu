#include "Particle.h"

#include <vector>
#include <ctime>

__host__ std::vector<Particle> createRainParticles(size_t size)
{
  std::vector<Particle> particles;
  for (size_t i = 0; i < size; ++i)
  {
    Vec3 direction = { 0, -1, 0 };
    Vec3 initPosition =
    {
      static_cast<float>(rand() % 200 - 100),
      100.0f,
      static_cast<float>(rand() % 200 - 100)
    };
    particles.push_back(Particle(static_cast<float>(rand() % 100), 100.0f, 5.0f, initPosition, direction, 5.0f));
  }
  return particles;
}

__host__ __device__ void rainParticleMove(Particle &particle)
{
  particle.Move();
}

__host__ void rainParticlesMoveCPU_execute(Particle *src, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    rainParticleMove(src[i]);
  }
}

__global__ void rainParticlesMoveGPU_execute(Particle *src, size_t size)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) rainParticleMove(src[index]);
}

__host__ void rainParticlesMoveCPU_prepare(size_t size, size_t iterSize)
{
  std::vector<Particle> particles = createRainParticles(size);

  std::cout << "Start moving" << std::endl;
  PrintResults(particles, "particles_CPU.txt");
  for (size_t i = 0; i < iterSize; ++i)
  {
    rainParticlesMoveCPU_execute(&particles[0], particles.size());
  }
  PrintResults(particles, "particles_CPU.txt", true);
}

__host__ void rainParticlesMoveGPU_prepare(size_t size, size_t iterSize)
{
  std::vector<Particle> particlesCPU = createRainParticles(size);
  Particle *particlesGPU = NULL;

  PrintResults(particlesCPU, "particles_GPU.txt");

  size_t byteSize = particlesCPU.size() * sizeof(Particle);

  HANDLE_ERROR(cudaMalloc(&particlesGPU, byteSize));
  HANDLE_ERROR(cudaMemcpy(particlesGPU, &particlesCPU[0], byteSize, cudaMemcpyHostToDevice));

  CUDAConfig cudaConfig(size);
  unsigned int gridSize = cudaConfig.GetGridSize();
  unsigned int blockSize = cudaConfig.GetBlockSize();
  std::cout << "Grid Size: " << gridSize << std::endl;
  std::cout << "Block Size: " << blockSize << std::endl;

  std::cout << "Start moving" << std::endl;
  PrintResults(particlesCPU, "particles_GPU.txt");
  for (size_t i = 0; i < iterSize; ++i)
  {
    rainParticlesMoveGPU_execute<<<gridSize, blockSize>>>(particlesGPU, particlesCPU.size());
  }
  HANDLE_ERROR(cudaMemcpy(&particlesCPU[0], particlesGPU, byteSize, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(particlesGPU));

  PrintResults(particlesCPU, "particles_GPU.txt", true);
}

int main_project(int argc, char **argv)
{
  size_t sizeParticles = 100000;
  size_t sizeIterations = 1000000;

  // CPU
  {
    std::cout << "Start CPU" << std::endl;
    clock_t t = clock();
    rainParticlesMoveCPU_prepare(sizeParticles, sizeIterations);
    t = clock() - t;
    std::cout << "Done: (" << t << "ms)" << std::endl;
  }
  std::cout << std::endl;
  // GPU
  {
    cudaEvent_t start, stop;
    float elapsedTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Start GPU" << std::endl;
    cudaEventRecord(start);
    rainParticlesMoveGPU_prepare(sizeParticles, sizeIterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Done: (" << elapsedTime << "ms)" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  return 0;
}
