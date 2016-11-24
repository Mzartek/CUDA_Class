#include "Helpers.h"

#include <vector>

__host__ __device__ unsigned int calculate(int x)
{
  int a, n;
  for (n = 0; x > 1; ++n)
  {
    a = x & 1;
    x = a * (3 * x + 1) + (a == 0 ? 1 : 0) * (x / 2);
  }
  return n;
}

__host__ void syracuseCPU_execute(unsigned int* dst, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    dst[i] = calculate(i + 1);
  }
}

__global__ void syracuseGPU_execute(unsigned int* dst, size_t size)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) dst[index] = calculate(index + 1);
}

__host__ void syracuseCPU_prepare(size_t size)
{
  std::vector<unsigned int> dst(size);

  syracuseCPU_execute(&dst[0], size);

  PrintResults<std::vector<unsigned int> >(dst, "output_CPU.txt");
}

__host__ void syracuseGPU_prepare(size_t size)
{
  size_t byteSize = size * sizeof(unsigned int);
  std::vector<unsigned int> dstCPU(size);
  unsigned int* dstGPU = NULL;

  HANDLE_ERROR(cudaMalloc(&dstGPU, byteSize));

  CUDAConfig cudaConfig(size);
  unsigned int gridSize = cudaConfig.GetGridSize();
  unsigned int blockSize = cudaConfig.GetBlockSize();
  std::cout << "Grid Size: " << gridSize << std::endl;
  std::cout << "Block Size: " << blockSize << std::endl;
  syracuseGPU_execute<<<gridSize, blockSize>>>(dstGPU, size);

  HANDLE_ERROR(cudaMemcpy(&dstCPU[0], dstGPU, byteSize, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(dstGPU));

  PrintResults<std::vector<unsigned int> >(dstCPU, "output_GPU.txt");
}

int main_tp(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cerr << "program [size]" << std::endl;
    return 1;
  }

  size_t size = atoi(argv[1]);
  std::cout << "Execute the CPU version" << std::endl;
  syracuseCPU_prepare(size);

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount > 0)
  {
    std::cout << "Execute the GPU version" << std::endl;
    syracuseGPU_prepare(size);
  }

  return 0;
}