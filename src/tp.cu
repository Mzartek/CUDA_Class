#include "Helpers.h"

#include <vector>
#include <cstdlib>

__host__ void printResults(const std::vector<unsigned int>& results)
{
  for (std::vector<unsigned int>::const_iterator it = results.begin(); it != results.end(); ++it)
    std::cout << *it << " ";
  std::cout << std::endl;
}

__host__ __device__ unsigned int calculate(int x)
{
  int n;
  for (n = 0; x > 1; ++n)
  {
    if (x & 1) x = 3 * x + 1;
    else x = x / 2;
  }
  return n;
}

__host__ void syracuseCPU_execute(unsigned int* dst, size_t size)
{
  for (int i = 0; i < size; ++i)
  {
    dst[i] = calculate(i + 1);
  }
}

__global__ void syracuseGPU_execute(unsigned int* dst)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  dst[index] = index + 1;
}

__host__ void syracuseCPU_prepare(size_t size)
{
  std::vector<unsigned int> dst(size);

  syracuseCPU_execute(&dst[0], size);

  printResults(dst);
}

__host__ void syracuseGPU_prepare(size_t size)
{
  size_t byteSize = size * sizeof(unsigned int);
  std::vector<unsigned int> dstCPU(size);
  unsigned int* dstGPU = NULL;

  HANDLE_ERROR(cudaMalloc(&dstGPU, byteSize));

  unsigned int blockSize = 1;
  unsigned int threadSize = static_cast<unsigned int>(size);
  syracuseGPU_execute<<<blockSize, threadSize>>>(dstGPU);

  HANDLE_ERROR(cudaMemcpy(&dstCPU[0], dstGPU, byteSize, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(dstGPU));

  printResults(dstCPU);
}

int main_tp(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cout << "program [size]" << std::endl;
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