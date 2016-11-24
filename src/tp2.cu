#include "Helpers.h"

#include <vector>
#include <iostream>

__host__ void printResults(const std::vector<int>& results)
{
  for (std::vector<int>::const_iterator it = results.begin(); it != results.end(); ++it)
    std::cout << *it << " ";
  std::cout << std::endl;
}

__host__ __device__ void sortElement(size_t element, const int* src, int* dst, size_t size)
{
  int offset = 0;
  int newIndex = 0;
  for (size_t i = 0; i < size; ++i)
  {
    newIndex += src[element] > src[i];
    offset += (i < element) & (src[element] == src[i]);
  }
  dst[newIndex + offset] = src[element];
}

__host__ void sortCPU_execute(const int* src, int* dst, size_t size)
{
  for (size_t i = 0; i < size; ++i) sortElement(i, src, dst, size);
}

__global__ void sortGPU_execute(const int* src, int* dst, size_t size)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) sortElement(index, src, dst, size);
}

__host__ void sortCPU_prepare(const std::vector<int>& src, std::vector<int>& dst)
{
  dst.resize(src.size());
  sortCPU_execute(&src[0], &dst[0], src.size());
}

__host__ void sortGPU_prepare(const std::vector<int>& srcCPU, std::vector<int>& dstCPU)
{
  dstCPU.resize(srcCPU.size());

  size_t byteSize = srcCPU.size() * sizeof(int);
  int* srcGPU = NULL;
  int* dstGPU = NULL;

  HANDLE_ERROR(cudaMalloc(&srcGPU, byteSize));
  HANDLE_ERROR(cudaMalloc(&dstGPU, byteSize));

  HANDLE_ERROR(cudaMemcpy(srcGPU, &srcCPU[0], byteSize, cudaMemcpyHostToDevice));

  CUDAConfig cudaConfig(srcCPU.size());
  unsigned int gridSize = cudaConfig.GetGridSize();
  unsigned int blockSize = cudaConfig.GetBlockSize();
  std::cout << "Grid Size: " << gridSize << std::endl;
  std::cout << "Block Size: " << blockSize << std::endl;
  sortGPU_execute<<<gridSize, blockSize>>>(srcGPU, dstGPU, srcCPU.size());

  HANDLE_ERROR(cudaMemcpy(&dstCPU[0], dstGPU, byteSize, cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(dstGPU));
  HANDLE_ERROR(cudaFree(srcGPU));

  printResults(dstCPU);
}

std::vector<int> generateVectorFromArgs(int argc, char **argv)
{
  std::vector<int> vectorGenerated;
  for (int i = 1; i < argc; ++i) vectorGenerated.push_back(atoi(argv[i]));
  return vectorGenerated;
}

int main_tp2(int argc, char **argv)
{
  std::vector<int> dst, src = generateVectorFromArgs(argc, argv);

  std::cout << "Execute the CPU version" << std::endl;
  sortCPU_prepare(src, dst);
  printResults(dst);

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount > 0)
  {
    std::cout << "Execute the GPU version" << std::endl;
    sortGPU_prepare(src, dst);
    printResults(dst);
  }

  return 0;
}