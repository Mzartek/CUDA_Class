#include "Helpers.h"

#include <vector>

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
  if (src.size() < 1)
  {
    std::cerr << "No elements to sort" << std::endl;
    return 1;
  }

  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::cout << "Execute the CPU version" << std::endl;
  {
    cudaEventRecord(start);
    sortCPU_prepare(src, dst);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
  }
  std::cout << "Elapsed time:" << elapsedTime << std::endl;
  PrintResults<std::vector<int>>(dst, "output_CPU.txt");

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount > 0)
  {
    std::cout << "Execute the GPU version" << std::endl;
    {
      cudaEventRecord(start);
      sortGPU_prepare(src, dst);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      elapsedTime = 0;
      cudaEventElapsedTime(&elapsedTime, start, stop);
    }
    std::cout << "Elapsed time:" << elapsedTime << std::endl;
    PrintResults<std::vector<int>>(dst, "ouput_GPU.txt");
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}