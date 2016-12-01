#ifndef HELPERS_HEADER
#define HELPERS_HEADER

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <curand.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_BLOCK_SIZE 256

class CUDAConfig
{
  size_t _gridSize;

public:
  __host__ explicit CUDAConfig(size_t size)
  {
    _gridSize = static_cast<size_t>(ceil(static_cast<float>(size) / CUDA_BLOCK_SIZE));
  }

  __host__ unsigned int GetGridSize() const { return static_cast<unsigned int>(_gridSize); }
  __host__ static unsigned int GetBlockSize() { return CUDA_BLOCK_SIZE; }
};

template <typename T>
__host__ void PrintResults(const T& results, const std::string& filename)
{
  std::ofstream ouput(filename.c_str());
  for (typename T::const_iterator it = results.begin(); it != results.end(); ++it)
    ouput << *it << " ";
  ouput << std::endl;
}

static __host__ void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#endif
