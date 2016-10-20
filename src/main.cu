#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

struct Complexe
{
  float _r;
  float _i;

  __host__ __device__ float GetMagnetudeSquare()
  {
    return _i * _i + _r * _r;
  }
};

__host__ __device__ Complexe operator*(Complexe lhs, const Complexe& rhs)
{
  Complexe res;
  res._r = lhs._r * rhs._r - lhs._i * rhs._i;
  res._i = lhs._i * rhs._r + lhs._r * rhs._i;
  return res;
}

__host__ __device__ Complexe operator+(Complexe lhs, const Complexe& rhs)
{
  Complexe res;
  res._r = lhs._r + rhs._r;
  res._i = lhs._i + rhs._i;
  return res;
}

__host__ __device__ unsigned int ComputePixel(Complexe Z)
{
  const Complexe C = { -0.8f, 0.156f };
  Complexe A;

  for (int i = 0; i < 200; ++i)
  {
    A = Z * Z;
    Z = A + C;
    if (Z.GetMagnetudeSquare() > 1000)
      return 0;
  }
  return 1;
}

__host__ void CPUDoIt(unsigned int* ptr, int width, int height)
{
  Complexe tmp;
  for (int i = 0; i < width; ++i)
  {
    for (int j = 0; j < height; ++j)
    {
      tmp._r = ((float)i / width) * 2 - 1;
      tmp._i = ((float)j / height) * 2 - 1;
      ptr[j * width + i] = ComputePixel(tmp);
    }
  }
}

__global__ void GPUDoIt(unsigned int* ptr, int width, int height)
{
  Complexe tmp;

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = index % width;
  int j = index / width;

  tmp._r = ((float)i / width) * 2 - 1;
  tmp._i = ((float)j / height) * 2 - 1;
  ptr[j * width + i] = ComputePixel(tmp);
}

template <int Width, int Height>
class GrayLevel
{
  unsigned int _array[Width * Height];
 
public:
  int GetWidth() { return Width; }
  int GetHeight() { return Height; }

  unsigned int* GetPtr()
  {
    return &_array[0];
  }

  void SaveToFile(const std::string& filename)
  {
    std::ofstream output(filename);
    for (int i = 0; i < Width; ++i)
    {
      for (int j = 0; j < Height; ++j)
      {
        output << (_array[j * Width + i] == 0 ? " " : "8");
      }
      output << std::endl;
    }
  }
};

int main(void)
{
  GrayLevel<100, 100> grayLevel;

  // CPU style
  {
    CPUDoIt(grayLevel.GetPtr(), grayLevel.GetWidth(), grayLevel.GetHeight());
    grayLevel.SaveToFile("output_CPU.txt");
  }

  // GPU stye
  {
    unsigned int* ptr;
    size_t ptrSize = grayLevel.GetWidth() * grayLevel.GetHeight() * sizeof(decltype(ptr));
    HANDLE_ERROR(cudaMalloc(&ptr, ptrSize));

    unsigned int blockSize = 1;
    unsigned int threadSize = grayLevel.GetWidth() * grayLevel.GetHeight();
    GPUDoIt<<<blockSize, threadSize>>>(ptr, grayLevel.GetWidth(), grayLevel.GetHeight());

    HANDLE_ERROR(cudaMemcpy(grayLevel.GetPtr(), ptr, ptrSize, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(ptr));
    grayLevel.SaveToFile("output_GPU.txt");
  }

  return 0;
}