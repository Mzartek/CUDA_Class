#include <fstream>

/*
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
    std::ofstream output(filename.c_str());
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
*/

#include "tp.h"
int main(int argc, char **argv)
{
  /*GrayLevel<100, 100> grayLevel;

  // CPU style
  {
    CPUDoIt(grayLevel.GetPtr(), grayLevel.GetWidth(), grayLevel.GetHeight());
    grayLevel.SaveToFile("output_CPU.txt");
  }

  // GPU stye
  {
    unsigned int* ptr;
    size_t ptrSize = grayLevel.GetWidth() * grayLevel.GetHeight() * sizeof(unsigned int);
    HANDLE_ERROR(cudaMalloc(&ptr, ptrSize));

    unsigned int blockSize = 1;
    unsigned int threadSize = grayLevel.GetWidth() * grayLevel.GetHeight();
    GPUDoIt<<<blockSize, threadSize>>>(ptr, grayLevel.GetWidth(), grayLevel.GetHeight());

    HANDLE_ERROR(cudaMemcpy(grayLevel.GetPtr(), ptr, ptrSize, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(ptr));
    grayLevel.SaveToFile("output_GPU.txt");
  }
  */

  return main_tp(argc, argv);
}