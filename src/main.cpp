#include <iostream>
#include <fstream>
#include <array>

struct Complexe
{
  float _r;
  float _i;

  Complexe() :
    _r(0.0f), _i(0.0f)
  {}

  Complexe(float r, float i) :
    _r(r), _i(i)
  {}

  float GetMagnetudeSquare()
  {
    return _i * _i + _r * _r;
  }
};

Complexe operator*(Complexe lhs, const Complexe& rhs)
{
  Complexe res;
  res._r = lhs._r * rhs._r - lhs._i * rhs._i;
  res._i = lhs._i * rhs._r + lhs._r * rhs._i;
  return res;
}

Complexe operator+(Complexe lhs, const Complexe& rhs)
{
  Complexe res;
  res._r = lhs._r + rhs._r;
  res._i = lhs._i + rhs._i;
  return res;
}

unsigned int ComputePixel(Complexe Z)
{
  const Complexe C(-0.8f, 0.156f);
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

template <int Width, int Height>
class GrayLevel
{
  std::array<unsigned int, Width * Height> _array;
 
public:
  int GetWidth() { return Width; }
  int GetHeight() { return Height; }

  void SetGrayLevel(int x, int y, unsigned int value)
  {
    _array[y * Width + x] = value;
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
  Complexe tmp;

  for (int i = 0; i < grayLevel.GetWidth(); ++i)
  {
    for (int j = 0; j < grayLevel.GetHeight(); ++j)
    {
      tmp._r = ((float)i / grayLevel.GetWidth()) * 2 - 1;
      tmp._i = ((float)j / grayLevel.GetHeight()) * 2 - 1;
      grayLevel.SetGrayLevel(i, j, ComputePixel(tmp));
    }
  }
  grayLevel.SaveToFile("output.txt");
  return 0;
}