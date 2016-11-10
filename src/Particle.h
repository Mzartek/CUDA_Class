#ifndef PARTICLE_HEADER
#define PARTICLE_HEADER

#include "Vec.h"

struct Particle
{
  Vec3<float> pos;
  Vec3<float> dir;
  float speed;
  float life;
};

#endif