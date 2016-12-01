#ifndef PARTICLE_HEADER
#define PARTICLE_HEADER

#include "Vec.h"

class Particle
{
  // Life
  float _life;
  float _initialLife;
  float _lifeSpeed;

  // Position
  Vec3 _position;
  Vec3 _initialPosition;

  // Direction
  Vec3 _direction;
  float _speedDirection;

public:

  __host__ Particle(float life, float initialLife, float lifeSpeed,
                    const Vec3& initialPosition, const Vec3& direction, float speedDirection) :
    _life(life), _initialLife(initialLife), _lifeSpeed(lifeSpeed),
    _initialPosition(initialPosition), _direction(direction), _speedDirection(speedDirection)
  {
    if (_life > _initialLife) throw std::exception();

    const float diff = (_initialLife - _life) / _lifeSpeed;
    _position = _initialPosition + ((_direction * _speedDirection) * diff);
  }

  __host__ __device__ void Rebirth()
  {
    _life = 100.0f;
    _position = _initialPosition;
  }

  __host__ __device__ void Move()
  {
    _life -= _lifeSpeed;
    _position += _direction * _speedDirection;
  }
};

#endif