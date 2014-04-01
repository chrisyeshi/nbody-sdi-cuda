#ifndef __NBODY_H__
#define __NBODY_H__

#include <cuda_runtime.h>
#include "cudabuffer.h"

class NBody
{
public:
	NBody(int N = 2);

	void advance(float deltaTime);
	std::vector<float3> getBodies() const { return dBodies.toHost(); }

protected:

private:
	int N;
	yy::CudaBuffer<float3> dBodies;
	yy::CudaBuffer<float2> dVelocities;
	yy::CudaBuffer<float2> dForces;
};

#endif // __NBODY_H__