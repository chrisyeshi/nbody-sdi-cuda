#ifndef __NBODY_H__
#define __NBODY_H__

#include <cuda_runtime.h>
#include <GL/gl.h>
#include "cudabuffer.h"
#include "cudaglbuffer.h"

class NBody
{
public:
	NBody(int N = 2);

	void initBodies(GLuint vbo);
	void advance(float deltaTime);
	std::vector<float3> getBodies() { return dBodies.toHost(); }

protected:

private:
	int N;
	yy::CudaGLBuffer<float3> dBodies;
	yy::CudaBuffer<float2> dVelocities;
	yy::CudaBuffer<float2> dForces;
};

#endif // __NBODY_H__