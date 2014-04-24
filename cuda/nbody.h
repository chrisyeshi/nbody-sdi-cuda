#ifndef __NBODY_H__
#define __NBODY_H__

#include <cuda_runtime.h>
#include <GL/gl.h>
#include "cudabuffer.h"
#include "cudaglbuffer.h"
#include "delaunay.h"

class NBody
{
public:
	NBody(int N = 2);

	void initBodies(GLuint vbo);
	void advance(float deltaTime);
	std::vector<float3> getBodies() { return dBodies.toHost(); }
	std::vector<ushort3> getTrias() const { return delaunay.getTrias(); }

protected:

private:
	int N;
	yy::CudaGLBuffer<float3> dBodies;
	yy::CudaBuffer<float2> dVelocities;
	yy::CudaBuffer<float2> dForces;
	Delaunay delaunay;
};

#endif // __NBODY_H__