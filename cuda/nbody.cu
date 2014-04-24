#include "nbody.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "delaunay.h"

#define EPS2 0.00001f
#define G 0.0000000667384f
#define BLOCKSIZE 256
#define MASS_RANGE make_float2(1.f, 10.f)
#define CENTER make_float3(0.5f, 0.5f, 100.f)
#define ROTATE make_float2(0.025f, 0.025f)
// #define ROTATE make_float2(0.f, 0.f)

__device__ float2 force_of_a_from_b(float3 aBody, float3 bBody)
{
	float2 r;
	r.x = bBody.x - aBody.x;
	r.y = bBody.y - aBody.y;
	float d2 = r.x * r.x + r.y * r.y + EPS2;
	float d6 = d2 * d2 * d2;
	float d3 = sqrtf(d6);
	float s = bBody.z / d3;
	float2 force;
	force.x = r.x * s * G * aBody.z;
	force.y = r.y * s * G * aBody.z;
	return force;
}

__global__ void kernelComputeForces(int N, float3* bodies, float2* forces)
{
	extern __shared__ float3 shBodies[];
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int nThreads = blockDim.x;
	int nBlocks = gridDim.x;
	float3 iBody;
	float2 iForce = make_float2(0.f, 0.f);
	if (i < N)
	{
		iBody = bodies[i];
	}

	for (int jBlock = 0; jBlock < nBlocks; ++jBlock)
	{
		//
		//
		// Utilize shared memory
		//
		//
		int jThread = jBlock * nThreads + threadIdx.x;
		if (i < N && jThread < N)
		{
			shBodies[threadIdx.x] = bodies[jThread];
		}
		__syncthreads();
		//
		//
		// Compute forces
		//
		//
		if (i < N)
		{
			for (int jThread = 0; jThread < nThreads; ++jThread)
			{
				int j = jThread + nThreads * jBlock;
				if (j >= N) break;
				float3 jBody = shBodies[jThread];
				float2 force = force_of_a_from_b(iBody, jBody);
				iForce.x += force.x;
				iForce.y += force.y;
			}
		}
		__syncthreads();
	}
	//
	//
	// Add a massive body in the middle so the bodies don't fly away
	//
	//
	if (i < N)
	{
		float3 jBody = CENTER;
		float2 force = force_of_a_from_b(iBody, jBody);
		iForce.x += force.x;
		iForce.y += force.y;
		forces[i] = iForce;
	}
}

__global__ void kernelUpdateVelocities(int N, float deltaTime, float3* bodies, float2* forces, float2* velocities)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= N) return;
	float2 iVelocity = velocities[i];
	float2 iForce = forces[i];
	float3 iBody = bodies[i];
	float2 iAcc;
	iAcc.x = iForce.x / iBody.z;
	iAcc.y = iForce.y / iBody.z;
	float2 v;
	v.x = iVelocity.x + iAcc.x * deltaTime;
	v.y = iVelocity.y + iAcc.y * deltaTime;
	velocities[i] = v;
}

__global__ void kernelAdvance(int N, float deltaTime, float2* velocities, float3* bodies)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= N) return;
	float2 iVelocity = velocities[i];
	float3 iBody = bodies[i];
	float3 b;
	b.x = iBody.x + iVelocity.x * deltaTime;
	b.y = iBody.y + iVelocity.y * deltaTime;
	b.z = iBody.z;
	bodies[i] = b;
}

NBody::NBody(int N)
  : N(N), dForces(N)
{
	delaunay.setBoardSize(1024);
}

void NBody::initBodies(GLuint vbo)
{
	// init positions
	std::vector<float3> hBodies(N);
	for (int i = 0; i < N; ++i)
	{
		hBodies[i].x = drand48();
		hBodies[i].y = drand48();
		// mass
		hBodies[i].z = drand48() * (MASS_RANGE.y - MASS_RANGE.x) + MASS_RANGE.x;
	}
	dBodies = yy::CudaGLBuffer<float3>(vbo);
	dBodies.upload(hBodies);
	// init velocities
	std::vector<float2> hVels(N);
	for (int i = 0; i < N; ++i)
	{
		float2 toCenter = make_float2(hBodies[i].x - CENTER.x, hBodies[i].y - CENTER.y);
		float2 perpen = make_float2(toCenter.y, -toCenter.x);
		hVels[i] = make_float2(perpen.x * ROTATE.x, perpen.y * ROTATE.y);
	}
	dVelocities = yy::CudaBuffer<float2>(hVels);
}

std::ostream& operator<<(std::ostream& os, const float2& val)
{
	os << "[" << val.x << "," << val.y << "]";
	return os;
}

void NBody::advance(float deltaTime = 1.f)
{
	dim3 grid(int(ceil(float(N)/BLOCKSIZE)));
	dim3 block(BLOCKSIZE);
	dBodies.map();

	// nbody
	kernelComputeForces<<<grid, block, BLOCKSIZE * sizeof(float3)>>>(N, dBodies, dForces);
	kernelUpdateVelocities<<<grid, block>>>(N, deltaTime, dBodies, dForces, dVelocities);
	kernelAdvance<<<grid, block>>>(N, deltaTime, dVelocities, dBodies);
	// delaunay
	thrust::device_ptr<float3> dev_ptr(dBodies.ptr());
	thrust::device_vector<float3> sites3d(dev_ptr, dev_ptr + dBodies.length());
	thrust::host_vector<float2> hSites(sites3d.size());
	for (unsigned int i = 0; i < hSites.size(); ++i)
	{
		float3 site = sites3d[i];
		hSites[i] = make_float2(site.x, site.y);
	}
	delaunay.build(hSites);

	dBodies.unmap();
}