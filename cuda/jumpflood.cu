#include "jumpflood.h"
#include <cassert>
#include <cmath>

//
//
// Device Functions
//
//

inline __device__ int toIdx(int2 idx2, int2 dim)
{
	if (idx2.x < 0 || idx2.x >= dim.x || idx2.y < 0 || idx2.y >= dim.y)
		return -1;
	return idx2.x + idx2.y * dim.x;
}

__device__ int querySiteIdx(int* siteIdx, int2 idx2, int2 dim)
{
	int idx = toIdx(idx2, dim);
	if (idx < 0)
		return -1;
	return siteIdx[idx];
}

__device__ float2 querySitePos(float2* sitePos, int2 idx2, int2 dim)
{
	int idx = toIdx(idx2, dim);
	if (idx < 0)
		return make_float2(-1.f, -1.f);
	return sitePos[idx];
}

__device__ void compareSite(int2 idx2, int2 dim, float2 cellpos,
		int* siteIdx, float2* sitePos,
		int* minSite, float* minDist2, float2* minPos)
{
	int site = querySiteIdx(siteIdx, idx2, dim);
	if (site >= 0)
	{
		float2 pos = sitePos[site]; //querySitePos(sitePos, idx2, dim);
		float2 distvec = make_float2(cellpos.x - pos.x, cellpos.y - pos.y);
		float dist2 = distvec.x * distvec.x + distvec.y * distvec.y;
		if (dist2 < *minDist2)
		{
			*minSite = site;
			*minDist2 = dist2;
			*minPos = pos;
		}
	}
}

//
//
// Kernels
//
//

__global__ void kernelSetCells(int n, const float2* sites, int2 dim, int* siteIdx, float2* sitePos)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= n)
		return;
	float2 pos = sites[i];
	if (pos.x < 0.0 || pos.x >= 1.0 || pos.y < 0 || pos.y >= 1.0)
		return;
	int2 idx2 = make_int2(pos.x * dim.x, pos.y * dim.y);
	if (idx2.x < 0 || idx2.x >= dim.x || idx2.y < 0 || idx2.y >= dim.y)
		return;
	int idx = idx2.x + idx2.y * dim.x;
	atomicCAS(&siteIdx[idx], -1, i);
	// if (-1 == atomicCAS(&siteIdx[idx], -1, i))
	// {
	// 	sitePos[idx] = pos;
	// }
	// siteIdx[idx] = i;
}

__global__ void kernelStepFlood(int2 dim, int* siteIdx, float2* sitePos, int d)
{
	int2 idx2 = make_int2(
			threadIdx.x + blockDim.x * blockIdx.x,
			threadIdx.y + blockDim.y * blockIdx.y);
	// if (idx2.x < 0 || idx2.x >= dim.x || idx2.y < 0 || idx2.y >= dim.y)
	// 	return;
	int idx = toIdx(idx2, dim);
	if (idx < 0)
		return;
	// check if it is the original site cell
	int currSiteIdx = querySiteIdx(siteIdx, idx2, dim);
	if (currSiteIdx >= 0)
	{
		float2 currSitePos = sitePos[currSiteIdx]; //querySitePos(sitePos, idx2, dim);
		int2 siteCell2 = make_int2(currSitePos.x * dim.x, currSitePos.y * dim.y);
		if (idx2.x == siteCell2.x && idx2.y == siteCell2.y)
			return;
	}
	// cell position
	float2 cellpos = make_float2((float(idx2.x) + 0.5f) / float(dim.x), (float(idx2.y) + 0.5f) / float(dim.y));
	// all candidates
	int minSite = -1;
	float minDist2 = 2.0;
	float2 minPos;
	compareSite(make_int2(idx2.x+0, idx2.y+0), dim, cellpos, siteIdx, sitePos, &minSite, &minDist2, &minPos);
	compareSite(make_int2(idx2.x+d, idx2.y+0), dim, cellpos, siteIdx, sitePos, &minSite, &minDist2, &minPos);
	compareSite(make_int2(idx2.x+d, idx2.y+d), dim, cellpos, siteIdx, sitePos, &minSite, &minDist2, &minPos);
	compareSite(make_int2(idx2.x+0, idx2.y+d), dim, cellpos, siteIdx, sitePos, &minSite, &minDist2, &minPos);
	compareSite(make_int2(idx2.x-d, idx2.y+d), dim, cellpos, siteIdx, sitePos, &minSite, &minDist2, &minPos);
	compareSite(make_int2(idx2.x-d, idx2.y+0), dim, cellpos, siteIdx, sitePos, &minSite, &minDist2, &minPos);
	compareSite(make_int2(idx2.x-d, idx2.y-d), dim, cellpos, siteIdx, sitePos, &minSite, &minDist2, &minPos);
	compareSite(make_int2(idx2.x+0, idx2.y-d), dim, cellpos, siteIdx, sitePos, &minSite, &minDist2, &minPos);
	compareSite(make_int2(idx2.x+d, idx2.y-d), dim, cellpos, siteIdx, sitePos, &minSite, &minDist2, &minPos);
	siteIdx[idx] = minSite;
	// sitePos[idx] = minPos;
}

//
//
// Constructor
//
//

JumpFlood::JumpFlood()
  : boardSize(4),
    dSiteIdx(std::vector<int>(boardSize * boardSize, -1))
    // dSitePos(boardSize * boardSize)
{}

//
//
// Configuration
//
//

void JumpFlood::setBoardSize(int boardSize)
{
	assert(boardSize >= 0);
	this->boardSize = boardSize;
	dSiteIdx = thrust::device_vector<int>(std::vector<int>(boardSize * boardSize, -1));
	// dSitePos = thrust::device_vector<float2>(boardSize * boardSize);
}

void JumpFlood::initSites(const thrust::device_vector<float2>& dSites)
{
	dSitePos = dSites;
	const float2* rSites = thrust::raw_pointer_cast(dSites.data());
	int block = 256;
	int grid = ceil(float(dSites.size()) / block);
	int2 dim = make_int2(boardSize, boardSize);
	kernelSetCells<<<grid, block>>>(dSites.size(), rSites, dim, rSiteIdx(), rSitePos());
}

//
//
// Run
//
//

void JumpFlood::flood()
{
	// stepFlood(1);
	for (int k = boardSize / 2; k > 0; k /= 2)
	{
		stepFlood(k);
	}
	stepFlood(1);
	stepFlood(1);
	stepFlood(1);
	stepFlood(1);
	// for (int k = boardSize / 2; k > 0; k /= 2)
	// {
		// stepFlood(k);
	// }
}

//
//
// Debug
//
//

void JumpFlood::print(std::ostream& os) const
{
	for (int row = boardSize-1; row >= 0; --row)
	{
		for (int col = 0; col < boardSize; ++col)
		{
			os << dSiteIdx[col + row * boardSize] << " ";
		}
		os << std::endl;
	}
}

//
//
// Protected Functions
//
//

void JumpFlood::stepFlood(int distance)
{
	int blockSide = 16;
	dim3 block(blockSide, blockSide);
	int gridSide = ceil(float(boardSize) / blockSide);
	dim3 grid(gridSide, gridSide);
	int2 dim = make_int2(boardSize, boardSize);
	kernelStepFlood<<<grid, block>>>(dim, rSiteIdx(), rSitePos(), distance);
}