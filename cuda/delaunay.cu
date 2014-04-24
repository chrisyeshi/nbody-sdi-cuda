#include "delaunay.h"
#include <iostream>
#include <iomanip>
#include <thrust/scan.h>
#include "jumpflood.h"

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

//
//
// Kernels
//
//

__global__ void kernelTriangleCounts(int2 dim, int* siteIdx, int* triCounts)
{
	int2 idx2 = make_int2(
			threadIdx.x + blockDim.x * blockIdx.x,
			threadIdx.y + blockDim.y * blockIdx.y);
	if (idx2.x < 0 || idx2.x >= dim.x - 1 || idx2.y < 0 || idx2.y >= dim.y - 1)
		return;
	int i0 = toIdx(make_int2(idx2.x+0, idx2.y+0), dim);
	int i1 = toIdx(make_int2(idx2.x+1, idx2.y+0), dim);
	int i2 = toIdx(make_int2(idx2.x+0, idx2.y+1), dim);
	int i3 = toIdx(make_int2(idx2.x+1, idx2.y+1), dim);
	int s0 = siteIdx[i0];
	int s1 = siteIdx[i1];
	int s2 = siteIdx[i2];
	int s3 = siteIdx[i3];
	if (s1 == s3 && s0 != s1 && s2 != s3 && s0 != s2)
		triCounts[i0] = 1;
	else if (s0 == s2 && s0 != s1 && s2 != s3 && s1 != s3)
		triCounts[i0] = 1;
	else if (s2 == s3 && s0 != s2 && s1 != s3 && s0 != s1)
		triCounts[i0] = 1;
	else if (s0 == s1 && s0 != s2 && s1 != s3 && s2 != s3)
		triCounts[i0] = 1;
	else if (s0 != s1 && s0 != s2 && s0 != s3 && s1 != s2 && s1 != s3 && s2 != s3)
		triCounts[i0] = 2;
	// otherwise, 0 is default
}

__global__ void kernelBuildTriangles(int2 dim, int* siteIdx, float2* sites, int* triOffset, int3* trias)
{
	int2 idx2 = make_int2(
			threadIdx.x + blockDim.x * blockIdx.x,
			threadIdx.y + blockDim.y * blockIdx.y);
	if (idx2.x < 0 || idx2.x >= dim.x - 1 || idx2.y < 0 || idx2.y >= dim.y - 1)
		return;
	int i0 = toIdx(make_int2(idx2.x+0, idx2.y+0), dim);
	int i1 = toIdx(make_int2(idx2.x+1, idx2.y+0), dim);
	int i2 = toIdx(make_int2(idx2.x+0, idx2.y+1), dim);
	int i3 = toIdx(make_int2(idx2.x+1, idx2.y+1), dim);
	int s0 = siteIdx[i0];
	int s1 = siteIdx[i1];
	int s2 = siteIdx[i2];
	int s3 = siteIdx[i3];
	int offset = triOffset[i0];
	if (s1 == s3 && s0 != s1 && s2 != s3 && s0 != s2)
	{ // 0 1
	  // 2 1
		trias[offset] = make_int3(s0, s2, s1);

	} else if (s0 == s2 && s0 != s1 && s2 != s3 && s1 != s3)
	{ // 0 1
	  // 0 3
		trias[offset] = make_int3(s0, s3, s1);

	} else if (s2 == s3 && s0 != s2 && s1 != s3 && s0 != s1)
	{ // 0 1
	  // 2 2
		trias[offset] = make_int3(s0, s2, s1);

	} else if (s0 == s1 && s0 != s2 && s1 != s3 && s2 != s3)
	{ // 0 0
	  // 2 3
		trias[offset] = make_int3(s0, s2, s3);

	} else if (s0 != s1 && s0 != s2 && s0 != s3 && s1 != s2 && s1 != s3 && s2 != s3)
	{ // 0 1
	  // 2 3
		// test which edge is shorter
		float2 p0 = sites[s0];
		float2 p1 = sites[s1];
		float2 p2 = sites[s2];
		float2 p3 = sites[s3];
		float dist03 = (p0.x - p3.x) * (p0.x - p3.x) + (p0.y - p3.y) * (p0.y - p3.y);
		float dist12 = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
		if (dist03 < dist12)
		{
			trias[offset+0] = make_int3(s2, s3, s0);
			trias[offset+1] = make_int3(s1, s0, s3);
		} else
		{
			trias[offset+0] = make_int3(s0, s2, s1);
			trias[offset+1] = make_int3(s3, s1, s2);
		}
	}
}

__global__ void kernelRemoveIslands(int2 dim, int* siteIdx, float2* sites, unsigned int* nIslands, int *islands)
{
	int2 idx2 = make_int2(
			threadIdx.x + blockDim.x * blockIdx.x,
			threadIdx.y + blockDim.y * blockIdx.y);
	if (idx2.x < 0 || idx2.x >= dim.x || idx2.y < 0 || idx2.y >= dim.y)
		return;
	int idx = toIdx(idx2, dim);
	float2 cellPos = make_float2((idx2.x + 0.5f) / dim.x, (idx2.y + 0.5f) / dim.y);
	int theSiteIdx = siteIdx[idx];
	float2 theSitePos = sites[theSiteIdx];
	float2 dirToSite = make_float2(theSitePos.x - cellPos.x, theSitePos.y - cellPos.y);
	int2 dir = make_int2(0, 0);
	if (dirToSite.x < 0)
		dir.x = -1;
	else
		dir.x = 1;
	if (dirToSite.y < 0)
		dir.y = -1;
	else
		dir.y = 1;
	int neiSiteIdx0, neiSiteIdx1, neiSiteIdx2;
	float2 neiSite0, neiSite1, neiSite2;
	{ // 1, 0
		int2 neiIdx2 = make_int2(idx2.x + dir.x, idx2.y);
		int neiIdx = toIdx(neiIdx2, dim);
		neiSiteIdx0 = neiIdx < 0 ? -1 : siteIdx[neiIdx];
		if (neiSiteIdx0 == theSiteIdx)
			return;
		neiSite0 = neiIdx < 0 ? make_float2(-1,-1) : sites[neiSiteIdx0];
	}
	{ // 0, 1
		int2 neiIdx2 = make_int2(idx2.x, idx2.y + dir.y);
		int neiIdx = toIdx(neiIdx2, dim);
		neiSiteIdx1 = neiIdx < 0 ? -1 : siteIdx[neiIdx];
		if (neiSiteIdx1 == theSiteIdx)
			return;
		neiSite1 = neiIdx < 0 ? make_float2(-1,-1) : sites[neiSiteIdx1];
	}
	{ // 1, 1
		int2 neiIdx2 = make_int2(idx2.x + dir.x, idx2.y + dir.y);
		int neiIdx = toIdx(neiIdx2, dim);
		neiSiteIdx2 = neiIdx < 0 ? -1 : siteIdx[neiIdx];
		if (neiSiteIdx2 == theSiteIdx)
			return;
		neiSite2 = neiIdx < 0 ? make_float2(-1,-1) : sites[neiSiteIdx2];
	}
	{ // this site only occupies one cell
		int2 theSiteCell2 = make_int2(theSitePos.x * dim.x, theSitePos.y * dim.y);
		int theSiteCell = toIdx(theSiteCell2, dim);
		if (theSiteCell == idx)
			return;
	}
	atomicAdd(nIslands, 1);

	islands[idx] = 1;

	// set to neighbor site index;
	float2 distvec0 = make_float2(neiSite0.x - cellPos.x, neiSite0.y - cellPos.y);
	float2 distvec1 = make_float2(neiSite1.x - cellPos.x, neiSite1.y - cellPos.y);
	float2 distvec2 = make_float2(neiSite2.x - cellPos.x, neiSite2.y - cellPos.y);
	float dist0 = distvec0.x * distvec0.x + distvec0.y * distvec0.y;
	float dist1 = distvec1.x * distvec1.x + distvec1.y * distvec1.y;
	float dist2 = distvec2.x * distvec2.x + distvec2.y * distvec2.y;
	float minDist = dist0;
	int minSiteIdx = neiSiteIdx0;
	if (dist1 < minDist)
	{
		minSiteIdx = neiSiteIdx1;
		minDist = dist1;
	}
	if (dist2 < minDist)
	{
		minSiteIdx = neiSiteIdx2;
		minDist = dist2;
	}
	// update
	siteIdx[idx] = minSiteIdx;
}

//
//
// Constructor
//
//

void printBoard(const thrust::device_vector<int>& board, int boardSize)
{
	for (int row = boardSize-1; row >= 0; --row)
	{
		for (int col = 0; col < boardSize; ++col)
		{
			std::cout << board[col + row * boardSize] << " ";
		}
		std::cout << std::endl;
	}
}

Delaunay::Delaunay()
{
}

void Delaunay::build(const thrust::device_vector<float2>& sites)
{
	verts = sites;
	build();
}

std::vector<ushort3> Delaunay::getTrias() const
{
	std::vector<ushort3> ret(trias.size());
	for (unsigned int i = 0; i < ret.size(); ++i)
	{
		int3 tria = trias[i];
		ret[i] = make_ushort3(tria.x, tria.y, tria.z);
	}
	return ret;
}

void Delaunay::build()
{
	// print verts
	// std::cout.precision(3);
	// thrust::host_vector<float2> hVerts = verts;
	// for (unsigned int i = 0; i < hVerts.size(); ++i)
	// 	std::cout << "[" << hVerts[i].x << "," << hVerts[i].y << "],";
	// std::cout << std::endl << std::endl;

	int blockSide = 16;
	dim3 block(blockSide, blockSide);
	int gridSide = ceil(float(boardSize) / blockSide);
	dim3 grid(gridSide, gridSide);
	int2 dim = make_int2(boardSize, boardSize);
	// flood
	JumpFlood jump;
	jump.setBoardSize(boardSize);
	jump.initSites(verts);
	jump.flood();
	thrust::device_vector<int> siteIdx = jump.getSiteIdx();

	// std::cout << "Jump: " << std::endl;
	// printBoard(siteIdx, boardSize); std::cout << std::endl;

	// remove islands
	unsigned int *nIslands;
	unsigned int hIslandCount;
	cudaMalloc(&nIslands, sizeof(unsigned int));
	int loopCount = 0;
	do {
		if (loopCount > boardSize)
		{
			std::cout << "Jump: " << std::endl;
			printBoard(siteIdx, boardSize); std::cout << std::endl;
		}

		thrust::device_vector<int> islands(siteIdx.size(), 0);
		cudaMemset(nIslands, 0, sizeof(unsigned int));
		kernelRemoveIslands<<<grid, block>>>(
				dim,
				thrust::raw_pointer_cast(siteIdx.data()),
				thrust::raw_pointer_cast(verts.data()),
				nIslands,
				thrust::raw_pointer_cast(islands.data()));
		cudaMemcpy(&hIslandCount, nIslands, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		if (loopCount > boardSize)
		{
			// print verts
			std::cout.precision(3);
			thrust::host_vector<float2> hVerts = verts;
			for (unsigned int i = 0; i < hVerts.size(); ++i)
				std::cout << "[" << hVerts[i].x << "," << hVerts[i].y << "],";
			std::cout << std::endl << std::endl;

			// std::cout << "Jump: " << std::endl;
			// jump.print(std::cout); std::cout << std::endl;

			std::cout << "nIslands: " << std::endl << hIslandCount << std::endl;

			std::cout << "Islands: " << std::endl;
			printBoard(islands, boardSize); std::cout << std::endl;

			std::cout << "After Islands: " << std::endl;
			printBoard(siteIdx, boardSize); std::cout << std::endl;
		}

		if (loopCount > boardSize * 4)
			exit(0);

		++loopCount;
	} while (hIslandCount > 0);
	cudaFree(nIslands);

	// locate voronoi vertices
	thrust::device_vector<int> triCounts(siteIdx.size(), 0);
	kernelTriangleCounts<<<grid, block>>>(
			dim,
			thrust::raw_pointer_cast(siteIdx.data()),
			thrust::raw_pointer_cast(triCounts.data()));

	// std::cout << "Vertices: " << std::endl;
	// printBoard(triCounts, boardSize); std::cout << std::endl;

	// scan for no. of triangles
	thrust::device_vector<int> triOffsets(triCounts.size());
	thrust::exclusive_scan(triCounts.begin(), triCounts.end(), triOffsets.begin());
	int nTriangles = triOffsets[triOffsets.size()-1] + triCounts[triCounts.size()-1];

	// std::cout << nTriangles << std::endl;

	// allocate space
	trias.resize(nTriangles);
	// connect delaunay vertices
	kernelBuildTriangles<<<grid, block>>>(
			dim,
			thrust::raw_pointer_cast(siteIdx.data()),
			thrust::raw_pointer_cast(verts.data()),
			thrust::raw_pointer_cast(triOffsets.data()),
			thrust::raw_pointer_cast(trias.data()));

	// for (unsigned int i = 0; i < trias.size(); ++i)
	// {
	// 	int3 tria = trias[i];
	// 	std::cout << "[" << tria.x << "," << tria.y << "," << tria.z << "],";
	// }
	// std::cout << std::endl;
}