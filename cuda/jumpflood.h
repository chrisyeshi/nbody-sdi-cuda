#ifndef __JumpFlood_h__
#define __JumpFlood_h__

#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

class JumpFlood
{
public:
	//
	//
	// Constructor
	//
	//
	JumpFlood();

	//
	//
	// Configuration
	//
	//
	void setBoardSize(int boardSize);
	int getBoardSize() const { return boardSize; }
	void initSites(const thrust::device_vector<float2>& dSites);

	//
	//
	// Run
	//
	//
	void flood();

	//
	//
	// Access
	//
	//
	void print(std::ostream& os) const;
	const thrust::device_vector<int>& getSiteIdx() const { return dSiteIdx; }
	const thrust::device_vector<float2>& getSitePos() const { return dSitePos; }

public:
	//
	//
	// Protected Functions
	//
	//
	void stepFlood(int distance);

private:
	//
	//
	// Member Variables
	//
	//
	int boardSize;
	thrust::device_vector<int> dSiteIdx;
	thrust::device_vector<float2> dSitePos;

	//
	//
	// Helper Functions
	//
	//
	int* rSiteIdx() { return thrust::raw_pointer_cast(dSiteIdx.data()); }
	float2* rSitePos() { return thrust::raw_pointer_cast(dSitePos.data()); }
};

#endif // __JumpFlood_h__