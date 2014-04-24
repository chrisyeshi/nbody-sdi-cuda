#ifndef __Delaunay_h__
#define __Delaunay_h__

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

class Delaunay
{
public:
	Delaunay();

	void setBoardSize(int boardSize) { this->boardSize = boardSize; }
	void build(const thrust::device_vector<float2>& sites);
	std::vector<ushort3> getTrias() const;

protected:
	void build();

private:
	int boardSize;
	thrust::device_vector<float2> verts;
	thrust::device_vector<int3> trias;
};

#endif // __Delaunay_h__