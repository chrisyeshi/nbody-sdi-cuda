#ifndef __yy_CudaBuffer_h__
#define __yy_CudaBuffer_h__

#include <vector>
#include <iostream>
#include "referencecount.h"

namespace yy
{

template <typename T>
class CudaBuffer
{
public:
	//
	//
	// Constructor / Destructor
	//
	//
    CudaBuffer();
	CudaBuffer(int length);
	CudaBuffer(const std::vector<T>& hBuffer);
	CudaBuffer(const CudaBuffer& buffer);
	CudaBuffer<T>& operator=(const CudaBuffer<T>& buffer);
	~CudaBuffer();
	//
	//
	// Device ==> Host
	//
	//
	std::vector<T> toHost() const;
	//
	//
	// Accessors
	//
	//
    operator T *() const { return dPtr; }
	T* ptr() const { return dPtr; }
	int length() const { return nItems; }
	int memory() const { return nItems * sizeof(T); }
	//
	//
	// For Debugging
	//
	//
	// std::ostream& operator<<(std::ostream&, const CudaBuffer<T>& buffer);

protected:

private:
	//
	//
	// Member Variables
	//
	//
	ReferenceCount* reference;
	T* dPtr;
	int nItems;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const CudaBuffer<T>& buffer)
{
    std::vector<T> hBuffer = buffer.toHost();
    for (unsigned int i = 0; i < hBuffer.size() - 1; ++i)
        os << hBuffer[i] << ",";
    os << hBuffer[hBuffer.size() - 1];
    return os;
}

} // namespace yy

#endif // __yy_CudaBuffer_h__