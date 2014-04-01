#include "cudabuffer.h"

namespace yy
{

//
//
// Constructor / Destructor
//
//

template <typename T>
CudaBuffer<T>::CudaBuffer() : reference(NULL), dPtr(NULL), nItems(1)
{
	reference = new ReferenceCount;
	reference->add();
	cudaMalloc((void**)&dPtr, nItems * sizeof(T));
}

template <typename T>
CudaBuffer<T>::CudaBuffer(int length) : reference(NULL), dPtr(NULL), nItems(length)
{
	reference = new ReferenceCount;
	reference->add();
	cudaMalloc((void**)&dPtr, nItems * sizeof(T));
}

template <typename T>
CudaBuffer<T>::CudaBuffer(const std::vector<T>& hBuffer)
  : reference(NULL), dPtr(NULL), nItems(hBuffer.size())
{
	reference = new ReferenceCount;
	reference->add();
	cudaMalloc((void**)&dPtr, nItems * sizeof(T));
	cudaMemcpy(dPtr, &hBuffer[0], nItems * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
CudaBuffer<T>::CudaBuffer(const CudaBuffer& buffer)
  : reference(buffer.reference), dPtr(buffer.dPtr), nItems(buffer.nItems)
{
	reference->add();
}

template <typename T>
CudaBuffer<T>& CudaBuffer<T>::operator=(const CudaBuffer<T>& buffer)
{
	if (this != &buffer)
	{
		if (reference->release() == 0)
		{
			delete reference; reference = NULL;
			cudaFree(dPtr); dPtr = NULL;
			nItems = 0;
		}
		reference = buffer.reference;
		reference->add();
		dPtr = buffer.dPtr;
		nItems = buffer.nItems;
	}
	return *this;
}

template <typename T>
CudaBuffer<T>::~CudaBuffer()
{
	if (reference->release() == 0)
	{
		delete reference; reference = NULL;
		cudaFree(dPtr); dPtr = NULL;
	}
}

//
//
// Device ==> Host
//
//

template <typename T>
std::vector<T> CudaBuffer<T>::toHost() const
{
	std::vector<T> ret(nItems);
	cudaMemcpy(&ret[0], dPtr, nItems * sizeof(T), cudaMemcpyDeviceToHost);
	return ret;
}

//
//
// For Debugging
//
//

// template <typename T>
// std::ostream& operator<<(std::ostream& os, const CudaBuffer<T>& buffer)
// {
// 	std::vector<T> hBuffer = buffer.toHost();
// 	for (unsigned int i = 0; i < hBuffer.size() - 1; ++i)
// 		os << hBuffer[i] << ",";
// 	os << hBuffer[hBuffer.size() - 1];
// 	return os;
// }

//
//
// Template Initiate
//
//

template class CudaBuffer<int>;
template class CudaBuffer<float>;
template class CudaBuffer<float3>;
template class CudaBuffer<float2>;

} // namespace yy