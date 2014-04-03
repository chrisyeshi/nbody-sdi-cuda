#include "cudaglbuffer.h"
#include "cuda_gl_interop.h"
#include <cassert>

namespace yy
{

//
//
// Constructor / Destructor
//
//

template <typename T>
CudaGLBuffer<T>::CudaGLBuffer()
  : reference(NULL), glBuffer(0), resource(NULL)
{
	reference = new ReferenceCount;
	reference->add();
}

template <typename T>
CudaGLBuffer<T>::CudaGLBuffer(GLuint bufId)
  : reference(NULL), glBuffer(0), resource(NULL)
{
	reference = new ReferenceCount;
	reference->add();
	glBuffer = bufId;
	cudaGraphicsGLRegisterBuffer(&resource, glBuffer, cudaGraphicsRegisterFlagsNone);
}

template <typename T>
CudaGLBuffer<T>::CudaGLBuffer(const CudaGLBuffer& buffer)
  : reference(buffer.reference), glBuffer(buffer.glBuffer), resource(buffer.resource)
{
	reference->add();
}

template <typename T>
CudaGLBuffer<T>& CudaGLBuffer<T>::operator=(const CudaGLBuffer<T>& buffer)
{
	if (this != &buffer)
	{
		if (reference->release() == 0)
		{
			this->unmap();
			delete reference; reference = NULL;
			glBuffer = 0;
			cudaGraphicsUnregisterResource(resource);
			resource = NULL;
		}
		reference = buffer.reference;
		reference->add();
		glBuffer = buffer.glBuffer;
		resource = buffer.resource;
	}
	return *this;
}

template <typename T>
CudaGLBuffer<T>::~CudaGLBuffer()
{
	if (reference->release() == 0)
	{
		unmap();
		delete reference; reference = NULL;
		glBuffer = 0;
		cudaGraphicsUnregisterResource(resource);
		resource = NULL;
	}
}

//
//
// Device ==> Host
//
//

template <typename T>
std::vector<T> CudaGLBuffer<T>::toHost()
{
	this->map();
	std::vector<T> ret(length());
	cudaMemcpy(&ret[0], this->ptr(), length() * sizeof(T), cudaMemcpyDeviceToHost);
	this->unmap();
	return ret;
}

//
//
// Accessors
//
//

template <typename T>
T* CudaGLBuffer<T>::ptr() const
{
	T* ptr = NULL;
	size_t size = 0;
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ptr), &size, resource);
	return ptr;
}

template <typename T>
int CudaGLBuffer<T>::length() const
{
	T* ptr = NULL;
	size_t size = 0;
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ptr), &size, resource);
	return size / sizeof(T);
}

template <typename T>
void CudaGLBuffer<T>::upload(const std::vector<T>& hBuffer)
{
	this->map();
	assert(hBuffer.size() == length());
	cudaMemcpy(ptr(), &hBuffer[0], length() * sizeof(T), cudaMemcpyHostToDevice);
	this->unmap();
}

//
//
// GL Operations
//
//

template <typename T>
void CudaGLBuffer<T>::map()
{
	assert(resource);
	cudaGraphicsMapResources(1, &resource);
}

template <typename T>
void CudaGLBuffer<T>::unmap()
{
	cudaGraphicsUnmapResources(1, &resource);
}

//
//
// Template Initiate
//
//

template class CudaGLBuffer<int>;
template class CudaGLBuffer<float>;
template class CudaGLBuffer<float3>;
template class CudaGLBuffer<float2>;

} // namespace yy