#ifndef __yy_CudaGLBuffer_h__
#define __yy_CudaGLBuffer_h__

#include <vector>
#include <iostream>
#include <GL/gl.h>
#include "referencecount.h"

namespace yy
{

template <typename T>
class CudaGLBuffer
{
public:
	//
	//
	// Constructor / Destructor
	//
	//
	CudaGLBuffer();
	CudaGLBuffer(GLuint bufId);
	CudaGLBuffer(const CudaGLBuffer& buffer);
	CudaGLBuffer<T>& operator=(const CudaGLBuffer<T>& buffer);
	~CudaGLBuffer();
	//
	//
	// Device ==> Host
	//
	//
	std::vector<T> toHost();
	//
	//
	// Accessors
	//
	//
	T* ptr() const;
	operator T *() const { return ptr(); }
	int length() const;
	int memory() const { return length() * sizeof(T); }
	void upload(const std::vector<T>& hBuffer);
	//
	//
	// GL Operations
	//
	//
	void map();
	void unmap();

protected:

private:
	//
	//
	// Member Variables
	//
	//
	ReferenceCount* reference;
	GLuint glBuffer;
	cudaGraphicsResource* resource;
};

} // namespace yy

#endif // __yy_CudaGLBuffer_h__