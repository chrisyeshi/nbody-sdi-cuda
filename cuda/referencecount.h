#ifndef __yy_ReferenceCount_h__
#define __yy_ReferenceCount_h__

namespace yy
{

class ReferenceCount
{
public:
	ReferenceCount() : count(0) {}

	void add() { ++count; }
	int release() { return --count; }

protected:

private:
	int count;

	ReferenceCount(int);					// Not implemented!!!
	void operator=(const ReferenceCount&);	// Not implemented!!!
};

} // namespace yy

#endif // __yy_ReferenceCount_h__