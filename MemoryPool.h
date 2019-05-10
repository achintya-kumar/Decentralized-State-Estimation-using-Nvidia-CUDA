#pragma once

// Data Structure includes
#include <map>
#include <stack>
#include <vector>

typedef std::map<size_t, std::vector<void *>>  Map_Of_Pointer_vectors;
typedef unsigned long ulong;
class MemoryPool
{
public:
	Map_Of_Pointer_vectors map_of_device_pointer_vectors;								// Represents a map(pool) of device-pointer vectors of different memory size. SUBJECT TO THREAD-SYNC LATER
	Map_Of_Pointer_vectors::iterator map_iterator;										// Used to iterate over the map.  SUBJECT TO THREAD-SYNC LATER
	bool insertDeviceMemoryPointerOfSize(size_t size, void *device_pointer);			// Inserts a device-pointer of given memory size into the pool
	void* getDeviceMemoryPointerOfSize(size_t size);									// Returns a device-pointer of requested memory size from the pool
	void appendReleasedStepAllocations(size_t size,										// Appends a vectors of device-pointers of given size to the pool currently in use.
		std::vector<void *> *device_pointer_vector);
	MemoryPool();
	~MemoryPool();
};

