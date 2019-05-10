
#include "MemoryPool.h"

MemoryPool::MemoryPool()
{
}


MemoryPool::~MemoryPool()
{
}

// Inserts a device-pointer to the pool.
bool MemoryPool::insertDeviceMemoryPointerOfSize(size_t size, void *device_pointer) {		// Could as well return void. Just to make sure things went well, I'm returning a bool.
	Map_Of_Pointer_vectors *map = &map_of_device_pointer_vectors;									// Aliasing the map name for convenience purposes. :)
	bool insertion_success_status = true;
	try {
		/**** Signature of memory pool: map<size_t, stack<void *>> ****/
		map_iterator = map->find(size);																// Retrieving map-entry (device-pointer vector) for the given size (key)
		if (map_iterator != map->end()) {															// In case a vector of device-pointers for the given size exists, do the following:								
			map_iterator->second.push_back(device_pointer);											// Pushing the device-pointer into the found vector.
		}
		else {
			std::vector<void *> new_device_pointer_vector;											// The vector may not exist. In that case, create a new vector!
			new_device_pointer_vector.push_back(device_pointer);									// Loading up the new vector with given device-pointer
			map->insert(std::pair <size_t, std::vector<void *>>(size, new_device_pointer_vector));		// Load up the new vector into the pool with key=size
		}
	}
	catch (std::exception& e) {
		printf("%d", e);
		insertion_success_status = false;															// In case something goes wrong during runtime, that will be caught in DEBUG MODE.
	}

	return insertion_success_status;
}


// Retrieves a device-pointer from the pool.
void* MemoryPool::getDeviceMemoryPointerOfSize(size_t size) {

	void* result_device_memory_pointer = nullptr;													// The required pointer may not exist. Starting off pessimistically.
	try {
		map_iterator = map_of_device_pointer_vectors.find(size);										// Finding the device-pointer stack for given size.
		if (map_iterator != map_of_device_pointer_vectors.end())	{									// If the device-pointer stack exists, do the following:
			if (map_iterator->second.empty() == false) {											// If the found device-pointer stack is not empty, do the following:
				result_device_memory_pointer = map_iterator->second.back();							// Retrieving the pointer at the top of device-pointer stack.
				map_iterator->second.pop_back();								// Since the top of the stack is retrieved, the stack must be popped.
			}
		}
	}
	catch (std::exception& e) {
		result_device_memory_pointer = nullptr;														// In case something goes wrong during runtime, that will be caught in DEBUG MODE.
	}

	return result_device_memory_pointer;
}

void MemoryPool::appendReleasedStepAllocations(size_t size, std::vector<void *> *per_step_allocations) {
	Map_Of_Pointer_vectors *map = &map_of_device_pointer_vectors;
	map_iterator = map_of_device_pointer_vectors.find(size);
	if (map_iterator != map->end()) {
		map_iterator->second.insert(map_iterator->second.end(), per_step_allocations->begin(), per_step_allocations->end());
	}
	else {
		std::vector<void *> new_device_pointer_vector;											// The vector may not exist. In that case, create a new vector!
		new_device_pointer_vector.insert(new_device_pointer_vector.end(), per_step_allocations->begin(), per_step_allocations->end());
		map->insert(std::pair <size_t, std::vector<void *>>(size, new_device_pointer_vector));		// Load up the new vector into the pool with key=size
	}
}