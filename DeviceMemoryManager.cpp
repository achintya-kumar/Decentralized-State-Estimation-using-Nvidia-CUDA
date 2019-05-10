#include "DeviceMemoryManager.h"


// CUDA includes
#include <cuda_runtime.h>
#include <cuda.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"


DeviceMemoryManager::DeviceMemoryManager()
{
	total_occupied_memory_in_bytes = 0;
}


DeviceMemoryManager::~DeviceMemoryManager()
{
	/**** Device memory allocation cleanup logic ****/
	memoryPoolCleanup(&DND_memory_pool);
	memoryPoolCleanup(&available_memory_pool);
	printf("\nDestroying device manager...");
}


// Returns a non-reusable device-memory allocation pointer
void* DeviceMemoryManager::getDndDeviceMemory(size_t size) {
	void *DND_device_memory_pointer;
	cudaMalloc((void**)&DND_device_memory_pointer, size);											// Allocates on Device-memory
	total_occupied_memory_in_bytes += size;
	DND_memory_pool.insertDeviceMemoryPointerOfSize(size, DND_device_memory_pointer);				// Inserts into DND pool for cleanup operation at the end of execution

	return DND_device_memory_pointer;
}

// Returns a reusable device-memory allocation pointer
void* DeviceMemoryManager::getDeviceMemory(size_t size, bool on_hold) {
	void *device_memory_pointer;


	device_memory_pointer = available_memory_pool.getDeviceMemoryPointerOfSize(size);				// Get a device-pointer from the alternating_memory_pool_1
	if (device_memory_pointer == nullptr) {															// Chances are, the returned device-pointer may be a nullptr
		cudaMalloc((void**)&device_memory_pointer, size);											// If nullptr, do a new device allocation
		total_occupied_memory_in_bytes += size;
	}

	if (on_hold)																					// The user may want to hold on to certain results.
		on_hold_pool.insertDeviceMemoryPointerOfSize(size, device_memory_pointer);					// Those results can be held in ON_HOLD_POOL
	else																							// The user may not be interested in intermediate results.
		per_step_pool.insertDeviceMemoryPointerOfSize(size, device_memory_pointer);					// They can be held in PER_STEP_POOL.

	return device_memory_pointer;

}

// Performs cleanup (cudaFree) of the given MemoryPool instance.
void DeviceMemoryManager::memoryPoolCleanup(MemoryPool *memory_pool) {
	/* Every MemoryPool instance is a hashmap of pointer vectors. */
	Map_Of_Pointer_vectors *map = &(memory_pool->map_of_device_pointer_vectors);					// Aliasing the map name for convenience purposes. :)
	Map_Of_Pointer_vectors::iterator map_iterator = memory_pool->map_iterator;
	for (map_iterator = map->begin(); map_iterator != map->end(); ++map_iterator) {					// Looping through the map pair-entries.
		std::vector<void *> *device_pointer_vector = &(map_iterator->second);						// Retrieving vector at the current map_iterator position
		if (!device_pointer_vector->empty()) {														// Looping is required only if the vector of device-pointers is not empty. 
			size_t device_pointer_vector_size = device_pointer_vector->size();
			for (int i = 0; i < device_pointer_vector_size; i++) {									// Looping through the vector of device-pointers
				void *device_pointer = device_pointer_vector->back();								// Retrieving the device-pointer at the top position
				if (device_pointer) {
					cudaError_t error_status = cudaFree(device_pointer);										// Freeing CUDA device allocation of the device-pointer
					if (error_status != cudaSuccess)
						printf("Memory deallocation failed!");
				}
				device_pointer_vector->pop_back();													// Deleting the device-pointer entry from the top of the vector
			}
		}
	}
}

// Moves an existing device-pointer from reusable pool to DND pool.
void DeviceMemoryManager::toDndDeviceMemory(DeviceMatrix *device_matrix) {
	int size;
	if (device_matrix->dtype == Double)
		size = device_matrix->height * device_matrix->width * sizeof(double);													// Computing size when the matrix is Double
	else if (device_matrix->dtype == ComplexZ)
		size = device_matrix->height * device_matrix->width * sizeof(cuDoubleComplex);											// Computing size when the matrix is ComplexZ
	else if (device_matrix->dtype == Float)
		size = device_matrix->height * device_matrix->width * sizeof(float);													// Computing size when the matrix is Double
	else if (device_matrix->dtype == ComplexC)
		size = device_matrix->height * device_matrix->width * sizeof(cuComplex);
	else
		exit(1000000);

	std::vector<void *>::iterator iterator;
	std::vector<void *> *pointer_vector = &(per_step_pool.map_of_device_pointer_vectors.find(size)->second);					// Finding the pointer-vector of the given size
	iterator = find(pointer_vector->begin(), pointer_vector->end(), device_matrix->device_pointer);								// Locating the device-pointer on the found pointer-vector
	pointer_vector->erase(iterator);																							// Removing the device-pointer from the pointer-vector.

	DND_memory_pool.insertDeviceMemoryPointerOfSize(size, device_matrix->device_pointer);										// Now, insert the device-pointer to the DND memory pool.
	return;
}

void DeviceMemoryManager::releaseStepAllocationsToPool() {

	/* Every MemoryPool instance is a hashmap of pointer vectors. */
	Map_Of_Pointer_vectors *map = &(per_step_pool.map_of_device_pointer_vectors);					// Aliasing the map name for convenience purposes. :)
	Map_Of_Pointer_vectors::iterator map_iterator = per_step_pool.map_iterator;
	for (map_iterator = map->begin(); map_iterator != map->end(); ++map_iterator) {					// Looping through the map pair-entries.
		std::vector<void *> *device_pointer_vector = &(map_iterator->second);						// Retrieving vector at the current map_iterator position
		if (!device_pointer_vector->empty()) {														// Checking if the vector of device-pointers is not empty for given allocation-size
			available_memory_pool.appendReleasedStepAllocations(map_iterator->first,				// If not empty, release all the device-pointers inside 'per-step-allocation' pool to the GPU allocation-buffer.
				&(map_iterator->second));
			map_iterator->second.clear();															// Clearing the per-step allocation vector for given size, since the device-pointers are now moved to the GPU allocation-buffer.
		}
	}
}



void DeviceMemoryManager::releaseOnHoldAllocationsToPool() {

	/* Every MemoryPool instance is a hashmap of pointer vectors. */
	Map_Of_Pointer_vectors *map = &(on_hold_pool.map_of_device_pointer_vectors);					// Aliasing the map name for convenience purposes. :)
	Map_Of_Pointer_vectors::iterator map_iterator = on_hold_pool.map_iterator;
	for (map_iterator = map->begin(); map_iterator != map->end(); ++map_iterator) {					// Looping through the map pair-entries.
		std::vector<void *> *device_pointer_vector = &(map_iterator->second);						// Retrieving vector at the current map_iterator position
		if (!device_pointer_vector->empty()) {														// Checking if the vector of device-pointers is not empty for the given allocation-size.
			available_memory_pool.appendReleasedStepAllocations(map_iterator->first,				// If not empty, release all the device-pointers inside 'on-hold' pool to the GPU allocation-buffer.
				&(map_iterator->second));
			map_iterator->second.clear();															// Clearing the 'on-hold' allocation vector the given size, since the device-pointers are now moved to the GPU allocation-buffer.
		}
	}
}

