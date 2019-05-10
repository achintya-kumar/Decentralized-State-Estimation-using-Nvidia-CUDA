#pragma once
#include "MemoryPool.h"
#include "DeviceMatrix.h"


enum LoopExecutionMode { Even, Odd };						// Every iteration within the State Estimation loop can be either odd or even.

class DeviceMemoryManager
{
private:
	MemoryPool per_step_pool;
	MemoryPool on_hold_pool;
	MemoryPool DND_memory_pool;								// Do-Not-Disturb Memory Pool. This is used for holding matrices on device-memory throughout the execution. Non-reusable.
	MemoryPool available_memory_pool;						// Memory pool to be used in even-iterations. Reusable.
	void memoryPoolCleanup(MemoryPool *memory_pool);		// Frees the device-allocations in the given memory pool.

public:
	size_t total_occupied_memory_in_bytes;
	void* getDndDeviceMemory(size_t size);					// Gets a DND device-memory allocation pointer.
	void* getDeviceMemory(size_t size,						// Gets a Re-usable device-memory allocation pointer.
		bool on_hold = false);
	void toDndDeviceMemory(DeviceMatrix *device_matrix);		// Moves a Re-usable device-memory allocation pointer to Non-reusable DND block.		
	void releaseStepAllocationsToPool();					// Releases device-pointers used during a step/operation to the pool for reuse.
	void releaseOnHoldAllocationsToPool();					// Releases device-pointers in ON-HOLD state to the pool for reuse.
	DeviceMemoryManager();
	~DeviceMemoryManager();
};

