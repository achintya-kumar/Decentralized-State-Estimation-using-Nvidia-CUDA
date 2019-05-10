#include "CudaLibraryHandles.h"


cublasHandle_t CudaLibraryHandles::cublasHandle = nullptr;		// Cublas handle
cusolverDnHandle_t CudaLibraryHandles::cusolverHandle = nullptr;	// Cusolver handle


CudaLibraryHandles::CudaLibraryHandles()
{
}


CudaLibraryHandles::~CudaLibraryHandles()
{
}

void CudaLibraryHandles::initialize() {
	if (cublasHandle == nullptr) {
		checkCudaErrors(cublasCreate(&cublasHandle));
	}

	if (cusolverHandle == nullptr) {
		checkCudaErrors(cusolverDnCreate(&cusolverHandle));
	}
}