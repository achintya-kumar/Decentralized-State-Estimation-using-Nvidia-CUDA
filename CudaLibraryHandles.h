#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include <cusparse_v2.h>
#include <cusolverSp.h>
#include "helper_cuda.h"

class CudaLibraryHandles
{
public:
	CudaLibraryHandles();
	~CudaLibraryHandles();
	static cusolverDnHandle_t cusolverHandle;	// Cusolver handle
	static cublasHandle_t cublasHandle;		// Cublas handle
	static void initialize();
};

