#include "CudaFunctions_Base.h"


CudaFunctions_Base::CudaFunctions_Base() 
{
	//printf("\nCreating CudaFunctions...");
	checkCudaErrors(cusolverDnCreate(&cusolverHandle));
	checkCudaErrors(cublasCreate(&cublasHandle));
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
}


CudaFunctions_Base::~CudaFunctions_Base()
{
	printf("\nDestroying CudaFunctions...");
	// Destroying library handles.
	if (cusolverHandle) { checkCudaErrors(cusolverDnDestroy(cusolverHandle)); }
	if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
}

void CudaFunctions_Base::setGridSize(unsigned int grid_size) {
	this->grid_size = grid_size;
}

void CudaFunctions_Base::setGridData(GridDataSet grid_data) {
	this->dataset = grid_data;
}

void CudaFunctions_Base::setSEresults(StateEstimationResults results) {
	this->SEstate_vector = results;
}