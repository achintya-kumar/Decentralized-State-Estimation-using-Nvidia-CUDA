#pragma once
#include <boost/algorithm/string.hpp>

#include <set>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include <cusparse_v2.h>
#include <cusolverSp.h>
#include "helper_cuda.h"

#include "DeviceMemoryManager.h"
#include "user_defined_kernels_fp64.h"
#include "DeviceMatrix.h"
#include "CudaLibraryHandles.h"

// Type-definitions for convenience
typedef std::map<std::string, float>  Map_Of_Execution_Times;

// Bundle to contain the results of SE
struct StateEstimationResults {
	DeviceMatrix voltages;
	DeviceMatrix phases;
	double timeTakenInSeconds;
};

// Bundle to contain data to feed into SE algorithm
struct GridDataSet {
	DeviceMatrix KKT;
	DeviceMatrix KKT_t;
	DeviceMatrix YT;
	DeviceMatrix YKK;
	DeviceMatrix INIT;
	DeviceMatrix MEAS;
	int grid_size;
};

class CudaFunctions_Base
{
protected:
	unsigned int grid_size;
	cudaEvent_t start, stop;
	cudaStream_t stream = 0;			// This value of stream will decide the execution lane on which the computation kernels are launched.
	virtual DeviceMatrix matrixElementWiseASMD_wrapper(DeviceMatrix a, DeviceMatrix b, unsigned int operation_choice, bool on_hold) = 0;
	virtual DeviceMatrix matrixElementWiseScalarASMD_wrapper(DeviceMatrix device_matrix, float scalar, int operation_choice, int order_of_operation, bool on_hold) = 0;
	virtual DeviceMatrix tdot(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) = 0;
	virtual DeviceMatrix matrixElementWiseSinOrCosOrAbs(DeviceMatrix device_matrix1, int choice_of_operation, int use_intrinsics, bool on_hold) = 0;
	virtual DeviceMatrix complexMatrixExtraction(DeviceMatrix device_matrix, int operation_choice, bool on_hold) = 0;
	virtual DeviceMatrix matrixEyeOrOnesOrZeros_wrapper(int width, int height, int operation_choice, bool on_hold) = 0;


public:
	CudaFunctions_Base();
	~CudaFunctions_Base();

	int precision;
	Map_Of_Execution_Times map_of_execution_times;
	DeviceMemoryManager dMManager;		// Device Memory Manager is responsible to device memory allocations of all sorts.
	cusolverDnHandle_t cusolverHandle;	// Cusolver handle
	cublasHandle_t cublasHandle;		// Cublas handle
	GridDataSet dataset;
	StateEstimationResults SEstate_vector;

	/* Non-virtual functions */
	void setGridSize(unsigned int grid_size);

	/* All the functions below are pure virtual functions and must be defined by the extending classes. */
	void setGridData(GridDataSet grid_data);
	void setSEresults(StateEstimationResults results);
	virtual void* memAlloc(size_t size, bool on_hold = false) = 0;
	virtual DeviceMatrix update_eK_fK(DeviceMatrix a, DeviceMatrix b) = 0;
	virtual void problematicExit(char* message) = 0;
	virtual double to_host(DeviceMatrix device_matrix) = 0;
	virtual DeviceMatrix extract_indices_for_non_zero(DeviceMatrix device_matrix) = 0;
	virtual DeviceMatrix to_device(double *array, int width, int height) = 0;
	virtual DeviceMatrix to_device(std::string file) = 0;
	virtual void to_DND_pool(DeviceMatrix device_matrix) = 0;
	virtual void printMatrix(DeviceMatrix device_matrix) = 0;
	virtual DeviceMatrix add(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false) = 0;
	virtual DeviceMatrix sub(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false) = 0;
	virtual DeviceMatrix mul(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false) = 0;
	virtual DeviceMatrix div(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false) = 0;
	virtual DeviceMatrix adds(DeviceMatrix device_matrix1, float scalar, bool on_hold = false) = 0;
	virtual DeviceMatrix subs(DeviceMatrix device_matrix1, float scalar, int order_of_operation, bool on_hold = false) = 0;
	virtual DeviceMatrix subs(DeviceMatrix device_matrix1, float scalar, bool on_hold = false) = 0;
	virtual DeviceMatrix muls(DeviceMatrix device_matrix1, float scalar, bool on_hold = false) = 0;
	virtual DeviceMatrix divs(DeviceMatrix device_matrix1, float scalar, int order_of_operation, bool on_hold = false) = 0;
	virtual DeviceMatrix divs(DeviceMatrix device_matrix1, float scalar, bool on_hold = false) = 0;
	virtual DeviceMatrix aTb(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false) = 0;
	virtual DeviceMatrix aTbT(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false) = 0;
	virtual DeviceMatrix dot(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false) = 0;
	virtual DeviceMatrix abT(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false) = 0;
	virtual DeviceMatrix complex(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false) = 0;
	virtual DeviceMatrix sin(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix cos(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix real(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix imag(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix abs(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix sign(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix eye(int width, bool on_hold = false) = 0;
	virtual DeviceMatrix ones(int rows, int columns, bool on_hold = false) = 0;
	virtual DeviceMatrix zeros(int rows, int columns, bool on_hold = false) = 0;
	virtual DeviceMatrix zerosInt(int width, int height, bool on_hold = false) = 0;
	virtual DeviceMatrix diagflat(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix diagWithPower(DeviceMatrix device_matrix, int power, bool on_hold = false) = 0;
	virtual DeviceMatrix conj(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix concatenate(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, char axis, bool on_hold = false) = 0;
	virtual DeviceMatrix concat(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, char axis, bool on_hold = false) = 0;
	virtual DeviceMatrix concat(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false) = 0;
	virtual DeviceMatrix transpose(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix complexify(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix solve(DeviceMatrix device_matrix_1, DeviceMatrix device_matrix_2, bool on_hold = false) = 0;
	virtual DeviceMatrix slice(DeviceMatrix device_matrix, int row_start, int row_end_exclusive, int column_start, int column_end_exclusive, bool on_hold = false) = 0;
	virtual DeviceMatrix slicei(DeviceMatrix device_matrix, DeviceMatrix indices_device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix specialSlicingOnR(DeviceMatrix device_matrix, DeviceMatrix indices, bool on_hold = false) = 0;
	virtual DeviceMatrix specialSlicingOnH(DeviceMatrix device_matrix, DeviceMatrix indices, bool on_hold = false) = 0;
	virtual double maxValue(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix insert(DeviceMatrix input_big_matrix, int row_start, int row_end_exclusive, int column_start, int column_end_exclusive, DeviceMatrix input_small_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix map_non_zero_elements(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix sort(DeviceMatrix device_matrix, bool on_hold = false) = 0;
	virtual DeviceMatrix host_array_wrapped_in_DeviceMatrix(DeviceMatrix device_matrix) = 0;
	virtual void setStream(cudaStream_t cudaStream) = 0;
	virtual void deviceSynchronize() = 0;
	virtual void write_to_file(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2) = 0;
	virtual void stepBegin() = 0;
	virtual void sanitizeMemoryPools() = 0;
	virtual void releaseOnHoldAllocationsToPool() = 0;
	virtual void insert_into_execution_times_map(std::string kernel_name, float milliseconds) = 0;
	virtual void print_execution_times() = 0;
	virtual DeviceMatrix angle(DeviceMatrix eK, DeviceMatrix fK, bool on_hold = false) = 0;
	virtual DeviceMatrix wrapPointersIntoPointerArrays(DeviceMatrix device_matrix_1, bool on_hold = false) = 0;
	virtual float getVectorElementAtIndex(DeviceMatrix device_matrix, int index) = 0;
};

