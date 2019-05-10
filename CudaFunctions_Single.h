#pragma once
#include <boost/algorithm/string.hpp>

#include <set>
#include <stdio.h>
#include <math.h>

#include "CudaFunctions_Base.h"
#include "DeviceMemoryManager.h"
#include "user_defined_kernels_fp32.h"
#include "DeviceMatrix.h"

class CudaFunctions_Single : public CudaFunctions_Base
{
protected:
	DeviceMatrix matrixElementWiseASMD_wrapper(DeviceMatrix a, DeviceMatrix b, unsigned int operation_choice, bool on_hold);
	DeviceMatrix matrixElementWiseScalarASMD_wrapper(DeviceMatrix device_matrix, float scalar, int operation_choice, int order_of_operation, bool on_hold);
	DeviceMatrix tdot(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold);
	DeviceMatrix matrixElementWiseSinOrCosOrAbs(DeviceMatrix device_matrix1, int choice_of_operation, int use_intrinsics, bool on_hold);
	DeviceMatrix complexMatrixExtraction(DeviceMatrix device_matrix, int operation_choice, bool on_hold);
	DeviceMatrix matrixEyeOrOnesOrZeros_wrapper(int width, int height, int operation_choice, bool on_hold);
	DeviceMatrix float_to_complex(DeviceMatrix device_matrix, bool on_hold);

public:
	CudaFunctions_Single() : CudaFunctions_Base() { printf("\nCreating CudaFunctions_Single...\n");  precision = 1; };
	~CudaFunctions_Single();
	int area(DeviceMatrix d, float scalar, bool sth = false);

	void* memAlloc(size_t size, bool on_hold = false);
	DeviceMatrix update_eK_fK(DeviceMatrix a, DeviceMatrix b);
	void problematicExit(char* message);
	double to_host(DeviceMatrix device_matrix);
	DeviceMatrix extract_indices_for_non_zero(DeviceMatrix device_matrix);
	DeviceMatrix to_device(double *array, int width, int height);
	DeviceMatrix to_device(std::string file);
	void to_DND_pool(DeviceMatrix device_matrix);
	void printMatrix(DeviceMatrix device_matrix);
	DeviceMatrix add(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false);
	DeviceMatrix sub(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false);
	DeviceMatrix mul(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false);
	DeviceMatrix div(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false);
	DeviceMatrix adds(DeviceMatrix device_matrix1, float scalar, bool on_hold = false);
	DeviceMatrix subs(DeviceMatrix device_matrix1, float scalar, int order_of_operation, bool on_hold = false);
	DeviceMatrix subs(DeviceMatrix device_matrix1, float scalar, bool on_hold = false);
	DeviceMatrix muls(DeviceMatrix device_matrix1, float scalar, bool on_hold = false);
	DeviceMatrix divs(DeviceMatrix device_matrix1, float scalar, int order_of_operation, bool on_hold = false);
	DeviceMatrix divs(DeviceMatrix device_matrix1, float scalar, bool on_hold = false);
	DeviceMatrix aTb(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false);
	DeviceMatrix aTbT(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false);
	DeviceMatrix dot(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false);
	DeviceMatrix abT(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false);
	DeviceMatrix complex(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false);
	DeviceMatrix sin(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix cos(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix real(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix imag(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix abs(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix sign(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix eye(int width, bool on_hold = false);
	DeviceMatrix ones(int rows, int columns, bool on_hold = false);
	DeviceMatrix zeros(int rows, int columns, bool on_hold = false);
	DeviceMatrix zerosInt(int width, int height, bool on_hold = false);
	DeviceMatrix diagflat(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix diagWithPower(DeviceMatrix device_matrix, int power, bool on_hold = false);
	DeviceMatrix conj(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix concatenate(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, char axis, bool on_hold = false);
	DeviceMatrix concat(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, char axis, bool on_hold = false);
	DeviceMatrix concat(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold = false);
	DeviceMatrix transpose(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix complexify(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix solve(DeviceMatrix device_matrix_1, DeviceMatrix device_matrix_2, bool on_hold = false);
	DeviceMatrix slice(DeviceMatrix device_matrix, int row_start, int row_end_exclusive, int column_start, int column_end_exclusive, bool on_hold = false);
	DeviceMatrix slicei(DeviceMatrix device_matrix, DeviceMatrix indices_device_matrix, bool on_hold = false);
	DeviceMatrix specialSlicingOnR(DeviceMatrix device_matrix, DeviceMatrix indices, bool on_hold = false);
	DeviceMatrix specialSlicingOnH(DeviceMatrix device_matrix, DeviceMatrix indices, bool on_hold = false);
	double maxValue(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix insert(DeviceMatrix input_big_matrix, int row_start, int row_end_exclusive, int column_start, int column_end_exclusive, DeviceMatrix input_small_matrix, bool on_hold = false);
	DeviceMatrix map_non_zero_elements(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix sort(DeviceMatrix device_matrix, bool on_hold = false);
	DeviceMatrix host_array_wrapped_in_DeviceMatrix(DeviceMatrix device_matrix);
	void setStream(cudaStream_t cudaStream);
	void deviceSynchronize();
	void write_to_file(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2);
	void stepBegin();
	void sanitizeMemoryPools();
	void releaseOnHoldAllocationsToPool();
	void insert_into_execution_times_map(std::string kernel_name, float milliseconds);
	void print_execution_times();
	DeviceMatrix angle(DeviceMatrix eK, DeviceMatrix fK, bool on_hold = false);
	DeviceMatrix wrapPointersIntoPointerArrays(DeviceMatrix device_matrix_1, bool on_hold = false);
	float getVectorElementAtIndex(DeviceMatrix device_matrix, int index);
};

