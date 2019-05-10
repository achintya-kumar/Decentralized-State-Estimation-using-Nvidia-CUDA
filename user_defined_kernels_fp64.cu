// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

//	CUDA's Complex number support header
#include <cuComplex.h>

// Thrust includes
#include <thrust/sort.h>
#include <thrust/device_vector.h>

void sort_on_device(double* h_vec, int width) {
	
	// wrap raw pointer with a device_ptr 
	thrust::device_ptr<double> d_vec = thrust::device_pointer_cast(h_vec);

	// sort data on the device
	thrust::sort(d_vec, d_vec + width);
}


/*void sort_on_device(thrust::host_vector<int>& h_vec)
{
	// transfer data to the device
	thrust::device_vector<int> d_vec = h_vec;

	// sort data on the device
	thrust::sort(d_vec.begin(), d_vec.end());

	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}*/

/*
This kernel locates the index of all the non-zero elements and returns their count.
*/
__global__ void countNonZeroElements(double *idata, int *count, int width, int height)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		if (idata[global_1D_index] != 0.0) { atomicAdd(count, 1); }

	}
}

/*====================================================================================================================*/
/*====================================================================================================================*/

/*
This kernel extracts the indices of all the non-zero elements.
*/
__global__ void filter_k(double *dst, int *nres, double *src, int width, int height) {

	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Calculating global 1D index for accessing the matrix in row-major fashion
	int global_1D_index = global_2D_x + global_2D_y * width;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height && src[global_1D_index] != 0.0) {
		dst[atomicAdd(nres, 1)] = global_1D_index;
	}
}

/*====================================================================================================================*/
/*====================================================================================================================*/

/*
This kernel maps indices with non-zero elements to the output matrix and the rest as -1.
*/
__global__ void mapNonZeroIndices(double *idata, double *odata, int width, int height)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		int isZero = (idata[global_1D_index] == 0.0);												// Converting two-way if-block to one liner.
		odata[global_1D_index] = global_1D_index * (!isZero)										// If the value at this index is non-zero, the index will get mapped.
			+ (-1) * isZero;										// Else, the mapped index will be -1, which indicates its invalidity.
	}
}

/*====================================================================================================================*/
/*====================================================================================================================*/

/*
This kernel handles complex matrix construction from given float matrices.
*/
__global__ void complexMatrixConstruction(double *idata1, double *idata2, cuDoubleComplex *odata, int width, int height)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;
		odata[global_1D_index] = make_cuDoubleComplex(idata1[global_1D_index], idata2[global_1D_index]);
	}
}

/*====================================================================================================================*/
/*====================================================================================================================*/

/*
This kernel handles matrix element-wise ASMD (Add, Subtract, Multiply or Divide) operations.
0 = Addition
1 = Subtraction
2 = Multiplication
3 = Division
*/
__global__ void matrixElementWiseASMD(double const *idata1, double const *idata2, double *odata, int width, int height,
	int operationChoice)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Depending on the operation choice, it either adds or subtracts
		if (operationChoice == 0)
			odata[global_1D_index] = idata1[global_1D_index] + idata2[global_1D_index];
		else if (operationChoice == 1)
			odata[global_1D_index] = idata1[global_1D_index] - idata2[global_1D_index];
		else if (operationChoice == 2)
			odata[global_1D_index] = idata1[global_1D_index] * idata2[global_1D_index];
		else if (operationChoice == 3)
			odata[global_1D_index] = idata1[global_1D_index] / idata2[global_1D_index];

	}
}

/*********************************************************************************************************************/
/*====================================================================================================================*/

/*
This kernel handles matrix element-wise ASMD (Add, Subtract, Multiply or Divide) operations when the latter of the
matrices is a single row matrix.
0 = Addition
1 = Subtraction
2 = Multiplication
3 = Division
*/
__global__ void matrixElementWiseASMDForSingleRow(double const *idata1, double const *idata2, double *odata, int width, int height,
	int operationChoice)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Depending on the operation choice, it either adds or subtracts
		// The latter of the matrices only has a single row. Hence, (global_2D_y * width) = 0
		if (operationChoice == 0)
			odata[global_1D_index] = idata1[global_1D_index] + idata2[global_2D_x];
		else if (operationChoice == 1)
			odata[global_1D_index] = idata1[global_1D_index] - idata2[global_2D_x];
		else if (operationChoice == 2)
			odata[global_1D_index] = idata1[global_1D_index] * idata2[global_2D_x];
		else if (operationChoice == 3)
			odata[global_1D_index] = idata1[global_1D_index] / idata2[global_2D_x];

	}
}


/*********************************************************************************************************************/
/*====================================================================================================================*/


/*
This kernel handles COMPLEX matrix element-wise ASMD (Add, Subtract, Multiply or Divide) operations.
0 = Addition
1 = Subtraction
2 = Multiplication
3 = Division
*/

__global__ void matrixComplexElementWiseASMD(cuDoubleComplex const *idata1, cuDoubleComplex const *idata2, cuDoubleComplex *odata,
	int width, int height, int operationChoice)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Depending on the operation choice, it either adds or subtracts
		if (operationChoice == 0)
			odata[global_1D_index] = cuCadd(idata1[global_1D_index], idata2[global_1D_index]);
		else if (operationChoice == 1)
			odata[global_1D_index] = cuCsub(idata1[global_1D_index], idata2[global_1D_index]);
		else if (operationChoice == 2)
			odata[global_1D_index] = cuCmul(idata1[global_1D_index], idata2[global_1D_index]);
		else if (operationChoice == 3)
			odata[global_1D_index] = cuCdiv(idata1[global_1D_index], idata2[global_1D_index]);

	}
}


/*********************************************************************************************************************/
/*********************************************************************************************************************/



/*
This kernel handles matrix element-wise scalar ASMD (Add, Subtract, Multiply or Divide) operations.
For example, adding a constant to all the elements of a matrix.
0 = Addition
1 = Subtraction
2 = Multiplication
3 = Division

Addition and Multiplication are commutative operations.

For Subtraction and Division in reverse order, commutative property does not hold.
Hence, the kernel provides for an order_of_operation argument to choose which value comes first.
For order_of_operation = 0, the corresponding matrix element comes first followed by the constant/scalar.
For order_of_operation = 1, the constant/scalar comes first followed by the matrix element.
*/
__global__ void matrixElementWiseScalarASMD(double *idata, float constant, double *odata, int width, int height,
	int choice_of_operation, int order_of_operation)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Depending on the operation choice, it either adds or subtracts.
		// In case I wonder later, this won't lead to thread divergence because all the threads
		// will go to a single branch of the following control flow block during runtime.
		if (choice_of_operation == 0)
			odata[global_1D_index] = idata[global_1D_index] + constant;
		else if (choice_of_operation == 1 && order_of_operation == 0) // Subtraction in regular order
			odata[global_1D_index] = idata[global_1D_index] - constant;
		else if (choice_of_operation == 1 && order_of_operation == 1) // Subtraction in reversed order
			odata[global_1D_index] = constant - idata[global_1D_index];
		else if (choice_of_operation == 2)
			odata[global_1D_index] = idata[global_1D_index] * constant;
		else if (choice_of_operation == 3 && order_of_operation == 0) // Division in regular order
			odata[global_1D_index] = idata[global_1D_index] / constant;
		else if (choice_of_operation == 3 && order_of_operation == 1) // Division in reversed order
			odata[global_1D_index] = constant / idata[global_1D_index];

	}
}


/*********************************************************************************************************************/
/*********************************************************************************************************************/



/*
This kernel handles complex matrix element-wise scalar ASMD (Add, Subtract, Multiply or Divide) operations.
For example, adding a float scalar or complex scalar to all the elements of a matrix.
0 = Addition
1 = Subtraction
2 = Multiplication
3 = Division

Addition and Multiplication are commutative operations.

For Subtraction and Division in reverse order, commutative property does not hold.
Hence, the kernel provides for an order_of_operation argument to choose which value comes first.
For order_of_operation = 0, the corresponding matrix element comes first followed by the scalar/scalar.
For order_of_operation = 1, the scalar comes first followed by the matrix element.
*/

__global__ void matrixComplexElementWiseScalarASMD(cuDoubleComplex *idata, float scalar, cuDoubleComplex *odata,
	int width, int height, int choice_of_operation, int order_of_operation)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Depending on the operation choice, it either adds or subtracts.
		// will go to a single branch of the following control flow block during runtime.
		if (choice_of_operation == 0)
			odata[global_1D_index] = cuCadd(idata[global_1D_index], make_cuDoubleComplex(scalar, 0));
		else if (choice_of_operation == 1 && order_of_operation == 0) // Subtraction in regular order
			odata[global_1D_index] = cuCsub(idata[global_1D_index], make_cuDoubleComplex(scalar, 0));
		else if (choice_of_operation == 1 && order_of_operation == 1) // Subtraction in reversed order
			odata[global_1D_index] = cuCsub(make_cuDoubleComplex(scalar, 0), idata[global_1D_index]);
		else if (choice_of_operation == 2)
			odata[global_1D_index] = cuCmul(idata[global_1D_index], make_cuDoubleComplex(scalar, 0));
		else if (choice_of_operation == 3 && order_of_operation == 0) // Division in regular order
			odata[global_1D_index] = cuCdiv(idata[global_1D_index], make_cuDoubleComplex(scalar, 0));
		else if (choice_of_operation == 3 && order_of_operation == 1) // Division in reversed order
			odata[global_1D_index] = cuCdiv(make_cuDoubleComplex(scalar, 0), idata[global_1D_index]);
	}
}


/*********************************************************************************************************************/
/*********************************************************************************************************************/

/*
This kernel handles matrix element-wise operations such as Numpy's sin and cos.
0 = Real part extraction
1 = Imaginary part extraction
2 = Absolute value for the given element
*/
__global__ void matrixElementWiseSinOrCosOrAbsForDoubles(double *idata, double *odata,
	int width, int height, int operationChoice, int use_intrinsics)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Depending on the operation choice, it either adds or subtracts
		if (operationChoice == 0) // Sin operation
			odata[global_1D_index] = sin(idata[global_1D_index]);
		else if (operationChoice == 1) // Cos operation
			odata[global_1D_index] = cos(idata[global_1D_index]);
		else if (operationChoice == 2)
			odata[global_1D_index] = fabs(idata[global_1D_index]);
	}
}


/*********************************************************************************************************************/
/*********************************************************************************************************************/

/*
This kernel computes angles from given eK and fK matrices.
*/
__global__ void matrixElementWiseAngles(double *eK, double *fK, double *odata,
	int width, int height)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Depending on the operation choice, it either adds or subtracts
		odata[global_1D_index] = atan2(fK[global_1D_index], eK[global_1D_index]) * 180/3.1415926536 ;
	}
}


/*********************************************************************************************************************/
/*********************************************************************************************************************/


/*
This kernel handles COMPLEX matrix to float64 element-wise operations (Numpy's real, imag, abs).
0 = Real part extraction
1 = Imaginary part extraction
2 = Absolute value for each element
3 = Sign of the Absolute Value for each element
*/
__global__ void matrixComplexElementWiseExtractions(cuDoubleComplex *idata1, double *odata,
	int width, int height, int operationChoice)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Depending on the operation choice, it either adds or subtracts
		if (operationChoice == 0)
			odata[global_1D_index] = cuCreal(idata1[global_1D_index]);
		else if (operationChoice == 1)
			odata[global_1D_index] = cuCimag(idata1[global_1D_index]);
		else if (operationChoice == 2)
			odata[global_1D_index] = cuCabs(idata1[global_1D_index]);
		else if (operationChoice == 3)
			odata[global_1D_index] = 1 * (cuCabs(idata1[global_1D_index]) > 0);

	}
}



/*********************************************************************************************************************/
/*********************************************************************************************************************/



/*
This kernel handles COMPLEX matrix conjugate operations.
*/
__global__ void matrixComplexConjugate(cuDoubleComplex *idata1, cuDoubleComplex *odata,
	int width, int height)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		odata[global_1D_index] = cuConj(idata1[global_1D_index]);

	}
}



/*********************************************************************************************************************/
/*********************************************************************************************************************/



/*
This kernel handles on-device matrix typecasting from float to double.
*/
__global__ void floatToDouble(float *idata, double *odata, int width, int height)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		odata[global_1D_index] = (double)(idata[global_1D_index]);

	}
}


/*********************************************************************************************************************/
/*********************************************************************************************************************/



/*
This kernel handles on-device matrix typecasting from double to float.
*/
__global__ void doubleToFloat(double *idata, float *odata, int width, int height)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;
		odata[global_1D_index] = (float)(idata[global_1D_index]);

	}
}



/*********************************************************************************************************************/
/*********************************************************************************************************************/



/*
This kernel handles on-device matrix typecasting from double to complex!
*/
__global__ void doubleToComplex(double *idata, cuDoubleComplex *odata, int width, int height)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		odata[global_1D_index] = make_cuDoubleComplex(idata[global_1D_index], 0);

	}
}


/*====================================================================================================================*/
/*====================================================================================================================*/

/*
On-device function to find the index of a given position on the indices array
*/
__device__ int isPresent(int position, double *indices, int lengthOfIndices)
{
	for (int i = 0; i < lengthOfIndices; i++) {
		if (position == (int)indices[i]) {
			return i;
		}
	}
	return -1;
}


/*====================================================================================================================*/
/*====================================================================================================================*/


// General purpose double slicing kernel.
__global__ void sliceDouble(double *input,
	double *output,
	const int row_start,
	const int row_end,
	const int column_start,
	const int column_end,
	const int input_matrix_width,
	const int output_matrix_width)
{
	int global_index_X = blockIdx.x * blockDim.x + threadIdx.x;
	int global_index_Y = blockIdx.y * blockDim.y + threadIdx.y;

	int row = global_index_Y;
	int column = global_index_X;

	if (row >= row_start && row < row_end && column >= column_start && column < column_end) {
		output[(global_index_X - column_start) + (global_index_Y - row_start) * output_matrix_width]
			= input[global_index_X + global_index_Y * input_matrix_width];
	}
}

// General purpose complex128 slicing kernel.
__global__ void sliceComplex128(cuDoubleComplex *input,
	cuDoubleComplex *output,
	const int row_start,
	const int row_end,
	const int column_start,
	const int column_end,
	const int input_matrix_width,
	const int output_matrix_width)
{
	int global_index_X = blockIdx.x * blockDim.x + threadIdx.x;
	int global_index_Y = blockIdx.y * blockDim.y + threadIdx.y;

	int row = global_index_Y;
	int column = global_index_X;

	if (row >= row_start && row < row_end && column >= column_start && column < column_end) {
		output[(global_index_X - column_start) + (global_index_Y - row_start) * output_matrix_width]
			= input[global_index_X + global_index_Y * input_matrix_width];
	}
}


/*====================================================================================================================*/
/*====================================================================================================================*/


// This kernel can be used to perform slicing of matrices with the help of given 1D indices matrix.
__global__ void slice_with_indices(double *input,
	double *output,
	double *indices,
	const int input_matrix_height,
	const int input_matrix_width,
	const int indices_matrix_length)
{
	int global_index_X = blockIdx.x * blockDim.x + threadIdx.x;
	int global_index_Y = blockIdx.y * blockDim.y + threadIdx.y;


	if (global_index_X < indices_matrix_length && global_index_Y < 1) {
		int value_at_current_index_in_indices = (int)indices[global_index_X];
		output[global_index_X] = input[value_at_current_index_in_indices];
		/*for (int i = 0; i < input_matrix_width; i++) {
			double value_at_current_index_in_input = input[i + value_at_current_index_in_indices * input_matrix_width];
			output[i + global_index_X * input_matrix_width] = value_at_current_index_in_input;
		}*/
	}
}


/*====================================================================================================================*/
/*====================================================================================================================*/

/*
This kernel performs special cutting of the R matrix.
*/
__global__ void specialSlicingOnR_kernel(double *input,
	double *output,
	double *indices,
	int input_matrix_height,
	int input_matrix_width,
	int indices_matrix_length)
{
	int global_index_X = blockIdx.x * blockDim.x + threadIdx.x;
	int global_index_Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (global_index_X < indices_matrix_length && global_index_Y < 1) {
		int value_at_current_index_in_indices = (int)indices[global_index_X];
		double value_at_current_index_in_input = pow(input[value_at_current_index_in_indices], 2);
		output[global_index_X] = value_at_current_index_in_input;
	}
}


/*====================================================================================================================*/
/*====================================================================================================================*/


/*
This kernel performs special cutting of the H matrix.
*/
__global__ void specialSlicingOnH_kernel(double *input,
	double *output,
	double *indices,
	int input_matrix_height,
	int input_matrix_width,
	int indices_matrix_length,
	int output_matrix_width)
{
	int global_index_X = blockIdx.x * blockDim.x + threadIdx.x;
	int global_index_Y = blockIdx.y * blockDim.y + threadIdx.y;


	if (global_index_X < indices_matrix_length && global_index_Y < 1) {
		int value_at_current_index_in_indices = (int)indices[global_index_X];

		for (int i = 0; i < input_matrix_width; i++) {
			double value_at_current_index_in_input = input[i + value_at_current_index_in_indices * input_matrix_width];
			output[i + global_index_X * input_matrix_width] = value_at_current_index_in_input;
		}
	}
}


/*====================================================================================================================*/
/*====================================================================================================================*/


__global__ void matrix_insert(double *input,
	double *matrix_to_insert,
	const int row_start,
	const int row_end,
	const int column_start,
	const int column_end,
	const int input_matrix_height,
	const int input_matrix_width)
{
	int global_index_X = blockIdx.x * blockDim.x + threadIdx.x;
	int global_index_Y = blockIdx.y * blockDim.y + threadIdx.y;

	int row = global_index_Y;
	int column = global_index_X;
	int matrix_to_insert_width = column_end - column_start;

	// Attempting replacement of IF-condition
	/*int value_to_insert = matrix_to_insert[(global_index_X - row_start) + (global_index_Y - column_start) * matrix_to_insert_width];
	int original_value = input[global_index_X + global_index_Y * input_matrix_width];
	input[global_index_X + global_index_Y * input_matrix_width] = (value_to_insert * insert_or_not) + (original_value * (!insert_or_not));*/

	if (row >= row_start && row < row_end && column >= column_start && column < column_end) {
		double value_to_insert = matrix_to_insert[(global_index_X - column_start) + (global_index_Y - row_start) * matrix_to_insert_width];
		input[global_index_X + global_index_Y * input_matrix_width] = value_to_insert;
	}
}

/*====================================================================================================================*/
/*====================================================================================================================*/


__global__ void transposeNaive(double *odata, double* idata, int width, int height)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if(xIndex >= width || yIndex >= height)
        return;

    int index_in = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;
    odata[index_out] = idata[index_in];
}

/*====================================================================================================================*/
/*====================================================================================================================*/

/*
This kernel handles matrix concatenation (horizontal or vertical) operations.

operationChoice:
0 = Horizontal concatenation
1 = Vertical concatenation
*/
__global__ void matrixConcatenateDouble(double *idata1, double *idata2, double *odata, int width, int height,
	int width1, int height1, int width2, int operationChoice)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {
		// Calculating global 1D index for accessing the target matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;
		if (operationChoice == 0) {
			if (global_2D_x < width1) {
				// Computing 1D index for the first matrix
				int idata1_1D_index = global_2D_x + global_2D_y * width1;
				// Storing the first matrix's elements into the concatenated matrix
				odata[global_1D_index] = idata1[idata1_1D_index];
			}
			else {
				// Calculating global 1D index for accessing the second matrix in row-major fashion
				int idata2_1D_index = (global_2D_x - width1) + global_2D_y * width2;
				// Storing the first matrix's elements into the concatenated matrix
				odata[global_1D_index] = idata2[idata2_1D_index];
			}
		}
		else {
			if (global_2D_y < height1) {
				// Computing 1D index for the first matrix
				int idata1_1D_index = global_2D_x + global_2D_y * width1;
				// Storing the first matrix's elements into the concatenated matrix
				odata[global_1D_index] = idata1[idata1_1D_index];
			}
			else {
				// Calculating global 1D index for accessing the second matrix in row-major fashion
				int idata2_1D_index = global_2D_x + (global_2D_y - height1) * width2;
				// Storing the first matrix's elements into the concatenated matrix
				odata[global_1D_index] = idata2[idata2_1D_index];
			}
		}


	}
}

__global__ void matrixConcatenateDouble2(double *idata1, double *idata2, double *odata, int width, int height,
	int width1, int height1, int width2, int height2, int operationChoice, int isFirstMatrix)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (isFirstMatrix) {
		if (global_2D_x < width1 && global_2D_y < height1) {
			int global_1D_index_source = global_2D_x + global_2D_y * width1;
			int global_1D_index_target = global_2D_x + global_2D_y * width;
			// Storing the first matrix's elements into the concatenated matrix
			odata[global_1D_index_target] = idata1[global_1D_index_source];
		}
	}
	else if (!isFirstMatrix) {
		if (global_2D_x < width2 && global_2D_y < height2) {
			int offset_1D_index_source, offset_1D_index_target;
			if (operationChoice == 0) {
				// Computing 1D index with x-offset=width1
				offset_1D_index_source = (global_2D_x)+global_2D_y * width2;
				offset_1D_index_target = (global_2D_x + width1) + global_2D_y * width;
			}
			else if (operationChoice == 1) {
				// Computing 1D index with x-offset=width1
				offset_1D_index_source = global_2D_x + global_2D_y * width2;
				offset_1D_index_target = global_2D_x + (global_2D_y + height1) * width;
			}
			odata[offset_1D_index_target] = idata2[offset_1D_index_source];
		}
	}
}
/*********************************************************************************************************************/
/*********************************************************************************************************************/



/*
This kernel handles matrix concatenation (horizontal or vertical) operations.

operationChoice:
0 = Horizontal concatenation
1 = Vertical concatenation
*/

__global__ void matrixConcatenateComplex(cuDoubleComplex *idata1, cuDoubleComplex *idata2, cuDoubleComplex *odata,
	int width, int height, int width1, int height1, int width2, int operationChoice)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {
		// Calculating global 1D index for accessing the target matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;
		if (operationChoice == 0) {
			if (global_2D_x < width1) {
				// Computing 1D index for the first matrix
				int idata1_1D_index = global_2D_x + global_2D_y * width1;
				// Storing the first matrix's elements into the concatenated matrix
				odata[global_1D_index] = idata1[idata1_1D_index];
			}
			else {
				// Calculating global 1D index for accessing the second matrix in row-major fashion
				int idata2_1D_index = (global_2D_x - width1) + global_2D_y * width2;
				// Storing the first matrix's elements into the concatenated matrix
				odata[global_1D_index] = idata2[idata2_1D_index];
			}
		}
		else {
			if (global_2D_y < height1) {
				// Computing 1D index for the first matrix
				int idata1_1D_index = global_2D_x + global_2D_y * width1;
				// Storing the first matrix's elements into the concatenated matrix
				odata[global_1D_index] = idata1[idata1_1D_index];
			}
			else {
				// Calculating global 1D index for accessing the second matrix in row-major fashion
				int idata2_1D_index = global_2D_x + (global_2D_y - height1) * width2;
				// Storing the first matrix's elements into the concatenated matrix
				odata[global_1D_index] = idata2[idata2_1D_index];
			}
		}


	}
}

/*====================================================================================================================*/
/*====================================================================================================================*/

/*
This kernel handles converting a empty matrix(with only zeros) to a
diagonal-flattened matrix using given input matrix.
*/
__global__ void matrixDiagflatDouble(double *idata, double *odata, int width, int height)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Insert elements from input matrix at diagonal positions
		odata[global_1D_index] = idata[global_2D_x] * (global_2D_x == global_2D_y);


	}
}

/*====================================================================================================================*/
/*====================================================================================================================*/

__global__ void matrixDiagflatDoubleWithPower(double *idata, double *odata, int power, int width, int height)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Insert elements from input matrix at diagonal positions
		odata[global_1D_index] = pow(idata[global_2D_x], power) * (global_2D_x == global_2D_y);


	}
}

/*====================================================================================================================*/
/*====================================================================================================================*/

__global__ void matrixDiagflatComplex(cuDoubleComplex *idata, cuDoubleComplex *odata, int width, int height)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Insert elements from input matrix at diagonal positions
		if (global_2D_x == global_2D_y)
			odata[global_1D_index] = idata[global_2D_x];
		else
			odata[global_1D_index] = make_cuDoubleComplex(0.0, 0.0); // Not setting this will lead to undefined behavior


	}
}

/*====================================================================================================================*/
/*====================================================================================================================*/

/*
This kernel handles matrix element-wise sign operation, identical to Numpy.sign()
*/

__global__ void matrixElementWiseSign(double *idata, double *odata, int width, int height)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Depending on the operation choice, it either adds or subtracts.
		// In case I wonder later, this will certainly lead to branch divergence. Better logic awaited!
		if (idata[global_1D_index] == 0)
			odata[global_1D_index] = 0;
		else if (idata[global_1D_index] > 0)
			odata[global_1D_index] = 1;
		else if (idata[global_1D_index] < 0)
			odata[global_1D_index] = -1;

	}
}

/*====================================================================================================================*/
/*====================================================================================================================*/

/*
This kernel handles creating identify, ones or zeros matrix (like numpy.eye, numpy.ones, numpy.zeros).
0 = Identity Matrix
1 = Ones
2 = Zeros
*/

__global__ void matrixEyeOrOnesOrZeros(double *odata, int width, int height, int operationChoice)
{
	// Calculating global 2D indices
	int global_2D_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_2D_y = blockDim.y * blockIdx.y + threadIdx.y;

	// Processing only those threads which are within matrix dimensions
	if (global_2D_x < width && global_2D_y < height) {

		// Calculating global 1D index for accessing the matrix in row-major fashion
		int global_1D_index = global_2D_x + global_2D_y * width;

		// Depending on the operation choice, it either adds or subtracts
		if (operationChoice == 0)
			odata[global_1D_index] = (global_2D_x == global_2D_y); // <-- Gives 1 if i=j, else 0
		else if (operationChoice == 1)
			odata[global_1D_index] = 1;
		else if (operationChoice == 2)
			odata[global_1D_index] = 0; // <-- Leaving this step would work too as the elements are automatically initialized to 0.
		// Actually about the comment above, NO! It turns out setting the memory location values is left to the programmer and no values are assigned by default.

	}
}

/*====================================================================================================================*/
/*====================================================================================================================*/

/*
This kernel is used to find the maximum/minimum value in a given matrix using shared memory.
The reduction is done in two stages.
Once, all the maximum/minimum values from each block is gathered into a single block.
Then, this single block obtained above is processed to find the maximum/minimum.

Setting is_max decides the max/min configuration.

Our implementation is limited to finding the maximum only.
*/

__global__ void shmem_min_max_reduce_kernel(double * d_out, const double * const d_in, int elements)
{


	int globalX = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ double sdata[];
	sdata[tid] = 1E-37;  // 1

	// load shared mem from global mem
	if (globalX < elements)
		sdata[tid] = d_in[globalX];

	__syncthreads();            // make sure entire block is loaded!

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s  && globalX < elements)
		{
			sdata[tid] = max(sdata[tid], sdata[tid + s]);
		}
		__syncthreads();        // make sure all adds at one stage are done!
	}

	// only thread 0 writes result for this block back to global mem
	if (tid == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}
}


__global__ void wrapPointersIntoPointerArrays_FP64(double* idata, double** odata)
{
	odata[0] = idata;
}

