#include <cuComplex.h>


// Kernels  declarations (from user_defined_kernels.cu file)
extern void sort_on_device(double* h_vec, int width);
extern void transposeCoalesced(double *odata, double *idata, int width, int height);
extern void countNonZeroElements(double *idata, int *nres, int width, int height);
extern void filter_k(double *dst, int *nres, double *src, int width, int height);
extern void mapNonZeroIndices(double *idata, double *odata, int width, int height);
extern void matrixElementWiseASMD(double const *idata1, double const *idata2, double *odata, int width, int height, int operationChoice);
extern void matrixElementWiseASMDForSingleRow(double const *idata1, double const *idata2, double *odata, int width, int height, int operationChoice);
extern void matrixComplexElementWiseASMD(cuDoubleComplex const *idata1, cuDoubleComplex const *idata2, cuDoubleComplex *odata, int width, int height, int operationChoice);
extern void matrixElementWiseScalarASMD(double *idata, float constant, double *odata, int width, int height, int choice_of_operation, int order_of_operation);
extern void matrixComplexElementWiseScalarASMD(cuDoubleComplex *idata, float scalar, cuDoubleComplex *odata, int width, int height, int choice_of_operation, int order_of_operation);
extern void complexMatrixConstruction(double *idata1, double *idata2, cuDoubleComplex *odata, int width, int height);
extern void matrixElementWiseSinOrCosOrAbsForDoubles(double *idata, double *odata, int width, int height, int operationChoice, int use_intrinsics);
extern void matrixComplexConjugate(cuDoubleComplex *idata1, cuDoubleComplex *odata, int width, int height);
extern void matrixComplexElementWiseExtractions(cuDoubleComplex *idata1, double *odata, int width, int height, int operationChoice);
extern void matrixComplexElementWiseExtractions(cuDoubleComplex *idata1, double *odata, int width, int height, int operationChoice);
extern void matrixEyeOrOnesOrZeros(double *odata, int width, int height, int operationChoice);
extern void matrixDiagflatDouble(double *idata, double *odata, int width, int height);
extern void matrixDiagflatComplex(cuDoubleComplex *idata, cuDoubleComplex *odata, int width, int height);
extern void matrixDiagflatDoubleWithPower(double *idata, double *odata, int power, int width, int height);
extern void matrixConcatenateDouble(double *idata1, double *idata2, double *odata, int width, int height, int width1, int height1, int width2, int operationChoice);
extern void matrixConcatenateComplex(cuDoubleComplex *idata1, cuDoubleComplex *idata2, cuDoubleComplex *odata, int width, int height, int width1, int height1, int width2, int operationChoice);
extern void doubleToComplex(double *idata, cuDoubleComplex *odata, int width, int height);
extern void sliceDouble(double *input, double *output, const int row_start, const int row_end, const int column_start, const int column_end, const int input_matrix_width, const int output_matrix_width);
extern void sliceComplex128(cuDoubleComplex *input, cuDoubleComplex *output, const int row_start, const int row_end, const int column_start, const int column_end, const int input_matrix_width, const int output_matrix_width);
extern void slice_with_indices(double *input, double *output, double *indices, const int input_matrix_height, const int input_matrix_width, const int indices_matrix_length);
extern void specialSlicingOnR_kernel(double *input, double *output, double *indices, int input_matrix_height, int input_matrix_width, int indices_matrix_length);
extern void specialSlicingOnH_kernel(double *input, double *output, double *indices, int input_matrix_height, int input_matrix_width, int indices_matrix_length, int output_matrix_width);
extern void shmem_min_max_reduce_kernel(double * d_out, const double * const d_in, int elements);
extern void matrix_insert(double *input, double *matrix_to_insert, const int row_start, const int row_end, const int column_start, const int column_end, const int input_matrix_height, const int input_matrix_width);
extern void matrixElementWiseAngles(double *eK, double *fK, double *odata, int width, int height);
extern void wrapPointersIntoPointerArrays_FP64(double* idata, double** odata);
extern void matrixConcatenateDouble2(double *idata1, double *idata2, double *odata, int width, int height,
	int width1, int height1, int width2, int height2, int operationChoice, int isFirstMatrix);