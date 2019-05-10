#include <cuComplex.h>


// Kernels  declarations (from user_defined_kernels.cu file)
extern void sort_on_device_fp32(float* h_vec, int width);
extern void transposeCoalesced_fp32(float *odata, float *idata, int width, int height);
extern void countNonZeroElements_fp32(float *idata, int *nres, int width, int height);
extern void filter_k_fp32(float *dst, int *nres, float *src, int width, int height);
extern void mapNonZeroIndices_fp32(float *idata, float *odata, int width, int height);
extern void matrixElementWiseASMD_fp32(float const *idata1, float const *idata2, float *odata, int width, int height, int operationChoice);
extern void matrixElementWiseASMDForSingleRow_fp32(float const *idata1, float const *idata2, float *odata, int width, int height, int operationChoice);
extern void matrixComplexElementWiseASMD_fp32(cuComplex const *idata1, cuComplex const *idata2, cuComplex *odata, int width, int height, int operationChoice);
extern void matrixElementWiseScalarASMD_fp32(float *idata, float constant, float *odata, int width, int height, int choice_of_operation, int order_of_operation);
extern void matrixComplexElementWiseScalarASMD_fp32(cuComplex *idata, float scalar, cuComplex *odata, int width, int height, int choice_of_operation, int order_of_operation);
extern void complexMatrixConstruction_fp32(float *idata1, float *idata2, cuComplex *odata, int width, int height);
extern void matrixElementWiseSinOrCosOrAbs_fp32(float *idata, float *odata, int width, int height, int operationChoice, int use_intrinsics);
extern void matrixComplexConjugate_fp32(cuComplex *idata1, cuComplex *odata, int width, int height);
extern void matrixComplexElementWiseExtractions_fp32(cuComplex *idata1, float *odata, int width, int height, int operationChoice);
extern void matrixComplexElementWiseExtractions_fp32(cuComplex *idata1, float *odata, int width, int height, int operationChoice);
extern void matrixEyeOrOnesOrZeros_fp32(float *odata, int width, int height, int operationChoice);
extern void matrixDiagflat_fp32(float *idata, float *odata, int width, int height);
extern void matrixDiagflatComplex_fp32(cuComplex *idata, cuComplex *odata, int width, int height);
extern void matrixDiagflatWithPower_fp32(float *idata, float *odata, int power, int width, int height);
extern void matrixConcatenate_fp32(float *idata1, float *idata2, float *odata, int width, int height, int width1, int height1, int width2, int operationChoice);
extern void matrixConcatenateComplex_fp32(cuComplex *idata1, cuComplex *idata2, cuComplex *odata, int width, int height, int width1, int height1, int width2, int operationChoice);
extern void floatToComplex(float *idata, cuComplex *odata, int width, int height);
extern void slice_fp32(float *input, float *output, const int row_start, const int row_end, const int column_start, const int column_end, const int input_matrix_width, const int output_matrix_width);
extern void sliceComplex64(cuComplex *input, cuComplex *output, const int row_start, const int row_end, const int column_start, const int column_end, const int input_matrix_width, const int output_matrix_width);
extern void slice_with_indices_fp32(float *input, float *output, float *indices, const int input_matrix_height, const int input_matrix_width, const int indices_matrix_length);
extern void specialSlicingOnR_kernel_fp32(float *input, float *output, float *indices, int input_matrix_height, int input_matrix_width, int indices_matrix_length);
extern void specialSlicingOnH_kernel_fp32(float *input, float *output, float *indices, int input_matrix_height, int input_matrix_width, int indices_matrix_length, int output_matrix_width);
extern void shmem_min_max_reduce_kernel_fp32(float * d_out, const float * const d_in, int elements);
extern void matrix_insert_fp32(float *input, float *matrix_to_insert, const int row_start, const int row_end, const int column_start, const int column_end, const int input_matrix_height, const int input_matrix_width);
extern void matrixElementWiseAngles_fp32(float *eK, float *fK, float *odata, int width, int height);