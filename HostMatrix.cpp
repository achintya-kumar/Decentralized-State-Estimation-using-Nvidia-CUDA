#include "HostMatrix.h"


HostMatrix::HostMatrix()
{
}


HostMatrix::~HostMatrix()
{
}

HostMatrix::HostMatrix(void* pointer, int matrix_width, int matrix_height, Dtype matrix_dtype) {
	host_pointer = pointer;
	width = matrix_width;
	height = matrix_height;
	dtype = matrix_dtype;
}