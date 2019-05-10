#pragma once
#include "DeviceMatrix.h"

class HostMatrix
{
public:
	HostMatrix();
	~HostMatrix();
	HostMatrix::HostMatrix(void* pointer, int matrix_width, int matrix_height, Dtype matrix_dtype);
	void* host_pointer;
	unsigned int width;
	unsigned int height;
	Dtype dtype;
};

