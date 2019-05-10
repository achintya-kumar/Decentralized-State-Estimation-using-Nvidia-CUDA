#include "DeviceMatrix.h"
#include <stdio.h>


// Destructor
DeviceMatrix::~DeviceMatrix()
{
	//printf("\nDestroying DeviceMatrix with properties: %d, %d, %d", width, height, dtype);
	//printf("\nDestroying DeviceMatrix with id: %d", id);
}


// Constructor
DeviceMatrix::DeviceMatrix() {}
DeviceMatrix::DeviceMatrix(void* matrix_device_pointer, int matrix_width, int matrix_height, Dtype matrix_dtype) {
	device_pointer = matrix_device_pointer;
	width = matrix_width;
	height = matrix_height;
	dtype = matrix_dtype;
}

//Empty constructor
DeviceMatrix::DeviceMatrix(int id_d)
{
	id = id_d;
	//printf("\nCreating DeviceMatrix with id: %d", id);
}