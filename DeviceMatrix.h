#pragma once
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>

// 5 possible types of matrices. Currently in use are Double and ComplexZ. Both are Double precision Floating Point and Complex Numbers
// Int was later introduced for storage of index-matrices.
enum Dtype { Int, Float, Double, ComplexC, ComplexZ };



// Class representing the matrix allocation on GPU, its dimensions and data-type.
class DeviceMatrix {
public:
	int id;
	void* device_pointer;
	int width;
	int height;
	Dtype dtype;
	DeviceMatrix();
	DeviceMatrix(void* matrix_device_pointer, int matrix_width, int matrix_height, Dtype matrix_dtype);
	DeviceMatrix(int id);
	~DeviceMatrix();
};

