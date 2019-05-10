#include "CudaFunctions_Single.h"

CudaFunctions_Single::~CudaFunctions_Single() {}

// Marks the beginning of a step
void CudaFunctions_Single::stepBegin() {}

// Marks the end of a step; releases allocations made during one step to the pool of available allocations
void CudaFunctions_Single::sanitizeMemoryPools() {
	cudaStreamSynchronize(stream);
	dMManager.releaseStepAllocationsToPool();
}

// Releases allocations in ON-HOLD state to the pool of available allocations
void CudaFunctions_Single::releaseOnHoldAllocationsToPool() {
	dMManager.releaseOnHoldAllocationsToPool();
}

// Returns a device-pointer of given size
void* CudaFunctions_Single::memAlloc(size_t size, bool on_hold) {
	return dMManager.getDeviceMemory(size, on_hold);
}

// To be used in case of abnormal exit
void CudaFunctions_Single::problematicExit(char* message) {
	printf(message);
	getchar();
	exit(1);

}

DeviceMatrix CudaFunctions_Single::to_device(double *array, int width, int height) {
	float *float_elements;
	cudaMalloc((void**)&float_elements, height * width * sizeof(float));
	cudaMemcpy(float_elements, array, height * width * sizeof(float), cudaMemcpyHostToDevice);
	dMManager.total_occupied_memory_in_bytes += height * width * sizeof(float);
	return DeviceMatrix(float_elements, width, height, Float);
}


DeviceMatrix CudaFunctions_Single::map_non_zero_elements(DeviceMatrix device_matrix, bool on_hold) {
	dim3 grid((int)ceil((float)device_matrix.width / 32), (int)ceil((float)device_matrix.height / 32), 1);

	// Allocate GPU buffers  .
	float *odata;
	//cudaMalloc((void**)&non_zero_count, sizeof(int));
	odata = static_cast<float*>(memAlloc(device_matrix.width * device_matrix.height * sizeof(float), on_hold));

	// Launch a kernel on the GPU.
	void* args[] = { &(device_matrix.device_pointer), &odata, &(device_matrix.width), &(device_matrix.height) };
	cudaLaunchKernel(
		(const void*)&mapNonZeroIndices_fp32, // pointer to kernel func.
		grid, // grid
		dim3(32, 32, 1), // block
		args,  // arguments
		0,
		stream);

	return DeviceMatrix(odata, device_matrix.width, device_matrix.height, Float);
}

// Returns a single-row matrix with indices of all non-zero elements of given on-device matrix
DeviceMatrix CudaFunctions_Single::extract_indices_for_non_zero(DeviceMatrix device_matrix) {

	dim3 grid((int)ceil((float)device_matrix.width / 1024), (int)ceil((float)device_matrix.height / 1), 1);

	// Allocate GPU buffers  .
	unsigned long *non_zero_count = NULL;
	//cudaMalloc((void**)&non_zero_count, sizeof(int));
	non_zero_count = static_cast<unsigned long*>(memAlloc(sizeof(unsigned long)));

	// Launch a kernel on the GPU.
	void* args[] = { &(device_matrix.device_pointer), &non_zero_count, &(device_matrix.width), &(device_matrix.height) };

	cudaLaunchKernel(
		(const void*)&countNonZeroElements_fp32, // pointer to kernel func.
		grid, // grid
		dim3(1, 1024, 1), // block
		args,  // arguments
		0,
		stream);

	int non_zero_count_h;
	cudaMemcpy(&non_zero_count_h, non_zero_count, sizeof(int), cudaMemcpyDeviceToHost);

	// Allocate GPU buffers 
	float *output;
	//cudaMalloc((void**)&output, non_zero_count_h * sizeof(float));
	output = static_cast<float*>(memAlloc(non_zero_count_h * sizeof(float)));

	// For atomic tracking of indices
	DeviceMatrix nres = zerosInt(1, 1);

	// Launch a kernel on the GPU.
	void* args2[] = { &output, &(nres.device_pointer), &(device_matrix.device_pointer), &(device_matrix.width), &(device_matrix.height) };

	cudaLaunchKernel(
		(const void*)&filter_k_fp32, // pointer to kernel func.
		grid, // grid
		dim3(1, 1024, 1), // block
		args2,  // arguments
		0,
		stream);

	return DeviceMatrix(output, non_zero_count_h, 1, Float);
}


// Returns single element host matrix after copying from device memory.
double CudaFunctions_Single::to_host(DeviceMatrix device_matrix) {
	float on_host;
	cudaMemcpyAsync(&on_host, device_matrix.device_pointer, sizeof(float), cudaMemcpyDeviceToHost, stream);
	return on_host;
}

// Loads the matrix from the given CSV file into device memory.
DeviceMatrix CudaFunctions_Single::to_device(std::string file) {

	// PHASE 1: Initial assessment of the matrix's csv file.
	std::ifstream  data(file);
	std::string line;
	int rows = 0, columns = 0;
	bool isComplexC = false;
	while (std::getline(data, line))
	{
		std::stringstream  lineStream(line);
		std::string        cell;
		while (std::getline(lineStream, cell, ','))
		{
			// You have a cell!!!!
			if (columns == 0) {
				try {
					float d = std::stof(cell);
					isComplexC = false;
				}
				catch (std::exception& e)
				{
					isComplexC = true;
				}
			}

			if (rows == 0)
				columns++;
		}
		rows++;
	}

	//printf("\nFile: %s", file);
	//printf("\nRows = %d, Columns = %d, Dtype = %s", rows, columns, isComplexC ? "ComplexC" : "Float");


	// PHASE 2: Loading the items into the memory.
	cuComplex *complex_elements;
	cuComplex *complex_elements_d; // For GPU Allocation
	float *float_elements;
	float *float_elements_d;			 // For GPU Allocation
	// Initializing array based on whether the elements are float or of complex types.
	if (isComplexC)
		complex_elements = new cuComplex[rows * columns];
	else
		float_elements = new float[rows * columns];

	std::ifstream  data2(file);
	std::string line2;
	int counter = 0;
	while (std::getline(data2, line2))
	{
		std::stringstream  lineStream(line2);
		std::string        cell;
		while (std::getline(lineStream, cell, ','))
		{
			// You have a cell!!!!
			if (!isComplexC) // When the elements are of type Float
				float_elements[counter] = std::stof(cell);

			else {			 // When the elements are of type ComplexC
				boost::replace_all(cell, "(", "");
				boost::replace_all(cell, ")", "");

				std::stringstream  innerLineStream(cell);
				std::string        innerCell;
				float complex_parts[2];
				int complex_parts_counter = 0;
				while (std::getline(innerLineStream, innerCell, '+')) {
					boost::replace_all(innerCell, "j", "");
					complex_parts[complex_parts_counter] = std::stof(innerCell);
					complex_parts_counter++;
				}
				complex_elements[counter] = make_cuComplex(complex_parts[0], complex_parts[1]); // Constructing complex number for CUDA
			}
			counter++;
		}
	}

	std::cout << std::endl;
	if (!isComplexC) {
		/*for (int i = 0; i < rows; i++) {																							// <-- For printing ComplexC matrix to console
		for (int j = 0; j < columns; j++) {
		printf("%f\t", float_elements[j + i*columns]);
		}
		std::cout << std::endl;
		}*/
		float_elements_d = static_cast<float*>(dMManager.getDeviceMemory(rows * columns * sizeof(float)));
		//checkCudaErrors(cudaMalloc((void**)&float_elements_d, rows * columns * sizeof(float)));
		checkCudaErrors(cudaMemcpy(float_elements_d, float_elements, rows * columns * sizeof(float), cudaMemcpyHostToDevice));
		delete[] float_elements; // Heap cleanup
		return DeviceMatrix(float_elements_d, columns, rows, Float);
	}
	else {
		/*for (int i = 0; i < rows; i++) {																							// <-- For printing Float matrix to console
		for (int j = 0; j < columns; j++) {
		printf("%f + %fj\t", cuCreal(complex_elements[j + i*columns]), cuCimag(complex_elements[j + i*columns]));
		}
		std::cout << std::endl;
		}*/
		complex_elements_d = static_cast<cuComplex*>(dMManager.getDeviceMemory(rows * columns * sizeof(cuComplex)));
		//checkCudaErrors(cudaMalloc((void**)&complex_elements_d, rows * columns * sizeof(cuComplex)));
		checkCudaErrors(cudaMemcpy(complex_elements_d, complex_elements, rows * columns * sizeof(cuComplex), cudaMemcpyHostToDevice));
		delete[] complex_elements; // Heap cleanup
		return DeviceMatrix(complex_elements_d, columns, rows, ComplexC);
	}
}

DeviceMatrix CudaFunctions_Single::host_array_wrapped_in_DeviceMatrix(DeviceMatrix device_matrix) {
	if (device_matrix.dtype == Float) {
		float *float_elements = new float[device_matrix.height * device_matrix.width];
		cudaMemcpy(float_elements, device_matrix.device_pointer, device_matrix.height * device_matrix.width * sizeof(float), cudaMemcpyDeviceToHost);
		return DeviceMatrix(float_elements, device_matrix.width, device_matrix.height, Float);
	}
}

void CudaFunctions_Single::printMatrix(DeviceMatrix device_matrix) {
	//system("CLS");
	int height_limit = device_matrix.height, width_limit = device_matrix.width;									// Limiting the dimensions while printing
	/*if (device_matrix.height > 9)
	height_limit = 9;
	if (device_matrix.width > 9)
	width_limit = 9;*/

	if (device_matrix.dtype == Float) {
		float *float_elements = new float[device_matrix.height * device_matrix.width];
		cudaMemcpy(float_elements, device_matrix.device_pointer, device_matrix.height * device_matrix.width * sizeof(float), cudaMemcpyDeviceToHost);
		printf("\n\n=== Printing Float matrix ===\n");
		for (int i = 0; i < height_limit; i++) {
			printf("Row#%d\t", i);
			for (int j = 0; j < width_limit; j++) {
				int current_index = j + i * device_matrix.width;
				printf("%.10f\t", float_elements[current_index]);
			}
			printf("\n");
		}
		delete[] float_elements;
	}
	else if (device_matrix.dtype == Int) {
		int *float_elements = new int[device_matrix.height * device_matrix.width];
		cudaMemcpy(float_elements, device_matrix.device_pointer, device_matrix.height * device_matrix.width * sizeof(int), cudaMemcpyDeviceToHost);
		printf("\n\n=== Printing Float matrix ===\n");
		for (int i = 0; i < height_limit; i++) {
			printf("Row#%d\t", i);
			for (int j = 0; j < width_limit; j++) {
				int current_index = j + i * device_matrix.width;
				printf("%d\t", float_elements[current_index]);
			}
			printf("\n");
		}
		delete[] float_elements;
	}
	else if (device_matrix.dtype == ComplexC) {
		cuComplex *complex_elements = new cuComplex[device_matrix.height * device_matrix.width];
		cudaMemcpy(complex_elements, device_matrix.device_pointer, device_matrix.height * device_matrix.width * sizeof(cuComplex), cudaMemcpyDeviceToHost);
		printf("\n\n=== Printing Complex matrix ===\n");
		for (int i = 0; i < height_limit; i++) {
			printf("Row#%d\t", i);
			for (int j = 0; j < width_limit; j++) {
				int current_index = j + i * device_matrix.width;
				printf("%.10f + %.10fj\t", cuCrealf(complex_elements[current_index]), cuCimagf(complex_elements[current_index]));
			}
			printf("\n");
		}
		delete[] complex_elements;
	}
}



void CudaFunctions_Single::insert_into_execution_times_map(std::string kernel_name, float milliseconds) {
	Map_Of_Execution_Times::iterator map_iterator = map_of_execution_times.find(kernel_name);

	if (map_iterator != map_of_execution_times.end()) {
		map_iterator->second += milliseconds;
	}
	else {
		map_of_execution_times.insert(std::pair <std::string, float>(kernel_name, milliseconds));		// Load up the new vector into the pool with key=size
	}
}

void CudaFunctions_Single::print_execution_times() {
	// Declaring the type of Predicate that accepts 2 pairs and return a bool
	typedef std::function<bool(std::pair<std::string, float>, std::pair<std::string, float>)> Comparator;

	// Defining a lambda function to compare two pairs. It will compare two pairs using second field
	Comparator compFunctor =
		[](std::pair<std::string, float> elem1, std::pair<std::string, float> elem2)
	{
		return elem1.second > elem2.second;
	};

	// Declaring a set that will store the pairs using above comparision logic
	std::set<std::pair<std::string, float>, Comparator> execution_times_sorted(
		map_of_execution_times.begin(), map_of_execution_times.end(), compFunctor);

	// Iterate over a set using range base for loop
	// It will display the items in sorted order of values
	for (std::pair<std::string, int> element : execution_times_sorted)
		std::cout << element.first << " ::\t" << element.second << std::endl;
}

/*All the kernel declarations can be found inside the header file for this class.
This is done to avoid repetitive declarations if kept inside the wrapper functions.*/

// Helper function for using CUDA to add vectors in parallel.

DeviceMatrix CudaFunctions_Single::update_eK_fK(DeviceMatrix a, DeviceMatrix b) {

	dim3 grid((int)ceil((float)a.width / 1), (int)ceil((float)a.height / 1024), 1);
	int operation_choice = 0;
	float *dev_c = 0;

	// Allocate GPU buffers  .
	//cudaMalloc((void**)&dev_c, a.height * a.width * sizeof(float));
	dev_c = static_cast<float*>(a.device_pointer);

	// Launch a kernel on the GPU.
	void* args[] = { &(a.device_pointer), &(b.device_pointer), &dev_c, &(a.width), &(a.height), &operation_choice };
	if (a.height == b.height && a.width == b.width) {
		//cudaEventRecord(start);
		cudaLaunchKernel(
			(const void*)&matrixElementWiseASMD_fp32, // pointer to kernel func.
			grid, // grid
			dim3(1, 1024, 1), // block
			args,  // arguments
			0,
			stream);

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("update_eK_fK", milliseconds);*/
	}

	else
		problematicExit("Incompatible matrix dimensions for in-place update of eK/fK.");

	return DeviceMatrix(dev_c, a.width, a.height, Float);
}


// ASMD operations on two matrices.
DeviceMatrix CudaFunctions_Single::matrixElementWiseASMD_wrapper(DeviceMatrix a, DeviceMatrix b, unsigned int operation_choice, bool on_hold) {

	dim3 grid((int)ceil((float)a.width / 32), (int)ceil((float)a.height / 32), 1);

	if (a.dtype == Float) {
		float *dev_c = 0;

		// Allocate GPU buffers  .
		//cudaMalloc((void**)&dev_c, a.height * a.width * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(a.height * a.width * sizeof(float), on_hold));

		// Launch a kernel on the GPU.
		void* args[] = { &(a.device_pointer), &(b.device_pointer), &dev_c, &(a.width), &(a.height), &operation_choice };

		if (a.height == b.height && a.width == b.width) {
			//cudaEventRecord(start);
			cudaLaunchKernel(
				(const void*)&matrixElementWiseASMD_fp32, // pointer to kernel func.
				grid, // grid
				dim3(32, 32, 1), // block
				args,  // arguments
				0,
				stream);
			/*cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			insert_into_execution_times_map("matrixElementWiseASMD", milliseconds);*/
		}
		else if (b.height) {
			//cudaEventRecord(start);
			cudaLaunchKernel(
				(const void*)&matrixElementWiseASMDForSingleRow_fp32, // pointer to kernel func.
				grid, // grid
				dim3(32, 32, 1), // block
				args,  // arguments
				0,
				stream);
			/*cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			insert_into_execution_times_map("matrixElementWiseASMDForSingleRow", milliseconds);*/
		}
		else
			problematicExit("Incompatible matrix dimensions for float elementwise ASMD.");

		return DeviceMatrix(dev_c, a.width, a.height, Float);
	}
	else if (a.dtype == ComplexC) {

		// Container pointer for result
		cuComplex *dev_c = 0;

		// Allocate GPU buffers for three vectors (two input, one output).
		//cudaMalloc((void**)&dev_c, a.height * a.width * sizeof(cuComplex));
		dev_c = static_cast<cuComplex*>(memAlloc(a.height * a.width * sizeof(cuComplex), on_hold));


		// Launch a kernel on the GPU.
		void* args[] = { &(a.device_pointer), &(b.device_pointer), &dev_c, &(a.width), &(a.height), &operation_choice };

		if (a.height == b.height && a.width == b.width) {
			//cudaEventRecord(start);
			cudaLaunchKernel(
				(const void*)&matrixComplexElementWiseASMD_fp32, // pointer to kernel func.
				grid, // grid
				dim3(32, 32, 1), // block
				args,  // arguments
				0,
				stream);
			/*cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			insert_into_execution_times_map("matrixComplexElementWiseASMD", milliseconds);*/
		}
		else if (b.height)
			problematicExit("Single-row complex elementwise ASMD source-modules currently not included in the library!");
		else
			problematicExit("Incompatible matrix dimensions for complexZ elementwise ASMD.");

		return DeviceMatrix(dev_c, a.width, a.height, ComplexC);
	}
}

// Add 2 matrices elementwise.
DeviceMatrix CudaFunctions_Single::add(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	return matrixElementWiseASMD_wrapper(device_matrix1, device_matrix2, 0, on_hold);
}

// Subtract 2 matrices elementwise.
DeviceMatrix CudaFunctions_Single::sub(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	return matrixElementWiseASMD_wrapper(device_matrix1, device_matrix2, 1, on_hold);
}

// Multiply 2 matrices elementwise.
DeviceMatrix CudaFunctions_Single::mul(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	return matrixElementWiseASMD_wrapper(device_matrix1, device_matrix2, 2, on_hold);
}

// Divide 2 matrices elementwise.
DeviceMatrix CudaFunctions_Single::div(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	return matrixElementWiseASMD_wrapper(device_matrix1, device_matrix2, 3, on_hold);
}

// ASMD operations on a scalar and a matrix.
DeviceMatrix CudaFunctions_Single::matrixElementWiseScalarASMD_wrapper(DeviceMatrix device_matrix, float scalar, int operation_choice, int order_of_operation, bool on_hold) {
	dim3 grid((int)ceil((float)device_matrix.width / 32), (int)ceil((float)device_matrix.height / 32), 1);

	if (device_matrix.dtype == Float) {
		//cudaEventRecord(start);
		float *dev_c = 0;

		// Allocate GPU buffers  .
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(device_matrix.height * device_matrix.width * sizeof(float), on_hold));

		// Launch a kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &(scalar), &dev_c, &(device_matrix.width), &(device_matrix.height), &operation_choice, &order_of_operation };
		cudaLaunchKernel(
			(const void*)&matrixElementWiseScalarASMD_fp32, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrixElementWiseScalarASMD", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, Float);
	}
	else if (device_matrix.dtype == ComplexC) {
		//cudaEventRecord(start);
		cuComplex *dev_c = 0;

		// Allocate GPU buffers  .
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(cuComplex));
		dev_c = static_cast<cuComplex*>(memAlloc(device_matrix.height * device_matrix.width * sizeof(cuComplex), on_hold));

		// Launch a kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &(scalar), &dev_c, &(device_matrix.width), &(device_matrix.height), &operation_choice, &order_of_operation };
		cudaLaunchKernel(
			(const void*)&matrixComplexElementWiseScalarASMD_fp32, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrixComplexElementWiseScalarASMD", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, ComplexC);
	}
}

// Add a scalar elementwise to a matrix.
DeviceMatrix CudaFunctions_Single::adds(DeviceMatrix device_matrix1, float scalar, bool on_hold) {
	return matrixElementWiseScalarASMD_wrapper(device_matrix1, scalar, 0, 0, on_hold);
}

// Subtract a scalar elementwise from a matrix. 'order_of_operation' argument specifies direction of the operation.
DeviceMatrix CudaFunctions_Single::subs(DeviceMatrix device_matrix1, float scalar, int order_of_operation, bool on_hold) {
	return matrixElementWiseScalarASMD_wrapper(device_matrix1, scalar, 1, order_of_operation, on_hold);
}

// Subtract a scalar elementwise from a matrix. 'order_of_operation' argument specifies direction of the operation.
DeviceMatrix CudaFunctions_Single::subs(DeviceMatrix device_matrix1, float scalar, bool on_hold) {
	return matrixElementWiseScalarASMD_wrapper(device_matrix1, scalar, 1, 0, on_hold);
}

// Multiply a scalar elementwise to a matrix.
DeviceMatrix CudaFunctions_Single::muls(DeviceMatrix device_matrix1, float scalar, bool on_hold) {
	return matrixElementWiseScalarASMD_wrapper(device_matrix1, scalar, 2, 0, on_hold);
}

// Divide matrix elementwise with a scalar. 'order_of_operation' argument specifies direction of the operation.
DeviceMatrix CudaFunctions_Single::divs(DeviceMatrix device_matrix1, float scalar, int order_of_operation, bool on_hold) {
	return matrixElementWiseScalarASMD_wrapper(device_matrix1, scalar, 3, order_of_operation, on_hold);
}

// Divide matrix elementwise with a scalar. 'order_of_operation' argument specifies direction of the operation.
DeviceMatrix CudaFunctions_Single::divs(DeviceMatrix device_matrix1, float scalar, bool on_hold) {
	return matrixElementWiseScalarASMD_wrapper(device_matrix1, scalar, 3, 0, on_hold);
}

// Transpose first matrix and then calculate matrix product with second matrix. Only works with DP floating point and complex numbers.
DeviceMatrix CudaFunctions_Single::tdot(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {

	if (device_matrix1.dtype == Float) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.width * device_matrix2.width * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(device_matrix1.width * device_matrix2.width * sizeof(float), on_hold));

		float alpha = 1.0f;
		float beta = 0.0f;
		cublasSgemm(
			cublasHandle,
			CUBLAS_OP_N,							// transA
			CUBLAS_OP_T,							// transB
			device_matrix2.width,					// m
			device_matrix1.width,					// n
			device_matrix1.height,					// k
			&alpha,									// alpha
			(float*)device_matrix2.device_pointer,	// A
			device_matrix2.width,					// lda
			(float*)device_matrix1.device_pointer,	// B
			device_matrix1.width,					// ldb
			&beta,									// beta
			dev_c,									// C
			device_matrix2.width);					// ldc

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("tdot Float", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.width, device_matrix1.width, Float);
	}
	else if (device_matrix1.dtype == ComplexC) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		cuComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.width * device_matrix2.width * sizeof(cuComplex));
		dev_c = static_cast<cuComplex*>(memAlloc(device_matrix1.width * device_matrix2.width * sizeof(cuComplex), on_hold));

		cuComplex alpha = make_cuComplex(1.0f, 0.0f);
		cuComplex beta = make_cuComplex(0.0f, 0.0f);
		cublasCgemm(
			cublasHandle,
			CUBLAS_OP_N,										// transA
			CUBLAS_OP_T,										// transB
			device_matrix2.width,								// m
			device_matrix1.width,								// n
			device_matrix1.height,								// k
			&alpha,												// alpha
			(cuComplex*)device_matrix2.device_pointer,	// A
			device_matrix2.width,								// lda
			(cuComplex*)device_matrix1.device_pointer,	// B
			device_matrix1.width,								// ldb
			&beta,												// beta
			dev_c,												// C
			device_matrix2.width);								// ldc
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("tdot Complex", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.width, device_matrix1.width, ComplexC);
	}
	else
		problematicExit("Unknown matrix datatype for tdot operation. Only Float and ComplexC are supported!");
}


DeviceMatrix CudaFunctions_Single::aTbT(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	if (device_matrix1.dtype == Float) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.width * device_matrix2.width * sizeof(double));
		dev_c = static_cast<float*>(memAlloc(device_matrix1.width * device_matrix2.height * sizeof(float), on_hold));

		float alpha = 1.0f;
		float beta = 0.0f;
		cublasSgemm(
			cublasHandle,
			CUBLAS_OP_T,							// transA
			CUBLAS_OP_T,							// transB
			device_matrix2.height,					// m
			device_matrix1.width,					// n
			device_matrix1.height,					// k
			&alpha,									// alpha
			(float*)device_matrix2.device_pointer,	// A
			device_matrix2.width,					// lda
			(float*)device_matrix1.device_pointer,	// B
			device_matrix1.width,					// ldb
			&beta,									// beta
			dev_c,									// C
			device_matrix2.height);					// ldc

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("tdot Double", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.height, device_matrix1.width, Float);
	}
	else if (device_matrix1.dtype == ComplexC) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		cuComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.width * device_matrix2.width * sizeof(cuDoubleComplex));
		dev_c = static_cast<cuComplex*>(memAlloc(device_matrix1.width * device_matrix2.height * sizeof(cuComplex), on_hold));

		cuComplex alpha = make_cuComplex(1.0f, 0.0f);
		cuComplex beta = make_cuComplex(0.0f, 0.0f);
		cublasCgemm(
			cublasHandle,
			CUBLAS_OP_T,										// transA
			CUBLAS_OP_T,										// transB
			device_matrix2.height,								// m
			device_matrix1.width,								// n
			device_matrix1.height,								// k
			&alpha,												// alpha
			(cuComplex*)device_matrix2.device_pointer,			// A
			device_matrix2.width,								// lda
			(cuComplex*)device_matrix1.device_pointer,			// B
			device_matrix1.width,								// ldb
			&beta,												// beta
			dev_c,												// C
			device_matrix2.height);								// ldc
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("tdot Complex", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.height, device_matrix1.width, ComplexC);
	}
	else
		problematicExit("Unknown matrix datatype for tdot operation. Only Float and ComplexC are supported!");
}


// Transpose first matrix and then calculate matrix product with second matrix. Only works with DP floating point and complex numbers.
DeviceMatrix CudaFunctions_Single::aTb(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	return tdot(device_matrix1, device_matrix2, on_hold);
}

// Calculate matrix product. Only works with DP floating point and complex numbers.
DeviceMatrix CudaFunctions_Single::dot(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	if (device_matrix1.dtype == Float) {
		//cudaEventRecord(start);

		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.height * device_matrix2.width * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(device_matrix1.height * device_matrix2.width * sizeof(float), on_hold));

		float alpha = 1;
		float beta = 0;
		cublasSgemm(
			cublasHandle,
			CUBLAS_OP_N,							// transA
			CUBLAS_OP_N,							// transB
			device_matrix2.width,					// m
			device_matrix1.height,					// n
			device_matrix1.width,					// k
			&alpha,									// alpha
			static_cast<float*>(device_matrix2.device_pointer),	// A
			device_matrix2.width,					// lda
			static_cast<float*>(device_matrix1.device_pointer),	// B
			device_matrix1.width,					// ldb
			&beta,									// beta
			dev_c,									// C
			device_matrix2.width);					// ldc
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("dot Float", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.width, device_matrix1.height, Float);
	}
	else if (device_matrix1.dtype == ComplexC) {
		//cudaEventRecord(start);

		// Allocate GPU buffers 
		cuComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.height * device_matrix2.width * sizeof(cuComplex));
		dev_c = static_cast<cuComplex*>(memAlloc(device_matrix1.height * device_matrix2.width * sizeof(cuComplex), on_hold));

		cuComplex alpha = make_cuComplex(1.0, 0.0);
		cuComplex beta = make_cuComplex(0.0, 0.0);
		cublasCgemm(
			cublasHandle,
			CUBLAS_OP_N,										// transA
			CUBLAS_OP_N,										// transB
			device_matrix2.width,								// m
			device_matrix1.height,								// n
			device_matrix1.width,								// k
			&alpha,												// alpha
			static_cast<cuComplex*>(device_matrix2.device_pointer),	// A
			device_matrix2.width,								// lda
			static_cast<cuComplex*>(device_matrix1.device_pointer),	// B
			device_matrix1.width,								// ldb
			&beta,												// beta
			dev_c,												// C
			device_matrix2.width);								// ldc

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("dot Complex", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.width, device_matrix1.height, ComplexC);
	}
	else
		problematicExit("Unknown matrix datatype for dot operation. Only Float and ComplexC are supported!");
}

// Transpose second matrix and then calculate matrix product with first matrix. Only works with DP floating point and complex numbers.
DeviceMatrix CudaFunctions_Single::abT(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	if (device_matrix1.dtype == Float) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.height * device_matrix2.height * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(device_matrix1.height * device_matrix2.height * sizeof(float), on_hold));

		float alpha = 1.0f;
		float beta = 0.0f;
		cublasSgemm(
			cublasHandle,
			CUBLAS_OP_T,							// transA
			CUBLAS_OP_N,							// transB
			device_matrix2.height,					// m
			device_matrix1.height,					// n
			device_matrix2.width,					// k
			&alpha,									// alpha
			(float*)device_matrix2.device_pointer,	// A
			device_matrix2.width,					// lda
			(float*)device_matrix1.device_pointer,	// B
			device_matrix1.width,					// ldb
			&beta,									// beta
			dev_c,									// C
			device_matrix2.height);					// ldc
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("abT Float", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.height, device_matrix1.height, Float);
	}
	else if (device_matrix1.dtype == ComplexC) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		cuComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.height * device_matrix2.height * sizeof(cuComplex));
		dev_c = static_cast<cuComplex*>(memAlloc(device_matrix1.height * device_matrix2.height * sizeof(cuComplex), on_hold));

		cuComplex alpha = make_cuComplex(1.0f, 0.0f);
		cuComplex beta = make_cuComplex(0.0f, 0.0f);
		cublasCgemm(
			cublasHandle,
			CUBLAS_OP_T,										// transA
			CUBLAS_OP_N,										// transB
			device_matrix2.height,								// m
			device_matrix1.height,								// n
			device_matrix2.width,								// k
			&alpha,												// alpha
			(cuComplex*)device_matrix2.device_pointer,	// A
			device_matrix2.width,								// lda
			(cuComplex*)device_matrix1.device_pointer,	// B
			device_matrix1.width,								// ldb
			&beta,												// beta
			dev_c,												// C
			device_matrix2.height);								// ldc

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("abT Complex", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.height, device_matrix1.height, ComplexC);
	}
	else
		problematicExit("Unknown matrix datatype for 'abT' operation. Only Float and ComplexC are supported!");
}

// This method combines to matrices A and B to generate complex matrix C with elements of form a+bj.
DeviceMatrix CudaFunctions_Single::complex(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {

	//cudaEventRecord(start);
	// Set grid dimensions
	dim3 grid((int)ceil((float)device_matrix1.width / 32), (int)ceil((float)device_matrix1.height / 32), 1);

	// Allocate GPU buffers 
	cuComplex *dev_c = 0;
	//cudaMalloc((void**)&dev_c, device_matrix1.height * device_matrix2.width * sizeof(cuComplex));
	dev_c = static_cast<cuComplex*>(memAlloc(device_matrix1.height * device_matrix2.width * sizeof(cuComplex), on_hold));

	// Set arguments and launch the required kernel on the GPU.
	void* args[] = { &(device_matrix1.device_pointer), &(device_matrix2.device_pointer), &dev_c, &(device_matrix1.width), &(device_matrix1.height) };
	cudaLaunchKernel(
		(const void*)&complexMatrixConstruction_fp32, // pointer to kernel func.
		grid, // grid
		dim3(32, 32, 1), // block
		args,  // arguments
		0,
		stream);

	/*cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	insert_into_execution_times_map("complexMatrixConstruction", milliseconds);*/

	return DeviceMatrix(dev_c, device_matrix1.width, device_matrix1.height, ComplexC);
}

// This method computes elementwise sin or cosine for a given matrix. Intrinsics sin/cos computations are off by default and can be turned on by setting use_intrinsics to 1.
DeviceMatrix CudaFunctions_Single::matrixElementWiseSinOrCosOrAbs(DeviceMatrix device_matrix, int choice_of_operation, int use_of_intrinsics, bool on_hold) {
	if (device_matrix.dtype == Float) {
		//cudaEventRecord(start);
		// Set grid dimensions

		// Taking 1024 as the height because no launch is encountered with width > 1
		dim3 grid((int)ceil((float)device_matrix.width / 1), (int)ceil((float)device_matrix.height / 1024), 1);

		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(device_matrix.height * device_matrix.width * sizeof(float), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &(device_matrix.width), &(device_matrix.height), &choice_of_operation, &use_of_intrinsics };
		cudaLaunchKernel(
			(const void*)&matrixElementWiseSinOrCosOrAbs_fp32, // pointer to kernel func.
			grid, // grid
			dim3(1, 1024, 1), // block
			args,  // arguments
			0,
			stream);

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrixElementWiseSinOrCosOrAbs", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, Float);
	}
	else
		problematicExit("Sin/cos method currently support float matrices only!");
}

// Computes elementwise sine of given matrix.
DeviceMatrix CudaFunctions_Single::sin(DeviceMatrix device_matrix, bool on_hold) {
	return matrixElementWiseSinOrCosOrAbs(device_matrix, 0, 0, on_hold);
}

// Computes elementwise cosine of given matrix.
DeviceMatrix CudaFunctions_Single::cos(DeviceMatrix device_matrix, bool on_hold) {
	return matrixElementWiseSinOrCosOrAbs(device_matrix, 1, 0, on_hold);
}

// Computes elementwise real/imag/conjugate values of given ComplexC matrix.
DeviceMatrix CudaFunctions_Single::complexMatrixExtraction(DeviceMatrix device_matrix, int operation_choice, bool on_hold) {
	if (operation_choice == 3) {
		//cudaEventRecord(start);
		// Set grid dimensions
		int blockHeight = 1024;
		if (device_matrix.height < blockHeight)
			blockHeight = device_matrix.height;
		dim3 grid((int)ceil((float)device_matrix.width / 1), (int)ceil((float)device_matrix.height / blockHeight), 1);


		// Allocate GPU buffers 
		cuComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(cuComplex));
		dev_c = static_cast<cuComplex*>(memAlloc(device_matrix.height * device_matrix.width * sizeof(cuComplex), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &(device_matrix.width), &(device_matrix.height) };
		cudaLaunchKernel(
			(const void*)&matrixComplexConjugate_fp32, // pointer to kernel func.
			grid, // grid
			dim3(1, blockHeight, 1), // block
			args,  // arguments
			0,
			stream);

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrixComplexConjugate", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, ComplexC);
	}
	else {

		if (device_matrix.dtype != ComplexC)
			problematicExit("real/imag methods support ComplexC type matrices only!");

		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)device_matrix.width / 32), (int)ceil((float)device_matrix.height / 32), 1);

		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(device_matrix.height * device_matrix.width * sizeof(float), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &(device_matrix.width), &(device_matrix.height), &operation_choice };
		cudaLaunchKernel(
			(const void*)&matrixComplexElementWiseExtractions_fp32, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrixComplexElementWiseExtractions", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, Float);
	}
}

// Computes elementwise real values of given ComplexC matrix.
DeviceMatrix CudaFunctions_Single::real(DeviceMatrix device_matrix, bool on_hold) {
	return complexMatrixExtraction(device_matrix, 0, on_hold);
}

// Computes elementwise imag values of given ComplexC matrix.
DeviceMatrix CudaFunctions_Single::imag(DeviceMatrix device_matrix, bool on_hold) {
	return complexMatrixExtraction(device_matrix, 1, on_hold);
}

// Computes elementwise abs values of given ComplexC matrix.
DeviceMatrix CudaFunctions_Single::abs(DeviceMatrix device_matrix, bool on_hold) {
	if (device_matrix.dtype == ComplexC)
		return complexMatrixExtraction(device_matrix, 2, on_hold);
	else if (device_matrix.dtype == Float)
		return matrixElementWiseSinOrCosOrAbs(device_matrix, 2, 0, on_hold);
	else
		problematicExit("Abs operation supports only Float and ComplexC datatypes!");
}

// Computes elementwise sign of the values of the given ComplexC matrix.
DeviceMatrix CudaFunctions_Single::sign(DeviceMatrix device_matrix, bool on_hold) {

	if (device_matrix.dtype == ComplexC) {
		//cudaEventRecord(start);

		// Set grid dimensions
		dim3 grid((int)ceil((float)device_matrix.width / 32), (int)ceil((float)device_matrix.height / 32), 1);

		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(device_matrix.height * device_matrix.width * sizeof(float), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		int operation_choice = 3;
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &(device_matrix.width), &(device_matrix.height), &operation_choice };
		cudaLaunchKernel(
			(const void*)&matrixComplexElementWiseExtractions_fp32, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrixComplexElementWiseExtractions", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, Float);
	}
	else
		problematicExit("Sign function not ready for matrices other than of type ComplexC!");
}

// Computes eye, ones or zeros matrix of given dimensions.
DeviceMatrix CudaFunctions_Single::matrixEyeOrOnesOrZeros_wrapper(int width, int height, int operation_choice, bool on_hold) {
	//cudaEventRecord(start);

	// Set grid dimensions
	dim3 grid((int)ceil((float)width / 32), (int)ceil((float)height / 32), 1);

	// Allocate GPU buffers 
	float *dev_c = 0;
	//cudaMalloc((void**)&dev_c, height * width * sizeof(float));
	dev_c = static_cast<float*>(memAlloc(height * width * sizeof(float), on_hold));

	// Set arguments and launch the required kernel on the GPU.
	void* args[] = { &dev_c, &width, &height, &operation_choice };
	cudaLaunchKernel(
		(const void*)&matrixEyeOrOnesOrZeros_fp32, // pointer to kernel func.
		grid, // grid
		dim3(32, 32, 1), // block
		args,  // arguments
		0,
		stream);
	/*cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	insert_into_execution_times_map("matrixEyeOrOnesOrZeros", milliseconds);*/

	return DeviceMatrix(dev_c, width, height, Float);
}

// Returns an int matrix filled with zeros
DeviceMatrix CudaFunctions_Single::zerosInt(int width, int height, bool on_hold) {

	int *dev_c = 0;
	//cudaMalloc((void**)&dev_c, height * width * sizeof(int));
	//cudaEventRecord(start);
	dev_c = static_cast<int*>(memAlloc(height * width * sizeof(int), on_hold));
	cudaMemset((void**)&dev_c, 0, sizeof(int) * height * width);


	/*cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	insert_into_execution_times_map("zerosInt", milliseconds);*/
	return DeviceMatrix(dev_c, width, height, Int);
}

// Computes identity matrix of given dimensions
DeviceMatrix CudaFunctions_Single::eye(int width, bool on_hold) {
	return matrixEyeOrOnesOrZeros_wrapper(width, width, 0, on_hold);
}

// Computes ones matrix of given dimensions
DeviceMatrix CudaFunctions_Single::ones(int rows, int columns, bool on_hold) {
	return matrixEyeOrOnesOrZeros_wrapper(columns, rows, 1, on_hold);
}

// Computes zeros matrix of given dimensions
DeviceMatrix CudaFunctions_Single::zeros(int rows, int columns, bool on_hold) {
	return matrixEyeOrOnesOrZeros_wrapper(columns, rows, 2, on_hold);
}

// Computes flattened matrix using a row or column matrix.
DeviceMatrix CudaFunctions_Single::diagflat(DeviceMatrix device_matrix, bool on_hold) {
	int output_matrix_width;

	if (device_matrix.width == 1) {
		output_matrix_width = device_matrix.height;
	}
	else if (device_matrix.height == 1) {
		output_matrix_width = device_matrix.width;
	}
	else
		problematicExit("Float Diagonal flattening operation failed!");

	if (device_matrix.dtype == Float) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)output_matrix_width / 32), (int)ceil((float)output_matrix_width / 32), 1);

		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, output_matrix_width * output_matrix_width * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(output_matrix_width * output_matrix_width * sizeof(float), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &output_matrix_width, &output_matrix_width };
		cudaLaunchKernel(
			(const void*)&matrixDiagflat_fp32, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("diagflat Float", milliseconds);*/
		return DeviceMatrix(dev_c, output_matrix_width, output_matrix_width, Float);
	}
	else {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)output_matrix_width / 32), (int)ceil((float)output_matrix_width / 32), 1);

		// Allocate GPU buffers 
		cuComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, output_matrix_width * output_matrix_width * sizeof(cuComplex));
		dev_c = static_cast<cuComplex*>(memAlloc(output_matrix_width * output_matrix_width * sizeof(cuComplex), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &output_matrix_width, &output_matrix_width };
		cudaLaunchKernel(
			(const void*)&matrixDiagflatComplex_fp32, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("diagflat Complex", milliseconds);*/
		return DeviceMatrix(dev_c, output_matrix_width, output_matrix_width, ComplexC);
	}
}


// Computes flattened matrix using a row or column matrix and raised to a power elementwise.
DeviceMatrix CudaFunctions_Single::diagWithPower(DeviceMatrix device_matrix, int power, bool on_hold) {

	int output_matrix_width;

	if (device_matrix.width == 1)
		output_matrix_width = device_matrix.height;
	else if (device_matrix.height == 1)
		output_matrix_width = device_matrix.width;
	else
		problematicExit("Float Diagonal flattening operation failed!");

	if (device_matrix.dtype == Float) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)output_matrix_width / 32), (int)ceil((float)output_matrix_width / 32), 1);

		// Allocate GPU buffers 
		float *dev_c;
		//cudaMalloc((void**)&dev_c, output_matrix_width * output_matrix_width * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(output_matrix_width * output_matrix_width * sizeof(float), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &power, &output_matrix_width, &output_matrix_width };
		cudaLaunchKernel(
			(const void*)&matrixDiagflatWithPower_fp32, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrixDiagflatFloatWithPower", milliseconds);*/

		return DeviceMatrix(dev_c, output_matrix_width, output_matrix_width, Float);
	}
	else
		problematicExit("Diagonal flattening with power only supported for Floats.");
}

// Returns conjugate of a given ComplexC matrix
DeviceMatrix CudaFunctions_Single::conj(DeviceMatrix device_matrix, bool on_hold) {
	if (device_matrix.dtype == ComplexC) {
		return complexMatrixExtraction(device_matrix, 3, on_hold);
	}
	else
		problematicExit("Complex conjugate for dtypes except ComplexC is not supported!");
}

// Returns the concatenated matrix based on given axis. Only two matrices can be concatenated at once.
DeviceMatrix CudaFunctions_Single::concatenate(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, char axis, bool on_hold) {

	int target_matrix_rows = NULL;
	int target_matrix_columns = NULL;
	int operation_choice = 0;
	if (axis == 'x') {
		target_matrix_rows = device_matrix1.height;
		target_matrix_columns = device_matrix1.width + device_matrix2.width;
	}
	else if (axis == 'y') {
		target_matrix_rows = device_matrix1.height + device_matrix2.height;
		target_matrix_columns = device_matrix1.width;
		operation_choice = 1;
	}
	else
		problematicExit("Concatenation failed! 'axis' argument must be either 'x' or 'y'.");

	// Set grid dimensions
	dim3 grid((int)ceil((float)target_matrix_columns / 32), (int)ceil((float)target_matrix_rows / 32), 1);

	if (device_matrix1.dtype == Float) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, target_matrix_columns * target_matrix_rows * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(target_matrix_columns * target_matrix_rows * sizeof(float), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix1.device_pointer), &(device_matrix2.device_pointer), &dev_c,
			&target_matrix_columns,
			&target_matrix_rows,
			&(device_matrix1.width), &(device_matrix1.height), &(device_matrix2.width), &operation_choice };

		cudaLaunchKernel(
			(const void*)&matrixConcatenate_fp32, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrixConcatenateFloat", milliseconds);*/

		return DeviceMatrix(dev_c, target_matrix_columns, target_matrix_rows, Float);

	}
	else if (device_matrix1.dtype == ComplexC) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		cuComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, target_matrix_columns * target_matrix_rows * sizeof(cuComplex));
		dev_c = static_cast<cuComplex*>(memAlloc(target_matrix_columns * target_matrix_rows * sizeof(cuComplex), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix1.device_pointer), &(device_matrix2.device_pointer), &dev_c,
			&target_matrix_columns,
			&target_matrix_rows,
			&(device_matrix1.width), &(device_matrix1.height), &(device_matrix2.width), &operation_choice };

		cudaLaunchKernel(
			(const void*)&matrixConcatenateComplex_fp32, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrixConcatenateComplex", milliseconds);*/
		return DeviceMatrix(dev_c, target_matrix_columns, target_matrix_rows, ComplexC);
	}
}

// Returns the concatenated matrix based on given axis. Only two matrices can be concatenated at once.
DeviceMatrix CudaFunctions_Single::concat(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, char axis, bool on_hold) {
	return concatenate(device_matrix1, device_matrix2, axis);
}

// Returns the concatenated matrix based on given axis. Only two matrices can be concatenated at once.
DeviceMatrix CudaFunctions_Single::concat(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	return concatenate(device_matrix1, device_matrix2, 'y', on_hold);
}

// Returns the transpose of a given matrix
DeviceMatrix CudaFunctions_Single::transpose(DeviceMatrix device_matrix, bool on_hold) {

	if (device_matrix.dtype == Float) {

		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.width * device_matrix.height * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(device_matrix.width * device_matrix.height * sizeof(float), on_hold));

		const float alpha = 1.0;
		const float beta = 0.0;
		cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, device_matrix.height, device_matrix.width, &alpha,
			(float*)device_matrix.device_pointer, device_matrix.width, &beta, (float*)device_matrix.device_pointer,
			device_matrix.height, dev_c, device_matrix.height);

		return DeviceMatrix(dev_c, device_matrix.height, device_matrix.width, Float);
	}
	else if (device_matrix.dtype == ComplexC)
		problematicExit("Transpose for complex numbers not implemented!");
}

// Returns complex equivalent of given float matrix
DeviceMatrix CudaFunctions_Single::float_to_complex(DeviceMatrix device_matrix, bool on_hold) {

	if (device_matrix.dtype == Float) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)device_matrix.width / 32), (int)ceil((float)device_matrix.height / 32), 1);

		// Allocate GPU buffers 
		cuComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.width * device_matrix.height * sizeof(cuComplex));
		dev_c = static_cast<cuComplex*>(memAlloc(device_matrix.width * device_matrix.height * sizeof(cuComplex), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &(device_matrix.width), &(device_matrix.height) };
		cudaLaunchKernel(
			(const void*)&floatToComplex, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("floatToComplex", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, ComplexC);

	}
	else
		problematicExit("Float-to-Complex typecasting failed. Matrix datatype is not float!");
}

// Returns complex equivalent of given float matrix
DeviceMatrix CudaFunctions_Single::complexify(DeviceMatrix device_matrix, bool on_hold) {
	return float_to_complex(device_matrix, on_hold);
}

// Returns solution of linear equations of the form: Ax=B
DeviceMatrix CudaFunctions_Single::solve(DeviceMatrix device_matrix_1, DeviceMatrix device_matrix_2, bool on_hold) {

	cudaEventRecord(start);
	//device_matrix_2 = transpose(device_matrix_2);
	DeviceMatrix devInfo = zerosInt(1, 1);

	DeviceMatrix devIpiv = zerosInt(device_matrix_1.height, 1);

	int lwork; /* size of workspace */

	float *d_work; /* device workspace for getrf */
	checkCudaErrors(cusolverDnSgetrf_bufferSize(
		cusolverHandle,
		device_matrix_1.height,
		device_matrix_1.height,
		(float*)device_matrix_1.device_pointer,
		device_matrix_1.height,
		&lwork));

	//cudaMalloc((void**)&d_work, sizeof(float)*lwork);
	d_work = static_cast<float*>(memAlloc(sizeof(float)*lwork, on_hold));
	checkCudaErrors(cusolverDnSgetrf(
		cusolverHandle,
		device_matrix_1.height,
		device_matrix_1.height,
		(float*)device_matrix_1.device_pointer,
		device_matrix_1.height,
		d_work,
		(int*)devIpiv.device_pointer,
		(int*)devInfo.device_pointer
		));
	checkCudaErrors(cusolverDnSgetrs(
		cusolverHandle,
		CUBLAS_OP_N,
		device_matrix_1.height,
		1,
		(float*)device_matrix_1.device_pointer,
		device_matrix_1.height,
		(int*)devIpiv.device_pointer,
		(float*)device_matrix_2.device_pointer,
		device_matrix_2.height,
		(int*)devInfo.device_pointer
		));
	//DeviceMatrix intermediate(device_matrix_2.device_pointer, device_matrix_2.height, device_matrix_2.width, Float);
	/*cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	insert_into_execution_times_map("solve", milliseconds);
	printf("\tsolve time = %f", milliseconds);*/
	return device_matrix_2;
	//return transpose(device_matrix_2, on_hold);







	//// --- Creating the array of pointers needed as input/output to the batched getrf
	//// A
	////cudaEventRecord(start);
	//float *device_matrix_1_copy = 0;
	//device_matrix_1_copy = (float*)memAlloc(device_matrix_1.height * device_matrix_1.height * sizeof(float));
	//cudaMemcpyAsync(device_matrix_1_copy, (float*)device_matrix_1.device_pointer, device_matrix_1.height * device_matrix_1.height * sizeof(float), cudaMemcpyDeviceToDevice, stream);

	//float **h_in_pointers = (float **)malloc(sizeof(float *));
	//h_in_pointers[0] = device_matrix_1_copy;

	//float **d_in_pointers = (float**)memAlloc(sizeof(float *), on_hold);
	//cudaMemcpyAsync(d_in_pointers, h_in_pointers, sizeof(float *), cudaMemcpyHostToDevice, stream);


	//int *d_pivotArray = (int*)memAlloc(device_matrix_1.height * sizeof(int), on_hold);
	//int *d_InfoArray = (int*)(memAlloc(sizeof(int)));

	//cudaEventRecord(start);
	//cublasSgetrfBatched(cublasHandle, device_matrix_1.height, d_in_pointers, device_matrix_1.height,
	//	d_pivotArray, d_InfoArray, 1);
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//float milliseconds = 0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("\tcublasSgetrfBatched time = %fms\t", milliseconds);
	///*cudaMemcpyAsync(h_InfoArray, d_InfoArray, sizeof(int), cudaMemcpyDeviceToHost, stream);
	//if (h_InfoArray[0] != 0) {
	//fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", 0);
	//cudaDeviceReset();
	//exit(EXIT_FAILURE);
	//}*/

	///*******************************************/
	///* APPROACH NR.1: THROUGH THE INVERSE OF A */
	///*******************************************/

	//// --- Allocate host space for the inverted matrices 
	//float *h_C = new float[device_matrix_1.width * device_matrix_1.width];

	//// --- Allocate device space for the inverted matrices 
	//float *d_C = (float*)memAlloc(device_matrix_1.width*device_matrix_1.width*sizeof(float), on_hold);

	//// --- Creating the array of pointers needed as output to the batched getri
	//float **h_out_pointers = (float **)malloc(sizeof(float *));
	//h_out_pointers[0] = (float *)((char*)d_C);

	//float **d_out_pointers = (float**)memAlloc(sizeof(float *), false);
	//cudaMemcpyAsync(d_out_pointers, h_out_pointers, sizeof(float *), cudaMemcpyHostToDevice, stream);
	//cudaEventRecord(start);
	//(cublasSgetriBatched(cublasHandle, device_matrix_1.width, (const float **)d_in_pointers, device_matrix_1.width, d_pivotArray, d_out_pointers, device_matrix_1.width, d_InfoArray, 1));
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//milliseconds = 0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("\tcublasSgetriBatched time = %fms\t", milliseconds);
	////(cudaMemcpy(h_C, d_C, device_matrix_1.width*device_matrix_1.width*sizeof(float), cudaMemcpyDeviceToHost));
	////// --- The output inverted matrix in column-major format
	////printf("\n\n");
	////for (int i = 0; i<device_matrix_1.width*device_matrix_1.width; i++) printf("C[%i]=%f\n", i, h_C[i]);





	//float alpha1 = 1.0;
	//float beta1 = 0.0;

	//float *d_X = (float*)memAlloc(device_matrix_2.height * sizeof(float), on_hold);
	//cudaEventRecord(start);
	//(cublasSgemv(cublasHandle, CUBLAS_OP_T, device_matrix_1.width, device_matrix_1.width, &alpha1, d_C, device_matrix_1.width, (float*)device_matrix_2.device_pointer, 1, &beta1, d_X, 1));
	////(cudaMemcpy(h_X, d_X, device_matrix_1.width*sizeof(float), cudaMemcpyDeviceToHost));
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//milliseconds = 0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("\tcublasSgetriBatched time = %fms\t", milliseconds);
	////// --- The output inverted matrix in column-major format
	////printf("\n\n");
	////for (int i = 0; i<device_matrix_1.width; i++) printf("X[%i]=%f\n", i, h_X[i]);
	//return DeviceMatrix((void*)d_X, device_matrix_2.width, device_matrix_2.height, Float);

}

// Returns sliced matrices based on given row-column coordinates.
DeviceMatrix CudaFunctions_Single::slice(DeviceMatrix device_matrix, int row_start, int row_end_exclusive, int column_start, int column_end_exclusive, bool on_hold) {

	if (device_matrix.dtype == Float) {
		//cudaEventRecord(start);
		int num_of_rows = row_end_exclusive - row_start;
		int num_of_cols = column_end_exclusive - column_start;

		// Set grid dimensions
		int blockHeight = 1024;
		if (device_matrix.height < blockHeight)
			blockHeight = device_matrix.height;
		dim3 grid((int)ceil((float)device_matrix.width / 1), (int)ceil((float)device_matrix.height / blockHeight), 1);

		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, num_of_cols * num_of_rows * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(num_of_cols * num_of_rows * sizeof(float), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c,
			&row_start, &row_end_exclusive,
			&column_start, &column_end_exclusive,
			&(device_matrix.width),
			&num_of_cols };

		cudaLaunchKernel(
			(const void*)&slice_fp32, // pointer to kernel func.
			grid, // grid
			dim3(1, blockHeight, 1), // block
			args,  // arguments
			0,
			stream);
	/*	cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("sliceFloat", milliseconds);*/
		return DeviceMatrix(dev_c, num_of_cols, num_of_rows, Float);
	}
	else if (device_matrix.dtype == ComplexC) {
		//cudaEventRecord(start);
		int num_of_rows = row_end_exclusive - row_start;
		int num_of_cols = column_end_exclusive - column_start;

		// Set grid dimensions
		int blockHeight = 1024;
		if (device_matrix.height < blockHeight)
			blockHeight = device_matrix.height;
		dim3 grid((int)ceil((float)device_matrix.width / 1), (int)ceil((float)device_matrix.height / blockHeight), 1);

		// Allocate GPU buffers 
		cuComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, num_of_cols * num_of_rows * sizeof(float));
		dev_c = static_cast<cuComplex*>(memAlloc(num_of_cols * num_of_rows * sizeof(cuComplex), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c,
			&row_start, &row_end_exclusive,
			&column_start, &column_end_exclusive,
			&(device_matrix.width),
			&num_of_cols };

		cudaLaunchKernel(
			(const void*)&sliceComplex64, // pointer to kernel func.
			grid, // grid
			dim3(1, blockHeight, 1), // block
			args,  // arguments
			0,
			stream);
		/*	cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("sliceFloat", milliseconds);*/
		return DeviceMatrix(dev_c, num_of_cols, num_of_rows, ComplexC);
	}
	else
		problematicExit("Slicing is currently supported only for Float and ComplexC!");
}

// This method can be used to perform slicing of matrices with the help of given 1D indices matrix.
DeviceMatrix CudaFunctions_Single::slicei(DeviceMatrix device_matrix, DeviceMatrix indices_device_matrix, bool on_hold) {

	if (device_matrix.dtype == Float) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)indices_device_matrix.width / 1024), (int)ceil((float)indices_device_matrix.height / 1), 1);

		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, indices_device_matrix.width * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(indices_device_matrix.width * sizeof(float), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c,
			&(indices_device_matrix.device_pointer),
			&(device_matrix.height), &(device_matrix.width),
			&(indices_device_matrix.width) };

		cudaLaunchKernel(
			(const void*)&slice_with_indices_fp32, // pointer to kernel func.
			grid, // grid
			dim3(1024, 1, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("slice_with_indices", milliseconds);*/
		return DeviceMatrix(dev_c, 1, indices_device_matrix.width, Float);
	}
	else
		problematicExit("Slicing with indices is currently supported only for Floats!");
}

//This method can be used to perform special slicing of matrix R. Please refrain from using it for general purpose.
DeviceMatrix CudaFunctions_Single::specialSlicingOnR(DeviceMatrix device_matrix, DeviceMatrix indices, bool on_hold) {

	if (device_matrix.dtype == Float) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)indices.width / 32), (int)ceil((float)indices.height / 32), 1);

		// Allocate GPU buffers 
		float *dev_c = 0;
		//cudaMalloc((void**)&dev_c, indices.width * sizeof(float));
		dev_c = static_cast<float*>(memAlloc(indices.width * sizeof(float), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c,
			&(indices.device_pointer),
			&(device_matrix.height), &(device_matrix.width),
			&(indices.width) };

		cudaLaunchKernel(
			(const void*)&specialSlicingOnR_kernel_fp32, // pointer to kernel func.
			grid, // grid
			dim3(1024, 1, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("specialSlicingOnR_kernel", milliseconds);*/

		return DeviceMatrix(dev_c, indices.width, 1, Float);
	}
	else
		problematicExit("Special Slicing for R matrix is currently supported for Floats!");
}

//This method can be used to perform special slicing of matrix H. Please refrain from using it for general purpose.
DeviceMatrix CudaFunctions_Single::specialSlicingOnH(DeviceMatrix device_matrix, DeviceMatrix indices, bool on_hold) {

	if (device_matrix.dtype == Float) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)indices.width / 1024), (int)ceil((float)indices.height / 1), 1);

		// Allocate GPU buffers 
		DeviceMatrix output_device_matrix = zeros(indices.width, device_matrix.width);

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &(output_device_matrix.device_pointer),
			&(indices.device_pointer),
			&(device_matrix.height), &(device_matrix.width),
			&(indices.width),
			&(device_matrix.width) };

		cudaLaunchKernel(
			(const void*)&specialSlicingOnH_kernel_fp32, // pointer to kernel func.
			grid, // grid
			dim3(1024, 1, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("specialSlicingOnH_kernel", milliseconds);*/
		return DeviceMatrix(output_device_matrix.device_pointer, device_matrix.width, indices.width, Float);
	}
	else
		problematicExit("Special Slicing for H matrix is currently supported for Floats!");
}

// Returns the maximum value within a given matrix.
double CudaFunctions_Single::maxValue(DeviceMatrix device_matrix, bool on_hold) {
	if (device_matrix.dtype == Float) {
		//cudaEventRecord(start);
		//int threads = 1024;
		//int blocks = (int)(ceil((float)device_matrix.height * (float)device_matrix.width / threads));

		//// Allocate GPU buffers 
		//float *output_intermediate_gpu = 0;
		//float *output_intermediate2_gpu = 0;
		//cudaMalloc((void**)&output_intermediate_gpu, blocks * sizeof(float));
		//cudaMalloc((void**)&output_intermediate2_gpu, (int)(ceil((float)blocks / 1024)) * sizeof(float));

		//// Set arguments and launch the required kernel on the GPU.
		//int dimensions_product = device_matrix.height * device_matrix.width;
		//void* args[] = { &output_intermediate_gpu, 
		//	&(device_matrix.device_pointer),
		//	&dimensions_product };
		//// Set grid dimensions
		//dim3 grid(blocks, 1, 1);

		//// Phase 1
		//cudaLaunchKernel(
		//	(const void*)&shmem_min_max_reduce_kernel, // pointer to kernel func.
		//	grid, // grid
		//	dim3(threads, 1, 1), // block
		//	args,  // arguments
		//	8 * threads,
		//	0
		//	);


		//// Set arguments and launch the required kernel on the GPU.
		//void* args2[] = { &output_intermediate2_gpu,
		//	&(output_intermediate_gpu),
		//	&blocks };

		//// Set grid dimensions
		//dim3 grid2((int)ceil((float)blocks/threads), 1, 1);
		//// Phase 2
		//cudaLaunchKernel(
		//	(const void*)&shmem_min_max_reduce_kernel, // pointer to kernel func.
		//	grid2, // grid
		//	dim3(threads, 1, 1), // block
		//	args2,  // arguments
		//	8 * threads,
		//	0
		//	);

		//// Allocate Final GPU Buffers
		//float *output_final_max_gpu = 0;
		//cudaMalloc((void**)&output_final_max_gpu, sizeof(float));

		//// Set arguments and launch the required kernel on the GPU.
		//int dimensions = (int)ceil((float)blocks / threads);
		//void* args3[] = { &output_final_max_gpu,
		//	&(output_intermediate2_gpu),
		//	&dimensions };
		//// Phase 3
		//cudaLaunchKernel(
		//	(const void*)&shmem_min_max_reduce_kernel, // pointer to kernel func.
		//	dim3(1, 1, 1), // grid
		//	dim3(threads, 1, 1), // block
		//	args3,  // arguments
		//	8 * threads,
		//	0
		//	);

		//return DeviceMatrix(output_final_max_gpu, 1, 1, Float);


		/***** Finding max value through CUBLAS. ******/
		int max_index;
		cublasIsamax(cublasHandle, device_matrix.height, (float*)device_matrix.device_pointer, 1, &max_index);
		max_index--; // To adjust from 1-based index to 0-based index.
		float max;
		cudaMemcpyAsync(&max, ((float*)(device_matrix.device_pointer)) + max_index, sizeof(float), cudaMemcpyDeviceToHost, stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("maxValue", milliseconds);*/

		return max;
	}
	else
		problematicExit("Max reduction is only supported for Floats.");
}

// Returns a matrix which is the result of insertion of a smaller matrix inside a bigger matrix.
// WARNING: The existing bigger matrix would be overwritten.
DeviceMatrix CudaFunctions_Single::insert(DeviceMatrix input_big_matrix, int row_start, int row_end_exclusive,
	int column_start, int column_end_exclusive, DeviceMatrix input_small_matrix, bool on_hold) {

	if (input_big_matrix.dtype == Float) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)input_big_matrix.width / 32), (int)ceil((float)input_big_matrix.height / 32), 1);

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(input_big_matrix.device_pointer),
			&(input_small_matrix.device_pointer),
			&row_start, &row_end_exclusive,
			&column_start, &column_end_exclusive,
			&(input_big_matrix.height), &(input_big_matrix.width) };
		cudaLaunchKernel(
			(const void*)&matrix_insert_fp32, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
	/*	cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrix_insert", milliseconds);*/
		return DeviceMatrix(input_big_matrix.device_pointer, input_big_matrix.width, input_big_matrix.height, Float);
	}
	else
		problematicExit("Matrix insertion is currently only supported for Floats!");
}


void CudaFunctions_Single::to_DND_pool(DeviceMatrix device_matrix) {
	return dMManager.toDndDeviceMemory(&device_matrix);
}

DeviceMatrix CudaFunctions_Single::sort(DeviceMatrix device_matrix, bool on_hold) {
	sort_on_device_fp32(static_cast<float*>(device_matrix.device_pointer), device_matrix.width);
	return device_matrix;
}

void CudaFunctions_Single::setStream(cudaStream_t cudaStream) {
	cusolverStatus_t status;
	stream = cudaStream;
	cublasSetStream(cublasHandle, stream);

	status = cusolverDnSetStream(cusolverHandle, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);
}

void CudaFunctions_Single::deviceSynchronize() {
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("\n\n");
		fprintf(stderr, "cudaDeviceSynchronize failed!");
	}
}

void CudaFunctions_Single::write_to_file(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2) {
	if (grid_size != NULL) {
		float *float_elements1 = new float[device_matrix1.height * device_matrix1.width];
		cudaMemcpy(float_elements1, device_matrix1.device_pointer, device_matrix1.height * device_matrix1.width * sizeof(float), cudaMemcpyDeviceToHost);
		float *float_elements2 = new float[device_matrix2.height * device_matrix2.width];
		cudaMemcpy(float_elements2, device_matrix2.device_pointer, device_matrix2.height * device_matrix2.width * sizeof(float), cudaMemcpyDeviceToHost);

		// current date/time based on current system
		time_t now = time(0);
		// convert now to string form
		std::string dt = ctime(&now);

		std::ofstream myfile;
		std::string result_file = "Results_FP32\\result_" + std::to_string(grid_size) + ".csv";;
		myfile.open(result_file, std::ios_base::trunc);

		std::cout << "\n*** Writing matrix contents of eK/fK matrices to a file. ***\n";
		std::string file_content = "";
		for (int i = 0; i < device_matrix1.height; i++) {
			file_content += ("Row#" + std::to_string(i) + ",");
			for (int j = 0; j < device_matrix1.width; j++) {
				int current_index = j + i * device_matrix1.width;
				file_content += std::to_string(float_elements1[current_index]) + "," + std::to_string(float_elements2[current_index]);
			}
			file_content += "\n";
		}
		myfile << file_content << std::endl;
		myfile.close();
	} else
		std::cout << "\n Cannot write results to file since grid-size is not set for CudaFunctions_Single instance." << std::endl;

}

DeviceMatrix CudaFunctions_Single::angle(DeviceMatrix eK, DeviceMatrix fK, bool on_hold) {
	if (eK.dtype == Float) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)eK.width / 32), (int)ceil((float)eK.height / 32), 1);

		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(eK.height * eK.width * sizeof(float), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(eK.device_pointer), &(fK.device_pointer), &dev_c, &(eK.width), &(eK.height) };
		cudaLaunchKernel(
			(const void*)&matrixElementWiseAngles_fp32, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("elementWiseAnglesDouble", milliseconds);*/

		return DeviceMatrix(dev_c, eK.width, eK.height, Float);
	}
	else
		problematicExit("Angle method currently support Float matrices only!");

}


DeviceMatrix CudaFunctions_Single::wrapPointersIntoPointerArrays(DeviceMatrix device_matrix_1, bool on_hold) {
	DeviceMatrix d;
	return d;
}

float CudaFunctions_Single::getVectorElementAtIndex(DeviceMatrix device_matrix, int index) {
	double element;
	cudaMemcpyAsync(&element, ((float*)(device_matrix.device_pointer)) + index, sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	return element;
}