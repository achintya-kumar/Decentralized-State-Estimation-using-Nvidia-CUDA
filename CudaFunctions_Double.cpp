#include "CudaFunctions_Double.h"

CudaFunctions_Double::~CudaFunctions_Double() {}

// Marks the beginning of a step
void CudaFunctions_Double::stepBegin() {}

// Marks the end of a step; releases allocations made during one step to the pool of available allocations
void CudaFunctions_Double::sanitizeMemoryPools() {
	cudaStreamSynchronize(stream);
	dMManager.releaseStepAllocationsToPool();
}

// Releases allocations in ON-HOLD state to the pool of available allocations
void CudaFunctions_Double::releaseOnHoldAllocationsToPool() {
	dMManager.releaseOnHoldAllocationsToPool();
}

// Returns a device-pointer of given size
void* CudaFunctions_Double::memAlloc(size_t size, bool on_hold) {
	return dMManager.getDeviceMemory(size, on_hold);
}

// To be used in case of abnormal exit
void CudaFunctions_Double::problematicExit(char* message) {
	printf(message);
	getchar();
	exit(1);

}

DeviceMatrix CudaFunctions_Double::to_device(double *array, int width, int height) {
	double *double_elements;
	cudaMalloc((void**)&double_elements, height * width * sizeof(double));
	cudaMemcpy(double_elements, array, height * width * sizeof(double), cudaMemcpyHostToDevice);
	dMManager.total_occupied_memory_in_bytes += height * width * sizeof(double);
	return DeviceMatrix(double_elements, width, height, Double);
}


DeviceMatrix CudaFunctions_Double::map_non_zero_elements(DeviceMatrix device_matrix, bool on_hold) {
	dim3 grid((int)ceil((float)device_matrix.width / 32), (int)ceil((float)device_matrix.height / 32), 1);

	// Allocate GPU buffers  .
	double *odata;
	//cudaMalloc((void**)&non_zero_count, sizeof(int));
	odata = static_cast<double*>(memAlloc(device_matrix.width * device_matrix.height * sizeof(double), on_hold));

	// Launch a kernel on the GPU.
	void* args[] = { &(device_matrix.device_pointer), &odata, &(device_matrix.width), &(device_matrix.height) };
	cudaLaunchKernel(
		(const void*)&mapNonZeroIndices, // pointer to kernel func.
		grid, // grid
		dim3(32, 32, 1), // block
		args,  // arguments
		0,
		stream);

	return DeviceMatrix(odata, device_matrix.width, device_matrix.height, Double);
}

// Returns a single-row matrix with indices of all non-zero elements of given on-device matrix
DeviceMatrix CudaFunctions_Double::extract_indices_for_non_zero(DeviceMatrix device_matrix) {

	dim3 grid((int)ceil((float)device_matrix.width / 1024), (int)ceil((float)device_matrix.height / 1), 1);

	// Allocate GPU buffers  .
	int *non_zero_count = NULL;
	//cudaMalloc((void**)&non_zero_count, sizeof(int));
	non_zero_count = static_cast<int*>(memAlloc(sizeof(int)));

	// Launch a kernel on the GPU.
	void* args[] = { &(device_matrix.device_pointer), &non_zero_count, &(device_matrix.width), &(device_matrix.height) };

	cudaLaunchKernel(
		(const void*)&countNonZeroElements, // pointer to kernel func.
		grid, // grid
		dim3(1, 1024, 1), // block
		args,  // arguments
		0,
		stream);

	int non_zero_count_h;
	cudaMemcpy(&non_zero_count_h, non_zero_count, sizeof(int), cudaMemcpyDeviceToHost);

	// Allocate GPU buffers 
	double *output;
	//cudaMalloc((void**)&output, non_zero_count_h * sizeof(double));
	output = static_cast<double*>(memAlloc(non_zero_count_h * sizeof(double)));

	// For atomic tracking of indices
	DeviceMatrix nres = zerosInt(1, 1);

	// Launch a kernel on the GPU.
	void* args2[] = { &output, &(nres.device_pointer), &(device_matrix.device_pointer), &(device_matrix.width), &(device_matrix.height) };

	cudaLaunchKernel(
		(const void*)&filter_k, // pointer to kernel func.
		grid, // grid
		dim3(1, 1024, 1), // block
		args2,  // arguments
		0,
		stream);

	return DeviceMatrix(output, non_zero_count_h, 1, Double);
}


// Returns single element host matrix after copying from device memory.
double CudaFunctions_Double::to_host(DeviceMatrix device_matrix) {
	double on_host;
	cudaMemcpyAsync(&on_host, device_matrix.device_pointer, sizeof(double), cudaMemcpyDeviceToHost, stream);
	return on_host;
}

// Loads the matrix from the given CSV file into device memory.
DeviceMatrix CudaFunctions_Double::to_device(std::string file) {

	// PHASE 1: Initial assessment of the matrix's csv file.
	std::ifstream  data(file);
	std::string line;
	int rows = 0, columns = 0;
	bool isComplexZ = false;
	while (std::getline(data, line))
	{
		std::stringstream  lineStream(line);
		std::string        cell;
		while (std::getline(lineStream, cell, ','))
		{
			// You have a cell!!!!
			if (columns == 0) {			// Examining the first cell for its data-type. It can either be a floating point number or a Complex floating point number.
				try {
					double d = std::stod(cell);
					isComplexZ = false;
				}
				catch (std::exception& e)
				{
					isComplexZ = true;
				}
			}

			if (rows == 0)
				columns++;
		}
		rows++;
	}

	//printf("\nFile: %s", file);
	//printf("\nRows = %d, Columns = %d, Dtype = %s", rows, columns, isComplexZ ? "ComplexZ" : "Double");


	// PHASE 2: Loading the items into the memory.
	cuDoubleComplex *complex_elements;
	cuDoubleComplex *complex_elements_d; // For GPU Allocation
	double *double_elements;
	double *double_elements_d;			 // For GPU Allocation
	// Initializing array based on whether the elements are double or of complex types.
	if (isComplexZ)
		complex_elements = new cuDoubleComplex[rows * columns];
	else
		double_elements = new double[rows * columns];

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
			if (!isComplexZ) // When the elements are of type Double
				double_elements[counter] = std::stod(cell);

			else {			 // When the elements are of type ComplexZ
				boost::replace_all(cell, "(", "");
				boost::replace_all(cell, ")", "");

				std::stringstream  innerLineStream(cell);
				std::string        innerCell;
				double complex_parts[2];
				int complex_parts_counter = 0;
				while (std::getline(innerLineStream, innerCell, '+')) {
					boost::replace_all(innerCell, "j", "");
					complex_parts[complex_parts_counter] = std::stod(innerCell);
					complex_parts_counter++;
				}
				complex_elements[counter] = make_cuDoubleComplex(complex_parts[0], complex_parts[1]); // Constructing complex number for CUDA
			}
			counter++;
		}
	}

	//std::cout << std::endl;
	if (!isComplexZ) {
		double_elements_d = static_cast<double*>(dMManager.getDeviceMemory(rows * columns * sizeof(double)));
		//checkCudaErrors(cudaMalloc((void**)&double_elements_d, rows * columns * sizeof(double)));
		checkCudaErrors(cudaMemcpy(double_elements_d, double_elements, rows * columns * sizeof(double), cudaMemcpyHostToDevice));
		delete[] double_elements; // Heap cleanup
		return DeviceMatrix(double_elements_d, columns, rows, Double);
	}
	else {
		complex_elements_d = static_cast<cuDoubleComplex*>(dMManager.getDeviceMemory(rows * columns * sizeof(cuDoubleComplex)));
		//checkCudaErrors(cudaMalloc((void**)&complex_elements_d, rows * columns * sizeof(cuDoubleComplex)));
		checkCudaErrors(cudaMemcpy(complex_elements_d, complex_elements, rows * columns * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
		delete[] complex_elements; // Heap cleanup
		return DeviceMatrix(complex_elements_d, columns, rows, ComplexZ);
	}
}

DeviceMatrix CudaFunctions_Double::host_array_wrapped_in_DeviceMatrix(DeviceMatrix device_matrix) {
	if (device_matrix.dtype == Double) {
		double *double_elements = new double[device_matrix.height * device_matrix.width];
		cudaMemcpy(double_elements, device_matrix.device_pointer, device_matrix.height * device_matrix.width * sizeof(double), cudaMemcpyDeviceToHost);
		return DeviceMatrix(double_elements, device_matrix.width, device_matrix.height, Double);
	}
}

void CudaFunctions_Double::printMatrix(DeviceMatrix device_matrix) {
	//system("CLS");
	int height_limit = device_matrix.height, width_limit = device_matrix.width;									// Limiting the dimensions while printing
	/*if (device_matrix.height > 9)
	height_limit = 9;
	if (device_matrix.width > 9)
	width_limit = 9;*/

	if (device_matrix.dtype == Double) {
		double *double_elements = new double[device_matrix.height * device_matrix.width];
		cudaMemcpy(double_elements, device_matrix.device_pointer, device_matrix.height * device_matrix.width * sizeof(double), cudaMemcpyDeviceToHost);
		printf("\n\n=== Printing Double matrix ===\n");
		for (int i = 0; i < height_limit; i++) {
			printf("Row#%d\t", i);
			for (int j = 0; j < width_limit; j++) {
				int current_index = j + i * device_matrix.width;
				printf("%.10f\t", double_elements[current_index]);
			}
			printf("\n");
		}
		delete[] double_elements;
	}
	else if (device_matrix.dtype == Int) {
		int *double_elements = new int[device_matrix.height * device_matrix.width];
		cudaMemcpy(double_elements, device_matrix.device_pointer, device_matrix.height * device_matrix.width * sizeof(int), cudaMemcpyDeviceToHost);
		printf("\n\n=== Printing Double matrix ===\n");
		for (int i = 0; i < height_limit; i++) {
			printf("Row#%d\t", i);
			for (int j = 0; j < width_limit; j++) {
				int current_index = j + i * device_matrix.width;
				printf("%d\t", double_elements[current_index]);
			}
			printf("\n");
		}
		delete[] double_elements;
	}
	else if (device_matrix.dtype == ComplexZ) {
		cuDoubleComplex *complex_elements = new cuDoubleComplex[device_matrix.height * device_matrix.width];
		cudaMemcpy(complex_elements, device_matrix.device_pointer, device_matrix.height * device_matrix.width * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
		printf("\n\n=== Printing Complex matrix ===\n");
		for (int i = 0; i < height_limit; i++) {
			printf("Row#%d\t", i);
			for (int j = 0; j < width_limit; j++) {
				int current_index = j + i * device_matrix.width;
				printf("%.10f + %.10fj\t", cuCreal(complex_elements[current_index]), cuCimag(complex_elements[current_index]));
			}
			printf("\n");
		}
		delete[] complex_elements;
	}
}



void CudaFunctions_Double::insert_into_execution_times_map(std::string kernel_name, float milliseconds) {
	Map_Of_Execution_Times::iterator map_iterator = map_of_execution_times.find(kernel_name);

	if (map_iterator != map_of_execution_times.end()) {
		map_iterator->second += milliseconds;
	}
	else {
		map_of_execution_times.insert(std::pair <std::string, float>(kernel_name, milliseconds));		// Load up the new vector into the pool with key=size
	}
}

void CudaFunctions_Double::print_execution_times() {
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

DeviceMatrix CudaFunctions_Double::update_eK_fK(DeviceMatrix a, DeviceMatrix b) {

	dim3 grid((int)ceil((float)a.width / 1), (int)ceil((float)a.height / 1024), 1);
	int operation_choice = 0;
	double *dev_c = 0;

	// Allocate GPU buffers  .
	//cudaMalloc((void**)&dev_c, a.height * a.width * sizeof(double));
	dev_c = static_cast<double*>(a.device_pointer);

	// Launch a kernel on the GPU.
	void* args[] = { &(a.device_pointer), &(b.device_pointer), &dev_c, &(a.width), &(a.height), &operation_choice };
	if (a.height == b.height && a.width == b.width) {
		////cudaEventRecord(start);
		cudaLaunchKernel(
			(const void*)&matrixElementWiseASMD, // pointer to kernel func.
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

	return DeviceMatrix(dev_c, a.width, a.height, Double);
}


// ASMD operations on two matrices.
DeviceMatrix CudaFunctions_Double::matrixElementWiseASMD_wrapper(DeviceMatrix a, DeviceMatrix b, unsigned int operation_choice, bool on_hold) {

	dim3 grid((int)ceil((float)a.width / 32), (int)ceil((float)a.height / 32), 1);

	if (a.dtype == Double) {
		double *dev_c = 0;

		// Allocate GPU buffers  .
		//cudaMalloc((void**)&dev_c, a.height * a.width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(a.height * a.width * sizeof(double), on_hold));

		// Launch a kernel on the GPU.
		void* args[] = { &(a.device_pointer), &(b.device_pointer), &dev_c, &(a.width), &(a.height), &operation_choice };

		if (a.height == b.height && a.width == b.width) {
			////cudaEventRecord(start);
			cudaLaunchKernel(
				(const void*)&matrixElementWiseASMD, // pointer to kernel func.
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
			////cudaEventRecord(start);
			cudaLaunchKernel(
				(const void*)&matrixElementWiseASMDForSingleRow, // pointer to kernel func.
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
			problematicExit("Incompatible matrix dimensions for double elementwise ASMD.");

		return DeviceMatrix(dev_c, a.width, a.height, Double);
	}
	else if (a.dtype == ComplexZ) {

		// Container pointer for result
		cuDoubleComplex *dev_c = 0;

		// Allocate GPU buffers for three vectors (two input, one output).
		//cudaMalloc((void**)&dev_c, a.height * a.width * sizeof(cuDoubleComplex));
		dev_c = static_cast<cuDoubleComplex*>(memAlloc(a.height * a.width * sizeof(cuDoubleComplex), on_hold));


		// Launch a kernel on the GPU.
		void* args[] = { &(a.device_pointer), &(b.device_pointer), &dev_c, &(a.width), &(a.height), &operation_choice };

		if (a.height == b.height && a.width == b.width) {
			////cudaEventRecord(start);
			cudaLaunchKernel(
				(const void*)&matrixComplexElementWiseASMD, // pointer to kernel func.
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

		return DeviceMatrix(dev_c, a.width, a.height, ComplexZ);
	}
}

// Add 2 matrices elementwise.
DeviceMatrix CudaFunctions_Double::add(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	return matrixElementWiseASMD_wrapper(device_matrix1, device_matrix2, 0, on_hold);
}

// Subtract 2 matrices elementwise.
DeviceMatrix CudaFunctions_Double::sub(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	return matrixElementWiseASMD_wrapper(device_matrix1, device_matrix2, 1, on_hold);
}

// Multiply 2 matrices elementwise.
DeviceMatrix CudaFunctions_Double::mul(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	return matrixElementWiseASMD_wrapper(device_matrix1, device_matrix2, 2, on_hold);
}

// Divide 2 matrices elementwise.
DeviceMatrix CudaFunctions_Double::div(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	return matrixElementWiseASMD_wrapper(device_matrix1, device_matrix2, 3, on_hold);
}

// ASMD operations on a scalar and a matrix.
DeviceMatrix CudaFunctions_Double::matrixElementWiseScalarASMD_wrapper(DeviceMatrix device_matrix, float scalar, int operation_choice, int order_of_operation, bool on_hold) {
	dim3 grid((int)ceil((float)device_matrix.width / 32), (int)ceil((float)device_matrix.height / 32), 1);

	if (device_matrix.dtype == Double) {
		//cudaEventRecord(start);
		double *dev_c = 0;

		// Allocate GPU buffers  .
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(device_matrix.height * device_matrix.width * sizeof(double), on_hold));

		// Launch a kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &(scalar), &dev_c, &(device_matrix.width), &(device_matrix.height), &operation_choice, &order_of_operation };
		cudaLaunchKernel(
			(const void*)&matrixElementWiseScalarASMD, // pointer to kernel func.
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

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, Double);
	}
	else if (device_matrix.dtype == ComplexZ) {
		//cudaEventRecord(start);
		cuDoubleComplex *dev_c = 0;

		// Allocate GPU buffers  .
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(cuDoubleComplex));
		dev_c = static_cast<cuDoubleComplex*>(memAlloc(device_matrix.height * device_matrix.width * sizeof(cuDoubleComplex), on_hold));

		// Launch a kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &(scalar), &dev_c, &(device_matrix.width), &(device_matrix.height), &operation_choice, &order_of_operation };
		cudaLaunchKernel(
			(const void*)&matrixComplexElementWiseScalarASMD, // pointer to kernel func.
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

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, ComplexZ);
	}
}

// Add a scalar elementwise to a matrix.
DeviceMatrix CudaFunctions_Double::adds(DeviceMatrix device_matrix1, float scalar, bool on_hold) {
	return matrixElementWiseScalarASMD_wrapper(device_matrix1, scalar, 0, 0, on_hold);
}

// Subtract a scalar elementwise from a matrix. 'order_of_operation' argument specifies direction of the operation.
DeviceMatrix CudaFunctions_Double::subs(DeviceMatrix device_matrix1, float scalar, int order_of_operation, bool on_hold) {
	return matrixElementWiseScalarASMD_wrapper(device_matrix1, scalar, 1, order_of_operation, on_hold);
}

// Subtract a scalar elementwise from a matrix. 'order_of_operation' argument specifies direction of the operation.
DeviceMatrix CudaFunctions_Double::subs(DeviceMatrix device_matrix1, float scalar, bool on_hold) {
	return matrixElementWiseScalarASMD_wrapper(device_matrix1, scalar, 1, 0, on_hold);
}

// Multiply a scalar elementwise to a matrix.
DeviceMatrix CudaFunctions_Double::muls(DeviceMatrix device_matrix1, float scalar, bool on_hold) {
	return matrixElementWiseScalarASMD_wrapper(device_matrix1, scalar, 2, 0, on_hold);
}

// Divide matrix elementwise with a scalar. 'order_of_operation' argument specifies direction of the operation.
DeviceMatrix CudaFunctions_Double::divs(DeviceMatrix device_matrix1, float scalar, int order_of_operation, bool on_hold) {
	return matrixElementWiseScalarASMD_wrapper(device_matrix1, scalar, 3, order_of_operation, on_hold);
}

// Divide matrix elementwise with a scalar. 'order_of_operation' argument specifies direction of the operation.
DeviceMatrix CudaFunctions_Double::divs(DeviceMatrix device_matrix1, float scalar, bool on_hold) {
	return matrixElementWiseScalarASMD_wrapper(device_matrix1, scalar, 3, 0, on_hold);
}

// Transpose first matrix and then calculate matrix product with second matrix. Only works with DP floating point and complex numbers.
DeviceMatrix CudaFunctions_Double::tdot(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {

	if (device_matrix1.dtype == Double) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.width * device_matrix2.width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(device_matrix1.width * device_matrix2.width * sizeof(double), on_hold));

		double alpha = 1.0f;
		double beta = 0.0f;
		cublasDgemm(
			cublasHandle,
			CUBLAS_OP_N,							// transA
			CUBLAS_OP_T,							// transB
			device_matrix2.width,					// m
			device_matrix1.width,					// n
			device_matrix1.height,					// k
			&alpha,									// alpha
			(double*)device_matrix2.device_pointer,	// A
			device_matrix2.width,					// lda
			(double*)device_matrix1.device_pointer,	// B
			device_matrix1.width,					// ldb
			&beta,									// beta
			dev_c,									// C
			device_matrix2.width);					// ldc

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("tdot Double", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.width, device_matrix1.width, Double);
	}
	else if (device_matrix1.dtype == ComplexZ) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		cuDoubleComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.width * device_matrix2.width * sizeof(cuDoubleComplex));
		dev_c = static_cast<cuDoubleComplex*>(memAlloc(device_matrix1.width * device_matrix2.width * sizeof(cuDoubleComplex), on_hold));

		cuDoubleComplex alpha = make_cuDoubleComplex(1.0f, 0.0f);
		cuDoubleComplex beta = make_cuDoubleComplex(0.0f, 0.0f);
		cublasZgemm(
			cublasHandle,
			CUBLAS_OP_N,										// transA
			CUBLAS_OP_T,										// transB
			device_matrix2.width,								// m
			device_matrix1.width,								// n
			device_matrix1.height,								// k
			&alpha,												// alpha
			(cuDoubleComplex*)device_matrix2.device_pointer,	// A
			device_matrix2.width,								// lda
			(cuDoubleComplex*)device_matrix1.device_pointer,	// B
			device_matrix1.width,								// ldb
			&beta,												// beta
			dev_c,												// C
			device_matrix2.width);								// ldc
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("tdot Complex", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.width, device_matrix1.width, ComplexZ);
	}
	else
		problematicExit("Unknown matrix datatype for tdot operation. Only Double and ComplexZ are supported!");
}


DeviceMatrix CudaFunctions_Double::aTbT(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	if (device_matrix1.dtype == Double) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.width * device_matrix2.width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(device_matrix1.width * device_matrix2.height * sizeof(double), on_hold));

		double alpha = 1.0;
		double beta = 0.0;
		cublasDgemm(
			cublasHandle,
			CUBLAS_OP_T,							// transA
			CUBLAS_OP_T,							// transB
			device_matrix2.height,					// m
			device_matrix1.width,					// n
			device_matrix1.height,					// k
			&alpha,									// alpha
			(double*)device_matrix2.device_pointer,	// A
			device_matrix2.width,					// lda
			(double*)device_matrix1.device_pointer,	// B
			device_matrix1.width,					// ldb
			&beta,									// beta
			dev_c,									// C
			device_matrix2.height);					// ldc

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("tdot Double", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.height, device_matrix1.width, Double);
	}
	else if (device_matrix1.dtype == ComplexZ) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		cuDoubleComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.width * device_matrix2.width * sizeof(cuDoubleComplex));
		dev_c = static_cast<cuDoubleComplex*>(memAlloc(device_matrix1.width * device_matrix2.height * sizeof(cuDoubleComplex), on_hold));

		cuDoubleComplex alpha = make_cuDoubleComplex(1.0f, 0.0f);
		cuDoubleComplex beta = make_cuDoubleComplex(0.0f, 0.0f);
		cublasZgemm(
			cublasHandle,
			CUBLAS_OP_T,										// transA
			CUBLAS_OP_T,										// transB
			device_matrix2.height,								// m
			device_matrix1.width,								// n
			device_matrix1.height,								// k
			&alpha,												// alpha
			(cuDoubleComplex*)device_matrix2.device_pointer,	// A
			device_matrix2.width,								// lda
			(cuDoubleComplex*)device_matrix1.device_pointer,	// B
			device_matrix1.width,								// ldb
			&beta,												// beta
			dev_c,												// C
			device_matrix2.height);								// ldc
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("tdot Complex", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.height, device_matrix1.width, ComplexZ);
	}
	else
		problematicExit("Unknown matrix datatype for tdot operation. Only Double and ComplexZ are supported!");
}


// Transpose first matrix and then calculate matrix product with second matrix. Only works with DP floating point and complex numbers.
DeviceMatrix CudaFunctions_Double::aTb(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	return tdot(device_matrix1, device_matrix2, on_hold);
}

// Calculate matrix product. Only works with DP floating point and complex numbers.
DeviceMatrix CudaFunctions_Double::dot(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	if (device_matrix1.dtype == Double) {
		//cudaEventRecord(start);

		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.height * device_matrix2.width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(device_matrix1.height * device_matrix2.width * sizeof(double), on_hold));

		double alpha = 1.0;
		double beta = 0.0;
		cublasDgemm(
			cublasHandle,
			CUBLAS_OP_N,							// transA
			CUBLAS_OP_N,							// transB
			device_matrix2.width,					// m
			device_matrix1.height,					// n
			device_matrix1.width,					// k
			&alpha,									// alpha
			static_cast<double*>(device_matrix2.device_pointer),	// A
			device_matrix2.width,					// lda
			static_cast<double*>(device_matrix1.device_pointer),	// B
			device_matrix1.width,					// ldb
			&beta,									// beta
			dev_c,									// C
			device_matrix2.width);					// ldc
	/*	cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("dot Double", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.width, device_matrix1.height, Double);
	}
	else if (device_matrix1.dtype == ComplexZ) {
		//cudaEventRecord(start);

		// Allocate GPU buffers 
		cuDoubleComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.height * device_matrix2.width * sizeof(cuDoubleComplex));
		dev_c = static_cast<cuDoubleComplex*>(memAlloc(device_matrix1.height * device_matrix2.width * sizeof(cuDoubleComplex), on_hold));

		cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
		cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
		cublasZgemm(
			cublasHandle,
			CUBLAS_OP_N,										// transA
			CUBLAS_OP_N,										// transB
			device_matrix2.width,								// m
			device_matrix1.height,								// n
			device_matrix1.width,								// k
			&alpha,												// alpha
			static_cast<cuDoubleComplex*>(device_matrix2.device_pointer),	// A
			device_matrix2.width,								// lda
			static_cast<cuDoubleComplex*>(device_matrix1.device_pointer),	// B
			device_matrix1.width,								// ldb
			&beta,												// beta
			dev_c,												// C
			device_matrix2.width);								// ldc

	/*	cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("dot Complex", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.width, device_matrix1.height, ComplexZ);
	}
	else
		problematicExit("Unknown matrix datatype for dot operation. Only Double and ComplexZ are supported!");
}

// Transpose second matrix and then calculate matrix product with first matrix. Only works with DP floating point and complex numbers.
DeviceMatrix CudaFunctions_Double::abT(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	if (device_matrix1.dtype == Double) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.height * device_matrix2.height * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(device_matrix1.height * device_matrix2.height * sizeof(double), on_hold));

		double alpha = 1.0f;
		double beta = 0.0f;
		cublasDgemm(
			cublasHandle,
			CUBLAS_OP_T,							// transA
			CUBLAS_OP_N,							// transB
			device_matrix2.height,					// m
			device_matrix1.height,					// n
			device_matrix2.width,					// k
			&alpha,									// alpha
			(double*)device_matrix2.device_pointer,	// A
			device_matrix2.width,					// lda
			(double*)device_matrix1.device_pointer,	// B
			device_matrix1.width,					// ldb
			&beta,									// beta
			dev_c,									// C
			device_matrix2.height);					// ldc
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("abT Double", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.height, device_matrix1.height, Double);
	}
	else if (device_matrix1.dtype == ComplexZ) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		cuDoubleComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix1.height * device_matrix2.height * sizeof(cuDoubleComplex));
		dev_c = static_cast<cuDoubleComplex*>(memAlloc(device_matrix1.height * device_matrix2.height * sizeof(cuDoubleComplex), on_hold));

		cuDoubleComplex alpha = make_cuDoubleComplex(1.0f, 0.0f);
		cuDoubleComplex beta = make_cuDoubleComplex(0.0f, 0.0f);
		cublasZgemm(
			cublasHandle,
			CUBLAS_OP_T,										// transA
			CUBLAS_OP_N,										// transB
			device_matrix2.height,								// m
			device_matrix1.height,								// n
			device_matrix2.width,								// k
			&alpha,												// alpha
			(cuDoubleComplex*)device_matrix2.device_pointer,	// A
			device_matrix2.width,								// lda
			(cuDoubleComplex*)device_matrix1.device_pointer,	// B
			device_matrix1.width,								// ldb
			&beta,												// beta
			dev_c,												// C
			device_matrix2.height);								// ldc

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("abT Complex", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix2.height, device_matrix1.height, ComplexZ);
	}
	else
		problematicExit("Unknown matrix datatype for 'abT' operation. Only Double and ComplexZ are supported!");
}

// This method combines to matrices A and B to generate complex matrix C with elements of form a+bj.
DeviceMatrix CudaFunctions_Double::complex(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {

	//cudaEventRecord(start);
	// Set grid dimensions
	dim3 grid((int)ceil((float)device_matrix1.width / 32), (int)ceil((float)device_matrix1.height / 32), 1);

	// Allocate GPU buffers 
	cuDoubleComplex *dev_c = 0;
	//cudaMalloc((void**)&dev_c, device_matrix1.height * device_matrix2.width * sizeof(cuDoubleComplex));
	dev_c = static_cast<cuDoubleComplex*>(memAlloc(device_matrix1.height * device_matrix2.width * sizeof(cuDoubleComplex), on_hold));

	// Set arguments and launch the required kernel on the GPU.
	void* args[] = { &(device_matrix1.device_pointer), &(device_matrix2.device_pointer), &dev_c, &(device_matrix1.width), &(device_matrix1.height) };
	cudaLaunchKernel(
		(const void*)&complexMatrixConstruction, // pointer to kernel func.
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

	return DeviceMatrix(dev_c, device_matrix1.width, device_matrix1.height, ComplexZ);
}

// This method computes elementwise sin or cosine for a given matrix. Intrinsics sin/cos computations are off by default and can be turned on by setting use_intrinsics to 1.
DeviceMatrix CudaFunctions_Double::matrixElementWiseSinOrCosOrAbs(DeviceMatrix device_matrix, int choice_of_operation, int use_of_intrinsics, bool on_hold) {
	if (device_matrix.dtype == Double) {
		//cudaEventRecord(start);
		// Set grid dimensions

		dim3 grid((int)ceil((float)device_matrix.width / 1), (int)ceil((float)device_matrix.height / 1024), 1);

		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(device_matrix.height * device_matrix.width * sizeof(double), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &(device_matrix.width), &(device_matrix.height), &choice_of_operation, &use_of_intrinsics };
		cudaLaunchKernel(
			(const void*)&matrixElementWiseSinOrCosOrAbsForDoubles, // pointer to kernel func.
			grid, // grid
			dim3(1, 1024, 1), // block
			args,  // arguments
			0,
			stream);

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrixElementWiseSinOrCosOrAbsForDoubles", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, Double);
	}
	else
		problematicExit("Sin/cos method currently support double matrices only!");
}

// Computes elementwise sine of given matrix.
DeviceMatrix CudaFunctions_Double::sin(DeviceMatrix device_matrix, bool on_hold) {
	return matrixElementWiseSinOrCosOrAbs(device_matrix, 0, 0, on_hold);
}

// Computes elementwise cosine of given matrix.
DeviceMatrix CudaFunctions_Double::cos(DeviceMatrix device_matrix, bool on_hold) {
	return matrixElementWiseSinOrCosOrAbs(device_matrix, 1, 0, on_hold);
}

// Computes elementwise real/imag/conjugate values of given ComplexZ matrix.
DeviceMatrix CudaFunctions_Double::complexMatrixExtraction(DeviceMatrix device_matrix, int operation_choice, bool on_hold) {
	if (operation_choice == 3) {
		//cudaEventRecord(start);
		// Set grid dimensions
		int blockHeight = 1024;
		if (device_matrix.height < blockHeight)
			blockHeight = device_matrix.height;
		dim3 grid((int)ceil((float)device_matrix.width / 1), (int)ceil((float)device_matrix.height / blockHeight), 1);

		// Allocate GPU buffers 
		cuDoubleComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(cuDoubleComplex));
		dev_c = static_cast<cuDoubleComplex*>(memAlloc(device_matrix.height * device_matrix.width * sizeof(cuDoubleComplex), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &(device_matrix.width), &(device_matrix.height) };
		cudaLaunchKernel(
			(const void*)&matrixComplexConjugate, // pointer to kernel func.
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

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, ComplexZ);
	}
	else {

		if (device_matrix.dtype != ComplexZ)
			problematicExit("real/imag methods support ComplexZ type matrices only!");

		//cudaEventRecord(start);
		// Set grid dimensions
		int blockHeight = 1024;
		if (device_matrix.height < blockHeight)
			blockHeight = device_matrix.height;

		dim3 grid((int)ceil((float)device_matrix.width / 32), (int)ceil((float)device_matrix.height / 32), 1);

		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(device_matrix.height * device_matrix.width * sizeof(double), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &(device_matrix.width), &(device_matrix.height), &operation_choice };
		cudaLaunchKernel(
			(const void*)&matrixComplexElementWiseExtractions, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);

		//cudaEventRecord(stop);
		//cudaEventSynchronize(stop);
		//float milliseconds = 0;
		//cudaEventElapsedTime(&milliseconds, start, stop);
		//insert_into_execution_times_map("matrixComplexElementWiseExtractions", milliseconds);

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, Double);
	}
}

// Computes elementwise real values of given ComplexZ matrix.
DeviceMatrix CudaFunctions_Double::real(DeviceMatrix device_matrix, bool on_hold) {
	return complexMatrixExtraction(device_matrix, 0, on_hold);
}

// Computes elementwise imag values of given ComplexZ matrix.
DeviceMatrix CudaFunctions_Double::imag(DeviceMatrix device_matrix, bool on_hold) {
	return complexMatrixExtraction(device_matrix, 1, on_hold);
}

// Computes elementwise abs values of given ComplexZ matrix.
DeviceMatrix CudaFunctions_Double::abs(DeviceMatrix device_matrix, bool on_hold) {
	if (device_matrix.dtype == ComplexZ)
		return complexMatrixExtraction(device_matrix, 2, on_hold);
	else if (device_matrix.dtype == Double)
		return matrixElementWiseSinOrCosOrAbs(device_matrix, 2, 0, on_hold);
	else
		problematicExit("Abs operation supports only Float and ComplexC datatypes!");
}

// Computes elementwise sign of the values of the given ComplexZ matrix.
DeviceMatrix CudaFunctions_Double::sign(DeviceMatrix device_matrix, bool on_hold) {

	if (device_matrix.dtype == ComplexZ) {
		//cudaEventRecord(start);

		// Set grid dimensions
		dim3 grid((int)ceil((float)device_matrix.width / 32), (int)ceil((float)device_matrix.height / 32), 1);

		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(device_matrix.height * device_matrix.width * sizeof(double), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		int operation_choice = 3;
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &(device_matrix.width), &(device_matrix.height), &operation_choice };
		cudaLaunchKernel(
			(const void*)&matrixComplexElementWiseExtractions, // pointer to kernel func.
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

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, Double);
	}
	else
		problematicExit("Sign function not ready for matrices other than of type ComplexZ!");
}

// Computes eye, ones or zeros matrix of given dimensions.
DeviceMatrix CudaFunctions_Double::matrixEyeOrOnesOrZeros_wrapper(int width, int height, int operation_choice, bool on_hold) {
	//cudaEventRecord(start);

	// Set grid dimensions
	dim3 grid((int)ceil((float)width / 32), (int)ceil((float)height / 32), 1);

	// Allocate GPU buffers 
	double *dev_c = 0;
	//cudaMalloc((void**)&dev_c, height * width * sizeof(double));
	dev_c = static_cast<double*>(memAlloc(height * width * sizeof(double), on_hold));

	// Set arguments and launch the required kernel on the GPU.
	void* args[] = { &dev_c, &width, &height, &operation_choice };
	cudaLaunchKernel(
		(const void*)&matrixEyeOrOnesOrZeros, // pointer to kernel func.
		grid, // grid
		dim3(32, 32, 1), // block
		args,  // arguments
		0,
		stream);
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//float milliseconds = 0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//insert_into_execution_times_map("matrixEyeOrOnesOrZeros", milliseconds);

	return DeviceMatrix(dev_c, width, height, Double);
}

// Returns an int matrix filled with zeros
DeviceMatrix CudaFunctions_Double::zerosInt(int width, int height, bool on_hold) {

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
DeviceMatrix CudaFunctions_Double::eye(int width, bool on_hold) {
	return matrixEyeOrOnesOrZeros_wrapper(width, width, 0, on_hold);
}

// Computes ones matrix of given dimensions
DeviceMatrix CudaFunctions_Double::ones(int rows, int columns, bool on_hold) {
	return matrixEyeOrOnesOrZeros_wrapper(columns, rows, 1, on_hold);
}

// Computes zeros matrix of given dimensions
DeviceMatrix CudaFunctions_Double::zeros(int rows, int columns, bool on_hold) {
	return matrixEyeOrOnesOrZeros_wrapper(columns, rows, 2, on_hold);
}

// Computes flattened matrix using a row or column matrix.
DeviceMatrix CudaFunctions_Double::diagflat(DeviceMatrix device_matrix, bool on_hold) {
	int output_matrix_width;

	if (device_matrix.width == 1) {
		output_matrix_width = device_matrix.height;
	}
	else if (device_matrix.height == 1) {
		output_matrix_width = device_matrix.width;
	}
	else
		problematicExit("Double Diagonal flattening operation failed!");

	if (device_matrix.dtype == Double) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)output_matrix_width / 32), (int)ceil((float)output_matrix_width / 32), 1);

		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, output_matrix_width * output_matrix_width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(output_matrix_width * output_matrix_width * sizeof(double), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &output_matrix_width, &output_matrix_width };
		cudaLaunchKernel(
			(const void*)&matrixDiagflatDouble, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("diagflat Double", milliseconds);*/
		return DeviceMatrix(dev_c, output_matrix_width, output_matrix_width, Double);
	}
	else {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)output_matrix_width / 32), (int)ceil((float)output_matrix_width / 32), 1);

		// Allocate GPU buffers 
		cuDoubleComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, output_matrix_width * output_matrix_width * sizeof(cuDoubleComplex));
		dev_c = static_cast<cuDoubleComplex*>(memAlloc(output_matrix_width * output_matrix_width * sizeof(cuDoubleComplex), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &output_matrix_width, &output_matrix_width };
		cudaLaunchKernel(
			(const void*)&matrixDiagflatComplex, // pointer to kernel func.
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
		return DeviceMatrix(dev_c, output_matrix_width, output_matrix_width, ComplexZ);
	}
}


// Computes flattened matrix using a row or column matrix and raised to a power elementwise.
DeviceMatrix CudaFunctions_Double::diagWithPower(DeviceMatrix device_matrix, int power, bool on_hold) {

	int output_matrix_width;

	if (device_matrix.width == 1)
		output_matrix_width = device_matrix.height;
	else if (device_matrix.height == 1)
		output_matrix_width = device_matrix.width;
	else
		problematicExit("Double Diagonal flattening operation failed!");

	if (device_matrix.dtype == Double) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)output_matrix_width / 32), (int)ceil((float)output_matrix_width / 32), 1);

		// Allocate GPU buffers 
		double *dev_c;
		//cudaMalloc((void**)&dev_c, output_matrix_width * output_matrix_width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(output_matrix_width * output_matrix_width * sizeof(double), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &power, &output_matrix_width, &output_matrix_width };
		cudaLaunchKernel(
			(const void*)&matrixDiagflatDoubleWithPower, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
	/*	cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrixDiagflatDoubleWithPower", milliseconds);*/

		return DeviceMatrix(dev_c, output_matrix_width, output_matrix_width, Double);
	}
	else
		problematicExit("Diagonal flattening with power only supported for Doubles.");
}

// Returns conjugate of a given ComplexZ matrix
DeviceMatrix CudaFunctions_Double::conj(DeviceMatrix device_matrix, bool on_hold) {
	if (device_matrix.dtype == ComplexZ) {
		return complexMatrixExtraction(device_matrix, 3, on_hold);
	}
	else
		problematicExit("Complex conjugate for dtypes except ComplexZ is not supported!");
}

// Returns the concatenated matrix based on given axis. Only two matrices can be concatenated at once.
DeviceMatrix CudaFunctions_Double::concatenate(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, char axis, bool on_hold) {

	//int target_matrix_rows = NULL;
	//int target_matrix_columns = NULL;
	//int operation_choice = 0;
	//if (axis == 'x') {
	//	target_matrix_rows = device_matrix1.height;
	//	target_matrix_columns = device_matrix1.width + device_matrix2.width;
	//}
	//else if (axis == 'y') {
	//	target_matrix_rows = device_matrix1.height + device_matrix2.height;
	//	target_matrix_columns = device_matrix1.width;
	//	operation_choice = 1;
	//}
	//else
	//	problematicExit("Concatenation failed! 'axis' argument must be either 'x' or 'y'.");

	//// Set grid dimensions
	//if (device_matrix1.dtype == Double) {
	//	//cudaEventRecord(start, stream);
	//	// Allocate GPU buffers 
	//	double *dev_c = 0;
	//	//cudaMalloc((void**)&dev_c, target_matrix_columns * target_matrix_rows * sizeof(double));
	//	dev_c = static_cast<double*>(memAlloc(target_matrix_columns * target_matrix_rows * sizeof(double), on_hold));

	//	int isFirstMatrix = 1;
	//	// Set arguments and launch the required kernel on the GPU.
	//	void* args1[] = { &(device_matrix1.device_pointer), &(device_matrix2.device_pointer), &dev_c,
	//		&target_matrix_columns,
	//		&target_matrix_rows,
	//		&(device_matrix1.width), &(device_matrix1.height), &(device_matrix2.width), &(device_matrix2.height), &operation_choice, &isFirstMatrix };

	//	dim3 grid1((int)ceil((float)(device_matrix1.width) / 32), (int)ceil((float)(device_matrix1.height) / 32), 1);
	//	cudaLaunchKernel(
	//		(const void*)&matrixConcatenateDouble2, // pointer to kernel func.
	//		grid1, // grid
	//		dim3(32, 32, 1), // block
	//		args1,  // arguments
	//		0,
	//		stream);

	//	isFirstMatrix = 0;
	//	// Set arguments and launch the required kernel on the GPU.
	//	void* args2[] = { &(device_matrix1.device_pointer), &(device_matrix2.device_pointer), &dev_c,
	//		&target_matrix_columns,
	//		&target_matrix_rows,
	//		&(device_matrix1.width), &(device_matrix1.height), &(device_matrix2.width), &(device_matrix2.height), &operation_choice, &isFirstMatrix };
	//	dim3 grid2((int)ceil((float)(device_matrix2.width) / 32), (int)ceil((float)(device_matrix2.height) / 32), 1);
	//	cudaLaunchKernel(
	//		(const void*)&matrixConcatenateDouble2, // pointer to kernel func.
	//		grid2, // grid
	//		dim3(32, 32, 1), // block
	//		args2,  // arguments
	//		0,
	//		stream);
	//	//cudaEventRecord(stop, stream);
	//	//cudaEventSynchronize(stop);
	//	//float milliseconds = 0;
	//	//cudaEventElapsedTime(&milliseconds, start, stop);
	//	//insert_into_execution_times_map("matrixConcatenateDouble", milliseconds);
	//	//printf("\nconcat time = %d", milliseconds);
	//	return DeviceMatrix(dev_c, target_matrix_columns, target_matrix_rows, Double);

	//}
	//else if (device_matrix1.dtype == ComplexZ) {
	//	dim3 grid((int)ceil((float)target_matrix_columns / 32), (int)ceil((float)target_matrix_rows / 32), 1);
	//	//cudaEventRecord(start);
	//	// Allocate GPU buffers 
	//	cuDoubleComplex *dev_c = 0;
	//	//cudaMalloc((void**)&dev_c, target_matrix_columns * target_matrix_rows * sizeof(cuDoubleComplex));
	//	dev_c = static_cast<cuDoubleComplex*>(memAlloc(target_matrix_columns * target_matrix_rows * sizeof(cuDoubleComplex), on_hold));

	//	// Set arguments and launch the required kernel on the GPU.
	//	void* args[] = { &(device_matrix1.device_pointer), &(device_matrix2.device_pointer), &dev_c,
	//		&target_matrix_columns,
	//		&target_matrix_rows,
	//		&(device_matrix1.width), &(device_matrix1.height), &(device_matrix2.width), &operation_choice };

	//	cudaLaunchKernel(
	//		(const void*)&matrixConcatenateComplex, // pointer to kernel func.
	//		grid, // grid
	//		dim3(32, 32, 1), // block
	//		args,  // arguments
	//		0,
	//		stream);
	//	/*cudaEventRecord(stop);
	//	cudaEventSynchronize(stop);
	//	float milliseconds = 0;
	//	cudaEventElapsedTime(&milliseconds, start, stop);
	//	insert_into_execution_times_map("matrixConcatenateComplex", milliseconds);*/
	//	return DeviceMatrix(dev_c, target_matrix_columns, target_matrix_rows, ComplexZ);
	//}




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

	if (device_matrix1.dtype == Double) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, target_matrix_columns * target_matrix_rows * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(target_matrix_columns * target_matrix_rows * sizeof(double), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix1.device_pointer), &(device_matrix2.device_pointer), &dev_c,
			&target_matrix_columns,
			&target_matrix_rows,
			&(device_matrix1.width), &(device_matrix1.height), &(device_matrix2.width), &operation_choice };

		cudaLaunchKernel(
			(const void*)&matrixConcatenateDouble, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrixConcatenateDouble", milliseconds);*/

		return DeviceMatrix(dev_c, target_matrix_columns, target_matrix_rows, Double);

	}
	else if (device_matrix1.dtype == ComplexZ) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		cuDoubleComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, target_matrix_columns * target_matrix_rows * sizeof(cuDoubleComplex));
		dev_c = static_cast<cuDoubleComplex*>(memAlloc(target_matrix_columns * target_matrix_rows * sizeof(cuDoubleComplex), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix1.device_pointer), &(device_matrix2.device_pointer), &dev_c,
			&target_matrix_columns,
			&target_matrix_rows,
			&(device_matrix1.width), &(device_matrix1.height), &(device_matrix2.width), &operation_choice };

		cudaLaunchKernel(
			(const void*)&matrixConcatenateComplex, // pointer to kernel func.
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
		return DeviceMatrix(dev_c, target_matrix_columns, target_matrix_rows, ComplexZ);
	}
}

// Returns the concatenated matrix based on given axis. Only two matrices can be concatenated at once.
DeviceMatrix CudaFunctions_Double::concat(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, char axis, bool on_hold) {
	return concatenate(device_matrix1, device_matrix2, axis);
}

// Returns the concatenated matrix based on given axis. Only two matrices can be concatenated at once.
DeviceMatrix CudaFunctions_Double::concat(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2, bool on_hold) {
	return concatenate(device_matrix1, device_matrix2, 'y', on_hold);
}

// Returns the transpose of a given matrix
DeviceMatrix CudaFunctions_Double::transpose(DeviceMatrix device_matrix, bool on_hold) {

	if (device_matrix.dtype == Double) {
		//cudaEventRecord(start);
		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.width * device_matrix.height * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(device_matrix.width * device_matrix.height * sizeof(double), on_hold));

		const double alpha = 1.0;
		const double beta = 0.0;
		cublasDgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, device_matrix.height, device_matrix.width, &alpha,
			(double*)device_matrix.device_pointer, device_matrix.width, &beta, (double*)device_matrix.device_pointer,
			device_matrix.height, dev_c, device_matrix.height);

		//cudaEventRecord(stop);
		//cudaEventSynchronize(stop);
		//float milliseconds = 0;
		//cudaEventElapsedTime(&milliseconds, start, stop);
		//insert_into_execution_times_map("solve", milliseconds);
		//printf("transpose time = %fms\t", milliseconds);

		return DeviceMatrix(dev_c, device_matrix.height, device_matrix.width, Double);
	}
	else if (device_matrix.dtype == ComplexZ)
		problematicExit("Transpose for complex numbers not implemented!");
}

// Returns complex equivalent of given double matrix
DeviceMatrix CudaFunctions_Double::double_to_complex(DeviceMatrix device_matrix, bool on_hold) {

	if (device_matrix.dtype == Double) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)device_matrix.width / 32), (int)ceil((float)device_matrix.height / 32), 1);

		// Allocate GPU buffers 
		cuDoubleComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.width * device_matrix.height * sizeof(cuDoubleComplex));
		dev_c = static_cast<cuDoubleComplex*>(memAlloc(device_matrix.width * device_matrix.height * sizeof(cuDoubleComplex), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c, &(device_matrix.width), &(device_matrix.height) };
		cudaLaunchKernel(
			(const void*)&doubleToComplex, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("doubleToComplex", milliseconds);*/

		return DeviceMatrix(dev_c, device_matrix.width, device_matrix.height, ComplexZ);

	}
	else
		problematicExit("Float-to-Complex typecasting failed. Matrix datatype is not float!");
}

// Returns complex equivalent of given double matrix
DeviceMatrix CudaFunctions_Double::complexify(DeviceMatrix device_matrix, bool on_hold) {
	return double_to_complex(device_matrix, on_hold);
}

// Returns solution of linear equations of the form: Ax=B
DeviceMatrix CudaFunctions_Double::solve(DeviceMatrix device_matrix_1, DeviceMatrix device_matrix_2, bool on_hold) {
	//cudaEventRecord(start);
	DeviceMatrix devInfo = zerosInt(1, 1);
	DeviceMatrix devIpiv = zerosInt(device_matrix_1.height, 1);
	int lwork; /* size of workspace */
	double *d_work; /* device workspace for getrf */
	checkCudaErrors(cusolverDnDgetrf_bufferSize(
		cusolverHandle,
		device_matrix_1.height,
		device_matrix_1.height,
		(double*)device_matrix_1.device_pointer,
		device_matrix_1.height,
		&lwork));
	//cudaMalloc((void**)&d_work, sizeof(double)*lwork);
	d_work = static_cast<double*>(memAlloc(sizeof(double)*lwork, on_hold));
	checkCudaErrors(cusolverDnDgetrf(
		cusolverHandle,
		device_matrix_1.height,
		device_matrix_1.height,
		(double*)device_matrix_1.device_pointer,
		device_matrix_1.height,
		d_work,
		(int*)devIpiv.device_pointer,
		(int*)devInfo.device_pointer
		));
	checkCudaErrors(cusolverDnDgetrs(
		cusolverHandle,
		CUBLAS_OP_N,
		device_matrix_1.height,
		1,
		(double*)device_matrix_1.device_pointer,
		device_matrix_1.height,
		(int*)devIpiv.device_pointer,
		(double*)device_matrix_2.device_pointer,
		device_matrix_2.height,
		(int*)devInfo.device_pointer
		));
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//float milliseconds = 0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//insert_into_execution_times_map("solve", milliseconds);
	//printf("\tsolve time = %fms\t", milliseconds);
	return device_matrix_2;
	//return transpose(device_matrix_2, on_hold);




	//// --- Creating the array of pointers needed as input/output to the batched getrf
	//// A
	//cudaEventRecord(start);
	//double *device_matrix_1_copy = 0;
	//device_matrix_1_copy = (double*)memAlloc(device_matrix_1.height * device_matrix_1.height * sizeof(double));
	//cudaMemcpyAsync(device_matrix_1_copy, (double*)device_matrix_1.device_pointer, device_matrix_1.height * device_matrix_1.height * sizeof(double), cudaMemcpyDeviceToDevice, stream);
	//
	//double **h_in_pointers = (double **)malloc(sizeof(double *));
	//h_in_pointers[0] = device_matrix_1_copy;

	//double **d_in_pointers = (double**)memAlloc(sizeof(double *), on_hold);
	//cudaMemcpyAsync(d_in_pointers, h_in_pointers, sizeof(double *), cudaMemcpyHostToDevice, stream);


	//int *d_pivotArray = (int*)memAlloc(device_matrix_1.height * sizeof(int), on_hold);
	//int *d_InfoArray = (int*)(memAlloc(sizeof(int)));

	//cublasDgetrfBatched(cublasHandle, device_matrix_1.height, d_in_pointers, device_matrix_1.height,
	//	d_pivotArray, d_InfoArray, 1);

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
	//double *h_C = new double[device_matrix_1.width * device_matrix_1.width];

	//// --- Allocate device space for the inverted matrices 
	//double *d_C = (double*)memAlloc(device_matrix_1.width*device_matrix_1.width*sizeof(double), on_hold);

	//// --- Creating the array of pointers needed as output to the batched getri
	//double **h_out_pointers = (double **)malloc(sizeof(double *));
	//h_out_pointers[0] = (double *)((char*)d_C);

	//double **d_out_pointers = (double**)memAlloc(sizeof(double *), false);
	//cudaMemcpyAsync(d_out_pointers, h_out_pointers, sizeof(double *), cudaMemcpyHostToDevice, stream);

	//(cublasDgetriBatched(cublasHandle, device_matrix_1.width, (const double **)d_in_pointers, device_matrix_1.width, d_pivotArray, d_out_pointers, device_matrix_1.width, d_InfoArray, 1));

	////(cudaMemcpy(h_C, d_C, device_matrix_1.width*device_matrix_1.width*sizeof(double), cudaMemcpyDeviceToHost));
	////// --- The output inverted matrix in column-major format
	////printf("\n\n");
	////for (int i = 0; i<device_matrix_1.width*device_matrix_1.width; i++) printf("C[%i]=%f\n", i, h_C[i]);





	//double alpha1 = 1.0;
	//double beta1 = 0.0;

	//double *d_X = (double*)memAlloc(device_matrix_2.height * sizeof(double), on_hold);
	//(cublasDgemv(cublasHandle, CUBLAS_OP_T, device_matrix_1.width, device_matrix_1.width, &alpha1, d_C, device_matrix_1.width, (double*)device_matrix_2.device_pointer, 1, &beta1, d_X, 1));
	////(cudaMemcpy(h_X, d_X, device_matrix_1.width*sizeof(double), cudaMemcpyDeviceToHost));

	////// --- The output inverted matrix in column-major format
	////printf("\n\n");
	////for (int i = 0; i<device_matrix_1.width; i++) printf("X[%i]=%f\n", i, h_X[i]);
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//float milliseconds = 0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("solve time = %fms\t", milliseconds);
	//return DeviceMatrix((void*)d_X, device_matrix_2.width, device_matrix_2.height, Double);
}

// Returns sliced matrices based on given row-column coordinates.
DeviceMatrix CudaFunctions_Double::slice(DeviceMatrix device_matrix, int row_start, int row_end_exclusive, int column_start, int column_end_exclusive, bool on_hold) {

	if (device_matrix.dtype == Double) {
		//cudaEventRecord(start);
		int num_of_rows = row_end_exclusive - row_start;
		int num_of_cols = column_end_exclusive - column_start;

		int height = 0;
		if (device_matrix.height < 1024)
			height = device_matrix.height;
		// Set grid dimensions
		dim3 grid((int)ceil((float)device_matrix.width / 1), (int)ceil((float)device_matrix.height / 1024), 1);

		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, num_of_cols * num_of_rows * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(num_of_cols * num_of_rows * sizeof(double), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c,
			&row_start, &row_end_exclusive,
			&column_start, &column_end_exclusive,
			&(device_matrix.width),
			&num_of_cols };

		cudaLaunchKernel(
			(const void*)&sliceDouble, // pointer to kernel func.
			grid, // grid
			dim3(1, 1024, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("sliceDouble", milliseconds);*/
		return DeviceMatrix(dev_c, num_of_cols, num_of_rows, Double);
	} else if (device_matrix.dtype == ComplexZ) {
		//cudaEventRecord(start);
		int num_of_rows = row_end_exclusive - row_start;
		int num_of_cols = column_end_exclusive - column_start;

		// Set grid dimensions
		dim3 grid((int)ceil((float)device_matrix.width / 32), (int)ceil((float)device_matrix.height / 32), 1);

		// Allocate GPU buffers 
		cuDoubleComplex *dev_c = 0;
		//cudaMalloc((void**)&dev_c, num_of_cols * num_of_rows * sizeof(double));
		dev_c = static_cast<cuDoubleComplex*>(memAlloc(num_of_cols * num_of_rows * sizeof(cuDoubleComplex), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c,
			&row_start, &row_end_exclusive,
			&column_start, &column_end_exclusive,
			&(device_matrix.width),
			&num_of_cols };

		cudaLaunchKernel(
			(const void*)&sliceComplex128, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("sliceDouble", milliseconds);*/
		return DeviceMatrix(dev_c, num_of_cols, num_of_rows, ComplexZ);
	}
	else
		problematicExit("Slicing is currently supported only for Double and ComplexZ!");
}

// This method can be used to perform slicing of matrices with the help of given 1D indices matrix.
DeviceMatrix CudaFunctions_Double::slicei(DeviceMatrix device_matrix, DeviceMatrix indices_device_matrix, bool on_hold) {

	if (device_matrix.dtype == Double) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)indices_device_matrix.width / 1024), (int)ceil((float)indices_device_matrix.height / 1), 1);

		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, indices_device_matrix.width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(indices_device_matrix.width * sizeof(double), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c,
			&(indices_device_matrix.device_pointer),
			&(device_matrix.height), &(device_matrix.width),
			&(indices_device_matrix.width) };

		cudaLaunchKernel(
			(const void*)&slice_with_indices, // pointer to kernel func.
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
		return DeviceMatrix(dev_c, 1, indices_device_matrix.width, Double);
	}
	else
		problematicExit("Slicing with indices is currently supported only for Doubles!");
}

//This method can be used to perform special slicing of matrix R. Please refrain from using it for general purpose.
DeviceMatrix CudaFunctions_Double::specialSlicingOnR(DeviceMatrix device_matrix, DeviceMatrix indices, bool on_hold) {

	if (device_matrix.dtype == Double) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)indices.width / 32), (int)ceil((float)indices.height / 32), 1);

		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, indices.width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(indices.width * sizeof(double), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix.device_pointer), &dev_c,
			&(indices.device_pointer),
			&(device_matrix.height), &(device_matrix.width),
			&(indices.width) };

		cudaLaunchKernel(
			(const void*)&specialSlicingOnR_kernel, // pointer to kernel func.
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

		return DeviceMatrix(dev_c, indices.width, 1, Double);
	}
	else
		problematicExit("Special Slicing for R matrix is currently supported for Doubles!");
}

//This method can be used to perform special slicing of matrix H. Please refrain from using it for general purpose.
DeviceMatrix CudaFunctions_Double::specialSlicingOnH(DeviceMatrix device_matrix, DeviceMatrix indices, bool on_hold) {

	if (device_matrix.dtype == Double) {
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
			(const void*)&specialSlicingOnH_kernel, // pointer to kernel func.
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
		return DeviceMatrix(output_device_matrix.device_pointer, device_matrix.width, indices.width, Double);
	}
	else
		problematicExit("Special Slicing for H matrix is currently supported for Doubles!");
}

// Returns the maximum value within a given matrix.
double CudaFunctions_Double::maxValue(DeviceMatrix device_matrix, bool on_hold) {
	if (device_matrix.dtype == Double) {
		//cudaEventRecord(start);
		//int threads = 1024;
		//int blocks = (int)(ceil((float)device_matrix.height * (float)device_matrix.width / threads));

		//// Allocate GPU buffers 
		//double *output_intermediate_gpu = 0;
		//double *output_intermediate2_gpu = 0;
		//cudaMalloc((void**)&output_intermediate_gpu, blocks * sizeof(double));
		//cudaMalloc((void**)&output_intermediate2_gpu, (int)(ceil((float)blocks / 1024)) * sizeof(double));

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
		//double *output_final_max_gpu = 0;
		//cudaMalloc((void**)&output_final_max_gpu, sizeof(double));

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

		//return DeviceMatrix(output_final_max_gpu, 1, 1, Double);


		/***** Finding max value through CUBLAS. ******/
		int max_index;
		cublasIdamax(cublasHandle, device_matrix.height, (double*)device_matrix.device_pointer, 1, &max_index);
		max_index--; // To adjust from 1-based index to 0-based index.
		//DeviceMatrix max = slice(device_matrix, max_index, max_index + 1, 0, 1);
		double max;
		cudaMemcpyAsync(&max, ((double*)(device_matrix.device_pointer)) + max_index, sizeof(double), cudaMemcpyDeviceToHost, stream);

		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("maxValue", milliseconds);*/

		return max;
	}
	else
		problematicExit("Max reduction is only supported for Doubles.");
}

// Returns a matrix which is the result of insertion of a smaller matrix inside a bigger matrix.
// WARNING: The existing bigger matrix would be overwritten.
DeviceMatrix CudaFunctions_Double::insert(DeviceMatrix input_big_matrix, int row_start, int row_end_exclusive,
	int column_start, int column_end_exclusive, DeviceMatrix input_small_matrix, bool on_hold) {

	if (input_big_matrix.dtype == Double) {
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
			(const void*)&matrix_insert, // pointer to kernel func.
			grid, // grid
			dim3(32, 32, 1), // block
			args,  // arguments
			0,
			stream);
		/*cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		insert_into_execution_times_map("matrix_insert", milliseconds);*/
		return DeviceMatrix(input_big_matrix.device_pointer, input_big_matrix.width, input_big_matrix.height, Double);
	}
	else
		problematicExit("Matrix insertion is currently only supported for Doubles!");
}


void CudaFunctions_Double::to_DND_pool(DeviceMatrix device_matrix) {
	return dMManager.toDndDeviceMemory(&device_matrix);
}

DeviceMatrix CudaFunctions_Double::sort(DeviceMatrix device_matrix, bool on_hold) {
	sort_on_device(static_cast<double*>(device_matrix.device_pointer), device_matrix.width);
	return device_matrix;
}

void CudaFunctions_Double::setStream(cudaStream_t cudaStream) {
	cusolverStatus_t status;
	stream = cudaStream;
	cublasSetStream(cublasHandle, stream);

	status = cusolverDnSetStream(cusolverHandle, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);
}

void CudaFunctions_Double::deviceSynchronize() {
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("\n\n");
		fprintf(stderr, "cudaDeviceSynchronize failed!");
	}
}

void CudaFunctions_Double::write_to_file(DeviceMatrix device_matrix1, DeviceMatrix device_matrix2) {
	if (grid_size != NULL) {
		double *double_elements1 = new double[device_matrix1.height * device_matrix1.width];
		cudaMemcpy(double_elements1, device_matrix1.device_pointer, device_matrix1.height * device_matrix1.width * sizeof(double), cudaMemcpyDeviceToHost);
		double *double_elements2 = new double[device_matrix2.height * device_matrix2.width];
		cudaMemcpy(double_elements2, device_matrix2.device_pointer, device_matrix2.height * device_matrix2.width * sizeof(double), cudaMemcpyDeviceToHost);

		// current date/time based on current system
		time_t now = time(0);
		// convert now to string form
		std::string dt = ctime(&now);

		std::ofstream myfile;
		std::string result_file = "Results_FP64\\result_" + std::to_string(grid_size) + ".csv";
		myfile.open(result_file, std::ios_base::trunc);

		std::cout << "\n*** Writing matrix contents of eK/fK matrices to a file. ***\n";
		std::string file_content = "";
		for (int i = 0; i < device_matrix1.height; i++) {
			file_content += ("Row#" + std::to_string(i) + ",");
			for (int j = 0; j < device_matrix1.width; j++) {
				int current_index = j + i * device_matrix1.width;
				file_content += std::to_string(double_elements1[current_index]) + "," + std::to_string(double_elements2[current_index]);
			}
			file_content += "\n";
		}
		myfile << file_content << std::endl;
		myfile.close();
	}
	else
		std::cout << "\n Cannot write results to file since grid-size is not set for CudaFunctions_Double instance." << std::endl;

}

DeviceMatrix CudaFunctions_Double::angle(DeviceMatrix eK, DeviceMatrix fK, bool on_hold) {
	if (eK.dtype == Double) {
		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid((int)ceil((float)eK.width / 32), (int)ceil((float)eK.height / 32), 1);

		// Allocate GPU buffers 
		double *dev_c = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(double));
		dev_c = static_cast<double*>(memAlloc(eK.height * eK.width * sizeof(double), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(eK.device_pointer), &(fK.device_pointer), &dev_c, &(eK.width), &(eK.height) };
		cudaLaunchKernel(
			(const void*)&matrixElementWiseAngles, // pointer to kernel func.
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


		return DeviceMatrix(dev_c, eK.width, eK.height, Double);
	}
	else
		problematicExit("Angle method currently support double matrices only!");

}


DeviceMatrix CudaFunctions_Double::wrapPointersIntoPointerArrays(DeviceMatrix device_matrix_1, bool on_hold) {

	if (device_matrix_1.dtype == Double) {

		//cudaEventRecord(start);
		// Set grid dimensions
		dim3 grid(1, 1, 1);

		// Allocate GPU buffers 
		double **odata = 0;
		//cudaMalloc((void**)&dev_c, device_matrix.height * device_matrix.width * sizeof(double));
		odata = static_cast<double**>(memAlloc(sizeof(double*), on_hold));

		// Set arguments and launch the required kernel on the GPU.
		void* args[] = { &(device_matrix_1.device_pointer), &odata };
		cudaLaunchKernel(
			(const void*)&wrapPointersIntoPointerArrays_FP64, // pointer to kernel func.
			grid, // grid
			dim3(1, 1, 1), // block
			args,  // arguments
			0,
			stream);

		return DeviceMatrix(odata, 1, 1, Double);
	}
}

float CudaFunctions_Double::getVectorElementAtIndex(DeviceMatrix device_matrix, int index) {
	double element;
	cudaMemcpyAsync(&element, ((double*)(device_matrix.device_pointer)) + index, sizeof(double), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	return element;
}