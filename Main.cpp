#pragma once
// System includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>
#include <thread>
#include <future>
#include <utility>

// User-defined includes
#include "DeviceMatrix.h"
#include "StateEstimationCUDA.h"

#include "MemoryPool.h"

// CudaFunctions v2 -- WORK IN PROGRESS
#include "CudaFunctions_Base.h"
#include "CudaFunctions_Double.h"
#include "CudaFunctions_Single.h"


#include "DotGenerator.h"
#include "DataImporter.h"

// Global Error variable
cudaError_t cudaStatus;

WholeAndTheParts loadDataAndCuda(CudaFunctions_Base *C) {
	// Extracting the grid size from the folder name
	auto wholeAndTheParts = DataImporter::load(C);
	return wholeAndTheParts;
}

int main (int argc, char *argv[])
{
	//generate_dot_file(read_from_files("C:\\Users\\kumar\\Desktop\\472"));
	{
		// Per launch thread, a CudaFunctions_Base instance is created.
		// The grid dataset is then acquired either from filePath or from split-operation
		// The result is then a StateEstimationResults struct which contains the voltages, phases and the time taken.

		CudaFunctions_Base *C = new CudaFunctions_Double;
		/*double *h_A = new double[9];

		h_A[0] = 4;
		h_A[1] = 3;
		h_A[2] = 8;
		h_A[3] = 9;
		h_A[4] = 5;
		h_A[5] = 1;
		h_A[6] = 2;
		h_A[7] = 7;
		h_A[8] = 6;
		double h_B[3] = { 1, 0.5, 3};
		auto d_A = C->to_device(h_A, 3, 3);
		auto d_B = C->to_device(h_B, 1, 3);
		auto c = C->solve(d_A, d_B, true);
		C->printMatrix(c);
		auto d = C->solve(d_A, d_B, true);
		C->printMatrix(d);*/
		CudaFunctions_Base *C2 = new CudaFunctions_Double;
		/*CudaFunctions_Base *C3 = new CudaFunctions_Single;
		CudaFunctions_Base *C4 = new CudaFunctions_Single;*/

		// State Estimation Function
		auto dataBundle1 = loadDataAndCuda(C);
		//auto dataAndCudaBundle2 = loadDataAndCuda(path);
		std::thread first(stateEstimationCUDAv3, std::make_pair(dataBundle1.partitions[0], C), false);
		std::thread second(stateEstimationCUDAv3, std::make_pair(dataBundle1.partitions[1], C2), false);
		/*std::thread third(stateEstimationCUDAv3, std::make_pair(dataBundle1.partitions[2], C3), false);
		std::thread fourth(stateEstimationCUDAv3, std::make_pair(dataBundle1.partitions[3], C4), false);*/
		first.join();
		second.join(); 
		
		auto voltage1 = C->slice(C->SEstate_vector.voltages, 0, (C->SEstate_vector.voltages.height) - 1, 0, 1);
		auto voltage = C->concat(voltage1, C2->SEstate_vector.voltages, 'y');
		auto phases1 = C->slice(C->SEstate_vector.phases, 0, (C->SEstate_vector.phases.height) - 1, 0, 1);
		auto phases2 = C->adds(C2->SEstate_vector.phases, C->getVectorElementAtIndex(C->SEstate_vector.phases, 117));
		auto phase = C->concat(phases1, phases2, 'y');
		cudaDeviceSynchronize();
		C->write_to_file(voltage, phase);
		/*third.join();
		fourth.join();*/
		/*std::thread second(stateEstimationCUDAv3, std::make_pair(dataBundle1.partitions[0], C2), false);
		std::thread third(stateEstimationCUDAv3, std::make_pair(dataBundle1.partitions[1], C3), false);*/
		//std::thread second(stateEstimationCUDAv3, dataAndCudaBundle2, false);

		/*second.join();
		third.join();*/
		getchar();

		delete C, C2/*, C3, C4*/;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		printf("\n\n");
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	} else
		printf("\ncudaDeviceReset succeeded!");


	


	//getchar();
    return 0;
}

