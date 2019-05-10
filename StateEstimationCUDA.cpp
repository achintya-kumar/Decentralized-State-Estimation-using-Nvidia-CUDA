#pragma once
#include "StateEstimationCUDA.h"
#include <iostream>
#include <ctime>
#include <chrono>

std::string deviceQuery() {
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
		exit(EXIT_FAILURE);
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
		if (deviceCount > 1)
			std::cout << "Consider multi-GPU setup for higher throughput.\n";

		// Getting name of the first GPU
		cudaSetDevice(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);

		printf("Device selected: \"%s\"\n", deviceProp.name);
		return deviceProp.name;
	}
}


void stateEstimationCUDAv3(std::pair<GridDataSet, CudaFunctions_Base*> dataAndCudaBundle, bool debug_mode) {
	debug_mode = false;
	auto dataset = dataAndCudaBundle.first;
	auto C = dataAndCudaBundle.second;
	//// Asking user for grid size 
	//int grid_size = 4;
	///*std::cout << "\nEnter the grid size: ";
	//std::cin >> grid_size;*/
	//const std::string folder = "C:\\Users\\kumar\\Desktop\\SEdata" + (std::to_string(grid_size)) + "bus\\";

	//CudaFunctions_Base *C;

	//int precision = 2;
	///*std::cout << "\n1. Single Precision\n2. Double Precision\n\tEnter the desired precision: ";
	//while (precision != 1 && precision != 2) {
	//	std::cin >> precision;
	//	if (precision != 1 && precision != 2)
	//		std::cout << "\n\tInvalid entry! Choose again: ";
	//}*/


	if (C == nullptr && dataset.KKT.dtype == Double)
		C = new CudaFunctions_Double;
	else if (C == nullptr && dataset.KKT.dtype == Float)
		C = new CudaFunctions_Single;


	C->setGridSize(dataset.grid_size);				// Not setting this will lead to corrupt result-file names
	
	// Setting per thread stream for independent computations
	cudaStream_t stream;
	auto deviceName = deviceQuery();
	if (deviceName == "Tesla V100-SXM2-16GB") {
		std::cout << "Creating stream with Non-blocking flag!" << std::endl;
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	}
	else {
		std::cout << "Creating stream without Non-blocking flag!" << std::endl;
		cudaStreamCreate(&stream);
	}
	C->setStream(stream);

	

	double EPS;
	if (C->precision == 2)
		EPS = 0.001;
	else
		EPS = 0.05;


	// Loading raw matrices
	//printf("\nLoading KKT...");
	auto KKT_d = dataset.KKT;
	auto KKT_d_T = dataset.KKT_t;

	//printf("\nLoading YT...");
	auto YT_d = dataset.YT;

	//printf("\nLoading YKK...");
	auto YKK_d = dataset.YKK;

	//printf("\nLoading INIT...");
	auto INIT_d = dataset.INIT;

	//printf("\nLoading MEAS...");
	auto MEAS_d = dataset.MEAS;
	printf("\tMemory occupied AFTER LOADING = %f", (float)C->dMManager.total_occupied_memory_in_bytes / (1024 * 1024 * 1024));
	//getchar();

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	// Measuring per iteration time...
	int maxIteration = 1000;
	if (C->precision == 1 && dataset.grid_size == 1200) {
		maxIteration = 30;
	}
	int nK_d = KKT_d.height, nT_d = KKT_d.width;

	if (debug_mode) {
		C->printMatrix(KKT_d);
		C->printMatrix(INIT_d);
		C->printMatrix(MEAS_d);
		C->printMatrix(YKK_d);
		C->printMatrix(YT_d);
	}

	// conversion of PMU measurements in real and imaginary part
	auto z_uTabs_d = C->slice(MEAS_d, (3 * nT_d + 2 * nK_d), (4 * nT_d + 2 * nK_d), 0, MEAS_d.width);
	auto z_uTang_d = C->slice(MEAS_d, (4 * nT_d + 2 * nK_d), (5 * nT_d + 2 * nK_d), 0, MEAS_d.width);
	auto z_iTabs_d = C->slice(MEAS_d, (5 * nT_d + 2 * nK_d), (6 * nT_d + 2 * nK_d), 0, MEAS_d.width);
	auto z_iTang_d = C->slice(MEAS_d, (6 * nT_d + 2 * nK_d), (7 * nT_d + 2 * nK_d), 0, MEAS_d.width);
	if (debug_mode)
		C->printMatrix(z_iTang_d);
	//printf("\nMemory occupied AFTER CONVERSION = %f", (float)C->dMManager.total_occupied_memory_in_bytes / (1024 * 1024 * 1024));
	//getchar();

	auto z_uTabs_d0 = C->slice(z_uTabs_d, 0, z_uTabs_d.height, 0, 1);  // Avoiding recomputation
	auto z_iTabs_d0 = C->slice(z_iTabs_d, 0, z_iTabs_d.height, 0, 1);
	auto z_uTang_d0sin = C->sin(C->slice(z_uTang_d, 0, z_uTang_d.height, 0, 1));
	auto z_uTang_d0cos = C->cos(C->slice(z_uTang_d, 0, z_uTang_d.height, 0, 1));
	auto z_iTang_d0sin = C->sin(C->slice(z_iTang_d, 0, z_uTang_d.height, 0, 1));
	auto z_iTang_d0cos = C->cos(C->slice(z_iTang_d, 0, z_uTang_d.height, 0, 1));
	auto z_eT_d = C->mul(z_uTabs_d0, z_uTang_d0cos);
	auto z_fT_d = C->mul(z_uTabs_d0, z_uTang_d0sin);
	auto z_aT_d = C->mul(z_iTabs_d0, z_iTang_d0cos);
	auto z_cT_d = C->mul(z_iTabs_d0, z_iTang_d0sin);
	if (debug_mode) {
		C->printMatrix(z_eT_d);
		C->printMatrix(z_fT_d);
		C->printMatrix(z_aT_d);
		C->printMatrix(z_cT_d);
	}
	//printf("\nMemory occupied AFTER CONVERSION 2 = %f", (float)C->dMManager.total_occupied_memory_in_bytes / (1024 * 1024 * 1024));
	//getchar();

	auto z_uTabs_d1 = C->slice(z_uTabs_d, 0, z_uTabs_d.height, 1, 2);
	auto z_iTabs_d1 = C->slice(z_iTabs_d, 0, z_iTabs_d.height, 1, 2);
	auto z_uTang_d1 = C->slice(z_uTang_d, 0, z_uTang_d.height, 1, 2);
	auto z_iTang_d1 = C->slice(z_iTang_d, 0, z_iTang_d.height, 1, 2);
	auto sgm_eT_d = C->add(C->mul(C->abs(z_uTang_d0cos), z_uTabs_d1), C->mul(C->abs(C->mul(z_uTabs_d0, z_uTang_d0sin)), z_uTang_d1));
	auto sgm_fT_d = C->add(C->mul(C->abs(z_uTang_d0sin), z_uTabs_d1), C->mul(C->abs(C->mul(z_uTabs_d0, z_uTang_d0cos)), z_uTang_d1));
	auto sgm_aT_d = C->add(C->mul(C->abs(z_iTang_d0cos), z_iTabs_d1), C->mul(C->abs(C->mul(z_iTabs_d0, z_iTang_d0sin)), z_iTang_d1));
	auto sgm_cT_d = C->add(C->mul(C->abs(z_iTang_d0sin), z_iTabs_d1), C->mul(C->abs(C->mul(z_iTabs_d0, z_iTang_d0cos)), z_iTang_d1));
	if (debug_mode) {
		C->printMatrix(sgm_eT_d);
		C->printMatrix(sgm_fT_d);
		C->printMatrix(sgm_aT_d);
		C->printMatrix(sgm_cT_d);
	}

	auto x1_d = C->concat(C->concat(C->concat(z_eT_d, sgm_eT_d, 'x'), C->concat(z_fT_d, sgm_fT_d, 'x')),
		C->concat(C->concat(z_aT_d, sgm_aT_d, 'x'), C->concat(z_cT_d, sgm_cT_d, 'x')));
	C->sanitizeMemoryPools();
	if (debug_mode)
		C->printMatrix(x1_d);

	MEAS_d = C->insert(MEAS_d, (3 * nT_d + 2 * nK_d), (7 * nT_d + 2 * nK_d), 0, MEAS_d.width, x1_d);
	if (debug_mode)
		C->printMatrix(MEAS_d);

	auto nT_diagflat_dot = C->dot(C->diagflat(C->ones(nT_d, 1)), KKT_d_T);
	auto nT_zeros_dot = C->dot(C->zeros(nT_d, nT_d), KKT_d_T); // why tf am I dotting sth with a zeros matrix??
	auto YT_real_KKT_dot = C->dot(C->real(YT_d), KKT_d_T);
	auto YT_imag = C->imag(YT_d);
	auto YT_imag_neg_dot = C->dot(C->muls(YT_imag, -1), KKT_d_T);
	auto YT_imag_dot = C->dot(YT_imag, KKT_d_T);

	auto H_eT_d = C->concat(nT_diagflat_dot, nT_zeros_dot, 'x'); /* Moving this to DND memory pool. */ C->to_DND_pool(H_eT_d);
	auto H_fT_d = C->concat(nT_zeros_dot, nT_diagflat_dot, 'x'); /* Moving this to DND memory pool. */ C->to_DND_pool(H_fT_d);
	auto H_aT_d = C->concat(YT_real_KKT_dot, YT_imag_neg_dot, 'x'); /* Moving this to DND memory pool. */ C->to_DND_pool(H_aT_d);
	auto H_cT_d = C->concat(YT_imag_dot, YT_real_KKT_dot, 'x'); /* Moving this to DND memory pool. */ C->to_DND_pool(H_cT_d);
	C->sanitizeMemoryPools();


	auto z_d = C->slice(MEAS_d, 0, MEAS_d.height, 0, 1); /* Moving this to DND memory pool. */ C->to_DND_pool(z_d);
	auto sgm_d = C->slice(MEAS_d, 0, MEAS_d.height, 1, 2); /* Moving this to DND memory pool. */ C->to_DND_pool(sgm_d);
	auto idx_z_d = C->extract_indices_for_non_zero(z_d); /* TODO Stream compaction pending */ C->to_DND_pool(idx_z_d);
	
	/*float *idx_z = 0;
	cudaMemcpyAsync(&idx_z, idx_z_d.device_pointer, idx_z_d.height * idx_z_d.width * sizeof(float), cudaMemcpyDeviceToHost, stream);*/
	/*cudaStreamSynchronize(stream);
	C->sort(idx_z_d, idx_z_d.width);*/
	auto eK_d = C->real(INIT_d);  /* Moving this to DND memory pool. */ C->to_DND_pool(eK_d);
	auto fK_d = C->imag(INIT_d);  /* Moving this to DND memory pool. */ C->to_DND_pool(fK_d);
	auto a = C->complexify(KKT_d);		 /* Moving this to DND memory pool. */ C->to_DND_pool(a);

	int ctr = 0;
	double delta = 2 * EPS;

	C->sanitizeMemoryPools();

	//C->printMatrix(z_d);
	//C->printMatrix(idx_z_d);
	auto tt_d = C->specialSlicingOnR(sgm_d, idx_z_d); 
	C->to_DND_pool(tt_d);

	//Instead of repeating the following operations on the GPU, they can just be reused on-device.
	// A good example of RECOMPUTATION AVOIDANCE
		// The values below are required outside as well as inside the loop
	auto subs_eye_nK_d_1 = C->subs(C->eye(nK_d), 1, 1); C->to_DND_pool(subs_eye_nK_d_1);
	auto subs_eye_nK_d = C->subs(C->eye(nK_d), 1); C->to_DND_pool(subs_eye_nK_d);
	auto adds_eye_nK_d = C->adds(C->eye(nK_d), 1); C->to_DND_pool(adds_eye_nK_d);
	auto real_YKK_d = C->real(YKK_d); C->to_DND_pool(real_YKK_d);
	auto imag_YKK_d = C->imag(YKK_d); C->to_DND_pool(imag_YKK_d);
	C->sanitizeMemoryPools();

		// The values below are only for use in the subsequent block
	auto imag_YT_d = C->imag(YT_d);
	auto real_YT_d = C->real(YT_d);
	auto subs_eye_nT_d_1 = C->subs(C->eye(nT_d), 1, 1);
	auto subs_eye_nT_d = C->subs(C->eye(nT_d), 1);
	auto adds_eye_nT_d = C->adds(C->eye(nT_d), 1);
		// These values are used in the loop. The values above should not be released until this block is over.
	auto abT_optimization_1 = C->dot(C->mul(subs_eye_nT_d, imag_YT_d), KKT_d_T); C->to_DND_pool(abT_optimization_1);
	auto abT_optimization_4 = C->dot(C->mul(adds_eye_nT_d, real_YT_d), KKT_d_T); C->to_DND_pool(abT_optimization_4);
	auto abT_optimization_5 = C->dot(C->mul(subs_eye_nT_d_1, imag_YT_d), KKT_d_T); C->to_DND_pool(abT_optimization_5);
	auto abT_optimization_6 = C->dot(C->mul(subs_eye_nT_d_1, real_YT_d), KKT_d_T); C->to_DND_pool(abT_optimization_6);
	auto dot_optimization_9 = C->dot(C->mul(subs_eye_nT_d, real_YT_d), KKT_d_T); C->to_DND_pool(dot_optimization_9);
	auto dot_optimization_100 = C->dot(C->mul(C->muls(adds_eye_nT_d, -1), imag_YT_d), KKT_d_T); C->to_DND_pool(dot_optimization_100);
	auto dot_optimization_101 = C->dot(C->mul(adds_eye_nT_d, imag_YT_d), KKT_d_T); C->to_DND_pool(dot_optimization_101);
	auto mul_optimization_102 = C->mul(adds_eye_nK_d, real_YKK_d); C->to_DND_pool(mul_optimization_102);
	auto dot_optimization_103 = C->dot(C->mul(subs_eye_nT_d_1, C->sign(YT_d)), KKT_d_T); C->to_DND_pool(dot_optimization_103);
	auto mul_optimization_104 = C->mul(subs_eye_nK_d_1, imag_YKK_d); C->to_DND_pool(mul_optimization_104);
	auto mul_optimization_105 = C->mul(C->muls(adds_eye_nK_d, -1), imag_YKK_d); C->to_DND_pool(mul_optimization_105);
	C->sanitizeMemoryPools();
	while (delta > EPS) {
	//while (ctr <= 38) {
		std::clock_t loop_start = std::clock();
		if (ctr > maxIteration - 1)
			break;

		auto diagflat_fK_d = C->diagflat(fK_d, true);
		auto diagflat_eK_d = C->diagflat(eK_d, true);

		auto uK_d = C->complex(eK_d, fK_d, true);

		auto uT_d = C->aTb(a, uK_d, true);
		C->sanitizeMemoryPools();

		auto dot_YTuT_d = C->dot(YT_d, uT_d, true); 
		auto dot_YKKuK_d = C->dot(YKK_d, uK_d, true);
		C->sanitizeMemoryPools();

		auto h_uT_d = C->abs(uT_d, true);

		auto h_pqT = C->dot(C->diagflat(C->muls(uT_d, 3)), C->conj(dot_YTuT_d));
		auto h_pT_d = C->real(h_pqT, true);

		auto h_qT_d = C->imag(h_pqT, true); 
		C->sanitizeMemoryPools();

		auto h_pqK = C->dot(C->muls(C->diagflat(uK_d), 3), C->conj(dot_YKKuK_d));
		auto h_pK_d = C->real(h_pqK, true);
		auto h_qK_d = C->imag(h_pqK, true);
		C->sanitizeMemoryPools();


		auto H_uT_d = C->aTb(KKT_d, C->concat(C->diagflat(C->div(eK_d, C->abs(uK_d))), C->diagflat(C->div(fK_d, C->abs(uK_d))), 'x'), true); 
		C->sanitizeMemoryPools();

		// In-loop recomputation avoidance
		auto dot_optimization_2 = C->dot(abT_optimization_1, eK_d, true); C->sanitizeMemoryPools();
		auto dot_optimization_3 = C->dot(abT_optimization_1, fK_d, true); C->sanitizeMemoryPools();
		auto dot_optimization_7 = C->dot(abT_optimization_6, eK_d, true); C->sanitizeMemoryPools();
		auto dot_optimization_8 = C->dot(abT_optimization_6, fK_d, true); C->sanitizeMemoryPools();

		auto temp1_d = C->add(C->dot(abT_optimization_4, eK_d), dot_optimization_3, true); C->sanitizeMemoryPools();
		auto temp2_d = C->add(C->dot(abT_optimization_5, eK_d), C->dot(abT_optimization_4, fK_d), true); C->sanitizeMemoryPools();
		auto temp3_d = C->add(dot_optimization_7, C->dot(abT_optimization_5, fK_d), true); C->sanitizeMemoryPools();
		auto temp4_d = C->add(dot_optimization_2, dot_optimization_8, true); C->sanitizeMemoryPools();
		auto idx_d = dot_optimization_103;
		auto H_pT_d = C->muls(C->add(C->concat(C->dot(C->diagflat(temp1_d), KKT_d_T), C->dot(C->diagflat(temp2_d), KKT_d_T), 'x'),
							C->concat(C->dot(C->diagflat(temp3_d), idx_d), C->dot(C->diagflat(temp4_d), idx_d), 'x')), 3, true);
		C->sanitizeMemoryPools();

		
		temp1_d = C->add(C->dot(dot_optimization_100, eK_d), C->dot(dot_optimization_9, fK_d), true); C->sanitizeMemoryPools();
		temp2_d = C->sub(dot_optimization_7, C->dot(dot_optimization_101, fK_d), true); C->sanitizeMemoryPools();
		temp3_d = C->add(dot_optimization_2, dot_optimization_8, true); C->sanitizeMemoryPools();
		temp4_d = C->add(C->dot(dot_optimization_9, eK_d), dot_optimization_3, true); C->sanitizeMemoryPools();
		auto H_qT_d = C->muls(C->add(C->concat(C->dot(C->diagflat(temp1_d), KKT_d_T), C->dot(C->diagflat(temp2_d), KKT_d_T), 'x'),
								C->concat(C->dot(C->diagflat(temp3_d), idx_d), C->dot(C->diagflat(temp4_d), idx_d), 'x')), 3, true);
		C->sanitizeMemoryPools();

		auto dot_optimization_10 = C->dot(diagflat_fK_d, subs_eye_nK_d_1, true);
		auto mul_optimization_11 = C->mul(dot_optimization_10, real_YKK_d, true);
		auto dot_optimization_12 = C->dot(diagflat_eK_d, subs_eye_nK_d, true);
		temp1_d = C->add(C->dot(mul_optimization_102, eK_d), C->dot(C->mul(subs_eye_nK_d, imag_YKK_d), fK_d), true); C->sanitizeMemoryPools();
		temp2_d = C->add(C->dot(mul_optimization_104, eK_d), C->dot(mul_optimization_102, fK_d), true); C->sanitizeMemoryPools();
		temp3_d = C->add(C->mul(C->dot(diagflat_eK_d, subs_eye_nK_d_1), real_YKK_d), C->mul(dot_optimization_10, imag_YKK_d), true); C->sanitizeMemoryPools();
		temp4_d = C->add(C->mul(dot_optimization_12, imag_YKK_d), mul_optimization_11, true); C->sanitizeMemoryPools();
		auto H_pK_d = C->muls(C->add(C->concat(C->diagflat(temp1_d), C->diagflat(temp2_d), 'x'), C->concat(temp3_d, temp4_d, 'x')), 3, true);
		C->sanitizeMemoryPools();

		temp1_d = C->add(C->dot(mul_optimization_105, eK_d), C->dot(C->mul(subs_eye_nK_d, real_YKK_d), fK_d), true); C->sanitizeMemoryPools();
		temp2_d = C->sub(C->dot(C->mul(subs_eye_nK_d_1, real_YKK_d), eK_d), C->dot(C->mul(adds_eye_nK_d, imag_YKK_d), fK_d), true); C->sanitizeMemoryPools();
		temp3_d = C->add(C->mul(dot_optimization_12, imag_YKK_d), mul_optimization_11, true); C->sanitizeMemoryPools();
		temp4_d = C->add(C->mul(dot_optimization_12, real_YKK_d), C->mul(C->dot(diagflat_fK_d, subs_eye_nK_d), imag_YKK_d), true); C->sanitizeMemoryPools();
		auto H_qK_d = C->muls(C->add(C->concat(C->diagflat(temp1_d), C->diagflat(temp2_d), 'x'), C->concat(temp3_d, temp4_d, 'x')), 3, true);
		C->sanitizeMemoryPools();

		auto eK_fK_d = C->concat(eK_d, fK_d, true);

		auto h_d = C->concat(C->concat(C->concat(C->concat(h_uT_d, h_pT_d), C->concat(h_qT_d, h_pK_d)), C->concat(C->concat(h_qK_d, C->dot(H_eT_d, eK_fK_d)),
								C->concat(C->dot(H_fT_d, eK_fK_d), C->dot(H_aT_d, eK_fK_d)))), C->dot(H_cT_d, eK_fK_d), true); 
		C->sanitizeMemoryPools();

		auto H_d = C->concat(C->concat(C->concat(C->concat(H_uT_d, H_pT_d), C->concat(H_qT_d, H_pK_d)), C->concat(C->concat(H_qK_d, H_eT_d), C->concat(H_fT_d, H_aT_d))), H_cT_d, true); C->sanitizeMemoryPools();

		auto tt2_d = C->specialSlicingOnH(H_d, idx_z_d, true);

		auto solved_d = C->div(C->transpose(tt2_d), tt_d);

		//auto G_d = C->aTb(tt2_d, solved_d, true); C->sanitizeMemoryPools();

		auto G_d = C->aTbT(tt2_d, solved_d, true); C->sanitizeMemoryPools();

		auto part2 = C->dot(C->div(C->transpose(tt2_d), tt_d), C->sub(C->slicei(z_d, idx_z_d), C->slicei(h_d, idx_z_d)), true); C->sanitizeMemoryPools();

	/*	C->printMatrix(G_d);
		C->printMatrix(part2);*/
		auto dx_d = C->solve(G_d, part2, true); C->sanitizeMemoryPools();

		/*C->printMatrix(G_d);
		C->printMatrix(part2);
		C->printMatrix(dx_d);*/

		delta = (C->maxValue(C->abs(dx_d)));
		C->sanitizeMemoryPools();


		eK_d = C->update_eK_fK(eK_d, C->slice(dx_d, 0, nK_d, 0, dx_d.width, true)); C->sanitizeMemoryPools();
		//C->printMatrix(eK_d);
		fK_d = C->update_eK_fK(fK_d, C->slice(dx_d, nK_d, 2 * nK_d, 0, dx_d.width, true)); C->sanitizeMemoryPools();
		//C->printMatrix(fK_d);

		C->releaseOnHoldAllocationsToPool();

		printf("\n\t\t\t\tIteration #%d complete! delta = %f", ctr, delta);
		//C->print_execution_times();
		ctr++;
		printf("\tMemory occupied = %f", (float)C->dMManager.total_occupied_memory_in_bytes / (1024 * 1024 * 1024));
		
		/*float seconds = (std::clock() - loop_start) / (double)(CLOCKS_PER_SEC / 1000) / 1000;
		std::cout << "\tTime: " << seconds << " s" << std::endl;*/
	}

	printf("\n\tMemory occupied = %f", (float)C->dMManager.total_occupied_memory_in_bytes / (1024 * 1024 * 1024));
	//getchar();
	//C->printMatrix(eK_d);
	//C->printMatrix(fK_d);
	//C->print_execution_times();

	//printf("\nDisplaying voltage magnitudes...");
	auto voltages = C->divs(C->abs(C->complex(eK_d, fK_d)), 1000);
	C->to_DND_pool(voltages);
	//C->printMatrix(voltages);
	//getchar();

	//printf("\nDisplaying voltage angles in degrees...");
	auto phases = C->angle(eK_d, fK_d);
	auto slackBusPhase = C->slice(phases, 0, 1, 0, 1);
	double slackBusPhaseOnHost = C->to_host(slackBusPhase);
	phases = C->subs(phases, slackBusPhaseOnHost);
	
	C->to_DND_pool(phases);
	/*C->printMatrix(phases);
	getchar();*/
	C->write_to_file(voltages, phases);
	printf("\nState Estimation computation concluded.");



	StateEstimationResults s;
	s.phases = phases;
	s.voltages = voltages;

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	std::cout << "\n\t Size: " << dataset.grid_size << ", Time: " << time_span.count() << " s, " << "Iterations: " << --ctr << std::endl;
	s.timeTakenInSeconds = time_span.count();
	C->setSEresults(s);
	//delete C;
}

