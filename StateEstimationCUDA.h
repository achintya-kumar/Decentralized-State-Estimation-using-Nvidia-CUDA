#pragma once
#include "DeviceMatrix.h"
#include "CudaFunctions_Base.h"
#include "CudaFunctions_Double.h"
#include "CudaFunctions_Single.h"
#include "DataImporter.h"

#include <string>

void stateEstimationCUDAv3(std::pair<GridDataSet, CudaFunctions_Base*> dataAndCudaBundle, bool debug_mode);