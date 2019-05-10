#pragma once
#include "DeviceMatrix.h"

#include <boost/algorithm/string.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include <cusparse_v2.h>
#include <cusolverSp.h>
#include "helper_cuda.h"

#include "CudaFunctions_Base.h"


struct WholeAndTheParts{
	GridDataSet theWhole;
	unsigned int number_of_partitions;
	std::vector<GridDataSet> partitions;
};


class DataImporter
{
private:
	// Loads the properties in the 'application.grid.properties' file. For internal use only.
	static std::map<std::string, std::string> loadGridProperties(std::string propertiesFilePath);


public:
	DataImporter();
	~DataImporter();

	// Loads the grid matrices and also the partitions as specified in the 'application.grid.properties' file
	static WholeAndTheParts load(CudaFunctions_Base*& C);
};								
