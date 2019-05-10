#include "DataImporter.h"


DataImporter::DataImporter()
{
}


DataImporter::~DataImporter()
{
}

// Loads the grid matrices and also the partitions as specified in the 'application.grid.properties' file
WholeAndTheParts DataImporter::load(CudaFunctions_Base*& C) {
	std::map<std::string, std::string> properties = loadGridProperties("application.grid.properties");
	if (properties.size() == 0) {
		std::cout << "Couldn't load properties. Please verify the path to 'properties.txt' file" << std::endl;
		exit(1); // Abnormal exit.
	}

	std::string dataSetDirectoryPath = properties["path"];
	int gridSize = stoi(properties["gridsize"]);
	if (dataSetDirectoryPath.substr(dataSetDirectoryPath.length() - 1) != "\\")
		dataSetDirectoryPath += "\\";

	// Loading raw matrices
	printf("\nLoading KKT...");
	DeviceMatrix KKT_d = C->to_device(dataSetDirectoryPath + "KKT.csv");
	DeviceMatrix KKT_d_T = C->transpose(KKT_d);
	C->to_DND_pool(KKT_d);
	C->to_DND_pool(KKT_d_T);

	printf("\nLoading YT...");
	/*DeviceMatrix YT_d = C->to_device(dataSetDirectoryPath + "YT.csv");*/
	DeviceMatrix YT_d_real = C->to_device(dataSetDirectoryPath + "YT_real.csv");
	DeviceMatrix YT_d_imag = C->to_device(dataSetDirectoryPath + "YT_imag.csv");
	DeviceMatrix YT_d = C->complex(YT_d_real, YT_d_imag);
	C->to_DND_pool(YT_d);

	printf("\nLoading YKK...");
	DeviceMatrix YKK_d = C->to_device(dataSetDirectoryPath + "YKK.csv");
	/*DeviceMatrix YKK_d_real = C->to_device(dataSetDirectoryPath + "YKK_real.csv");
	DeviceMatrix YKK_d_imag = C->to_device(dataSetDirectoryPath + "YKK_imag.csv");
	DeviceMatrix YKK_d = C->complex(YKK_d_real, YKK_d_imag);*/
	C->to_DND_pool(YKK_d);

	printf("\nLoading INIT...");
	DeviceMatrix INIT_d = C->to_device(dataSetDirectoryPath + "INIT.csv");
	/*DeviceMatrix INIT_d_real = C->to_device(dataSetDirectoryPath + "INIT_real.csv");
	DeviceMatrix INIT_d_imag = C->to_device(dataSetDirectoryPath + "INIT_imag.csv");
	DeviceMatrix INIT_d = C->complex(INIT_d_real, INIT_d_imag);*/
	C->to_DND_pool(INIT_d);

	printf("\nLoading MEAS...");
	/*DeviceMatrix MEAS_d = C->to_device(dataSetDirectoryPath + "MEAS.csv"); C->to_DND_pool(MEAS_d);*/

	//Let us assume that MEAS matrix is supplied separately and assembled together inside application
	DeviceMatrix z_uT, z_pT, z_qT, z_pK, z_qK, z_uTabs, z_uTang, z_iTabs, z_iTang;
	z_uT = C->to_device(dataSetDirectoryPath + "z_uT.csv");
	z_pT = C->to_device(dataSetDirectoryPath + "z_pT.csv");
	z_qT = C->to_device(dataSetDirectoryPath + "z_qT.csv");
	z_pK = C->to_device(dataSetDirectoryPath + "z_pK.csv");
	z_qK = C->to_device(dataSetDirectoryPath + "z_qK.csv");
	z_uTabs = C->to_device(dataSetDirectoryPath + "z_uTabs.csv");
	z_uTang = C->to_device(dataSetDirectoryPath + "z_uTang.csv");
	z_iTabs = C->to_device(dataSetDirectoryPath + "z_iTabs.csv");
	z_iTang = C->to_device(dataSetDirectoryPath + "z_iTang.csv");
	DeviceMatrix MEAS_d = C->concat(C->concat(C->concat(C->concat(C->concat(C->concat(C->concat(
		C->concat(z_uT, z_pT), z_qT), z_pK), z_qK), z_uTabs), z_uTang), z_iTabs), z_iTang);
	C->to_DND_pool(MEAS_d);
	
	GridDataSet dataset;// (KKT_d, KKT_d_T, YT_d, YKK_d, INIT_d, MEAS_d);
	dataset.KKT = KKT_d;
	dataset.KKT_t = KKT_d_T;
	dataset.YT = YT_d;
	dataset.YKK = YKK_d;
	dataset.INIT = INIT_d;
	dataset.MEAS = MEAS_d;
	dataset.grid_size = gridSize;

	//Loading grid-properties from the properties file
	WholeAndTheParts wholeAndTheParts;
	wholeAndTheParts.number_of_partitions = stoi(properties["number_of_partitions"]);
	//wholeAndTheParts.partitions.reserve(wholeAndTheParts.number_of_partitions);
	for (int i = 0; i < wholeAndTheParts.number_of_partitions; i++) {
		std::string startToEndString = properties["partition" + std::to_string(i+1)];
		std::vector<std::string> splits;
		boost::split(splits, startToEndString, boost::is_any_of(","));
		
		//Bus coordinates
		int bus_start = stoi(splits[0]) - 1; // Adjusting from real world to computer world.
		int bus_end = stoi(splits[1]);
		// Terminal coordinates
		int terminal_start = stoi(splits[2]) - 1;
		int terminal_end = stoi(splits[3]);
		
		GridDataSet partition_dataset;
		partition_dataset.KKT = C->slice(KKT_d, bus_start, bus_end, terminal_start, terminal_end); 
			C->to_DND_pool(partition_dataset.KKT);
		partition_dataset.KKT_t = C->transpose(partition_dataset.KKT); 
			C->to_DND_pool(partition_dataset.KKT_t);
		partition_dataset.YT = C->slice(YT_d, terminal_start, terminal_end, terminal_start, terminal_end);  
			C->to_DND_pool(partition_dataset.YT);
		partition_dataset.YKK = C->dot(C->dot(C->muls(C->complexify(partition_dataset.KKT), -1), partition_dataset.YT), C->complexify(partition_dataset.KKT_t));
			C->to_DND_pool(partition_dataset.YKK);
		partition_dataset.INIT = C->slice(INIT_d, bus_start, bus_end, 0, 1);
			C->to_DND_pool(partition_dataset.INIT);

		// Extract data for partition MEAS and assemble it
		DeviceMatrix z_uT_p, z_pT_p, z_qT_p, z_pK_p, z_qK_p, z_uTabs_p, z_uTang_p, z_iTabs_p, z_iTang_p;
		z_uT_p = C->slice(z_uT, terminal_start, terminal_end, 0, 2);
		z_pT_p = C->slice(z_pT, terminal_start, terminal_end, 0, 2);
		z_qT_p = C->slice(z_qT, terminal_start, terminal_end, 0, 2);
		z_pK_p = C->slice(z_pK, bus_start, bus_end, 0, 2);
		z_qK_p = C->slice(z_qK, bus_start, bus_end, 0, 2);
		z_uTabs_p = C->slice(z_uTabs, terminal_start, terminal_end, 0, 2);
		z_uTang_p = C->slice(z_uTang, terminal_start, terminal_end, 0, 2);
		z_iTabs_p = C->slice(z_iTabs, terminal_start, terminal_end, 0, 2);
		z_iTang_p = C->slice(z_iTang, terminal_start, terminal_end, 0, 2);
		partition_dataset.MEAS = C->concat(C->concat(C->concat(C->concat(C->concat(C->concat(C->concat(
									C->concat(z_uT_p, z_pT_p), z_qT_p), z_pK_p), z_qK_p), z_uTabs_p), z_uTang_p), z_iTabs_p), z_iTang_p);

		C->to_DND_pool(partition_dataset.MEAS);
		partition_dataset.grid_size = bus_end - bus_start; 
		wholeAndTheParts.partitions.push_back(partition_dataset);
	}

	wholeAndTheParts.theWhole = dataset;
	return wholeAndTheParts;
}

std::map<std::string, std::string> DataImporter::loadGridProperties(std::string propertiesFilePath) {
	std::map<std::string, std::string> properties;

	//Reading from file
	std::ifstream  data(propertiesFilePath);
	std::string line;
	while (std::getline(data, line))
	{
		boost::trim(line);
		if (line[0] != '#' && line != "") {
			std::vector<std::string> splits;
			std::cout << line << std::endl;
			boost::replace_all(line, "}", "");
			boost::replace_all(line, "{", "");
			boost::split(splits, line, boost::is_any_of("="));
			properties[splits[0]] = splits[1];
		}
	}

	return properties;
}