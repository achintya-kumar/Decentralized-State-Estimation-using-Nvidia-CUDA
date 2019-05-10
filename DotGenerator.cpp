#include "DotGenerator.h"
using std::map;

typedef unsigned int uint;

enum NodeType{
	BUS, PMU
};

struct VertexData
{
	std::string id;
	NodeType type;
	std::string color;
};

template <typename Map>
struct my_node_writer {
	// my_node_writer() {}
	my_node_writer(Map& g_) : g(g_) {};
	template <class Vertex>
	void operator()(std::ostream& out, Vertex v) {
		out << " [label=\"" << g[v].id << "\", color=\"" + g[v].color + "\"]" << std::endl;
	};
	Map g;
};

template <typename Map>
my_node_writer<Map> node_writer(Map& map) { return my_node_writer<Map>(map); }


void generate_dot_file(std::tuple<cuDoubleComplex*, int, int, double*, int, int> tuple) {

	cuDoubleComplex* YKK = std::get<0>(tuple);
	int YKK_width = std::get<1>(tuple);
	int YKK_height = std::get<2>(tuple);

	double* KKT = std::get<3>(tuple);
	int no_of_terminals = std::get<4>(tuple);

	std::vector<int> pmu_location_indices = pmuLocations(YKK_height);
	uint no_of_PMUs = pmu_location_indices.size();
	int& no_of_buses = YKK_height;

	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexData, boost::no_property> Graph;
	typedef boost::graph_traits<Graph>::vertex_descriptor vertex_general;
	Graph g;

	std::vector<vertex_general> bus_vertices;
	std::vector<vertex_general> pmu_vertices;

	// Preparation of different vertices of the graph
	//std::cout << "no of buses = " << no_of_buses << std::endl;
	//std::cout << "no of PMUs = " << no_of_PMUs << std::endl;
	for (int g_index = 0; g_index < no_of_buses + no_of_PMUs; g_index++) {
		if (g_index < no_of_buses) {
			VertexData vd = { "B" + std::to_string(g_index + 1), BUS, "gray" };								// Preparing data associated with the bus vertex
			auto bus_vertex_pointer = boost::add_vertex(g);												// Bus Vertex pointer is returned after adding the vertex
			bus_vertices.push_back(bus_vertex_pointer);													// Holding the bus vertex pointers for easy querying
			g[bus_vertex_pointer] = vd;																	// Inserting the data associated with the bus vertex into it.
		}
		else if (g_index >= no_of_buses && g_index < no_of_buses + no_of_PMUs) {
			//VertexData vd = { "uP" + std::to_string(g_index + 1 - no_of_buses), PMU, "gold" };	// Preparing data associated with the PMU vertex
			//auto pmu_vertex_pointer = boost::add_vertex(g);												// PMU Vertex pointer is returned after adding the vertex
			//pmu_vertices.push_back(pmu_vertex_pointer);													// Holding the PMU vertex pointers for easy querying
			//g[pmu_vertex_pointer] = vd;																	// Inserting the data associated with the PMU vertex into it.
		}
	}


	std::map<int, std::vector<int>> leftMap;
	std::map<int, std::vector<int>> rightMap;
	std::map<int, std::vector<int>>::iterator leftMapIterator;
	std::map<int, std::vector<int>>::iterator rightMapIterator;
	//std::cout << KKT[0] << " " << KKT[1] << std::endl;
	if (no_of_PMUs > 0) {
		for (int i = 0; i < no_of_PMUs; i++) {
			uint pmu_terminal_id = pmu_location_indices[i] - 1;
			uint pmu_next_terminal_id;
			if (pmu_terminal_id % 2 == 0)																	// Finding the other terminal (id) connected to given PMU terminal
				pmu_next_terminal_id = pmu_terminal_id + 1;
			else
				pmu_next_terminal_id = pmu_terminal_id - 1;

			uint nearestBus, secondNearestBus;
			for (int j = 0; j < no_of_buses; j++) {
				if (KKT[pmu_terminal_id + j * no_of_terminals] == 1.0) {
					nearestBus = j;
				}
				if (KKT[pmu_next_terminal_id + j * no_of_terminals] == 1.0){
					secondNearestBus = j;
				}
			}
			//boost::add_edge(bus_vertices[nearestBus], pmu_vertices[i], g);									// Connecting vPMU vertex to the correct BUS vertex
			//boost::add_edge(bus_vertices[secondNearestBus], pmu_vertices[i], g);							// Connecting vPMU vertex to the second nearest BUS vertex						

			//leftMapIterator = leftMap.find(nearestBus);

			//if (leftMapIterator == leftMap.end()) {
			//	std::vector<int> connections = { (int)secondNearestBus };
			//	leftMap.insert(std::pair<int, std::vector<int>>(nearestBus, connections));
			//}
			//else {
			//	leftMapIterator->second.push_back(secondNearestBus);
			//}

			//rightMapIterator = rightMap.find(secondNearestBus);
			//if (rightMapIterator == rightMap.end()) {
			//	std::vector<int> connections = { (int)nearestBus };
			//	rightMap.insert(std::pair<int, std::vector<int>>(secondNearestBus, connections));
			//}
			//else {
			//	rightMapIterator->second.push_back(nearestBus);
			//}
			//leftMap.insert(std::pair<int, int>(nearestBus, secondNearestBus));								// Maintaining a bidirectional map to prevent connecting the -
			//rightMap.insert(std::pair<int, int>(secondNearestBus, nearestBus));									// nearest and the second nearest busses.
			
			g[nearestBus].color = "red";
		}
	}


	// Connecting bus-vertices based on the YKK matrix elements
	for (int i = 0; i < YKK_width; i++) {
		for (int j = 0; j < YKK_height; j++) {
			if (i > j) {								// Only checking upper triangular matrix. Lower triangular matrix is a repitition.
				if (cuCabs(YKK[i + j * YKK_width]) != 0.0) {
					//boost::add_edge(i, j, g);

					leftMapIterator = leftMap.find(i);
					rightMapIterator = rightMap.find(i);

					if (leftMapIterator != leftMap.end()) {
						auto connectionsVector = leftMapIterator->second;
						if (find(connectionsVector.begin(), connectionsVector.end(), j) != connectionsVector.end())
							continue;
						else
							boost::add_edge(bus_vertices[i], bus_vertices[j], g);
					}
					else if (rightMapIterator != rightMap.end()) {
						auto connectionsVector = rightMapIterator->second;
						if (find(connectionsVector.begin(), connectionsVector.end(), j) != connectionsVector.end())
							continue;
						else
							boost::add_edge(bus_vertices[i], bus_vertices[j], g);
					}
					/*if (leftMapIterator != leftMap.end()) {
						if (leftMapIterator->second == j)
							continue;
					}
					else if (rightMapIterator != rightMap.end()) {
						if (rightMapIterator->second == j)
							continue;
					}*/
					else 
						boost::add_edge(bus_vertices[i], bus_vertices[j], g);
				}
			}
		}
	}

	std::ofstream f("C:\\Users\\kumar\\Google Drive\\Masterarbeit ROOT\\Masterarbeit Implementation\\DecentralizedStateEstimationWithCuda\\AGrid.dot");
	boost::write_graphviz(f, g, node_writer(g));
	f.close();
}

std::tuple<cuDoubleComplex*, int, int, double*, int, int> read_from_files(std::string directory) {
	std::tuple<std::string, void*, int, int> YKK_data = read_from_file(directory + "\\YKK.csv");
	std::tuple<std::string, void*, int, int> KKT_data = read_from_file(directory + "\\KKT.csv");

	(cuDoubleComplex*)std::get<1>(YKK_data);

	std::tuple<cuDoubleComplex*, int, int, double*, int, int> result = std::make_tuple((cuDoubleComplex*)std::get<1>(YKK_data), std::get<2>(YKK_data), std::get<3>(YKK_data),
								(double*)std::get<1>(KKT_data), std::get<2>(KKT_data), std::get<3>(KKT_data));

	//std::cout << std::get<1>(result);
	return result;

}

std::tuple<std::string, void*, int, int> read_from_file(std::string file) {

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
				catch (std::exception& e) { isComplexZ = true; }
			}

			if (rows == 0)
				columns++;
		}
		rows++;
	}
	data.close();

	//printf("\nFile: %s", file);
	//printf("\nRows = %d, Columns = %d, Dtype = %s", rows, columns, isComplexZ ? "ComplexZ" : "Double");


	// PHASE 2: Loading the items into the memory.
	cuDoubleComplex *complex_elements;
	double *double_elements;
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
			if (!isComplexZ)  { // When the elements are of type Double
				double_elements[counter] = std::stod(cell);
				////std::cout << (double_elements[counter] == 0.0 ? 0 : 1) << " ";
			}
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
				//std::cout << (cuCabs(complex_elements[counter]) == 0.0? 0 : 1 ) << " ";
			}
			counter++;
		}
		//std::cout << std::endl;
	}

	data2.close();
	return std::make_tuple((isComplexZ ? "Complex" : "Double"),
		(isComplexZ ? (void*)complex_elements : (void*)double_elements),
		columns,
		rows);
}

std::vector<int> pmuLocations(int grid_size) {
	std::vector<int> pmu_location_indices;
	if (grid_size == 4) {
		pmu_location_indices = { 2 };							// Matlab says 2. Subtracting 1 gives the C-appropriate value
	}
	else if (grid_size == 14 || grid_size == 30) {										
		pmu_location_indices = { 2, 7, 11, 13 };				// Matlab says 2. Subtracting 1 gives the C-appropriate value
	}
	else if (grid_size%118 == 0) {
		/*pmu_location_indices = { 3, 5, 9, 12, 15, 17, 21, 25, 28, 34, 37, 40, 
								45, 49, 53, 56, 62, 64, 68, 70, 71, 75, 77, 80, 
								85, 86, 91, 94, 101, 105, 110, 114 };*/
		pmu_location_indices = { 4, 6, 9, 14, 17, 21, 26, 30, 33, 36, 39, 45, 
								51, 54, 55, 62, 64, 68, 69, 72, 77, 87, 94, 97, 
								102, 110, 111, 113, 119, 122, 123, 130, 138, 139, 
								141, 146, 151, 158, 159, 163, 171, 182, 188, 189, 
								199, 201, 208, 218, 223, 225, 234, 236, 238, 241, 
								250, 251, 255, 262, 264, 267, 269, 271, 280, 290, 
								292, 293, 299, 301, 303, 305, 309, 323, 336, 337, 
								339, 341, 348, 350, 351, 353, 355, 360, 363, 365, 367, 369 };
	}
	else if (grid_size == 300) {
		/*pmu_location_indices = { 3, 5, 7, 8, 9, 10, 12, 14, 15, 16, 22, 23, 26,
			31, 32, 33, 35, 36, 38, 41, 45, 46, 51, 56, 57,
			61, 62, 64, 65, 70, 74, 76, 82, 83, 89, 90, 91,
			99, 100, 107, 108, 109, 112, 113, 114, 122, 132,
			136, 141, 151, 156, 161, 164, 167, 170, 174, 175,
			180, 183, 184, 188, 191, 193, 195, 197, 201, 204,
			205, 230, 247, 257, 262, 263, 266, 268, 270, 275,
			278, 280, 284, 285, 286, 287, 289, 290, 293, 297 };*/

		pmu_location_indices = { 4, 9, 11, 13, 15, 17, 20, 27, 33, 37, 39,
								41, 43, 47, 49, 51, 56, 57, 59, 61, 63, 65,
								67, 69, 71, 73, 77, 79, 81, 83, 85, 102, 103,
								110, 111, 113, 115, 124, 128, 131, 133, 137,
								139, 141, 143, 145, 175, 182, 189, 191, 193,
								195, 202, 209, 214, 221, 228, 230, 233, 235,
								238, 243, 247, 250, 260, 262, 263, 265, 267, 269, 276, 279, 281, 284,
								285, 287, 296, 300, 302, 303, 305, 307, 316, 328, 331, 333, 335, 339,
								346, 348, 350, 354, 358, 363, 367, 370, 380, 390, 396, 398, 403, 405,
								411, 413, 415, 417, 424, 429, 435, 437, 439, 443, 445, 456, 460, 464,
								468, 469, 472, 480, 484, 490, 497, 499, 501, 510, 511, 513, 520, 522,
								524, 525, 528, 538, 542, 552, 554, 568, 570, 572, 574, 575, 582, 586,
								587, 589, 594, 595, 602, 606, 609, 612, 614, 617, 620, 622, 626, 629,
								631, 640, 642, 644, 645, 654, 657, 659, 666, 667, 673, 680, 686, 692,
								694, 699, 703, 711, 717, 719, 721, 723, 737, 741, 744, 746, 747, 750,
								762, 763, 766, 767, 773, 777, 781, 783, 786, 788, 790, 792, 794, 796,
								798, 800, 802, 804, 806, 808, 810, 812, 814, 816, 818, 820, 822 };

	}
	else
		exit(10000);

	return pmu_location_indices;
}