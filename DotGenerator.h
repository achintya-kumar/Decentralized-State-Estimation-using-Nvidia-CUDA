#pragma once
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/bimap.hpp>

//	CUDA's Complex number support header
#include <cuComplex.h>


void generate_dot_file(std::tuple<cuDoubleComplex*, int, int, double*, int, int> tuple);
std::tuple<std::string, void*, int, int> read_from_file(std::string file);
std::tuple<cuDoubleComplex*, int, int, double*, int, int> read_from_files(std::string directory);
std::vector<int> pmuLocations(int grid_size);