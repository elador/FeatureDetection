/*
 * Hdf5Utils.hpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */
#pragma once

#ifndef HDF5UTILS_HPP_
#define HDF5UTILS_HPP_

#ifdef WITH_MORPHABLEMODEL_HDF5

#include "render/Mesh.hpp"

#include "H5Cpp.h"

#include "opencv2/core/core.hpp"

#include <vector>

namespace morphablemodel {
	namespace hdf5utils { // TODO: Update the filename to lowercase

// Note: Hmm, all those methods below aren't used anywhere in our code. Check the code and update the code to use them?

// readonly
H5::H5File openFile(const std::string& filename);
H5::Group openPath(H5::H5File& file, const std::string& path);

cv::Mat readMatrixFloat(const H5::CommonFG& fg, std::string name);
void readMatrixInt(const H5::CommonFG& fg, std::string name, cv::Mat& matrix);
void readVector(const H5::CommonFG& fg, std::string name, std::vector<float>& vector);
std::string readString(const H5::CommonFG& fg, std::string name);

bool existsObjectWithName(const H5::CommonFG& fg, const std::string& name);

render::Mesh readReference(std::string filename);

// Below: New:



	} /* namespace hdf5utils */
} /* namespace morphablemodel */

#endif /* WITH_MORPHABLEMODEL_HDF5 */

#endif /* HDF5UTILS_HPP_ */
