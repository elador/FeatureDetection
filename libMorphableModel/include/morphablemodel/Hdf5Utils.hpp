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

// Todo: Class with static methods? Or just functions? I don't know which method is better.

namespace morphablemodel {
	namespace utils {

		class Hdf5Utils
		{
		public:
			static H5::H5File openFile(const std::string filename);
			static H5::Group openPath(H5::H5File& file, const std::string& path);

			static cv::Mat readMatrixFloat(const H5::CommonFG& fg, std::string name);
			static void readMatrixInt(const H5::CommonFG& fg, std::string name, cv::Mat& matrix);
			static void readVector(const H5::CommonFG& fg, std::string name, std::vector<float>& vector);
			static std::string readString(const H5::CommonFG& fg, std::string name);

			static bool existsObjectWithName(const H5::CommonFG& fg, const std::string& name);

			static render::Mesh readFromHdf5(std::string filename);
		};

	} /* namespace utils */
} /* namespace morphablemodel */

#endif /* WITH_MORPHABLEMODEL_HDF5 */

#endif /* HDF5UTILS_HPP_ */
