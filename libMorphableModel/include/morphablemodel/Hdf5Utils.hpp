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

bool existsObjectWithName(const H5::CommonFG& fg, const std::string& name); // ok

render::Mesh readReference(std::string filename);

// Below: New:
H5::H5File openOrCreate(const std::string& filename);

// Note: Do all those methods leave the DataSet "open", i.e. do we have to call close() on them
// outside the functions? That wouldn't be so nice?
// The d'tor closes the DataSet. Meaning, if I capture the return value, I _have to_ (!) call
// close() on it. If I don't capture it, the d'tor takes care of it.
// Todo: Change them all to return void.
// create a new entry for an int in the fg, with the given name, and write the value to it.
// return the created dataset.
H5::DataSet writeInt(const H5::CommonFG& fg, const std::string& name, int value);

// create a new entry for a float in the fg, with the given name, and write the value to it.
// return the created dataset.
H5::DataSet writeFloat(const H5::CommonFG& fg, const std::string& name, float value);

// at least 1 row and col (no empty matrices), 32FC1, continuous
H5::DataSet writeMatrixInt(const H5::CommonFG& fg, const std::string& name, const cv::Mat& matrix);
H5::DataSet writeMatrixFloat(const H5::CommonFG& fg, const std::string& name, const cv::Mat& matrix);
H5::DataSet writeMatrixDouble(const H5::CommonFG& fg, const std::string& name, const cv::Mat& matrix);

// cv::Mat: input: Col-vector!
H5::DataSet writeVectorInt(const H5::CommonFG& fg, const std::string& name, const cv::Mat& vector);
H5::DataSet writeVectorFloat(const H5::CommonFG& fg, const std::string& name, const cv::Mat& vector);
H5::DataSet writeVectorDouble(const H5::CommonFG& fg, const std::string& name, const cv::Mat& vector);

// I think the 'int' type depends on the platform. So we should use a 32-bit int type from cstd... instead.
H5::DataSet writeArrayInt(const H5::CommonFG& fg, const std::string& name, const std::vector<int>& array);

H5::DataSet writeString(const H5::CommonFG& fg, const std::string& name, const std::string& s);

void writeStringAttribute(const H5::H5Object& fg, const std::string& name, const std::string& s);
void writeIntAttribute(const H5::H5Object& fg, const std::string& name, int value);

	} /* namespace hdf5utils */
} /* namespace morphablemodel */

#endif /* WITH_MORPHABLEMODEL_HDF5 */

#endif /* HDF5UTILS_HPP_ */
