/*
 * Hdf5Utils.cpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */

#include "morphablemodel/Hdf5Utils.hpp"

namespace render {
	namespace utils {

H5::H5File Hdf5Utils::openFile(const std::string filename) {
	H5::H5File file;

	try {
		file = H5::H5File(filename, H5F_ACC_RDONLY);
	}
	catch (H5::Exception& e) {
		std::string msg( std::string( "Could not open HDF5 file \n" ) + e.getCDetailMsg() );
		throw msg;
	}
	return file;
}

H5::Group Hdf5Utils::openPath(H5::H5File& file, const std::string& path) {
	H5::Group group;

	// take the first part of the path
	size_t curpos = 1;
	size_t nextpos = path.find_first_of("/", curpos);
	H5::Group g = file.openGroup("/");

	std::string name = path.substr(curpos, nextpos-1);

	while (curpos != std::string::npos && name != "") {

		if (existsObjectWithName(g, name)) {
			g = g.openGroup(name);
		} else {
			std::string msg = std::string("the path ") +path +" does not exist";
			throw msg.c_str();
		}

		curpos = nextpos+1;
		nextpos = path.find_first_of("/", curpos);
		if ( nextpos != std::string::npos )
			name = path.substr(curpos, nextpos-curpos);
		else
			name = path.substr(curpos);
	}

	return g;
}

cv::Mat Hdf5Utils::readMatrixFloat(const H5::CommonFG& fg, std::string name) {
	
	H5::DataSet ds = fg.openDataSet( name );
	hsize_t dims[2];
	ds.getSpace().getSimpleExtentDims(dims, NULL);
	cv::Mat matrix((int)dims[0], (int)dims[1], CV_32FC1); // r, c?
	// simply read the whole dataspace
	ds.read(matrix.data, H5::PredType::NATIVE_FLOAT);

	return matrix;
}
void Hdf5Utils::readMatrixInt(const H5::CommonFG& fg, std::string name, cv::Mat& matrix) {
	H5::DataSet ds = fg.openDataSet( name ); // ./triangle-list
	hsize_t dims[2];
	ds.getSpace().getSimpleExtentDims(dims, NULL);

	matrix.resize(dims[0], dims[1]);
	ds.read(matrix.data, H5::PredType::NATIVE_INT32);
	if ( matrix.cols != 3 )
		throw std::runtime_error("Reference reading failed, triangle-list has not 3 indices per entry");

}

void Hdf5Utils::readVector(const H5::CommonFG& fg, std::string name, std::vector<float>& vector) {
	H5::DataSet ds = fg.openDataSet( name );
	hsize_t dims[1];
	ds.getSpace().getSimpleExtentDims(dims, NULL);
	vector.resize(dims[0], 1);
	ds.read(vector.data(), H5::PredType::NATIVE_FLOAT);
}

std::string Hdf5Utils::readString(const H5::CommonFG& fg, std::string name) {
	std::string outputString;
	H5::DataSet ds = fg.openDataSet(name);
	ds.read(outputString, ds.getStrType());
	return outputString;
}

bool Hdf5Utils::existsObjectWithName(const H5::CommonFG& fg, const std::string& name) {
	for (hsize_t i = 0; i < fg.getNumObjs(); ++i) {
		std::string objname= 	fg.getObjnameByIdx(i);
		if (objname == name) {
			return true;
		}
	}
	return false;
}

	} /* namespace utils */
} /* namespace render */
