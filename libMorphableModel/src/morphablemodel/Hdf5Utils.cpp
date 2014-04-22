/*
 * Hdf5Utils.cpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */
#ifdef WITH_MORPHABLEMODEL_HDF5

#include "morphablemodel/Hdf5Utils.hpp"

namespace morphablemodel {
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

render::Mesh Hdf5Utils::readFromHdf5(std::string filename)
{
	render::Mesh mesh;

	// Open the HDF5 file
	H5::H5File h5Model;
	try
	{
		h5Model = H5::H5File(filename, H5F_ACC_RDONLY);
	}
	catch (H5::Exception& e)
	{
		std::string msg(std::string("could not open HDF5 file \n") + e.getCDetailMsg());
		throw msg.c_str();
	}

	// make a MM and load both the shape and color models (maybe in another function). For now, we only load the mesh & color info.

	H5::Group fg = h5Model.openGroup("/color");

	//if ( Hdf5Utils::existsObjectWithName( fg, "reference-mesh" ) )
	//{
	//H5::Group fgRef = fg.openGroup( "representer/reference-mesh" );
	fg = fg.openGroup("representer/reference-mesh");

	// Vertex coordinates
	cv::Mat matVertex = Hdf5Utils::readMatrixFloat(fg, "./vertex-coordinates");
	if (matVertex.cols != 3)
		throw std::runtime_error("Reference reading failed, vertex-coordinates have too many dimensions");
	mesh.vertex.resize(matVertex.rows);
	for (unsigned int i = 0; i < matVertex.rows; ++i)
	{
		mesh.vertex[i].position = cv::Vec4f(matVertex.at<float>(i, 0), matVertex.at<float>(i, 1), matVertex.at<float>(i, 2), 1.0f);
	}

	// triangle list
	// get the integer matrix
	H5::DataSet ds = fg.openDataSet("./triangle-list");
	hsize_t dims[2];
	ds.getSpace().getSimpleExtentDims(dims, NULL);
	cv::Mat matTL((int)dims[0], (int)dims[1], CV_32SC1);	// int
	//matTL.resize(dims[0], dims[1]);
	ds.read(matTL.data, H5::PredType::NATIVE_INT32);
	ds.close();
	if (matTL.cols != 3)
		throw std::runtime_error("Reference reading failed, triangle-list has not 3 indices per entry");

	mesh.tvi.resize(matTL.rows);
	for (size_t i = 0; i < matTL.rows; ++i) {
		mesh.tvi[i][0] = matTL.at<int>(i, 0);
		mesh.tvi[i][1] = matTL.at<int>(i, 1);
		mesh.tvi[i][2] = matTL.at<int>(i, 2);
	}

	// color coordinates
	cv::Mat matColor = Hdf5Utils::readMatrixFloat(fg, "./vertex-colors");
	if (matColor.cols != 3)
		throw std::runtime_error("Reference reading failed, vertex-colors have too many dimensions");
	//pReference->color.resize( matColor.rows );
	for (size_t i = 0; i < matColor.rows; ++i)
	{
		mesh.vertex[i].color = cv::Vec3f(matColor.at<float>(i, 2), matColor.at<float>(i, 1), matColor.at<float>(i, 0));	// order in hdf5: RGB. Order in OCV/vertex.color: BGR
	}

	// 		// triangle list
	// 		// get the integer matrix
	// 		ds = fg.openDataSet( "./triangle-color-indices" );
	// 		//hsize_t dims[2];
	// 		ds.getSpace().getSimpleExtentDims(dims, NULL);
	// 		cv::Mat matTLC((int)dims[0], (int)dims[1], CV_32SC1);	// int
	// 		//matTLC.resize(dims[0], dims[1]);
	// 		ds.read( matTLC.data, H5::PredType::NATIVE_INT32);
	// 		ds.close();
	// 		
	// 		if ( matTLC.cols != 3 )
	// 			throw std::runtime_error("Reference reading failed, triangle-color-indices has not 3 indices per entry");
	// 
	// 		mesh.tci.resize( matTLC.rows );
	// 		for ( size_t i = 0; i < matTLC.rows; ++i ) {
	// 			mesh.tci[i][0] = matTLC.at<int>(i, 0);
	// 			mesh.tci[i][1] = matTLC.at<int>(i, 1);
	// 			mesh.tci[i][2] = matTLC.at<int>(i, 2);
	// 		}
	mesh.tci.resize(matTL.rows);
	for (size_t i = 0; i < matTL.rows; ++i) {
		mesh.tci[i][0] = matTL.at<int>(i, 0);
		mesh.tci[i][1] = matTL.at<int>(i, 1);
		mesh.tci[i][2] = matTL.at<int>(i, 2);
	}

	fg.close();
	//}

	h5Model.close();

	mesh.hasTexture = false;

	return mesh; // pReference
}


	} /* namespace utils */
} /* namespace morphablemodel */

#endif /* WITH_MORPHABLEMODEL_HDF5 */
