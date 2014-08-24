/*
 * Hdf5Utils.cpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */
#ifdef WITH_MORPHABLEMODEL_HDF5

#include "morphablemodel/Hdf5Utils.hpp"

using std::string;
using cv::Mat;

namespace morphablemodel {
	namespace hdf5utils {

H5::H5File openFile(const string& filename) {
	H5::H5File file;
	try {
		file = H5::H5File(filename, H5F_ACC_RDONLY);
	}
	catch (H5::Exception& e) {
		string msg("Could not open HDF5 file: " + e.getDetailMsg());
		throw std::runtime_error(msg);
	}
	return file;
}

H5::Group openPath(H5::H5File& file, const string& path) {
	H5::Group group;

	// take the first part of the path
	size_t curpos = 1;
	size_t nextpos = path.find_first_of("/", curpos);
	H5::Group g = file.openGroup("/");

	string name = path.substr(curpos, nextpos-1);

	while (curpos != string::npos && name != "") {

		if (existsObjectWithName(g, name)) {
			g = g.openGroup(name);
		} else {
			string msg("The path " + path + " does not exist.");
			throw std::runtime_error(msg);
		}

		curpos = nextpos + 1;
		nextpos = path.find_first_of("/", curpos);
		if (nextpos != string::npos)
			name = path.substr(curpos, nextpos-curpos);
		else
			name = path.substr(curpos);
	}

	return g;
}

cv::Mat readMatrixFloat(const H5::CommonFG& fg, string name) {
	
	H5::DataSet ds = fg.openDataSet(name);
	hsize_t dims[2];
	ds.getSpace().getSimpleExtentDims(dims, NULL);
	cv::Mat matrix((int)dims[0], (int)dims[1], CV_32FC1); // r, c?
	// simply read the whole dataspace
	ds.read(matrix.data, H5::PredType::NATIVE_FLOAT);

	return matrix;
}
void readMatrixInt(const H5::CommonFG& fg, string name, cv::Mat& matrix) {
	H5::DataSet ds = fg.openDataSet(name); // ./triangle-list
	hsize_t dims[2];
	ds.getSpace().getSimpleExtentDims(dims, NULL);

	matrix.resize(dims[0], dims[1]); // Todo: Is this correct? The second argument is a double?
	ds.read(matrix.data, H5::PredType::NATIVE_INT32);
	if (matrix.cols != 3)
		throw std::runtime_error("Reference reading failed, triangle-list doesn't have 3 indices per entry.");

}

void readVector(const H5::CommonFG& fg, string name, std::vector<float>& vector) {
	H5::DataSet ds = fg.openDataSet( name );
	hsize_t dims[1];
	ds.getSpace().getSimpleExtentDims(dims, NULL);
	vector.resize(dims[0], 1);
	ds.read(vector.data(), H5::PredType::NATIVE_FLOAT);
}

string readString(const H5::CommonFG& fg, string name) {
	string outputString;
	H5::DataSet ds = fg.openDataSet(name);
	ds.read(outputString, ds.getStrType());
	return outputString;
}

bool existsObjectWithName(const H5::CommonFG& fg, const string& name) {
	for (hsize_t i = 0; i < fg.getNumObjs(); ++i) {
		string objname = fg.getObjnameByIdx(i);
		if (objname == name) {
			return true;
		}
	}
	return false;
}

render::Mesh readReference(string filename)
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
		string msg("Could not open HDF5 file: " + e.getDetailMsg());
		throw std::runtime_error(msg);
	}

	// make a MM and load both the shape and color models (maybe in another function). For now, we only load the mesh & color info.

	H5::Group fg = h5Model.openGroup("/color");

	//if ( Hdf5Utils::existsObjectWithName( fg, "reference-mesh" ) )
	//{
	//H5::Group fgRef = fg.openGroup( "representer/reference-mesh" );
	fg = fg.openGroup("representer/reference-mesh");

	// Vertex coordinates
	cv::Mat matVertex = readMatrixFloat(fg, "./vertex-coordinates");
	if (matVertex.cols != 3)
		throw std::runtime_error("Reference reading failed, vertex-coordinates have too many dimensions");
	mesh.vertex.resize(matVertex.rows);
	for (decltype(matVertex.rows) i = 0; i < matVertex.rows; ++i)
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
	for (decltype(matTL.rows) i = 0; i < matTL.rows; ++i) {
		mesh.tvi[i][0] = matTL.at<int>(i, 0);
		mesh.tvi[i][1] = matTL.at<int>(i, 1);
		mesh.tvi[i][2] = matTL.at<int>(i, 2);
	}

	// color coordinates
	cv::Mat matColor = readMatrixFloat(fg, "./vertex-colors");
	if (matColor.cols != 3)
		throw std::runtime_error("Reference reading failed, vertex-colors have too many dimensions");
	//pReference->color.resize( matColor.rows );
	for (decltype(matColor.rows) i = 0; i < matColor.rows; ++i)
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
	for (decltype(matTL.rows) i = 0; i < matTL.rows; ++i) {
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

H5::DataSet writeInt(const H5::CommonFG& fg, const string& name, int value)
{
	H5::IntType h5floatType(H5::PredType::NATIVE_INT32);
	H5::DataSet ds = fg.createDataSet(name, h5floatType, H5::DataSpace(H5S_SCALAR));
	ds.write(&value, h5floatType);
	return ds;
}

H5::DataSet writeFloat(const H5::CommonFG& fg, const string& name, float value)
{
	H5::FloatType h5floatType(H5::PredType::NATIVE_FLOAT);
	H5::DataSet ds = fg.createDataSet(name, h5floatType, H5::DataSpace(H5S_SCALAR));
	ds.write(&value, h5floatType);
	return ds;
}

H5::DataSet writeMatrixUInt(const H5::CommonFG& fg, const string& name, const Mat& matrix)
{
	// HDF5 does not like empty matrices (Todo: Check that).
	if (matrix.rows == 0 || matrix.cols == 0) {
		string errorMsg("Empty matrix provided to writeMatrixFloat.");
		throw std::runtime_error(errorMsg);
	}
	if (matrix.type() != CV_8UC1) {
		string errorMsg("Given matrix is not of type 1-channel float.");
		throw std::runtime_error(errorMsg);
	}
	if (!matrix.isContinuous()) {
		string errorMsg("Given matrix is not continuous in memory.");
		throw std::runtime_error(errorMsg);
	}

	hsize_t dims[2] = { matrix.rows, matrix.cols };
	H5::DataSet ds = fg.createDataSet(name, H5::PredType::NATIVE_UINT, H5::DataSpace(2, dims));
	ds.write(matrix.data, H5::PredType::NATIVE_UINT);
	return ds;
}

H5::DataSet writeMatrixFloat(const H5::CommonFG& fg, const string& name, const Mat& matrix)
{
	// HDF5 does not like empty matrices (Todo: Check that).
	if (matrix.rows == 0 || matrix.cols == 0) {
		string errorMsg("Empty matrix provided to writeMatrixFloat.");
		throw std::runtime_error(errorMsg);
	}
	if (matrix.type() != CV_32FC1) {
		string errorMsg("Given matrix is not of type 1-channel float.");
		throw std::runtime_error(errorMsg);
	}
	if (!matrix.isContinuous()) {
		string errorMsg("Given matrix is not continuous in memory.");
		throw std::runtime_error(errorMsg);
	}

	hsize_t dims[2] = { matrix.rows, matrix.cols };
	H5::DataSet ds = fg.createDataSet(name, H5::PredType::NATIVE_FLOAT, H5::DataSpace(2, dims));
	ds.write(matrix.data, H5::PredType::NATIVE_FLOAT);
	return ds;
}

H5::DataSet writeMatrixDouble(const H5::CommonFG& fg, const string& name, const Mat& matrix)
{
	// HDF5 does not like empty matrices (Todo: Check that).
	if (matrix.rows == 0 || matrix.cols == 0) {
		string errorMsg("Empty matrix provided to writeMatrixFloat.");
		throw std::runtime_error(errorMsg);
	}
	if (matrix.type() != CV_32FC1) {
		string errorMsg("Given matrix is not of type 1-channel float.");
		throw std::runtime_error(errorMsg);
	}
	if (!matrix.isContinuous()) {
		string errorMsg("Given matrix is not continuous in memory.");
		throw std::runtime_error(errorMsg);
	}

	hsize_t dims[2] = { matrix.rows, matrix.cols };
	H5::DataSet ds = fg.createDataSet(name, H5::PredType::NATIVE_DOUBLE, H5::DataSpace(2, dims));
	ds.write(matrix.data, H5::PredType::NATIVE_DOUBLE);
	return ds;
}

H5::DataSet writeVectorInt(const H5::CommonFG& fg, const string& name, const Mat& vector)
{
	// Check for empty?
	// Todo: Check if the dims are 1 x n or n x 1
	if (vector.cols != 1) {
		string errorMsg("Given vector doesn't have 1 column. (Did you supply a column-vector?)");
		throw std::runtime_error(errorMsg);
	}
	if (vector.type() != CV_8SC1) {
		string errorMsg("Given vector is not of type 1-channel int.");
		throw std::runtime_error(errorMsg);
	}
	if (!vector.isContinuous()) {
		string errorMsg("Given vector is not continuous in memory.");
		throw std::runtime_error(errorMsg);
	}
	hsize_t dims[1] = { vector.rows };
	H5::DataSet ds = fg.createDataSet(name, H5::PredType::NATIVE_INT, H5::DataSpace(1, dims));
	ds.write(vector.data, H5::PredType::NATIVE_INT);
	return ds;
}

H5::DataSet writeVectorFloat(const H5::CommonFG& fg, const string& name, const Mat& vector)
{
	// Check for empty?
	// Todo: Check if the dims are 1 x n or n x 1
	if (vector.cols != 1) {
		string errorMsg("Given vector doesn't have 1 column. (Did you supply a column-vector?)");
		throw std::runtime_error(errorMsg);
	}
	if (vector.type() != CV_32FC1) {
		string errorMsg("Given vector is not of type 1-channel float.");
		throw std::runtime_error(errorMsg);
	}
	if (!vector.isContinuous()) {
		string errorMsg("Given vector is not continuous in memory.");
		throw std::runtime_error(errorMsg);
	}
	hsize_t dims[1] = { vector.rows };
	H5::DataSet ds = fg.createDataSet(name, H5::PredType::NATIVE_FLOAT, H5::DataSpace(1, dims));
	ds.write(vector.data, H5::PredType::NATIVE_FLOAT);
	return ds;
}

H5::DataSet writeVectorDouble(const H5::CommonFG& fg, const string& name, const Mat& vector)
{
	// Check for empty?
	// Todo: Check if the dims are 1 x n or n x 1
	if (vector.cols != 1) {
		string errorMsg("Given vector doesn't have 1 column. (Did you supply a column-vector?)");
		throw std::runtime_error(errorMsg);
	}
	if (vector.type() != CV_64FC1) {
		string errorMsg("Given vector is not of type 1-channel double.");
		throw std::runtime_error(errorMsg);
	}
	if (!vector.isContinuous()) {
		string errorMsg("Given vector is not continuous in memory.");
		throw std::runtime_error(errorMsg);
	}
	hsize_t dims[1] = { vector.rows };
	H5::DataSet ds = fg.createDataSet(name, H5::PredType::NATIVE_DOUBLE, H5::DataSpace(1, dims));
	ds.write(vector.data, H5::PredType::NATIVE_DOUBLE);
	return ds;
}

H5::DataSet writeArrayInt(const H5::CommonFG& fg, const string& name, const std::vector<int>& array)
{
	hsize_t dims[1] = { array.size() };
	H5::DataSet ds = fg.createDataSet(name, H5::PredType::NATIVE_INT32, H5::DataSpace(1, dims));
	ds.write(&array[0], H5::PredType::NATIVE_INT32);
	return ds;
}

H5::DataSet writeString(const H5::CommonFG& fg, const string& name, const string& s)
{
	H5::StrType h5stringType(H5::PredType::C_S1, s.length() + 1); // + 1 for trailing zero
	H5::DataSet ds = fg.createDataSet(name, h5stringType, H5::DataSpace(H5S_SCALAR));
	ds.write(s, h5stringType);
	return ds;
}

void writeStringAttribute(const H5::H5Object& fg, const string& name, const string& s)
{
	H5::StrType h5stringType(H5::PredType::C_S1, s.length() + 1); // + 1 for trailing 0
	H5::Attribute att = fg.createAttribute(name, h5stringType, H5::DataSpace(H5S_SCALAR));
	att.write(h5stringType, s);
	att.close();
}

void writeIntAttribute(const H5::H5Object& fg, const string& name, int value)
{
	H5::IntType h5intType(H5::PredType::NATIVE_INT32);
	H5::Attribute att = fg.createAttribute(name, h5intType, H5::DataSpace(H5S_SCALAR));
	att.write(h5intType, &value);
	att.close();
}

	} /* namespace hdf5utils */
} /* namespace morphablemodel */

#endif /* WITH_MORPHABLEMODEL_HDF5 */
