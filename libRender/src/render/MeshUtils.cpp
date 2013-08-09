/*
 * MeshUtils.cpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */

#include "render/MeshUtils.hpp"
#include "render/Hdf5Utils.hpp"

#include "opencv2/core/core.hpp"

#include <array>
#include <iostream>
#include <fstream>

namespace render {
	namespace utils {

Mesh MeshUtils::createCube()
{
	Mesh cube;
	cube.vertex.resize(24);

	for (int i = 0; i < 24; i++)
		cube.vertex[i].color = cv::Vec3f(1.0f, 1.0f, 1.0f);

	cube.vertex[0].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[0].texcrd = cv::Vec2f(0.0f, 0.0f);
	cube.vertex[1].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[1].texcrd = cv::Vec2f(0.0f, 1.0f);
	cube.vertex[2].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[2].texcrd = cv::Vec2f(1.0f, 1.0f);
	cube.vertex[3].position = cv::Vec4f(0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[3].texcrd = cv::Vec2f(1.0f, 0.0f);

	cube.vertex[4].position = cv::Vec4f(0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[4].texcrd = cv::Vec2f(0.0f, 0.0f);
	cube.vertex[5].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[5].texcrd = cv::Vec2f(0.0f, 1.0f);
	cube.vertex[6].position = cv::Vec4f(0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[6].texcrd = cv::Vec2f(1.0f, 1.0f);
	cube.vertex[7].position = cv::Vec4f(0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[7].texcrd = cv::Vec2f(1.0f, 0.0f);

	cube.vertex[8].position = cv::Vec4f(0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[8].texcrd = cv::Vec2f(0.0f, 0.0f);
	cube.vertex[9].position = cv::Vec4f(0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[9].texcrd = cv::Vec2f(0.0f, 1.0f);
	cube.vertex[10].position = cv::Vec4f(-0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[10].texcrd = cv::Vec2f(1.0f, 1.0f);
	cube.vertex[11].position = cv::Vec4f(-0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[11].texcrd = cv::Vec2f(1.0f, 0.0f);

	cube.vertex[12].position = cv::Vec4f(-0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[12].texcrd = cv::Vec2f(0.0f, 0.0f);
	cube.vertex[13].position = cv::Vec4f(-0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[13].texcrd = cv::Vec2f(0.0f, 1.0f);
	cube.vertex[14].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[14].texcrd = cv::Vec2f(1.0f, 1.0f);
	cube.vertex[15].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[15].texcrd = cv::Vec2f(1.0f, 0.0f);

	cube.vertex[16].position = cv::Vec4f(-0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[16].texcrd = cv::Vec2f(0.0f, 0.0f);
	cube.vertex[17].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[17].texcrd = cv::Vec2f(0.0f, 1.0f);
	cube.vertex[18].position = cv::Vec4f(0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[18].texcrd = cv::Vec2f(1.0f, 1.0f);
	cube.vertex[19].position = cv::Vec4f(0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[19].texcrd = cv::Vec2f(1.0f, 0.0f);

	cube.vertex[20].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[20].texcrd = cv::Vec2f(0.0f, 0.0f);
	cube.vertex[21].position = cv::Vec4f(-0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[21].texcrd = cv::Vec2f(0.0f, 1.0f);
	cube.vertex[22].position = cv::Vec4f(0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[22].texcrd = cv::Vec2f(1.0f, 1.0f);
	cube.vertex[23].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[23].texcrd = cv::Vec2f(1.0f, 0.0f);

	// the efficiency of this might be improvable...
	std::array<int, 3> vi;
	vi[0] = 0; vi[1] = 1; vi[2] = 2;
	cube.tvi.push_back(vi);
	vi[0] = 0; vi[1] = 2; vi[2] = 3;
	cube.tvi.push_back(vi);
	vi[0] = 4; vi[1] = 5; vi[2] = 6;
	cube.tvi.push_back(vi);
	vi[0] = 4; vi[1] = 6; vi[2] = 7;
	cube.tvi.push_back(vi);
	vi[0] = 8; vi[1] = 9; vi[2] = 10;
	cube.tvi.push_back(vi);
	vi[0] = 8; vi[1] = 10; vi[2] = 11;
	cube.tvi.push_back(vi);
	vi[0] = 12; vi[1] = 13; vi[2] = 14;
	cube.tvi.push_back(vi);
	vi[0] = 12; vi[1] = 14; vi[2] = 15;
	cube.tvi.push_back(vi);
	vi[0] = 16; vi[1] = 17; vi[2] = 18;
	cube.tvi.push_back(vi);
	vi[0] = 16; vi[1] = 18; vi[2] = 19;
	cube.tvi.push_back(vi);
	vi[0] = 20; vi[1] = 21; vi[2] = 22;
	cube.tvi.push_back(vi);
	vi[0] = 20; vi[1] = 22; vi[2] = 23;
	cube.tvi.push_back(vi);
	/*cube.triangleList.push_back(render::Triangle(cube.vertex[0], cube.vertex[1], cube.vertex[2]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[0], cube.vertex[2], cube.vertex[3]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[4], cube.vertex[5], cube.vertex[6]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[4], cube.vertex[6], cube.vertex[7]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[8], cube.vertex[9], cube.vertex[10]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[8], cube.vertex[10], cube.vertex[11]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[12], cube.vertex[13], cube.vertex[14]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[12], cube.vertex[14], cube.vertex[15]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[16], cube.vertex[17], cube.vertex[18]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[16], cube.vertex[18], cube.vertex[19]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[20], cube.vertex[21], cube.vertex[22]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[20], cube.vertex[22], cube.vertex[23]));*/

	cube.texture = std::make_shared<Texture>();
	cube.texture->createFromFile("C:\\Users\\Patrik\\Cloud\\PhD\\up.png");
	cube.hasTexture = true;

	return cube;
}

Mesh MeshUtils::createPlane()
{
	Mesh plane;
	plane.vertex.resize(4);
	plane.vertex[0].position = cv::Vec4f(-0.5f, 0.0f, -0.5f, 1.0f);
	plane.vertex[1].position = cv::Vec4f(-0.5f, 0.0f, 0.5f, 1.0f);
	plane.vertex[2].position = cv::Vec4f(0.5f, 0.0f, 0.5f, 1.0f);
	plane.vertex[3].position = cv::Vec4f(0.5f, 0.0f, -0.5f, 1.0f);
	/*
	plane.vertex[0].position = cv::Vec4f(-1.0f, 1.0f, 0.0f);
	plane.vertex[1].position = cv::Vec4f(-1.0f, -1.0f, 0.0f);
	plane.vertex[2].position = cv::Vec4f(1.0f, -1.0f, 0.0f);
	plane.vertex[3].position = cv::Vec4f(1.0f, 1.0f, 0.0f);
	*/
	plane.vertex[0].color = cv::Vec3f(1.0f, 0.0f, 0.0f);
	plane.vertex[1].color = cv::Vec3f(0.0f, 1.0f, 0.0f);
	plane.vertex[2].color = cv::Vec3f(0.0f, 0.0f, 1.0f);
	plane.vertex[3].color = cv::Vec3f(1.0f, 1.0f, 1.0f);

	plane.vertex[0].texcrd = cv::Vec2f(0.0f, 0.0f);
	plane.vertex[1].texcrd = cv::Vec2f(0.0f, 4.0f);
	plane.vertex[2].texcrd = cv::Vec2f(4.0f, 4.0f);
	plane.vertex[3].texcrd = cv::Vec2f(4.0f, 0.0f);

	std::array<int, 3> vi;
	vi[0] = 0; vi[1] = 1; vi[2] = 2;
	plane.tvi.push_back(vi);
	vi[0] = 0; vi[1] = 2; vi[2] = 3;
	plane.tvi.push_back(vi);

	//plane.triangleList.push_back(render::Triangle(plane.vertex[0], plane.vertex[1], plane.vertex[2]));
	//plane.triangleList.push_back(render::Triangle(plane.vertex[0], plane.vertex[2], plane.vertex[3]));

	plane.texture = std::make_shared<Texture>();
	plane.texture->createFromFile("C:\\Users\\Patrik\\Cloud\\PhD\\rocks.png");
	plane.hasTexture = true;

	return plane;
}

Mesh MeshUtils::createPyramid()
{
	Mesh pyramid;
	pyramid.vertex.resize(4);

	pyramid.vertex[0].position = cv::Vec4f(-0.5f, 0.0f, 0.5f, 1.0f);
	pyramid.vertex[0].color = cv::Vec3f(1.0f, 0.0f, 0.0f);

	pyramid.vertex[1].position = cv::Vec4f(0.5f, 0.0f, 0.5f, 1.0f);
	pyramid.vertex[1].color = cv::Vec3f(0.0f, 1.0f, 0.0f);

	pyramid.vertex[2].position = cv::Vec4f(0.0f, 0.0f, -0.5f, 1.0f);
	pyramid.vertex[2].color = cv::Vec3f(0.0f, 0.0f, 1.0f);
	
	pyramid.vertex[3].position = cv::Vec4f(0.0f, 1.0f, 0.0f, 1.0f);
	pyramid.vertex[3].color = cv::Vec3f(0.5f, 0.5f, 0.5f);

	// the efficiency of this might be improvable...
	std::array<int, 3> vi;
	vi[0] = 0; vi[1] = 2; vi[2] = 1; // the bottom plate, draw so that visible from below
	pyramid.tvi.push_back(vi);
	vi[0] = 0; vi[1] = 1; vi[2] = 3; // front
	pyramid.tvi.push_back(vi);
	vi[0] = 2; vi[1] = 3; vi[2] = 1; // right side
	pyramid.tvi.push_back(vi);
	vi[0] = 3; vi[1] = 2; vi[2] = 0; // left side
	pyramid.tvi.push_back(vi);

	pyramid.hasTexture = false;

	return pyramid;
}

shared_ptr<Mesh> MeshUtils::createTriangle()
{
	shared_ptr<Mesh> triangle = std::make_shared<Mesh>();
	triangle->vertex.resize(3);

	triangle->vertex[0].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	triangle->vertex[0].color = cv::Vec3f(1.0f, 0.0f, 0.0f);
	triangle->vertex[0].texcrd = cv::Vec2f(0.0f, 0.0f);
	
	triangle->vertex[1].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	triangle->vertex[1].color = cv::Vec3f(0.0f, 1.0f, 0.0f);
	triangle->vertex[1].texcrd = cv::Vec2f(0.0f, 1.0f);
	
	triangle->vertex[2].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	triangle->vertex[2].color = cv::Vec3f(0.0f, 0.0f, 1.0f);
	triangle->vertex[2].texcrd = cv::Vec2f(1.0f, 1.0f);
	

	// the efficiency of this might be improvable...
	std::array<int, 3> vi;
	vi[0] = 0; vi[1] = 1; vi[2] = 2;
	triangle->tvi.push_back(vi);
	
	triangle->texture = std::make_shared<Texture>();
	triangle->texture->createFromFile("C:\\Users\\Patrik\\Cloud\\PhD\\up.png");
	triangle->hasTexture = false;

	return triangle;
}

Mesh MeshUtils::readFromHdf5(std::string filename)
{
	Mesh mesh;

	// Open the HDF5 file
	H5::H5File h5Model;
	try
	{
		h5Model = H5::H5File( filename, H5F_ACC_RDONLY );
	}
	catch ( H5::Exception& e )
	{
		std::string msg( std::string( "could not open HDF5 file \n" ) + e.getCDetailMsg() );
		throw msg.c_str();
	}

	// make a MM and load both the shape and color models (maybe in another function). For now, we only load the mesh & color info.
	 
	H5::Group fg = h5Model.openGroup( "/color" );

	//if ( Hdf5Utils::existsObjectWithName( fg, "reference-mesh" ) )
	//{
		//H5::Group fgRef = fg.openGroup( "representer/reference-mesh" );
		fg = fg.openGroup( "representer/reference-mesh" );

		// Vertex coordinates
		cv::Mat matVertex = Hdf5Utils::readMatrixFloat( fg, "./vertex-coordinates" );
		if ( matVertex.cols != 3 )
			throw std::runtime_error("Reference reading failed, vertex-coordinates have too many dimensions");
		mesh.vertex.resize(matVertex.rows);
		for ( unsigned int i = 0; i < matVertex.rows; ++i )
		{
			mesh.vertex[i].position = cv::Vec4f(matVertex.at<float>(i, 0), matVertex.at<float>(i, 1), matVertex.at<float>(i, 2), 1.0f);
		}

		// triangle list
		// get the integer matrix
		H5::DataSet ds = fg.openDataSet( "./triangle-list" );
		hsize_t dims[2];
		ds.getSpace().getSimpleExtentDims(dims, NULL);
		cv::Mat matTL((int)dims[0], (int)dims[1], CV_32SC1);	// int
		//matTL.resize(dims[0], dims[1]);
		ds.read( matTL.data, H5::PredType::NATIVE_INT32);
		ds.close();
		if ( matTL.cols != 3 )
			throw std::runtime_error("Reference reading failed, triangle-list has not 3 indices per entry");

		mesh.tvi.resize( matTL.rows );
		for ( size_t i = 0; i < matTL.rows; ++i ) {
			mesh.tvi[i][0] = matTL.at<int>(i, 0);
			mesh.tvi[i][1] = matTL.at<int>(i, 1);
			mesh.tvi[i][2] = matTL.at<int>(i, 2);
		}
		
		// color coordinates
		cv::Mat matColor = Hdf5Utils::readMatrixFloat( fg, "./vertex-colors" );
		if ( matColor.cols != 3 )
			throw std::runtime_error("Reference reading failed, vertex-colors have too many dimensions");
		//pReference->color.resize( matColor.rows );
		for ( size_t i = 0; i < matColor.rows; ++i )
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
		mesh.tci.resize( matTL.rows );
		for ( size_t i = 0; i < matTL.rows; ++i ) {
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

MorphableModel MeshUtils::readFromScm(std::string filename)
{
	MorphableModel mm;
	Mesh mesh;

	// Shape:
	unsigned int numVertices = 0;
	unsigned int numTriangles = 0;
	std::vector<unsigned int> triangles;
	unsigned int numShapePcaCoeffs = 0;
	unsigned int numShapeDims = 0;	// what's that?
	std::vector<double> pcaBasisMatrixShp;	// numShapePcaCoeffs * numShapeDims
	unsigned int numMean = 0; // what's that?
	//std::vector<double> meanVertices;
	unsigned int numEigenVals = 0;
	std::vector<double> eigenVals;
	
	// Texture:
	unsigned int numTexturePcaCoeffs = 0;
	unsigned int numTextureDims = 0;
	std::vector<double> pcaBasisMatrixTex;	// numTexturePcaCoeffs * numTextureDims
	unsigned int numMeanTex = 0; // what's that?
	//std::vector<double> meanVerticesTex; // color mean for each vertex of the mean
	unsigned int numEigenValsTex = 0;
	std::vector<double> eigenValsTex;

	if(sizeof(unsigned int) != 4) {
		std::cout << "Warning: We're reading 4 Bytes from the file but sizeof(unsigned int) != 4. Check the code/behaviour." << std::endl;
	}
	if(sizeof(double) != 8) {
		std::cout << "Warning: We're reading 8 Bytes from the file but sizeof(double) != 8. Check the code/behaviour." << std::endl;
	}

	std::ifstream modelFile;
	modelFile.open(filename, std::ios::binary);
	if (!modelFile.is_open()) {
		std::cout << "Could not open model file: " << filename << std::endl;
		exit(EXIT_FAILURE); // Todo use the logger & stuff
	}

	// make a MM and load both the shape and color models (maybe in another function). For now, we only load the mesh & color info.

	// 1 char = 1 byte. uint32=4bytes. float64=8bytes.

	//READING SHAPE MODEL
	// Read (reference?) num triangles and vertices
	modelFile.read(reinterpret_cast<char*>(&numVertices), 4);
	modelFile.read(reinterpret_cast<char*>(&numTriangles), 4);

	//Read triangles
	mesh.tvi.resize(numTriangles);
	mesh.tci.resize(numTriangles);
	unsigned int v0, v1, v2;
	for (unsigned int i=0; i < numTriangles; ++i) {
		v0 = v1 = v2 = 0;
		modelFile.read(reinterpret_cast<char*>(&v0), 4);	// would be nice to pass a &vector and do it in one
		modelFile.read(reinterpret_cast<char*>(&v1), 4);	// go, but didn't work. Maybe a cv::Mat would work?
		modelFile.read(reinterpret_cast<char*>(&v2), 4);
		mesh.tvi[i][0] = v0;
		mesh.tvi[i][1] = v1;
		mesh.tvi[i][2] = v2;	// do probably the same for tci
		
		mesh.tci[i][0] = v0;
		mesh.tci[i][1] = v1;
		mesh.tci[i][2] = v2;
		
	}
	
	//Read number of rows and columns of the shape projection matrix (pcaBasis)
	modelFile.read(reinterpret_cast<char*>(&numShapePcaCoeffs), 4);
	modelFile.read(reinterpret_cast<char*>(&numShapeDims), 4);

	//Read shape projection matrix
	mm.matPcaBasisShp = cv::Mat(numShapeDims, numShapePcaCoeffs, CV_64FC1);
	// m x n (rows x cols) = numShapeDims x numShapePcaCoeffs
	std::cout << mm.matPcaBasisShp.rows << ", " << mm.matPcaBasisShp.cols << std::endl;
	for (unsigned int col = 0; col < numShapePcaCoeffs; ++col) {
		for (unsigned int row = 0; row < numShapeDims; ++row) {
			double var = 0.0;
			modelFile.read(reinterpret_cast<char*>(&var), 8);
			mm.matPcaBasisShp.at<double>(row, col) = var;
		}
	}

	//Read mean shape vector
	modelFile.read(reinterpret_cast<char*>(&numMean), 4);
	mm.matMeanShp = cv::Mat(numMean, 1, CV_64FC1);
	unsigned int matCounter = 0;
	mesh.vertex.resize(numMean/3);
	double vd0, vd1, vd2;
	for (unsigned int i=0; i < numMean/3; ++i) {
		vd0 = vd1 = vd2 = 0.0;
		modelFile.read(reinterpret_cast<char*>(&vd0), 8);
		modelFile.read(reinterpret_cast<char*>(&vd1), 8);
		modelFile.read(reinterpret_cast<char*>(&vd2), 8);
		//meanVertices.push_back(var);
		mesh.vertex[i].position = cv::Vec4f(vd0, vd1, vd2, 1.0f);
		
		mm.matMeanShp.at<double>(matCounter, 0) = vd0;
		++matCounter;
		mm.matMeanShp.at<double>(matCounter, 0) = vd1;
		++matCounter;
		mm.matMeanShp.at<double>(matCounter, 0) = vd2;
		++matCounter;
	}

	//Read shape eigenvalues
	modelFile.read(reinterpret_cast<char*>(&numEigenVals), 4);
	mm.matEigenvalsShp = cv::Mat(numEigenVals, 1, CV_64FC1);
	for (unsigned int i=0; i < numEigenVals; ++i) {
		double var = 0.0;
		modelFile.read(reinterpret_cast<char*>(&var), 8);
		eigenVals.push_back(var);
		mm.matEigenvalsShp.at<double>(i, 0) = var;
	}

	//READING TEXTURE MODEL
	//Read number of rows and columns of projection matrix 
	modelFile.read(reinterpret_cast<char*>(&numTexturePcaCoeffs), 4);
	modelFile.read(reinterpret_cast<char*>(&numTextureDims), 4);
	//Read texture projection matrix
	mm.matPcaBasisTex = cv::Mat(numTextureDims, numTexturePcaCoeffs, CV_64FC1);
	std::cout << mm.matPcaBasisTex.rows << ", " << mm.matPcaBasisTex.cols << std::endl;
	for (unsigned int col = 0; col < numTexturePcaCoeffs; ++col) {
		for (unsigned int row = 0; row < numTextureDims; ++row) {
			double var = 0.0;
			modelFile.read(reinterpret_cast<char*>(&var), 8);
			mm.matPcaBasisTex.at<double>(row, col) = var;
		}
	}

	//Read mean texture vector
	modelFile.read(reinterpret_cast<char*>(&numMeanTex), 4);
	mm.matMeanTex = cv::Mat(numMeanTex, 1, CV_64FC1);
	matCounter = 0;
	for (unsigned int i=0; i < numMeanTex/3; ++i) {
		//double var = 0.0;
		vd0 = vd1 = vd2 = 0.0;
		modelFile.read(reinterpret_cast<char*>(&vd0), 8);
		modelFile.read(reinterpret_cast<char*>(&vd1), 8);
		modelFile.read(reinterpret_cast<char*>(&vd2), 8);
		//meanVerticesTex.push_back(var);
		mesh.vertex[i].color = cv::Vec3f(vd0, vd1, vd2);	// order in hdf5: RGB. Order in OCV: BGR. But order in vertex.color: RGB

		mm.matMeanTex.at<double>(matCounter, 0) = vd0;
		++matCounter;
		mm.matMeanTex.at<double>(matCounter, 0) = vd1;
		++matCounter;
		mm.matMeanTex.at<double>(matCounter, 0) = vd2;
		++matCounter;
	}

	//Read texture eigenvalues
	modelFile.read(reinterpret_cast<char*>(&numEigenValsTex), 4);
	mm.matEigenvalsTex = cv::Mat(numEigenValsTex, 1, CV_64FC1);
	for (unsigned int i=0; i < numEigenValsTex; ++i) {
		double var = 0.0;
		modelFile.read(reinterpret_cast<char*>(&var), 8);
		eigenValsTex.push_back(var);
		mm.matEigenvalsTex.at<double>(i, 0) = var;
	}

	modelFile.close();

	mesh.hasTexture = false;

	mm.mesh = mesh;
	return mm; // pReference
}

	} /* namespace utils */
} /* namespace render */
