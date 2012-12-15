/*!
 * \file MeshUtils.cpp
 *
 * \author Patrik Huber
 * \date December 12, 2012
 *
 * [comment here]
 */

#include "render/MeshUtils.hpp"
#include "render/Hdf5Utils.hpp"

#include <opencv2/core/core.hpp>

#include <array>
#include <iostream>

namespace render {
	namespace utils {

Mesh MeshUtils::createCube(void)
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

	cube.texture.createFromFile("data/pwr.png");
	cube.hasTexture = true;

	return cube;
}

Mesh MeshUtils::createPlane(void)
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

	plane.texture.createFromFile("data/rocks.png");
	plane.hasTexture = true;

	return plane;
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

		// triangle list
		// get the integer matrix
		ds = fg.openDataSet( "./triangle-color-indices" );
		//hsize_t dims[2];
		ds.getSpace().getSimpleExtentDims(dims, NULL);
		cv::Mat matTLC((int)dims[0], (int)dims[1], CV_32SC1);	// int
		//matTLC.resize(dims[0], dims[1]);
		ds.read( matTLC.data, H5::PredType::NATIVE_INT32);
		ds.close();
		
		if ( matTLC.cols != 3 )
			throw std::runtime_error("Reference reading failed, triangle-color-indices has not 3 indices per entry");

		mesh.tci.resize( matTLC.rows );
		for ( size_t i = 0; i < matTLC.rows; ++i ) {
			mesh.tci[i][0] = matTLC.at<int>(i, 0);
			mesh.tci[i][1] = matTLC.at<int>(i, 1);
			mesh.tci[i][2] = matTLC.at<int>(i, 2);
		}

		fg.close();
	//}

	h5Model.close();

	mesh.hasTexture = false;

	return mesh; // pReference
}

	} /* END namespace utils */
} /* END namespace render */
