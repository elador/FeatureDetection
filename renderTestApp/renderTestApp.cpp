// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

#ifdef WIN32
	#include <crtdbg.h>
#endif

#ifdef _DEBUG
   #ifndef DBG_NEW
      #define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
      #define new DBG_NEW
   #endif
#endif  // _DEBUG

#include "render/SRenderer.hpp"
#include "render/Vertex.hpp"
#include "render/Triangle.hpp"
#include "render/Camera.hpp"

#include <iostream>
#include <fstream>

#include <boost/math/constants/constants.hpp>
#include "opencv2/core/core.hpp" // for FdImage.h
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include <boost/program_options.hpp>


namespace po = boost::program_options;
using namespace std;

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
    copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
    return os;
}

void initCamera(render::Camera camera)
{
	camera.verticalAngle = -boost::math::constants::pi<float>()/4.0f;
	camera.updateFixed(cv::Vec3f(0.0f, 3.0f, 3.0f), cv::Vec3f());
}

std::vector< std::vector<render::Triangle> > init(void)
{
	std::vector< std::vector<render::Triangle> > objectsList;

	render::Vertex cubeVertices[24];

	for (int i = 0; i < 24; i++)
		cubeVertices[i].color = cv::Vec3f(1.0f, 1.0f, 1.0f);

	cubeVertices[0].position = cv::Vec4f(-0.5f, 0.5f, 0.5f);  // set w (4th comp.) = 1.0f !!! (1 or 0?)
	cubeVertices[0].texCoord = cv::Vec2f(0.0f, 0.0f);
	cubeVertices[1].position = cv::Vec4f(-0.5f, -0.5f, 0.5f);
	cubeVertices[1].texCoord = cv::Vec2f(0.0f, 1.0f);
	cubeVertices[2].position = cv::Vec4f(0.5f, -0.5f, 0.5f);
	cubeVertices[2].texCoord = cv::Vec2f(1.0f, 1.0f);
	cubeVertices[3].position = cv::Vec4f(0.5f, 0.5f, 0.5f);
	cubeVertices[3].texCoord = cv::Vec2f(1.0f, 0.0f);

	cubeVertices[4].position = cv::Vec4f(0.5f, 0.5f, 0.5f);
	cubeVertices[4].texCoord = cv::Vec2f(0.0f, 0.0f);
	cubeVertices[5].position = cv::Vec4f(0.5f, -0.5f, 0.5f);
	cubeVertices[5].texCoord = cv::Vec2f(0.0f, 1.0f);
	cubeVertices[6].position = cv::Vec4f(0.5f, -0.5f, -0.5f);
	cubeVertices[6].texCoord = cv::Vec2f(1.0f, 1.0f);
	cubeVertices[7].position = cv::Vec4f(0.5f, 0.5f, -0.5f);
	cubeVertices[7].texCoord = cv::Vec2f(1.0f, 0.0f);

	cubeVertices[8].position = cv::Vec4f(0.5f, 0.5f, -0.5f);
	cubeVertices[8].texCoord = cv::Vec2f(0.0f, 0.0f);
	cubeVertices[9].position = cv::Vec4f(0.5f, -0.5f, -0.5f);
	cubeVertices[9].texCoord = cv::Vec2f(0.0f, 1.0f);
	cubeVertices[10].position = cv::Vec4f(-0.5f, -0.5f, -0.5f);
	cubeVertices[10].texCoord = cv::Vec2f(1.0f, 1.0f);
	cubeVertices[11].position = cv::Vec4f(-0.5f, 0.5f, -0.5f);
	cubeVertices[11].texCoord = cv::Vec2f(1.0f, 0.0f);

	cubeVertices[12].position = cv::Vec4f(-0.5f, 0.5f, -0.5f);
	cubeVertices[12].texCoord = cv::Vec2f(0.0f, 0.0f);
	cubeVertices[13].position = cv::Vec4f(-0.5f, -0.5f, -0.5f);
	cubeVertices[13].texCoord = cv::Vec2f(0.0f, 1.0f);
	cubeVertices[14].position = cv::Vec4f(-0.5f, -0.5f, 0.5f);
	cubeVertices[14].texCoord = cv::Vec2f(1.0f, 1.0f);
	cubeVertices[15].position = cv::Vec4f(-0.5f, 0.5f, 0.5f);
	cubeVertices[15].texCoord = cv::Vec2f(1.0f, 0.0f);

	cubeVertices[16].position = cv::Vec4f(-0.5f, 0.5f, -0.5f);
	cubeVertices[16].texCoord = cv::Vec2f(0.0f, 0.0f);
	cubeVertices[17].position = cv::Vec4f(-0.5f, 0.5f, 0.5f);
	cubeVertices[17].texCoord = cv::Vec2f(0.0f, 1.0f);
	cubeVertices[18].position = cv::Vec4f(0.5f, 0.5f, 0.5f);
	cubeVertices[18].texCoord = cv::Vec2f(1.0f, 1.0f);
	cubeVertices[19].position = cv::Vec4f(0.5f, 0.5f, -0.5f);
	cubeVertices[19].texCoord = cv::Vec2f(1.0f, 0.0f);

	cubeVertices[20].position = cv::Vec4f(-0.5f, -0.5f, 0.5f);
	cubeVertices[20].texCoord = cv::Vec2f(0.0f, 0.0f);
	cubeVertices[21].position = cv::Vec4f(-0.5f, -0.5f, -0.5f);
	cubeVertices[21].texCoord = cv::Vec2f(0.0f, 1.0f);
	cubeVertices[22].position = cv::Vec4f(0.5f, -0.5f, -0.5f);
	cubeVertices[22].texCoord = cv::Vec2f(1.0f, 1.0f);
	cubeVertices[23].position = cv::Vec4f(0.5f, -0.5f, 0.5f);
	cubeVertices[23].texCoord = cv::Vec2f(1.0f, 0.0f);

	std::vector<render::Triangle> cubeTriangles;
	// the efficiency of this might be improvable
	cubeTriangles.push_back(render::Triangle(cubeVertices[0], cubeVertices[1], cubeVertices[2]));
	cubeTriangles.push_back(render::Triangle(cubeVertices[0], cubeVertices[2], cubeVertices[3]));
	cubeTriangles.push_back(render::Triangle(cubeVertices[4], cubeVertices[5], cubeVertices[6]));
	cubeTriangles.push_back(render::Triangle(cubeVertices[4], cubeVertices[6], cubeVertices[7]));
	cubeTriangles.push_back(render::Triangle(cubeVertices[8], cubeVertices[9], cubeVertices[10]));
	cubeTriangles.push_back(render::Triangle(cubeVertices[8], cubeVertices[10], cubeVertices[11]));
	cubeTriangles.push_back(render::Triangle(cubeVertices[12], cubeVertices[13], cubeVertices[14]));
	cubeTriangles.push_back(render::Triangle(cubeVertices[12], cubeVertices[14], cubeVertices[15]));
	cubeTriangles.push_back(render::Triangle(cubeVertices[16], cubeVertices[17], cubeVertices[18]));
	cubeTriangles.push_back(render::Triangle(cubeVertices[16], cubeVertices[18], cubeVertices[19]));
	cubeTriangles.push_back(render::Triangle(cubeVertices[20], cubeVertices[21], cubeVertices[22]));
	cubeTriangles.push_back(render::Triangle(cubeVertices[20], cubeVertices[22], cubeVertices[23]));

	objectsList.push_back(cubeTriangles);

	render::Vertex planeVertices[4];

	planeVertices[0].position = cv::Vec4f(-0.5f, 0.0f, -0.5f); // set w (4th comp.) = 1.0f !!!
	planeVertices[1].position = cv::Vec4f(-0.5f, 0.0f, 0.5f);
	planeVertices[2].position = cv::Vec4f(0.5f, 0.0f, 0.5f);
	planeVertices[3].position = cv::Vec4f(0.5f, 0.0f, -0.5f);
/*
	planeVertices[0].position = cv::Vec4f(-1.0f, 1.0f, 0.0f);
	planeVertices[1].position = cv::Vec4f(-1.0f, -1.0f, 0.0f);
	planeVertices[2].position = cv::Vec4f(1.0f, -1.0f, 0.0f);
	planeVertices[3].position = cv::Vec4f(1.0f, 1.0f, 0.0f);
*/
	planeVertices[0].color = cv::Vec3f(1.0f, 0.0f, 0.0f);
	planeVertices[1].color = cv::Vec3f(0.0f, 1.0f, 0.0f);
	planeVertices[2].color = cv::Vec3f(0.0f, 0.0f, 1.0f);
	planeVertices[3].color = cv::Vec3f(1.0f, 1.0f, 1.0f);

	planeVertices[0].texCoord = cv::Vec2f(0.0f, 0.0f);
	planeVertices[1].texCoord = cv::Vec2f(0.0f, 4.0f);
	planeVertices[2].texCoord = cv::Vec2f(4.0f, 4.0f);
	planeVertices[3].texCoord = cv::Vec2f(4.0f, 0.0f);

	std::vector<render::Triangle> planeTriangles;

	planeTriangles.push_back(render::Triangle(planeVertices[0], planeVertices[1], planeVertices[2]));
	planeTriangles.push_back(render::Triangle(planeVertices[0], planeVertices[2], planeVertices[3]));

	objectsList.push_back(planeTriangles);

	//texture1.createFromFile(commonDataDir + "pwr.png");
	//texture2.createFromFile(commonDataDir + "rocks.png");

	return objectsList;
}

void updateCamera(render::Camera camera, int deltaTime)
{
	cv::Vec3f eye;

	float speed = 0.02f;

	eye = camera.getEye();
	eye += speed * deltaTime * camera.getForwardVector();	// 'w' key
	eye -= speed * deltaTime * camera.getForwardVector();	// 's' key
	eye -= speed * deltaTime * camera.getRightVector();	// 'a' key
	eye += speed * deltaTime * camera.getRightVector();	// 'd' key

	camera.updateFree(eye);
}


int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(3759128);
	#endif

	std::string filename; // Create vector to hold the filenames
	
	try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("input-file,i", po::value<string>(), "input image")
        ;

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).
                  options(desc).run(), vm);
        po::notify(vm);
    
        if (vm.count("help")) {
            cout << "[ffpDetectApp] Usage: options_description [options]\n";
            cout << desc;
            return 0;
        }
        if (vm.count("input-file"))
        {
            cout << "[ffpDetectApp] Using input images: " << vm["input-file"].as< vector<string> >() << "\n";
			filename = vm["input-file"].as<string>();
        }
    }
    catch(std::exception& e) {
        cout << e.what() << "\n";
        return 1;
    }
	render::Camera camera;
	render::Renderer->create();
	initCamera(camera);
	std::vector< std::vector<render::Triangle> > objectsList = init();
	
	const float& aspect = 640.0f/480.0f; // ScreenAspectRatio - this might be incorrect (comes from SDL)

	float frustumLeft = -0.25f * aspect;
	float frustumRight = 0.25f * aspect;
	float frustumBottom = -0.25f;
	float frustumTop = 0.25f;
	float frustumNear = 0.5f;
	float frustumFar = 500.0f;

	// loop start
	updateCamera(camera, 1);

	cv::Mat vt = render::Renderer->constructViewTransform(camera.getEye(), camera.getRightVector(), camera.getUpVector(), -camera.getForwardVector());
	cv::Mat pt = render::Renderer->constructProjTransform(frustumLeft, frustumRight, frustumBottom, frustumTop, frustumNear, frustumFar);
	cv::Mat viewProjTransform = vt * pt;

	std::cout << viewProjTransform;

	render::Renderer->setTrianglesBuffer(objectsList[0]);	// The cube is the first object
	//render::Renderer->setTexture(texture1);
	for (int i = 0; i < 15; i++)	// draw 15 cubes in each direction on the plane
	{
		for (int j = 0; j < 15; j++)
		{
			cv::Mat worldTransform;// = ::translate(3.0f*(i - 7), 0.5f, 3.0f*(j - 7));
			render::Renderer->setTransform(worldTransform * viewProjTransform);
			render::Renderer->draw();
		}
	}

	render::Renderer->setTrianglesBuffer(objectsList[1]);	// The plane is the second object
	render::Renderer->setTransform(/*mtx::scale(15.0f, 1.0f, 15.0f) * */viewProjTransform);
	//render::Renderer->setTexture(texture2);
	render::Renderer->draw();

	render::Renderer->end();
	// loop end - measure the time here


	return 0;
}

