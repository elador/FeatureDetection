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
#include "render/MatrixUtils.hpp"
#include "render/MeshUtils.hpp"
#include "render/Texture.hpp"
#include "render/Mesh.hpp"

#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

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
            cout << "[renderTestApp] Usage: options_description [options]\n";
            cout << desc;
            return 0;
        }
        if (vm.count("input-file"))
        {
            cout << "[renderTestApp] Using input images: " << vm["input-file"].as< vector<string> >() << "\n";
			filename = vm["input-file"].as<string>();
        }
    }
    catch(std::exception& e) {
        cout << e.what() << "\n";
        return 1;
    }
	render::Camera camera;
	render::Renderer->create();
	camera.init();

	render::Mesh cube = render::utils::MeshUtils::createCube();
	render::Mesh plane = render::utils::MeshUtils::createPlane();

	const float& aspect = 640.0f/480.0f;

	float frustumLeft = -0.25f * aspect;
	float frustumRight = 0.25f * aspect;
	float frustumBottom = -0.25f;
	float frustumTop = 0.25f;
	float frustumNear = 0.5f;
	float frustumFar = 500.0f;

	// loop start
	bool running = true;
	while(running) {
		camera.update(1);
	
		cv::Mat vt = render::Renderer->constructViewTransform(camera.getEye(), camera.getRightVector(), camera.getUpVector(), -camera.getForwardVector());
		cv::Mat pt = render::Renderer->constructProjTransform(frustumLeft, frustumRight, frustumBottom, frustumTop, frustumNear, frustumFar);
		cv::Mat viewProjTransform = pt * vt;
/*
		render::Renderer->setMesh(&cube);	// The cube is the first object
		render::Renderer->setTexture(cube.texture);	// not necessary anymore
		for (int i = 0; i < 15; i++)	// draw 15 cubes in each direction on the plane
		{
			for (int j = 0; j < 15; j++)
			{
				cv::Mat worldTransform = render::utils::MatrixUtils::createTranslationMatrix(3.0f*(i - 7), 0.5f, 3.0f*(j - 7));
				render::Renderer->setTransform(viewProjTransform * worldTransform);
				render::Renderer->draw();
			}
		}
*/
		render::Renderer->setMesh(&plane);	// The plane is the second object
		render::Renderer->setTransform(viewProjTransform * render::utils::MatrixUtils::createScalingMatrix(15.0f, 1.0f, 15.0f));
		render::Renderer->setTexture(plane.texture);
		render::Renderer->draw();
		
		render::Mesh mmHeadL4 = render::utils::MeshUtils::readFromHdf5("H:\\projects\\Software Renderer\\statismo_l4_head.h5");
		render::Renderer->setMesh(&mmHeadL4);
		cv::Mat headWorld = render::utils::MatrixUtils::createScalingMatrix(1.0f/120.0f, 1.0f/120.0f, 1.0f/120.0f);
		render::Renderer->setTransform(viewProjTransform * headWorld);
		render::Renderer->draw();
		
		render::Renderer->end();

		cv::namedWindow("renderOutput");
		cv::imshow("renderOutput", render::Renderer->getRendererImage());

		float speed = 0.4f;
		float mouseSpeed = 0.2f;
		cv::Vec3f eye = camera.getEye();
		float deltaTime = 1.0f;

		char c = (char)cv::waitKey(0);
		if (c == 'q')
			running = false;
		else if (c == 't')
			std::cout << "a" << std::endl;
		else if (c == 'w')
			eye += speed * deltaTime * camera.getForwardVector();	// 'w' key
		else if (c == 's')
			eye -= speed * deltaTime * camera.getForwardVector();	// 's' key
		else if (c == 'a')
			eye -= speed * deltaTime * camera.getRightVector();	// 'a' key
		else if (c == 'd')
			eye += speed * deltaTime * camera.getRightVector();	// 'd' key
		else if (c == 'i')
			camera.verticalAngle += mouseSpeed;		// mouse up
		else if (c == 'k')
			camera.verticalAngle -= mouseSpeed;		// mouse down
		else if (c == 'j')
			camera.horizontalAngle += mouseSpeed;	// mouse left
		else if (c == 'l')
			camera.horizontalAngle -= mouseSpeed;	// mouse right
		std::cout << "next frame" << std::endl;

		camera.updateFree(eye);
	}
	// loop end - measure the time here

	cv::imwrite("colorBuffer.png", render::Renderer->getRendererImage());
	cv::imwrite("depthBuffer.png", render::Renderer->getRendererDepthBuffer());

	return 0;
}

// TODO: All those vec3 * mat4 cases... Do I have to add the homogeneous coordinate and make it vec4 * mat4, instead of how I'm doing it now? Difference?