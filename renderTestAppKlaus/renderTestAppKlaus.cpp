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

#include "render/MorphableModel.hpp"
#include "render/Renderer.hpp"
#include "render/SoftwareDevice.hpp"
#include "render/Vertex.hpp"
#include "render/Triangle.hpp"
#include "render/Camera.hpp"
#include "render/MatrixUtils.hpp"
#include "render/MeshUtils.hpp"
#include "render/Texture.hpp"
#include "render/Mesh.hpp"

#include "imageio/LandmarkCollection.hpp"
#include "imageio/ModelLandmark.hpp"
#include "imageio/DidLandmarkFormatParser.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"

#include <iostream>
#include <fstream>

namespace po = boost::program_options;
using namespace std;
using namespace cv;
using namespace imageio;


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

	//render::Mesh mmHeadL4 = render::utils::MeshUtils::readFromHdf5("D:\\model2012_l6_rms.h5");
	render::MorphableModel mmHeadL4 = render::utils::MeshUtils::readFromScm("C:\\Users\\Patrik\\Cloud\\PhD\\MorphModel\\ShpVtxModelBin.scm");

	const float& aspect = 640.0f/480.0f;
	shared_ptr<render::SoftwareDevice> swr = make_shared<render::SoftwareDevice>(640, 480);
	render::Renderer r(swr);
	r.renderDevice->getCamera().setFrustum(-1.0f*aspect, 1.0f*aspect, 1.0f, -1.0f, 0.1f, 100.0f);
	r.renderDevice->setWorldTransform(render::utils::MatrixUtils::createScalingMatrix(1.0f/100.0f, -1.0f/100.0f, 1.0f/100.0f));
	Vec4f test(0.5f, 0.5f, 0.5f, 1.0f);
	Vec2f res = r.renderDevice->renderVertex(test);

	vector<int> vertexIds;
	vertexIds.push_back(177); // left-eye-left - right.eye.corner_outer
	vertexIds.push_back(181); // left-eye-right - right.eye.corner_inner
	vertexIds.push_back(614); // right-eye-left - left.eye.corner_inner
	vertexIds.push_back(610); // right-eye-right - left.eye.corner_outer
	vertexIds.push_back(398); // mouth-left - right.lips.corner
	vertexIds.push_back(812); // mouth-right - left.lips.corner
	vertexIds.push_back(11140); // bridge of the nose - 
	vertexIds.push_back(114); // nose-tip - center.nose.tip
	vertexIds.push_back(270); // nasal septum - 
	vertexIds.push_back(3284); // left-alare - right nose ...
	vertexIds.push_back(572); // right-alare - left nose ...


	for (int angle = -45; angle <= 45; ++angle) {
		vector<shared_ptr<Landmark>> lms;

		for (const auto& vid : vertexIds) {
			r.renderDevice->setWorldTransform(render::utils::MatrixUtils::createRotationMatrixX(angle * (CV_PI/180.0f)));
			Vec2f res = r.renderDevice->renderVertex(mmHeadL4.mesh.vertex[vid].position);
			string name = DidLandmarkFormatParser::didToTlmsName(vid);
			shared_ptr<Landmark> lm = make_shared<ModelLandmark>(name, res);
			lms.push_back(lm);
		}

		Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
		for (const auto& lm : lms) {
			lm->draw(img);
		}
	}



	return 0;
}

// TODO: All those vec3 * mat4 cases... Do I have to add the homogeneous coordinate and make it vec4 * mat4, instead of how I'm doing it now? Difference?