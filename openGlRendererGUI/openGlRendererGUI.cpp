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

#include "render/MeshUtils.hpp"
#include "render/MatrixUtils.hpp"
#include "render/RenderDevice.hpp"
#include "render/Camera.hpp"
#include "render/OpenGlDevice.hpp"

#include "shapemodels/MorphableModel.hpp"

#include "logging/LoggerFactory.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/lexical_cast.hpp"

#ifdef WIN32
	#define NOMINMAX
	#include <windows.h>
#endif
#include "GL/GL.h"

//#include <cmath> // log2 - not on win...
#include <iostream>
#include <fstream>
#include <algorithm> // min/max

namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::loglevel;
using namespace std;
using namespace cv;
using namespace render;
using boost::lexical_cast;

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
	copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
	return os;
}

#ifdef WIN32
	template<class T>
	T log2(T n) {  
		return log(n)/log(2);
	}
#endif

void copyImgToTex(const Mat& _tex_img, GLuint* texID, double* _twr, double* _thr) {
	Mat tex_img = _tex_img;
	flip(_tex_img,tex_img,0); // reverses the order of the rows, columns or both in a matrix
	Mat tex_pow2(pow(2.0,ceil(log2(tex_img.rows))),pow(2.0,ceil(log2(tex_img.cols))),CV_8UC3);
	std::cout << tex_pow2.rows <<"x"<<tex_pow2.cols<<std::endl;
	Mat region = tex_pow2(Rect(0,0,tex_img.cols,tex_img.rows));
	if (tex_img.type() == region.type()) {
		tex_img.copyTo(region);
	} else if (tex_img.type() == CV_8UC1) {
		cvtColor(tex_img, region, CV_GRAY2BGR);
	} else {
		tex_img.convertTo(region, CV_8UC3, 255.0);
	}

	if (_twr != 0 && _thr != 0) {
		*_twr = (double)tex_img.cols/(double)tex_pow2.cols;
		*_thr = (double)tex_img.rows/(double)tex_pow2.rows;
	}
	glBindTexture( GL_TEXTURE_2D, *texID );
	glTexImage2D(GL_TEXTURE_2D, 0, 3, tex_pow2.cols, tex_pow2.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, tex_pow2.data);
}

typedef struct my_texture {
	GLuint tex_id;
	double twr,thr,aspect_w2h;
	Mat image;
	my_texture():tex_id(-1),twr(1.0),thr(1.0) {}
	bool initialized;
	void set(const Mat& ocvimg) { 
		ocvimg.copyTo(image); 
		copyImgToTex(image, &tex_id, &twr, &thr); // only works for power of 2 width/height?
		aspect_w2h = (double)ocvimg.cols/(double)ocvimg.rows;
	}
} OpenCVGLTexture;

OpenCVGLTexture MakeOpenCVGLTexture(const Mat& _tex_img) {
	OpenCVGLTexture _ocvgl;

	glGenTextures( 1, &_ocvgl.tex_id );
	glBindTexture( GL_TEXTURE_2D, _ocvgl.tex_id );
	glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

	if (!_tex_img.empty()) { //image may be dummy, just to generate pointer to texture object in GL
		copyImgToTex(_tex_img,&_ocvgl.tex_id,&_ocvgl.twr,&_ocvgl.thr);
		_ocvgl.aspect_w2h = (double)_tex_img.cols/(double)_tex_img.rows;
	}

	return _ocvgl;
}

static string controlWindowName = "Controls";
static std::array<float, 55> pcVals;
static std::array<int, 55> pcValsInt;

static shapemodels::MorphableModel mm;
static shared_ptr<Mesh> meshToDraw;

static string windowName = "RenderOutput";
static float horizontalAngle = 0.0f;
static float verticalAngle = 0.0f;
static int lastX = 0;
static int lastY = 0;
static float movingFactor = 0.5f;
static bool moving = false;

static float zNear = -0.1f;
static float zFar = -100.0f;
static bool perspective = false;
static bool freeCamera = true;

static void winOnMouse(int event, int x, int y, int, void* userdata)
{
	if (event == EVENT_MOUSEMOVE && moving == false) {
		lastX = x;
		lastY = y;
	}
	if (event == EVENT_MOUSEMOVE && moving == true) {
		if (x != lastX || y != lastY) {
			horizontalAngle += (x-lastX)*movingFactor;
			verticalAngle += (y-lastY)*movingFactor;
			lastX = x;
			lastY = y;
		}
	}
	if (event == EVENT_LBUTTONDOWN) {
		moving = true;
		lastX = x;
		lastY = y;
	}
	if (event == EVENT_LBUTTONUP) {
		moving = false;
	}

}

static void controlWinOnMouse(int test, void* userdata)
{
	// TODO: This breaks when the has less than 55 PCA comps. Make more dynamic...
	for (int i = 0; i < pcValsInt.size(); ++i) {
		pcVals[i] = 0.1f * static_cast<float>(pcValsInt[i]) - 5.0f;
	}
	Mat coefficients = Mat::zeros(mm.getShapeModel().getNumberOfPrincipalComponents(), 1, CV_32FC1);
	for (int row=0; row < pcVals.size(); ++row) {
		coefficients.at<float>(row, 0) = pcVals[row];
	}

	meshToDraw.reset();
	meshToDraw = make_shared<Mesh>(mm.drawSample(coefficients, vector<float>()));
}


float angx=55, angy=45;
float angstep=10;

std::string winname = "opengl";

// opengl callback
void on_opengl(void* param)
{
	// INIT standard
	glShadeModel(GL_SMOOTH);                        // Enable Smooth Shading
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);                   // Black Background
	glClearDepth(1.0f);                         // Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);                        // Enables Depth Testing
	glDepthFunc(GL_LEQUAL);                         // The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);          // Really Nice Perspective Calculations
	// end standard

	const GLfloat light_ambient[]  = { 0.0f, 0.0f, 0.0f, 1.0f };
	const GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
	const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	const GLfloat light_position[] = { 0.0f, 0.0f, 1.0f, 0.0f };

	const GLfloat mat_ambient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
	const GLfloat mat_diffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
	const GLfloat mat_specular[]   = { 1.0f, 1.0f, 1.0f, 1.0f };
	const GLfloat high_shininess[] = { 100.0f };
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LIGHT0);
	glEnable(GL_NORMALIZE);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial ( GL_FRONT, GL_AMBIENT_AND_DIFFUSE );

	glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);
	glEnable(GL_LIGHTING);
	// end INIT advanced

	glLoadIdentity();

	glTranslated(0.0, 0.0, -1.0);
	Mat img = imread("C:\\Users\\Patrik\\Documents\\GitHub\\image00239.png");
	cv::GlTexture2D tex;
	
	//tex.copyFrom(img);

	//GlArrays bla;
	//GlBuffer bla;


	glRotatef( angx, 1, 0, 0 );
	glRotatef( angy, 0, 1, 0 );
	glRotatef( 0, 0, 0, 1 );

	//cv::render(tex);

	static const int coords[6][4][3] = {
		{ { +1, -1, -1 }, { -1, -1, -1 }, { -1, +1, -1 }, { +1, +1, -1 } },
		{ { +1, +1, -1 }, { -1, +1, -1 }, { -1, +1, +1 }, { +1, +1, +1 } },
		{ { +1, -1, +1 }, { +1, -1, -1 }, { +1, +1, -1 }, { +1, +1, +1 } },
		{ { -1, -1, -1 }, { -1, -1, +1 }, { -1, +1, +1 }, { -1, +1, -1 } },
		{ { +1, -1, +1 }, { -1, -1, +1 }, { -1, -1, -1 }, { +1, -1, -1 } },
		{ { -1, -1, +1 }, { +1, -1, +1 }, { +1, +1, +1 }, { -1, +1, +1 } }
	};

	for (int i = 0; i < 6; ++i) {
		glColor3ub( i*20, 100+i*10, i*42 );
		glBegin(GL_QUADS);
		for (int j = 0; j < 4; ++j) {
			glVertex3d(0.2 * coords[i][j][0], 0.2 * coords[i][j][1], 0.2 * coords[i][j][2]);
		}
		glEnd();
	}
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF); // dump leaks at return
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
			cout << "[renderTestApp] Using input images: " << vm["input-file"].as<vector<string>>() << "\n";
			filename = vm["input-file"].as<string>();
		}
	}
	catch (std::exception& e) {
		cout << e.what() << endl;
		return 1;
	}
	
	Loggers->getLogger("shapemodels").addAppender(std::make_shared<logging::ConsoleAppender>(loglevel::TRACE));

	//render::MorphableModel mmHeadL4 = render::utils::MeshUtils::readFromScm("C:\\Users\\Patrik\\Cloud\\PhD\\MorphModel\\ShpVtxModelBin.scm");
	render::Mesh cube = render::utils::MeshUtils::createCube();
	render::Mesh pyramid = render::utils::MeshUtils::createPyramid();
	render::Mesh plane = render::utils::MeshUtils::createPlane();
	shared_ptr<render::Mesh> tri = render::utils::MeshUtils::createTriangle();
	
	//mm = shapemodels::MorphableModel::loadScmModel("C:\\Users\\Patrik\\Cloud\\PhD\\MorphModel\\ShpVtxModelBin.scm", "C:\\Users\\Patrik\\Documents\\GitHub\\featurePoints_SurreyScm.txt");
	mm = shapemodels::MorphableModel::loadScmModel("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\SurreyLowResGuosheng\\NON3448\\ShpVtxModelBin_NON3448.scm", "C:\\Users\\Patrik\\Documents\\GitHub\\featurePoints_SurreyScm.txt");
	//mm = shapemodels::MorphableModel::loadOldBaselH5Model("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\model2012p.h5", "featurePoints_head_newfmt.txt");
	//mm = shapemodels::MorphableModel::loadStatismoModel("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\2012.2\\head\\model2012_l4_head.h5");

	int wn = 512;
	int hn = 512;
	Mat image = Mat::zeros(hn, wn, CV_8UC1);
	OpenGlDevice ogl(wn, hn);
	OpenCVGLTexture imgTex, imgWithDrawing;

	// Create some 2D points
	map<string, Point2f> landmarkPointsAsPoint2f;
	vector<Point2f> imagePoints;
	Point2f p1(200.0f, 150.0f); landmarkPointsAsPoint2f.insert(make_pair("right.eye.corner_outer", p1)); imagePoints.emplace_back(p1);
	Point2f p2(400.0f, 150.0f); landmarkPointsAsPoint2f.insert(make_pair("left.eye.corner_outer", p2)); imagePoints.emplace_back(p2);
	Point2f p3(250.0f, 300.0f); landmarkPointsAsPoint2f.insert(make_pair("right.lips.corner", p3)); imagePoints.emplace_back(p3);
	Point2f p4(350.0f, 300.0f); landmarkPointsAsPoint2f.insert(make_pair("left.lips.corner", p4)); imagePoints.emplace_back(p4);
	Point2f p5(300.0f, 220.0f); landmarkPointsAsPoint2f.insert(make_pair("center.nose.tip", p5)); imagePoints.emplace_back(p5);

	//imgTex.set(image); //TODO: what if different size??  // only works for power of 2 width/height?
	// input img with 2d lms
	for(const auto& landmark : imagePoints) {
		circle(image, landmark, 2, Scalar(255,0,255), CV_FILLED);
	}

	// Get and create the model points
	map<string, Point3f> mmPoints;
	vector<Point3f> modelPoints;
	for (const auto& p : landmarkPointsAsPoint2f) {
		Point3f v = mm.getShapeModel().getMeanAtPoint(p.first);
		mmPoints.insert(std::make_pair(p.first, v));
		modelPoints.push_back(Point3f(v));
	}

	// do PnP, set camMatrix, set rotM = rotM.t()
	
	//Estimate the pose
	int max_d = std::max(image.rows, image.cols); // should be the focal length? (don't forget the aspect ratio!)
	Mat camMatrix = (cv::Mat_<double>(3,3) << max_d, 0,		image.cols/2.0,
		0,	 max_d, image.rows/2.0,
		0,	 0,		1.0);
	Mat rvec(3, 1, CV_64FC1);
	Mat tvec(3, 1, CV_64FC1);
	solvePnP(modelPoints, imagePoints, camMatrix, vector<float>(), rvec, tvec, false, CV_ITERATIVE); // CV_ITERATIVE (3pts) | CV_P3P (4pts) | CV_EPNP (4pts)
	//solvePnPRansac(modelPoints, imagePoints, camMatrix, distortion, rvec, tvec, false); // min 4 points

	Mat rotation_matrix(3, 3, CV_64FC1);
	Rodrigues(rvec, rotation_matrix);
	rotation_matrix.convertTo(rotation_matrix, CV_32FC1);
	Mat translation_vector = tvec;
	translation_vector.convertTo(translation_vector, CV_32FC1);

	camMatrix.convertTo(camMatrix, CV_32FC1);

	Mat cameraMatrix = camMatrix;
	Mat rodrRotVec = rvec;
	Mat rodrTransVec = tvec;


	// rotate it
	rotation_matrix = rotation_matrix.t();
	ogl.setMatrices(translation_vector, rotation_matrix);

	ogl.renderMesh(mm.getMean());

	while (true) {
		waitKey(10);
		cv::updateWindow("OpenGlRenderContext");
	}

	// the rvec that comes out of PnP: rodrigues, makes it into a Mat. That one -> .t()
	//reproject object points - check validity of found projection matrix
	
	// draw reprojections in 2d
	//circle(image, opt_p_img, 4, Scalar(0,0,255), 1);
	//rotM = rotM.t();// transpose to conform with majorness of opengl matrix

	//imgWithDrawing.set(image);

	//ogl.
	/*
	while (true) {
		cv::waitKey(30);
	}*/

	/*
	for (const auto& p : landmarkPoints) {
		cv::rectangle(img, cv::Point(cvRound(p.second->getX()-2.0f), cvRound(p.second->getY()-2.0f)), cv::Point(cvRound(p.second->getX()+2.0f), cvRound(p.second->getY()+2.0f)), cv::Scalar(255, 0, 0));
		drawFfpsText(img, make_pair(p.first, Point2f(p.second->getX(), p.second->getY())));
	}
	//vector<Point2f> projectedPoints;
	//projectPoints(modelPoints, rvec, tvec, camMatrix, vector<float>(), projectedPoints); // same result as below
	for (const auto& v : mmVertices) {
		Mat vertex(v.second);
		Mat v2 = rotation_matrix * vertex;
		Mat v3 = v2 + translation_vector;
		Mat v4 = camMatrix * v3;
		Point3f v4p(v4);
		Point2f v4p2d(v4p.x/v4p.z, v4p.y/v4p.z); // if != 0
		drawFfpsCircle(img, make_pair(v.first, v4p2d));
		drawFfpsText(img, make_pair(v.first, v4p2d));
	}
	*/




	meshToDraw = std::make_shared<Mesh>(mm.getMean());

	int screenWidth = 640;
	int screenHeight = 480;
	const float aspect = (float)screenWidth/(float)screenHeight;

	//Camera camera(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));
	Camera camera(Vec3f(0.0f, 0.0f, 0.0f), horizontalAngle*(CV_PI/180.0f), verticalAngle*(CV_PI/180.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));

	RenderDevice r(screenWidth, screenHeight, camera);

	namedWindow(windowName, WINDOW_AUTOSIZE);
	setMouseCallback(windowName, winOnMouse);

	namedWindow(controlWindowName, WINDOW_NORMAL);
	for (auto& val : pcValsInt)	{
		val = 50;
	}
	for (unsigned int i = 0; i < pcVals.size(); ++i) {
		createTrackbar("PC" + lexical_cast<string>(i) + ": " + lexical_cast<string>(pcVals[i]), controlWindowName, &pcValsInt[i], 100, controlWinOnMouse);
	}
	

	bool running = true;
	while (running) {
		int key = waitKey(30);
		if (key==-1) {
			// no key pressed
		}
		if (key == 'q') {
			running = false;
		}
		if (key == 'w') {
			r.camera.eye += 0.05f * -camera.getForwardVector();
		}
		if (key == 'a') {
			r.camera.eye -= 0.05f * camera.getRightVector();
		}
		if (key == 's') {
			r.camera.eye -= 0.05f * -camera.getForwardVector();
		}
		if (key == 'd') {
			r.camera.eye += 0.05f * camera.getRightVector();
		}
		if (key == 'r') {
			r.camera.eye += 0.05f * camera.getUpVector();
		}
		if (key == 'f') {
			r.camera.eye -= 0.05f * camera.getUpVector();
		}

		if (key == 'z') {
			r.camera.gaze += 0.05f * camera.getUpVector();
		}
		if (key == 'h') {
			r.camera.gaze -= 0.05f * camera.getUpVector();
		}
		if (key == 'g') {
			r.camera.gaze -= 0.05f * camera.getRightVector();
		}
		if (key == 'j') {
			r.camera.gaze += 0.05f * camera.getRightVector();
		}

		if (key == 'i') {
			r.camera.frustum.n += 0.1f;
		}
		if (key == 'k') {
			r.camera.frustum.n -= 0.1f;
		}
		if (key == 'o') {
			r.camera.frustum.f += 1.0f;
		}
		if (key == 'l') {
			r.camera.frustum.f -= 1.0f;
		}
		if (key == 'p') {
			perspective = !perspective;
		}
		if (key == 'c') {
			freeCamera = !freeCamera;
		}
		if (key == 'n') {
			meshToDraw.reset();
			meshToDraw = make_shared<Mesh>(mm.drawSample(1.0f));
		}
		
		r.resetBuffers();

		r.camera.horizontalAngle = horizontalAngle*(CV_PI/180.0f);
		r.camera.verticalAngle = verticalAngle*(CV_PI/180.0f);
		if(freeCamera) {
			r.camera.updateFree(r.camera.eye);
		} else {
			r.camera.updateFixed(r.camera.eye, r.camera.gaze);
		}

		r.updateViewTransform();
		r.updateProjectionTransform(perspective);

		Vec4f origin(0.0f, 0.0f, 0.0f, 1.0f);
		Vec4f xAxis(1.0f, 0.0f, 0.0f, 1.0f);
		Vec4f yAxis(0.0f, 1.0f, 0.0f, 1.0f);
		Vec4f zAxis(0.0f, 0.0f, 1.0f, 1.0f);
		Vec4f mzAxis(0.0f, 0.0f, -1.0f, 1.0f);
		r.renderLine(origin, xAxis, Scalar(0.0f, 0.0f, 255.0f));
		r.renderLine(origin, yAxis, Scalar(0.0f, 255.0f, 0.0f));
		r.renderLine(origin, zAxis, Scalar(255.0f, 0.0f, 0.0f));
		r.renderLine(origin, mzAxis, Scalar(0.0f, 255.0f, 255.0f));
		/*
		for (const auto& tIdx : cube.tvi) {
			Vertex v0 = cube.vertex[tIdx[0]];
			Vertex v1 = cube.vertex[tIdx[1]];
			Vertex v2 = cube.vertex[tIdx[2]];
			r.renderLine(v0.position, v1.position, Scalar(0.0f, 0.0f, 255.0f));
			r.renderLine(v1.position, v2.position, Scalar(0.0f, 0.0f, 255.0f));
			r.renderLine(v2.position, v0.position, Scalar(0.0f, 0.0f, 255.0f));
		}

		// Test-vertex for rasterizing:
		Vertex v0 = cube.vertex[0];
		Vertex v1 = cube.vertex[1];
		Vertex v2 = cube.vertex[2];
		r.renderLine(v0.position, v1.position, Scalar(255.0f, 0.0f, 0.0f));
		r.renderLine(v1.position, v2.position, Scalar(255.0f, 0.0f, 0.0f));
		r.renderLine(v2.position, v0.position, Scalar(255.0f, 0.0f, 0.0f));
		*/
		//r.draw(tri, tri->texture);
		
		Mat modelScaling = render::utils::MatrixUtils::createScalingMatrix(1.0f/140.0f, 1.0f/140.0f, 1.0f/140.0f);
		//Mat modelScaling = render::utils::MatrixUtils::createScalingMatrix(7.0f, 7.0f, 7.0f);
		Mat modelTrans = render::utils::MatrixUtils::createTranslationMatrix(0.0f, 0.0f, -1.5f);
		Mat rot = Mat::eye(4, 4, CV_32FC1);
		Mat modelMatrix = modelTrans * rot * modelScaling;
		r.setWorldTransform(modelMatrix);
		//r.draw(meshToDraw, nullptr);
		
		r.setWorldTransform(Mat::eye(4, 4, CV_32FC1));
		// End test
		
		//r.renderLine(Vec4f(1.5f, 0.0f, 0.5f, 1.0f), Vec4f(-1.5f, 0.0f, 0.5f, 1.0f), Scalar(0.0f, 0.0f, 255.0f));

		Mat screen = r.getImage();
		putText(screen, "(" + lexical_cast<string>(lastX) + ", " + lexical_cast<string>(lastY) + ")", Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "horA: " + lexical_cast<string>(horizontalAngle) + ", verA: " + lexical_cast<string>(verticalAngle), Point(10, 38), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "moving: " + lexical_cast<string>(moving), Point(10, 56), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "zNear: " + lexical_cast<string>(r.camera.frustum.n) + ", zFar: " + lexical_cast<string>(r.camera.frustum.f) + ", p: " + lexical_cast<string>(perspective), Point(10, 74), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "eye: " + lexical_cast<string>(r.camera.eye[0]) + ", " + lexical_cast<string>(r.camera.eye[1]) + ", " + lexical_cast<string>(r.camera.eye[2]), Point(10, 92), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		imshow(windowName, screen);

	}
	
	return 0;
}

