/*
 * crop-pasc-video-heads-affinealign.cpp
 *
 *  Created on: 28.10.2014
 *      Author: Patrik Huber
 *
 * Example:
 * crop-pasc-video-heads-affinealign ...
 *   
 */

#include <memory>
#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/range.hpp"
#include "boost/range/algorithm.hpp"
#include "boost/range/adaptor/filtered.hpp"

#include "facerecognition/pasc.hpp"
#include "facerecognition/utils.hpp"
#include "facerecognition/alignment.hpp"

#include "imageio/SimpleModelLandmarkFormatParser.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using boost::filesystem::path;
using cv::Mat;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::make_shared;

using namespace cv;

// Expects the given 5 points, in the following order:
// re_c, le_c, mouth_c, nt, botn
vector<Point2f> addArtificialPoints(vector<Point2f> points)
{
	// Create some 'artificial', additional points:
	auto re_m = Vec2f(points[0]);
	auto le_m = Vec2f(points[1]);
	auto ied_m = cv::norm(le_m - re_m, cv::NORM_L2);
	auto eyeLine_m = le_m - re_m;
	auto rightOuter_m = re_m - eyeLine_m * 0.6f; // move 0.5 times the ied in the eye-line direction
	auto leftOuter_m = le_m + eyeLine_m * 0.6f;
	auto mouth_m = Vec2f(points[2]);
	auto betweenEyes_m = (le_m + re_m) / 2.0f;
	auto eyesMouthLine_m = mouth_m - betweenEyes_m;
	auto belowMouth_m = mouth_m + eyesMouthLine_m;
	auto aboveRightEyebrow_m = re_m - eyesMouthLine_m;
	auto aboveLeftEyebrow_m = le_m - eyesMouthLine_m;
	auto rightOfMouth_m = mouth_m - 0.8f * eyeLine_m;
	auto leftOfMouth_m = mouth_m + 0.8f * eyeLine_m;
	points.emplace_back(Point2f(rightOuter_m));
	points.emplace_back(Point2f(leftOuter_m));
	points.emplace_back(Point2f(belowMouth_m));
	points.emplace_back(Point2f(aboveRightEyebrow_m));
	points.emplace_back(Point2f(aboveLeftEyebrow_m));
	points.emplace_back(Point2f(rightOfMouth_m));
	points.emplace_back(Point2f(leftOfMouth_m));
	return points;
};

// Returns true if inside the tri or on the border
bool isPointInTriangle(cv::Point2f point, cv::Point2f triV0, cv::Point2f triV1, cv::Point2f triV2) {
	/* See http://www.blackpawn.com/texts/pointinpoly/ */
	// Compute vectors        
	cv::Point2f v0 = triV2 - triV0;
	cv::Point2f v1 = triV1 - triV0;
	cv::Point2f v2 = point - triV0;

	// Compute dot products
	float dot00 = v0.dot(v0);
	float dot01 = v0.dot(v1);
	float dot02 = v0.dot(v2);
	float dot11 = v1.dot(v1);
	float dot12 = v1.dot(v2);

	// Compute barycentric coordinates
	float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// Check if point is in triangle
	return (u >= 0) && (v >= 0) && (u + v < 1);
}

// Draws the visible (i.e. inside the image boundaries) triangles.
void draw_subdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		if (t[0] >= img.cols || t[1] >= img.rows || t[2] >= img.cols || t[3] >= img.rows || t[4] >= img.cols || t[5] >= img.rows) {
			continue;
		}
		if (t[0] < 0 || t[1] < 0 || t[2] < 0 || t[3] < 0 || t[4] < 0 || t[5] < 0) {
			continue;
		}
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
		line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
		line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
	}
}

// Paints the Voronoi diagram of the found triangles
void paint_voronoi(Mat& img, Subdiv2D& subdiv)
{
	vector<vector<Point2f> > facets;
	vector<Point2f> centers;
	subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

	vector<Point> ifacet;
	vector<vector<Point> > ifacets(1);

	for (size_t i = 0; i < facets.size(); i++)
	{
		ifacet.resize(facets[i].size());
		for (size_t j = 0; j < facets[i].size(); j++)
			ifacet[j] = facets[i][j];

		Scalar color;
		color[0] = rand() & 255;
		color[1] = rand() & 255;
		color[2] = rand() & 255;
		fillConvexPoly(img, ifacet, color, 8, 0);

		ifacets[0] = ifacet;
		polylines(img, ifacets, true, Scalar(), 1, CV_AA, 0);
		circle(img, centers[i], 3, Scalar(), -1, CV_AA, 0);
	}
}

// Find a delauney triangulation of given points. Return the triangles.
// In: Points
// Out: Triangle list, with indices corresponding to the input points.
std::vector<std::array<int, 3>> delaunayTriangulate(std::vector<cv::Point2f> points)
{
	auto minMaxX = std::minmax_element(begin(points), end(points), [](const Point2f& lhs, const Point2f& rhs) { return lhs.x < rhs.x; });
	auto minMaxY = std::minmax_element(begin(points), end(points), [](const Point2f& lhs, const Point2f& rhs) { return lhs.y < rhs.y; });
	auto width = minMaxX.second->x - minMaxX.first->x; // max - min
	auto height = minMaxY.second->y - minMaxY.first->y; // max - min
	auto offsetX = minMaxX.first->x; // subtract to each point
	auto offsetY = minMaxY.first->y; // Maybe we want to subtract floor(val) instead?
	cv::Point2f offset(offsetX, offsetY);
	Rect rect(0, 0, std::ceil(width), std::ceil(height));
	Subdiv2D subdiv(rect);
	
	Mat tmp = Mat::zeros(rect.size(), CV_8UC3);

	for (auto& p : points)
	{
		subdiv.insert(p - offset);
	}
	vector<Vec6f> triangleListTmp;
	subdiv.getTriangleList(triangleListTmp);
	vector<Point2f> pt(3);

	vector<std::array<int, 3>> triangleList;

	for (size_t i = 0; i < triangleListTmp.size(); i++)
	{
		Vec6f t = triangleListTmp[i];
		if (t[0] >= tmp.cols || t[1] >= tmp.rows || t[2] >= tmp.cols || t[3] >= tmp.rows || t[4] >= tmp.cols || t[5] >= tmp.rows) {
			continue;
		}
		if (t[0] < 0 || t[1] < 0 || t[2] < 0 || t[3] < 0 || t[4] < 0 || t[5] < 0) {
			continue;
		}
		pt[0] = Point2f(t[0], t[1]);
		pt[1] = Point2f(t[2], t[3]);
		pt[2] = Point2f(t[4], t[5]);
		// Find these points in 'points':
		auto ret = std::find(begin(points), end(points), pt[0] + offset);
		if (ret == end(points)) {
			throw std::runtime_error("Shouldn't happen, floating point equality comparison failed?");
		}
		auto idx0 = std::distance(begin(points), ret);
		ret = std::find(begin(points), end(points), pt[1] + offset);
		if (ret == end(points)) {
			throw std::runtime_error("Shouldn't happen, floating point equality comparison failed?");
		}
		auto idx1 = std::distance(begin(points), ret);
		ret = std::find(begin(points), end(points), pt[2] + offset);
		if (ret == end(points)) {
			throw std::runtime_error("Shouldn't happen, floating point equality comparison failed?");
		}
		auto idx2 = std::distance(begin(points), ret);
		triangleList.emplace_back(std::array<int, 3>({ idx0, idx1, idx2 }));
	}
	return triangleList;
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path sigsetFile, metadataFile, landmarksDirectory, inputDirectory;
	path outputFolder;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
//			("sigset,s", po::value<path>(&sigsetFile)->required(),
//				"PaSC video XML sigset of the frames to be cropped")
			("input,i", po::value<path>(&inputDirectory)->required(),
				"directory containing the frames. Files should be in the format 'videoname.012.png'")
			("metadata,m", po::value<path>(&metadataFile)->required(),
				"PaSC video detections metadata in boost::serialization format")
			("landmarks,l", po::value<path>(&landmarksDirectory)->required(),
				"Directory containing landmarks to perform the piecewise affine alignment with")
			("output,o", po::value<path>(&outputFolder)->default_value("."),
				"path to save the cropped patches to")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: crop-pasc-video-heads-affinealign [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);

	}
	catch (po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	LogLevel logLevel;
	if(boost::iequals(verboseLevelConsole, "PANIC")) logLevel = LogLevel::Panic;
	else if(boost::iequals(verboseLevelConsole, "ERROR")) logLevel = LogLevel::Error;
	else if(boost::iequals(verboseLevelConsole, "WARN")) logLevel = LogLevel::Warn;
	else if(boost::iequals(verboseLevelConsole, "INFO")) logLevel = LogLevel::Info;
	else if(boost::iequals(verboseLevelConsole, "DEBUG")) logLevel = LogLevel::Debug;
	else if(boost::iequals(verboseLevelConsole, "TRACE")) logLevel = LogLevel::Trace;
	else {
		cout << "Error: Invalid LogLevel." << endl;
		return EXIT_FAILURE;
	}
	
	Loggers->getLogger("facerecognition").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("app").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	if (!fs::exists(inputDirectory) || fs::is_regular_file(inputDirectory)) {
		return EXIT_FAILURE;
	}
	
	// Create the output directory if it doesn't exist yet
	if (!boost::filesystem::exists(outputFolder)) {
		boost::filesystem::create_directory(outputFolder);
	}

	// Read the video detections metadata (eyes, face-coords):
	vector<facerecognition::PascVideoDetection> pascVideoDetections;
	{
		std::ifstream ifs(metadataFile.string());
		boost::archive::text_iarchive ia(ifs);
		ia >> pascVideoDetections;
	}

	//auto sigset = facerecognition::utils::readPascSigset(sigsetFile, true);
	vector<path> files;
	try	{
		fs::directory_iterator end_itr;
		for (boost::filesystem::directory_iterator i(inputDirectory); i != end_itr; ++i)
		{
			if (fs::is_regular_file(i->status()) && i->path().extension() == ".png")
				files.emplace_back(i->path());
		}
	}
	catch (const fs::filesystem_error& ex)
	{
		cout << ex.what() << endl;
	}

	for (auto& file : files) {
		appLogger.info("Processing " + file.string());

		string frameNumExtension = file.stem().extension().string(); // from videoname.123.png to ".123"
		frameNumExtension.erase(std::remove(frameNumExtension.begin(), frameNumExtension.end(), '.'), frameNumExtension.end());
		int frameNum = boost::lexical_cast<int>(frameNumExtension);
		path videoName = file.stem().stem(); // from videoname.123.png to "videoname"
		videoName.replace_extension(".mp4");
		string frameName = facerecognition::getPascFrameName(videoName, frameNum);
		
		// Load our own landmark detections:
		auto landmarksFile = landmarksDirectory / file.stem().stem() / file.filename();
		landmarksFile.replace_extension(".txt");
		imageio::SimpleModelLandmarkFormatParser lmParser;
		// Todo: This can crash, if we don't have PP eyes, we won't have our own LMs!
		auto lms = lmParser.read(landmarksFile).at(file.stem().string());

		Mat frame = cv::imread(file.string());

		// Delaunay triangulation:
		// Todo: Calculate the triangulation only once per app-start, or even 'offline'
		// Mean, reference:
		vector<Point2f> referencePoints;
		referencePoints.emplace_back(Point2f(561.2688f, 528.9627f)); // re_c
		referencePoints.emplace_back(Point2f(645.7872f, 528.7401f)); // le_c
		referencePoints.emplace_back(Point2f(603.2419f, 624.6601f)); // mouth_c
		referencePoints.emplace_back(Point2f(603.2317f, 577.7201f)); // nt
		referencePoints.emplace_back(Point2f(603.4039f, 529.2591f)); // botn
		referencePoints = addArtificialPoints(referencePoints);
		for (auto& p : referencePoints) {
			//cv::circle(frame, p, 4, { 255, 0, 0 });
		}
		auto triangleList = delaunayTriangulate(referencePoints);

		// Convert to texture coordinates, i.e. rescale the points to lie in [0, 1] x [0, 1]
		auto minMaxX = std::minmax_element(begin(referencePoints), end(referencePoints), [](const Point2f& lhs, const Point2f& rhs) { return lhs.x < rhs.x; });
		auto minMaxY = std::minmax_element(begin(referencePoints), end(referencePoints), [](const Point2f& lhs, const Point2f& rhs) { return lhs.y < rhs.y; });
		auto width = minMaxX.second->x - minMaxX.first->x; // max - min
		auto height = minMaxY.second->y - minMaxY.first->y; // max - min
		auto offsetX = minMaxX.first->x; // subtract to each point
		auto offsetY = minMaxY.first->y; // Maybe we want to subtract floor(val) instead?
		cv::Point2f offset(offsetX, offsetY);
		auto maxY = minMaxY.second->y; // minMaxY is only an iterator, so we have to store it
		for (auto& p : referencePoints) {
			p = (p - offset) * (1.0 / (maxY-offset.y)); // y is usually bigger. To keep the aspect ratio, we rescale y to [0, 1]. Then, x will be [0, 1] too.
		}

		// Now the same for the detected landmarks:
		vector<Point2f> landmarkPoints;
		for (auto& l : lms.getLandmarks()) {
			landmarkPoints.emplace_back(l->getPoint2D()); // order: re_c, le_c, mouth_c, nt, botn
		}
		landmarkPoints = addArtificialPoints(landmarkPoints);
		for (auto& p : landmarkPoints) {
			//cv::circle(frame, p, 4, { 255, 0, 0 });
		}
		/*
		for (auto& tri : triangleList) {
			cv::line(frame, referencePoints[tri[0]], referencePoints[tri[1]], { 255, 0, 0 });
			cv::line(frame, referencePoints[tri[1]], referencePoints[tri[2]], { 255, 0, 0 });
			cv::line(frame, referencePoints[tri[2]], referencePoints[tri[0]], { 255, 0, 0 });

			cv::line(frame, landmarkPoints[tri[0]], landmarkPoints[tri[1]], { 0, 255, 0 });
			cv::line(frame, landmarkPoints[tri[1]], landmarkPoints[tri[2]], { 0, 255, 0 });
			cv::line(frame, landmarkPoints[tri[2]], landmarkPoints[tri[0]], { 0, 255, 0 });
		}
		*/
		// Now warp the detected landmarks triangles to the reference (mean), using a triangle-wise (piecewise) affine mapping:
		Mat textureMap = Mat::zeros(100, 100, frame.type());
		for (auto& tri : triangleList) {
			Mat dstTriImg;
			vector<Point2f> srcTri, dstTri;
			srcTri.emplace_back(landmarkPoints[tri[0]]);
			srcTri.emplace_back(landmarkPoints[tri[1]]);
			srcTri.emplace_back(landmarkPoints[tri[2]]);
			dstTri.emplace_back(referencePoints[tri[0]]);
			dstTri.emplace_back(referencePoints[tri[1]]);
			dstTri.emplace_back(referencePoints[tri[2]]);

			// ROI in the source image:
			// Todo: Check if the triangle is on screen. If it's outside, we crash here.
			float src_tri_min_x = std::min(srcTri[0].x, std::min(srcTri[1].x, srcTri[2].x)); // note: might be better to round later (i.e. use the float points for getAffineTransform for a more accurate warping)
			float src_tri_max_x = std::max(srcTri[0].x, std::max(srcTri[1].x, srcTri[2].x));
			float src_tri_min_y = std::min(srcTri[0].y, std::min(srcTri[1].y, srcTri[2].y));
			float src_tri_max_y = std::max(srcTri[0].y, std::max(srcTri[1].y, srcTri[2].y));

			Mat inputImageRoi = frame.rowRange(cvFloor(src_tri_min_y), cvCeil(src_tri_max_y)).colRange(cvFloor(src_tri_min_x), cvCeil(src_tri_max_x)); // We round down and up. ROI is possibly larger. But wrong pixels get thrown away later when we check if the point is inside the triangle? Correct?
			srcTri[0] -= Point2f(src_tri_min_x, src_tri_min_y);
			srcTri[1] -= Point2f(src_tri_min_x, src_tri_min_y);
			srcTri[2] -= Point2f(src_tri_min_x, src_tri_min_y); // shift all the points to correspond to the roi

			dstTri[0] = cv::Point2f(textureMap.cols*dstTri[0].x, textureMap.rows*dstTri[0].y - 1.0f);
			dstTri[1] = cv::Point2f(textureMap.cols*dstTri[1].x, textureMap.rows*dstTri[1].y - 1.0f);
			dstTri[2] = cv::Point2f(textureMap.cols*dstTri[2].x, textureMap.rows*dstTri[2].y - 1.0f);

			/// Get the Affine Transform
			Mat warp_mat = getAffineTransform(srcTri, dstTri);

			/// Apply the Affine Transform just found to the src image
			Mat tmpDstBuffer = Mat::zeros(textureMap.rows, textureMap.cols, frame.type()); // I think using the source-size here is not correct. The dst might be larger. We should warp the endpoints and set to max-w/h. No, I think it would be even better to directly warp to the final textureMap size. (so that the last step is only a 1:1 copy)
			warpAffine(inputImageRoi, tmpDstBuffer, warp_mat, tmpDstBuffer.size(), cv::INTER_LANCZOS4, cv::BORDER_TRANSPARENT); // last row/col is zeros, depends on interpolation method. Maybe because of rounding or interpolation? So it cuts a little. Maybe try to implement by myself?

			// only copy to final img if point is inside the triangle (or on the border)
			for (int x = std::min(dstTri[0].x, std::min(dstTri[1].x, dstTri[2].x)); x < std::max(dstTri[0].x, std::max(dstTri[1].x, dstTri[2].x)); ++x) {
				for (int y = std::min(dstTri[0].y, std::min(dstTri[1].y, dstTri[2].y)); y < std::max(dstTri[0].y, std::max(dstTri[1].y, dstTri[2].y)); ++y) {
					if (isPointInTriangle(cv::Point2f(x, y), dstTri[0], dstTri[1], dstTri[2])) {
						textureMap.at<cv::Vec3b>(y, x) = tmpDstBuffer.at<cv::Vec3b>(y, x);
					}
				}
			}
		}
		Mat finalMap = textureMap.colRange(0, 66);
			
		//string logMessage("Throwing away patch because it goes outside the image bounds. This shouldn't happen, or rather, we do not want it to happen, because we didn't select a frame where this should happen.");
		//string logMessage("Hmm no PP eyes. How are we going to do the affine alignment? Skipping the image for now.");

		// Normalise:
		//croppedFace = equaliseIntensity(croppedFace);
		
		/*
		Mat mask(0, 0, IPL_DEPTH_8U, 1);
		Mat gr;
		Mat gf; // out
		cv::cvtColor(croppedFace, gr, cv::COLOR_BGR2GRAY);
		// gamma stuff:
		Mat temp;
		cvConvertScale(gr, temp, 1.0 / 255, 0);
		cvPow(temp, temp, 0.2);
		cvConvertScale(temp, gf, 255, 0);

		cvSmooth(gf, b1, CV_GAUSSIAN, 1);
		cvSmooth(gf, b2, CV_GAUSSIAN, 23);
		cvSub(b1, b2, b2, mask);
		cvConvertScale(b2, gr, 127, 127);
		cvEqualizeHist(gr, gr); // ==> use CvNormalize
		//cvThreshold(gr,tr,255,0,CV_THRESH_TRUNC);
		*/

		path croppedImageFilename = outputFolder / file.filename();
		cv::imwrite(croppedImageFilename.string(), frame);
	}
	appLogger.info("Finished cropping all files in the directory.");

	return EXIT_SUCCESS;
}
