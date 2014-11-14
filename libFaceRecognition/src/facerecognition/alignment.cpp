/*
 * alignment.cpp
 *
 *  Created on: 09.11.2014
 *      Author: Patrik Huber
 */
#include "facerecognition/alignment.hpp"

#include "logging/LoggerFactory.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include "boost/lexical_cast.hpp"

#include <algorithm>
#include <array>
#include <vector>
#include <fstream>

using logging::LoggerFactory;
using cv::Mat;
using cv::Point2f;
using std::vector;

namespace facerecognition {

cv::Mat getRotationMatrixFromEyePairs(cv::Vec2f rightEye, cv::Vec2f leftEye)
{
	// Angle calc:
	cv::Vec2f reToLeLandmarksLine(leftEye - rightEye);
	float angle = std::atan2(reToLeLandmarksLine[1], reToLeLandmarksLine[0]);
	float angleDegrees = angle * (180.0 / 3.141592654);
	// IED:
	float ied = cv::norm(reToLeLandmarksLine, cv::NORM_L2);
	// Rotate it:
	cv::Vec2f centerOfRotation = (rightEye + leftEye) / 2; // between the eyes
	return cv::getRotationMatrix2D(centerOfRotation, angleDegrees, 1.0f);
}

cv::Mat cropAligned(cv::Mat image, cv::Vec2f rightEye, cv::Vec2f leftEye, float widthFactor/*=1.1f*/, float heightFactor/*=0.8f*/, float* translationX/*=nullptr*/, float* translationY/*=nullptr*/)
{
	// Crop, place eyes in "middle" horizontal, and at around 1/3 vertical
	cv::Vec2f cropCenter = (rightEye + leftEye) / 2; // between the eyes
	auto ied = cv::norm(leftEye - rightEye, cv::NORM_L2);
	cv::Rect roi(cropCenter[0] - widthFactor * ied, cropCenter[1] - heightFactor * ied, 2 * widthFactor * ied, (heightFactor + 2 * heightFactor) * ied);
	return image(roi);
}

std::vector<std::array<int, 3>> delaunayTriangulate(std::vector<cv::Point2f> points)
{
	auto minMaxX = std::minmax_element(begin(points), end(points), [](const Point2f& lhs, const Point2f& rhs) { return lhs.x < rhs.x; });
	auto minMaxY = std::minmax_element(begin(points), end(points), [](const Point2f& lhs, const Point2f& rhs) { return lhs.y < rhs.y; });
	auto width = minMaxX.second->x - minMaxX.first->x; // max - min
	auto height = minMaxY.second->y - minMaxY.first->y; // max - min
	auto offsetX = minMaxX.first->x; // subtract to each point
	auto offsetY = minMaxY.first->y; // Maybe we want to subtract floor(val) instead?
	cv::Point2f offset(offsetX, offsetY);
	cv::Rect rect(0, 0, std::ceil(width), std::ceil(height));
	cv::Subdiv2D subdiv(rect);

	Mat tmp = Mat::zeros(rect.size(), CV_8UC3);
	// Note/Todo: We might want to do the triangulation on a higher resolution? This can
	// end up being 80x75 which is pretty low when we have e.g. 68 landmarks and they are
	// close to each other. Or does the triangulation use the floating point values and it
	// doesn't matter?

	for (auto& p : points)
	{
		subdiv.insert(p - offset);
	}
	vector<cv::Vec6f> triangleListTmp;
	subdiv.getTriangleList(triangleListTmp);
	vector<Point2f> pt(3);

	vector<std::array<int, 3>> triangleList;

	for (size_t i = 0; i < triangleListTmp.size(); i++)
	{
		cv::Vec6f t = triangleListTmp[i];
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

/*	for (auto& tri : triangleList) {
		cv::line(tmp, points[tri[0]], points[tri[1]], { 255, 0, 0 });
		cv::line(tmp, points[tri[1]], points[tri[2]], { 255, 0, 0 });
		cv::line(tmp, points[tri[2]], points[tri[0]], { 255, 0, 0 });
	}*/

	return triangleList;
}

FivePointModel::FivePointModel()
{
	using cv::Point2f;
	// 5-point model:
	/*
	points.emplace_back(Point2f(561.2688f, 528.9627f)); // re_c
	points.emplace_back(Point2f(645.7872f, 528.7401f)); // le_c
	points.emplace_back(Point2f(603.2419f, 624.6601f)); // mouth_c
	points.emplace_back(Point2f(603.2317f, 577.7201f)); // nt
	points.emplace_back(Point2f(603.4039f, 529.2591f)); // botn
	points = addArtificialPoints(points);
	*/
	// 68 iBug model:
	// Read the mean from a file...
	{
		std::ifstream modelMeanFile(R"(C:\Users\Patrik\Documents\GitHub\model_mean_sdm_pasc_ibug_68pts.txt)");
		if (!modelMeanFile.is_open() || !modelMeanFile.good()) {
			throw std::runtime_error("Error opening mean file.");
		}
		std::string line;
		vector<float> lines;
		while (std::getline(modelMeanFile, line)) {
			lines.emplace_back(boost::lexical_cast<float>(line));
		}
		modelMeanFile.close();
		if (lines.size() != 68 * 2) {
			throw std::runtime_error("File with mean doesn't contain 68 LMs");
		}
		for (int i = 0; i < 68; ++i) {
			points.emplace_back(Point2f(lines[i], lines[i + 68]));
		}
	}

	triangleList = delaunayTriangulate(points);
	
	// Convert to texture coordinates, i.e. rescale the points to lie in [0, 1] x [0, 1]
	auto minMaxX = std::minmax_element(begin(points), end(points), [](const Point2f& lhs, const Point2f& rhs) { return lhs.x < rhs.x; });
	auto minMaxY = std::minmax_element(begin(points), end(points), [](const Point2f& lhs, const Point2f& rhs) { return lhs.y < rhs.y; });
	auto width = minMaxX.second->x - minMaxX.first->x; // max - min
	auto height = minMaxY.second->y - minMaxY.first->y; // max - min
	auto offsetX = minMaxX.first->x; // subtract to each point
	auto offsetY = minMaxY.first->y; // Maybe we want to subtract floor(val) instead?
	cv::Point2f offset(offsetX, offsetY);
	auto maxY = minMaxY.second->y; // minMaxY is only an iterator, so we have to store it
	for (auto& p : points) {
		p = (p - offset) * (1.0 / (maxY - offset.y)); // y is usually bigger. To keep the aspect ratio, we rescale y to [0, 1]. Then, x will be inside [0, 1] too. Note: This doesn't hold for the 68-point model? Odd?
	}
}

cv::Mat FivePointModel::extractTexture2D(cv::Mat image, std::vector<cv::Point2f> landmarkPoints)
{
	Mat textureMap = Mat::zeros(100, 100, image.type());
	for (auto& tri : triangleList) {
		Mat dstTriImg;
		vector<Point2f> srcTri, dstTri;
		srcTri.emplace_back(landmarkPoints[tri[0]]);
		srcTri.emplace_back(landmarkPoints[tri[1]]);
		srcTri.emplace_back(landmarkPoints[tri[2]]);
		dstTri.emplace_back(points[tri[0]]);
		dstTri.emplace_back(points[tri[1]]);
		dstTri.emplace_back(points[tri[2]]);

		// ROI in the source image:
		// Todo: Check if the triangle is on screen. If it's outside, we crash here.
		float src_tri_min_x = std::min(srcTri[0].x, std::min(srcTri[1].x, srcTri[2].x)); // note: might be better to round later (i.e. use the float points for getAffineTransform for a more accurate warping)
		float src_tri_max_x = std::max(srcTri[0].x, std::max(srcTri[1].x, srcTri[2].x));
		float src_tri_min_y = std::min(srcTri[0].y, std::min(srcTri[1].y, srcTri[2].y));
		float src_tri_max_y = std::max(srcTri[0].y, std::max(srcTri[1].y, srcTri[2].y));

		Mat inputImageRoi = image.rowRange(cvFloor(src_tri_min_y), cvCeil(src_tri_max_y)).colRange(cvFloor(src_tri_min_x), cvCeil(src_tri_max_x)); // We round down and up. ROI is possibly larger. But wrong pixels get thrown away later when we check if the point is inside the triangle? Correct?
		srcTri[0] -= Point2f(src_tri_min_x, src_tri_min_y);
		srcTri[1] -= Point2f(src_tri_min_x, src_tri_min_y);
		srcTri[2] -= Point2f(src_tri_min_x, src_tri_min_y); // shift all the points to correspond to the roi

		dstTri[0] = cv::Point2f(textureMap.cols*dstTri[0].x, textureMap.rows*dstTri[0].y - 1.0f);
		dstTri[1] = cv::Point2f(textureMap.cols*dstTri[1].x, textureMap.rows*dstTri[1].y - 1.0f);
		dstTri[2] = cv::Point2f(textureMap.cols*dstTri[2].x, textureMap.rows*dstTri[2].y - 1.0f);

		/// Get the Affine Transform
		Mat warp_mat = getAffineTransform(srcTri, dstTri);

		/// Apply the Affine Transform just found to the src image
		Mat tmpDstBuffer = Mat::zeros(textureMap.rows, textureMap.cols, image.type()); // I think using the source-size here is not correct. The dst might be larger. We should warp the endpoints and set to max-w/h. No, I think it would be even better to directly warp to the final textureMap size. (so that the last step is only a 1:1 copy)
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
	return finalMap;
}

std::vector<cv::Point2f> addArtificialPoints(std::vector<cv::Point2f> points)
{
	// Create some 'artificial', additional points:
	auto re_m = cv::Vec2f(points[0]);
	auto le_m = cv::Vec2f(points[1]);
	auto ied_m = cv::norm(le_m - re_m, cv::NORM_L2);
	auto eyeLine_m = le_m - re_m;
	auto rightOuter_m = re_m - eyeLine_m * 0.6f; // move 0.5 times the ied in the eye-line direction
	auto leftOuter_m = le_m + eyeLine_m * 0.6f;
	auto mouth_m = cv::Vec2f(points[2]);
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
}

bool isPointInTriangle(cv::Point2f point, cv::Point2f triV0, cv::Point2f triV1, cv::Point2f triV2)
{
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

} /* namespace facerecognition */
