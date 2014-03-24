/*
 * SdmLandmarkModel.hpp
 *
 *  Created on: 02.02.2014
 *      Author: Patrik Huber
 */

#pragma once

#ifndef SDMLANDMARKMODEL_HPP_
#define SDMLANDMARKMODEL_HPP_

#include "superviseddescentmodel/DescriptorExtractor.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"

extern "C" {
	#include "superviseddescentmodel/hog.h"
}

using cv::Mat;
using cv::Scalar;
using std::vector;
using std::string;
using boost::lexical_cast;

namespace superviseddescentmodel {

/**
 * A class representing a landmark model trained
 * with the supervised descent method.
 *
 * Todo: Write something about how the landmarks are represented?
 */
class SdmLandmarkModel  {

public:
	// should consider deleting this
	SdmLandmarkModel();

	/**
	* Constructs a new SdmModel.
	*
	* @param[in] a b
	*/
	SdmLandmarkModel(cv::Mat meanLandmarks, std::vector<std::string> landmarkIdentifier, std::vector<cv::Mat> regressorData, std::vector<std::shared_ptr<DescriptorExtractor>> descriptorExtractors, std::vector<std::string> descriptorTypes);

	struct HogParameter
	{
		int cellSize;
		int numBins;
	};

	int getNumLandmarks() const;

	int getNumCascadeSteps() const;

	HogParameter getHogParameters(int cascadeLevel) {
		return hogParameters[cascadeLevel];
	}

	/**
	* Returns the mean of the shape- and color model
	* as a Mesh.
	*
	* @return The mean of the model.
	*/
	// returns a copy. col-vec. ext always col, internal row
	cv::Mat getMeanShape() const;

	// returns  a header that points to the original data
	cv::Mat getRegressorData(int cascadeLevel);

	std::shared_ptr<DescriptorExtractor> getDescriptorExtractor(int cascadeLevel);
	
	std::string getDescriptorType(int cascadeLevel);

	//std::vector<cv::Point2f> getLandmarksAsPoints(cv::Mat or vector<float> alphas or empty(=mean));
	std::vector<cv::Point2f> getMeanAsPoints() const;

	cv::Point2f getLandmarkAsPoint(std::string landmarkIdentifier, cv::Mat modelInstance=cv::Mat()) const;

	void save(boost::filesystem::path filename, std::string comment="");

	/**
	* Load a SdmLandmarkModel model TODO a property tree node in a config file.
	* The function uses the file extension to determine which load
	* Throws a std::... ....
	*
	* @param[in] configTree A node of a ptree.
	* @return A morphable model.
	*/
	static SdmLandmarkModel load(boost::filesystem::path filename);

private:
	cv::Mat meanLandmarks; // 1 x numLandmarks*2. First all the x-coordinates, then all the y-coordinates.
	std::vector<std::string> landmarkIdentifier; //
	std::vector<cv::Mat> regressorData; // Holds the training data, one cv::Mat for each cascade level. Every Mat is (numFeatureDim+1) x numLandmarks*2 (for x & y)

	std::vector<HogParameter> hogParameters;
	std::vector<std::shared_ptr<DescriptorExtractor>> descriptorExtractors;
	std::vector<std::string> descriptorTypes; //

};


/*
Some notes:
- The current model ('SDM_Model_HOG_Zhenhua_11012014.txt') uses roughly 1/10 of
the training data of the original model from the paper, and has no expressions

- One problem: Running the optimization several times doesn't result in better
performance. Two possible reasons:
* In the training, what we train is the step from the mean to the groundtruth.
So we only train a big step.
- Actually, that means that it's very important to get the rigid alignment
right to get the first update-step right?
* The update-step for one landmark is dependent on the other landmarks

Test: To calculate the face-box (Zhenhua): Take all 68 LMs; Take the min/max x and y
for the face-box. (so the face-box is quite small)
*/

class SdmLandmarkModelFitting
{
public:
	SdmLandmarkModelFitting(SdmLandmarkModel model)/* : model(model)*/ {
		this->model = model;
	};

	// out: aligned modelShape
	// in: Rect, ocv with tl x, tl y, w, h (?) and calcs center
	// directly modifies modelShape
	// could move to parent-class
	// assumes mean -0.5, 0.5 and just places inside FB
	cv::Mat alignRigid(cv::Mat modelShape, cv::Rect faceBox) const {
		// we assume we get passed a col-vec. For convenience, we keep it.
		if (modelShape.cols != 1) {
			// error bzw convert
		}
		Mat xCoords = modelShape.rowRange(0, modelShape.rows / 2);
		Mat yCoords = modelShape.rowRange(modelShape.rows / 2, modelShape.rows);		
		// b) Align the model to the current face-box. (rigid, only centering of the mean). x_0
		// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
		// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
		// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
		xCoords = (xCoords + 0.5f) * faceBox.width + faceBox.x;
		yCoords = (yCoords + 0.5f) * faceBox.height + faceBox.y;

		/*
		// Old algorithm Zhenhua:
		// scale the model:
		double minX, maxX, minY, maxY;
		cv::minMaxLoc(xCoords, &minX, &maxX);
		cv::minMaxLoc(yCoords, &minY, &maxY);
		float faceboxScaleFactor = 1.25f; // 1.25f: value of Zhenhua Matlab FD. Mine: 1.35f
		float modelWidth = maxX - minX;
		float modelHeight = maxY - minY;
		// scale it:
		modelShape = modelShape * (faceBox.width / modelWidth + faceBox.height / modelHeight) / (2.0f * faceboxScaleFactor);
		// translate the model:
		Scalar meanX = cv::mean(xCoords);
		double meanXd = meanX[0];
		Scalar meanY = cv::mean(yCoords);
		double meanYd = meanY[0];
		// move it:
		xCoords += faceBox.x + faceBox.width / 2.0f - meanXd;
		yCoords += faceBox.y + faceBox.height / 1.8f - meanYd; // we use another value for y because we don't want to center the model right in the middle of the face-box
		*/
		return modelShape;
	};

	// out: optimized model-shape
	// in: GRAY img
	// in: evtl zusaetzlicher param um scale-level/iter auszuwaehlen
	// calculates shape updates (deltaShape) for one or more iter/scales and returns...
	// assume we get a col-vec.
	cv::Mat optimize(cv::Mat modelShape, cv::Mat image) {

		for (int cascadeStep = 0; cascadeStep < model.getNumCascadeSteps(); ++cascadeStep) {
			//feature_current = obtain_features(double(TestImg), New_Shape, 'HOG', hogScale);

			vector<cv::Point2f> points;
			for (int i = 0; i < model.getNumLandmarks(); ++i) { // in case of HOG, need integers?
				points.emplace_back(cv::Point2f(modelShape.at<float>(i), modelShape.at<float>(i + model.getNumLandmarks())));
			}
			Mat currentFeatures = model.getDescriptorExtractor(cascadeStep)->getDescriptors(image, points);
			currentFeatures = currentFeatures.reshape(0, currentFeatures.cols * model.getNumLandmarks()).t();

			//delta_shape = AAM.RF(1).Regressor(hogScale).A(1:end - 1, : )' * feature_current + AAM.RF(1).Regressor(hogScale).A(end,:)';
			Mat regressorData = model.getRegressorData(cascadeStep);
			//Mat deltaShape = regressorData.rowRange(0, regressorData.rows - 1).t() * currentFeatures + regressorData.row(regressorData.rows - 1).t();
			Mat deltaShape = currentFeatures * regressorData.rowRange(0, regressorData.rows - 1) + regressorData.row(regressorData.rows - 1);
			modelShape = modelShape + deltaShape.t();
			/*
			for (int i = 0; i < m.getNumLandmarks(); ++i) {
			cv::circle(landmarksImage, Point2f(modelShape.at<float>(i, 0), modelShape.at<float>(i + m.getNumLandmarks(), 0)), 6 - hogScale, Scalar(51.0f*(float)hogScale, 51.0f*(float)hogScale, 0.0f));
			}*/
		}

		return modelShape;
	};

private:
	SdmLandmarkModel model;
};


} /* namespace shapemodels */
#endif /* SDMLANDMARKMODEL_HPP_ */
