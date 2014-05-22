/*
 * LinearShapeFitting.cpp
 *
 *  Created on: 22.05.2014
 *      Author: Patrik Huber
 */

#include "morphablemodel/LinearShapeFitting.hpp"

#include "logging/LoggerFactory.hpp"

using logging::LoggerFactory;
using cv::Mat;
using std::vector;

namespace morphablemodel {

vector<float> fitShapeToLandmarksLinear(MorphableModel morphablemodel, Mat affineCameraMatrix, vector<int> landmarks, int numCoefficientsToFit/*=0*/, float detectorVariance/*=0.0f*/, float modelVariance/*=0.0f*/)
{

	return vector<float>();
}

} /* namespace morphablemodel */