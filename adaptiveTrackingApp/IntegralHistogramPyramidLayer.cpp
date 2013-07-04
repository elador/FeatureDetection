/*
 * ImagePyramidLayer.cpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#include "IntegralHistogramPyramidLayer.hpp"

namespace imageprocessing {

IntegralHistogramPyramidLayer::IntegralHistogramPyramidLayer(int index, double scaleFactor, const vector<Mat>& scaledImage) :
		index(index), scaleFactor(scaleFactor), scaledImage(scaledImage) {}

IntegralHistogramPyramidLayer::~IntegralHistogramPyramidLayer() {}

} /* namespace imageprocessing */
