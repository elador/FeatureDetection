/*
 * ImagePyramidLayer.cpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/ImagePyramidLayer.hpp"

namespace imageprocessing {

ImagePyramidLayer::ImagePyramidLayer(int index, double scaleFactor, const Mat& scaledImage) :
		index(index), scaleFactor(scaleFactor), scaledImage(scaledImage) {}

ImagePyramidLayer::~ImagePyramidLayer() {}

} /* namespace imageprocessing */
