/*
 * PyramidLayer.cpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#include "PyramidLayer.hpp"

namespace imageprocessing {

PyramidLayer::PyramidLayer(double scaleFactor, const Mat& scaledImage) :
		scaleFactor(scaleFactor), scaledImage(scaledImage) {}

PyramidLayer::~PyramidLayer() {}

} /* namespace imageprocessing */
