#include "imageprocessing/ImagePyramid.hpp"

namespace imageprocessing {

ImagePyramid::ImagePyramid(double incrementalScaleFactor, double minScaleFactor, double maxScaleFactor) :
		incrementalScaleFactor(incrementalScaleFactor), firstLayer(0), layers() {}

ImagePyramid::~ImagePyramid() {}

const PyramidLayer* ImagePyramid::getLayer(double scaleFactor) const {
	double power = log(scaleFactor) / log(incrementalScaleFactor);
	int index = cvRound(power) - firstLayer;
	if (index < 0 || index >= (int)layers.size())
		return NULL;
	return layers[index];
}

} /* namespace imageprocessing */
