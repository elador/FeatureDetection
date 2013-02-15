/*
 * ImagePyramid.hpp
 *
 *  Created on: 15.02.2013
 *      Author: huber & poschmann
 */

#pragma once

#ifndef IMAGEPYRAMID_H
#define IMAGEPYRAMID_H

#include "imageprocessing/PyramidLayer.hpp"
#include <vector>

using std::vector;

namespace imageprocessing {

/**
 * Image pyramid consisting of scaled representations of an image.
 */
class ImagePyramid {
public:

	/**
	 * Constructs a new image pyramid without layers.
	 *
	 * TODO min- und max-scale nur für sub-klassen interessant
	 */
	explicit ImagePyramid(double incrementalScaleFactor, double minScaleFactor, double maxScaleFactor);

	virtual ~ImagePyramid();

	/**
	 * Updates this image pyramid from the information of a new image.
	 * TODO sub-klasse erstellt pyramide neu aufgrund von bild oder reicht es an andere pyramide weiter und erstellt dann neu
	 *
	 * @param[in] image The image.
	 */
	virtual void update(Mat image) = 0;

	/**
	 * Determines the pyramid layer that is closest to the given scale factor.
	 *
	 * @param[in] scaleFactor The approximate scale factor of the layer.
	 * @return The pyramid layer or NULL if no layer has an appropriate scale factor.
	 */
	const PyramidLayer* getLayer(double scaleFactor) const;

	/**
	 * @return A reference to the pyramid layers.
	 */
	const vector<PyramidLayer>& getLayers() const {
		return layers;
	}

	/**
	 * @return A reference to the pyramid layers.
	 */
	vector<PyramidLayer>& getLayers() {
		return layers;
	}

protected:

	double incrementalScaleFactor; ///< The incremental scale factor between two layers of the pyramid.
	int firstLayer;                ///< The index of the first stored pyramid layer.
	vector<PyramidLayer> layers;   ///< The pyramid layers.
};

} /* namespace imageprocessing */
#endif // IMAGEPYRAMID_H
