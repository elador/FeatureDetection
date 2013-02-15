/*
 * RowFeatureTransformer.cpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/RowFeatureTransformer.hpp"

namespace imageprocessing {

RowFeatureTransformer::RowFeatureTransformer() {
	// TODO Auto-generated constructor stub

}

RowFeatureTransformer::~RowFeatureTransformer() {
	// TODO Auto-generated destructor stub
}

Mat RowFeatureTransformer::transform(const Mat& patch) {
	Mat filteredPatch;
	// TODO liste von image filters und so
	// TODO in-place filter chain (außer dem ersten)
	filteredPatch.reshape(0, 1);
	return filteredPatch;
}

} /* namespace imageprocessing */
