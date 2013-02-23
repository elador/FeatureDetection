/*
 * SlidingWindowDetector.cpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#include "detection/SlidingWindowDetector.hpp"
#include "classification/BinaryClassifier.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/ImagePyramid.hpp"

namespace detection {

SlidingWindowDetector::SlidingWindowDetector( shared_ptr<BinaryClassifier> classifier ) : classifier(classifier)
{

}

vector<pair<Patch, pair<bool, double>>> SlidingWindowDetector::detect( const ImagePyramid& imagePyramid ) const
{
	vector<pair<Patch, pair<bool, double>>> classifiedPatches;
	return classifiedPatches;
}

} /* namespace detection */
