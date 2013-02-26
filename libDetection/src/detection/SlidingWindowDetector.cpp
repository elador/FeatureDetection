/*
 * SlidingWindowDetector.cpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#include "detection/SlidingWindowDetector.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/PyramidPatchExtractor.hpp"
#include "imageprocessing/FilteringFeatureTransformer.hpp"
#include "classification/ProbabilisticClassifier.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using imageprocessing::PyramidPatchExtractor;

namespace detection {

SlidingWindowDetector::SlidingWindowDetector( shared_ptr<ProbabilisticClassifier> classifier, shared_ptr<FilteringFeatureTransformer> patchTransformer, int stepSizeX, int stepSizeY ) :
		classifier(classifier), patchTransformer(patchTransformer), stepSizeX(stepSizeX), stepSizeY(stepSizeY)
{

}

vector<pair<shared_ptr<Patch>, pair<bool, double>>> SlidingWindowDetector::detect( shared_ptr<ImagePyramid> imagePyramid ) const
{
	vector<pair<shared_ptr<Patch>, pair<bool, double>>> classifiedPatches;
	// For now, this just loops over all pyramid layers given. In the long run, we have to think about pyramid sharing.
	PyramidPatchExtractor ppe(imagePyramid, 20, 20);	// Could only create once in constructor
	vector<shared_ptr<Patch>> pyramidPatches = ppe.extract(stepSizeX, stepSizeY);

	// TODO: Try with a FeatureExtractor: (combines PatchExtractor & FeatureTransformer). But atm missing extract(...) all patches.
	//shared_ptr<FeatureExtractor> featEx = make_shared<FeatureExtractor>(ppe, fft);
	// Or maybe don't use a FeatureExtractor because the PatchExtractor is always a ppe

	for(unsigned int i=0; i<pyramidPatches.size(); ++i) {
		patchTransformer->transform(pyramidPatches[i]->getData());
		pair<bool, double> res = classifier->classify(pyramidPatches[i]->getData());
		if(res.first==true)
			classifiedPatches.push_back(make_pair(pyramidPatches[i], res));
	}

	

	return classifiedPatches;
}

} /* namespace detection */
