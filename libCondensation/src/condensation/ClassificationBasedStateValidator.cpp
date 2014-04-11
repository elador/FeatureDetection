/*
 * ClassificationBasedStateValidator.cpp
 *
 *  Created on: 10.04.2014
 *      Author: poschmann
 */

#include "condensation/ClassificationBasedStateValidator.hpp"
#include "condensation/Sample.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "classification/BinaryClassifier.hpp"

using imageprocessing::Patch;
using imageprocessing::VersionedImage;
using imageprocessing::FeatureExtractor;
using classification::BinaryClassifier;
using std::vector;
using std::shared_ptr;

namespace condensation {

ClassificationBasedStateValidator::ClassificationBasedStateValidator(
		shared_ptr<FeatureExtractor> extractor, shared_ptr<BinaryClassifier> classifier) :
				extractor(extractor), classifier(classifier) {}

bool ClassificationBasedStateValidator::isValid(const Sample& target,
		const vector<shared_ptr<Sample>>& samples, shared_ptr<VersionedImage> image) {
	// TODO replace magic numbers by parameters given to the constructor
	extractor->update(image);
	double scale = 0.87;
	for (int s = -1; s <= 1; ++s) {
		int size = static_cast<int>(std::round(target.getSize() * std::pow(scale, s)));
		double step = size / 19.;
		for (int xs = -2; xs <= 2; ++xs) {
			int x = target.getX() + static_cast<int>(std::round(xs * step));
			for (int ys = -2; ys <= 2; ++ys) {
				int y = target.getY() + static_cast<int>(std::round(ys * step));
				Sample s(x, y, size);
				shared_ptr<Patch> patch = extractor->extract(s.getX(), s.getY(), s.getWidth(), s.getHeight());
				if (patch && classifier->classify(patch->getData()))
					return true;
			}
		}
	}
	return false;
}

} /* namespace imageprocessing */
