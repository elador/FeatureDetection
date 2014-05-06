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
		shared_ptr<FeatureExtractor> extractor, shared_ptr<BinaryClassifier> classifier, vector<double> sizes, vector<double> displacements) :
				extractor(extractor), classifier(classifier), sizes(sizes), displacements(displacements) {}

bool ClassificationBasedStateValidator::isValid(const Sample& target,
		const vector<shared_ptr<Sample>>& samples, shared_ptr<VersionedImage> image) {
	extractor->update(image);
	for (double scale : sizes) {
		int size = static_cast<int>(std::round(scale * target.getSize()));
		for (double offsetX : displacements) {
			int x = target.getX() + static_cast<int>(std::round(offsetX * size));
			for (double offsetY : displacements) {
				int y = target.getY() + static_cast<int>(std::round(offsetY * size));
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
