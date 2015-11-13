/*
 * DilatedLabeledImageSource.hpp
 *
 *  Created on: 23.10.2015
 *      Author: poschmann
 */

#ifndef DILATEDLABELEDIMAGESOURCE_HPP_
#define DILATEDLABELEDIMAGESOURCE_HPP_

#include "imageio/LabeledImageSource.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "imageio/RectLandmark.hpp"
#include <memory>

/**
 * Labeled image source wrapper that dilates the landmarks.
 */
class DilatedLabeledImageSource : public imageio::LabeledImageSource {
public:

	DilatedLabeledImageSource(const std::shared_ptr<imageio::LabeledImageSource>& source,
			double widthScaleFactor, double heightScaleFactor) :
					imageio::LabeledImageSource(source->getSourceName()),
					wrapped(source),
					widthScaleFactor(widthScaleFactor),
					heightScaleFactor(heightScaleFactor) {}

	void reset() {
		wrapped->reset();
	}

	bool next() {
		return wrapped->next();
	}

	const cv::Mat getImage() const {
		return wrapped->getImage();
	}

	boost::filesystem::path getName() const {
		return wrapped->getName();
	}

	std::vector<boost::filesystem::path> getNames() const  {
		return wrapped->getNames();
	}

	const imageio::LandmarkCollection getLandmarks() const {
		const imageio::LandmarkCollection collection = wrapped->getLandmarks();
		imageio::LandmarkCollection scaledCollection;
		for (const std::shared_ptr<imageio::Landmark>& landmark : collection.getLandmarks())
			scaledCollection.insert(createScaled(landmark));
		return scaledCollection;
	}

private:

	std::shared_ptr<imageio::RectLandmark> createScaled(const std::shared_ptr<imageio::Landmark>& landmark) const {
		if (landmark->getType() != imageio::Landmark::LandmarkType::RECT)
			throw std::runtime_error("DilatedLabeledImageSource: cannot handle landmarks other than RectLandmark");
		const std::string& name = landmark->getName();
		float centerX = landmark->getX();
		float centerY = landmark->getY();
		float newWidth = widthScaleFactor * landmark->getWidth();
		float newHeight = heightScaleFactor * landmark->getHeight();
		return std::make_shared<imageio::RectLandmark>(name, centerX, centerY, newWidth, newHeight);
	}

	double widthScaleFactor;
	double heightScaleFactor;
	std::shared_ptr<imageio::LabeledImageSource> wrapped;
};

#endif /* DILATEDLABELEDIMAGESOURCE_HPP_ */
