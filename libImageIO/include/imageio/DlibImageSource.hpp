/*
 * DlibImageSource.hpp
 *
 *  Created on: 31.07.2015
 *      Author: poschmann
 */

#ifndef DLIBIMAGESOURCE_HPP_
#define DLIBIMAGESOURCE_HPP_

#include "imageio/LabeledImageSource.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "boost/property_tree/ptree.hpp"

namespace imageio {

/**
 * Labeled image source that reads the file names and annotations from an XML file created with
 * the imglab-tool of dlib.
 */
class DlibImageSource : public LabeledImageSource {
public:

	/**
	 * Constructs a new dlib image source.
	 *
	 * @param[in] filename Name of the XML file.
	 */
	DlibImageSource(const std::string& filename);

	void reset();

	bool next();

	const cv::Mat getImage() const;

	boost::filesystem::path getName() const;

	std::vector<boost::filesystem::path> getNames() const;

	const LandmarkCollection getLandmarks() const;

private:

	boost::filesystem::path directory; ///< Image directory.
	boost::property_tree::ptree info; ///< Image and annotation information.
	boost::property_tree::ptree::const_assoc_iterator imagesBegin; ///< Iterator pointing to the first image entry.
	boost::property_tree::ptree::const_assoc_iterator imagesEnd; ///< Iterator pointing behind the last image entry.
	boost::property_tree::ptree::const_assoc_iterator imagesNext; ///< Iterator pointing to the next image entry.
	boost::filesystem::path name; ///< Current image name.
	cv::Mat image; ///< Current image.
	LandmarkCollection landmarks; ///< Current landmarks.
};

} /* namespace imageio */

#endif /* DLIBIMAGESOURCE_HPP_ */
