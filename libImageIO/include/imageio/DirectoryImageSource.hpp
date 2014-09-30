/*
 * DirectoryImageSource.hpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#ifndef DIRECTORYIMAGESOURCE_HPP_
#define DIRECTORYIMAGESOURCE_HPP_

#include "imageio/ImageSource.hpp"
#include "opencv2/core/core.hpp"
#include <vector>

namespace imageio {

/**
 * Image source that takes the images of a directory.
 */
class DirectoryImageSource : public ImageSource {
public:

	/**
	 * Constructs a new directory image source.
	 *
	 * @param[in] directory The directory containing image files.
	 */
	explicit DirectoryImageSource(const std::string& directory);

	void reset();

	bool next();

	const cv::Mat getImage() const;

private:
	std::vector<std::string> files; ///< The image files of the given directory, ordered by name.
	int index; ///< The index of the next file.

};

} /* namespace imageio */
#endif /* DIRECTORYIMAGESOURCE_HPP_ */
