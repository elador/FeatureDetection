/*
 * FileImageSource.hpp
 *
 *  Created on: 26.04.2013
 *      Author: Patrik Huber
 */

#ifndef FILEIMAGESOURCE_HPP_
#define FILEIMAGESOURCE_HPP_

#include "imageio/ImageSource.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include "opencv2/core/core.hpp"
#include <vector>

namespace imageio {

/**
 * Image source that takes a single image or a vector of images.
 */
class FileImageSource : public ImageSource {
public:

	/**
	 * Constructs a new file image source from a single file.
	 *
	 * @param[in] filePath The path and filename of the image.
	 */
	explicit FileImageSource(const std::string& filePath);

	/**
	 * Constructs a new file image source from a vector of files.
	 *
	 * @param[in] filePaths A vector with all the filenames.
	 */
	explicit FileImageSource(std::vector<std::string> filePaths);

	~FileImageSource();

	void reset();

	bool next();

	const cv::Mat getImage() const;

	boost::filesystem::path getName() const;

	std::vector<boost::filesystem::path> getNames() const;

private:
	std::vector<boost::filesystem::path> files; ///< The files of the given directory, ordered by name.
	int index; ///< The index of the next file.

};

} /* namespace imageio */
#endif /* FILEIMAGESOURCE_HPP_ */
