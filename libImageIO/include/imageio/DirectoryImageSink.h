/*
 * DirectoryImageSink.h
 *
 *  Created on: 18.12.2012
 *      Author: poschmann
 */

#ifndef DIRECTORYIMAGESINK_H_
#define DIRECTORYIMAGESINK_H_

#include "imageio/ImageSink.h"

using std::string;

namespace imageio {

/**
 * Image sink that stores images into a directory.
 */
class DirectoryImageSink : public ImageSink {
public:

	/**
	 * Constructs a new directory image sink.
	 *
	 * @param[in] directory The name of the directory.
	 * @param[in] ending The file ending of the image files.
	 */
	explicit DirectoryImageSink(string directory, string ending = "png");

	~DirectoryImageSink();

	void add(const Mat& image);

private:

	string directory;   ///< The name of the directory.
	string ending;      ///< The file ending of the image files.
	unsigned int index; ///< The index of the next file.
};

} /* namespace imageio */
#endif /* DIRECTORYIMAGESINK_H_ */
