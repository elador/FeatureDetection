/*
 * SingleLandmarkSink.hpp
 *
 *  Created on: 21.01.2014
 *      Author: poschmann
 */

#ifndef SIMPLELANDMARKSINK_HPP_
#define SIMPLELANDMARKSINK_HPP_

#include "imageio/OrderedLandmarkSink.hpp"
#include <fstream>
#include <memory>

namespace imageio {

class Landmark;

/**
 * Landmark sink that writes rectangular landmarks to a file, one per frame and line. One landmark of
 * each added collection will be written to the file and results in one line. There will be four values:
 * the x and y coordinate of the upper left corner, followed by the width and height. If the landmark is
 * invisible, all values will be zero. There will be whitespaces between the values.
 */
class SingleLandmarkSink : public OrderedLandmarkSink {
public:

	/**
	 * Constructs a new simple landmark sink.
	 *
	 * @param[in] precision The decimal precision of the output.
	 * @param[in] landmarkName The name of the landmark. Can be omitted if there is only one landmark per image.
	 */
	explicit SingleLandmarkSink(size_t precision = 0, const std::string& landmarkName = "");

	/**
	 * Constructs a new simple landmark sink and opens a file to write into.
	 *
	 * @param[in] filename The name of the file to write the landmark data into.
	 * @param[in] precision The decimal precision of the output.
	 * @param[in] landmarkName The name of the landmark. Can be ommitted if there is only one landmark per image.
	 */
	explicit SingleLandmarkSink(const std::string& filename, size_t precision = 0, const std::string& landmarkName = "");

	bool isOpen();

	void open(const std::string& filename);

	void close();

	void add(const LandmarkCollection& collection);

private:
	SingleLandmarkSink(SingleLandmarkSink& other);        // Copy c'tor and assignment operator private
	SingleLandmarkSink operator=(SingleLandmarkSink rhs); // since we own an ofstream object.

	/**
	 * Extracts the landmark that should be written to the file. If the landmark name is an empty string, the first
	 * landmark will be taken (there will be assumed to be only one landmark within the collection). Otherwise, the
	 * landmark with the specified name will be taken.
	 *
	 * @param[in] collection The landmark collection.
	 * @return The landmark.
	 */
	const std::shared_ptr<Landmark> getLandmark(const LandmarkCollection& collection);

	const std::string landmarkName; ///< The name of the landmark.
	std::ofstream output;           ///< The file output stream.
};

} /* namespace imageio */
#endif /* SIMPLELANDMARKSINK_HPP_ */
