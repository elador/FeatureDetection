/*
 * BobotLandmarkSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef BOBOTLANDMARKSOURCE_HPP_
#define BOBOTLANDMARKSOURCE_HPP_

#include "imageio/NamedLandmarkSource.hpp"
#include "imageio/OrderedLandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "opencv2/core/core.hpp"
#include "boost/filesystem/path.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace imageio {

class ImageSource;

/**
 * Landmark source that reads rectangular landmarks from a Bonn Benchmark on Tracking (BoBoT) file. Uses the
 * associated image source to determine the size of the images. Each image will have one associated landmark
 * whose name is "target".
 */
class BobotLandmarkSource : public NamedLandmarkSource, public OrderedLandmarkSource {
public:

	/**
	 * Constructs a new BoBoT landmark source.
	 *
	 * @param[in] imageSource The source of the images. Is assumed to be at the same position as this landmark source.
	 * @param[in] filename The name of the file containing the landmark data in BoBoT format.
	 */
	BobotLandmarkSource(shared_ptr<ImageSource> imageSource, const std::string& filename);

	void reset();

	bool next();

	LandmarkCollection get();

	LandmarkCollection get(const boost::filesystem::path& imagePath); 	// Note: Modifies the state of this LandmarkSource

	LandmarkCollection getLandmarks() const;

private:

	static const string landmarkName;         ///< The name of the landmarks.
	shared_ptr<ImageSource> imageSource;      ///< The source of the images. Is assumed to be at the same position as this landmark source.
	vector<Rect_<float>> positions;           ///< The target positions inside each image.
	std::unordered_map<string, size_t> name2index; ///< Mapping between image name and position index.
	int index;                                ///< The index of the current target position.
};

} /* namespace imageio */
#endif /* BOBOTLANDMARKSOURCE_HPP_ */
