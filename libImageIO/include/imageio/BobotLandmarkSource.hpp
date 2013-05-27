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
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

using cv::Rect_;
using std::string;
using std::vector;
using std::unordered_map;
using std::shared_ptr;

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
	 * @param(in] imageSource The source of the images. Is assumed to be at the same position as this landmark source.
	 * @param[in] filename The name of the file containing the landmark data in BoBoT format.
	 */
	BobotLandmarkSource(shared_ptr<ImageSource> imageSource, const string& filename);

	~BobotLandmarkSource();

	const bool next();

	const LandmarkCollection& get();

	const LandmarkCollection& get(const path& imagePath);

	const LandmarkCollection& getLandmarks() const;

private:

	static const string landmarkName;         ///< The name of the landmarks.
	shared_ptr<ImageSource> imageSource;      ///< The source of the images. Is assumed to be at the same position as this landmark source.
	vector<Rect_<float>> positions;           ///< The target positions inside each image.
	unordered_map<string, size_t> name2index; ///< Mapping between image name and position index.
	int index;                                ///< The index of the current target position.
	mutable LandmarkCollection collection;    ///< The current landmark collection.
};

} /* namespace imageio */
#endif /* BOBOTLANDMARKSOURCE_HPP_ */
