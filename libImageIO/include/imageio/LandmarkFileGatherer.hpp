/*
 * LandmarkFileGatherer.hpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#ifndef LANDMARKFILEGATHERER_HPP_
#define LANDMARKFILEGATHERER_HPP_

#include "imageio/ImageSource.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <map>
#include <memory>
#include <string>

using boost::filesystem::path;
using std::map;
using std::shared_ptr;
using std::string;

namespace imageio {

/**
 * Represents the different possible methods for gathering landmark files.
 */
enum class GatherMethod { // Case inconsistent with "loglevels"
	SEPARATE_FILES,		// ...
	ONE_FILE_PER_IMAGE_SAME_DIR,	// ...
	ONE_FILE_PER_IMAGE_DIFFERENT_DIRS // ...
	// Todo: separate SAME_DIR / DIFFERENT_DIRS, assign powers of 2 and use logical operators?
};

class LandmarkCollection;
class FilebasedImageSource;
class LandmarkFormatParser;

/**
 * Provides different means of loading landmark collections
 * from files.
 */
class LandmarkFileGatherer {
public:

	LandmarkFileGatherer();

	virtual ~LandmarkFileGatherer();

	/**
	 * Loads landmark-files
	 *
	 * Note: If these loaders get too many, we could encapsulate them into separate classes, and pass them to the constructor to LandmarkSource.
	 *       e.g. SingleFileLandmarkLoader, {Directory|MultipleFiles}LandmarkLoader, ...
	 *       We should distinguish between loading from WHERE (1 file, directory, ...) and loading WHAT FORMAT (.tlms, .did, ...), although 
	 *       that is sometimes coupled (e.g. .lst is always 1 file). But maybe that's not a problem - what scenarios do we have?
	 *         - Loading from 1 file (.lst, .xml), all lm's in there
	 *         - Loading multiple files, 1 lm file per image
	 *               * we can have multiple formats, .tlms, .did, ...
	 *
	 * Note2: Do we want to load all landmark-files at the start, or just load each one
	 *        when LandmarkSource::get(path) is called? Advantage of the latter would be
	 *        that we already know the path. Disadvantage: Maybe more complicated in the
	 *        case we load all landmarks from one image?
	 *
	 *        Add a FilebasedImageSource::getFilepaths()?
	 *
	 * TODO: To pass the imageSource here is probably ugly, because the LabeledImageSource we
	 *       create in our code already knows the ImageSource!
	 *
	 * @param[in] imageSource An ImageSource, needed for knowing the filenames to load.
	 * @param[in] fileFxtension The file extension of the landmark files to load.
	 * @param[in] gatherMethod The method with which to gather the landmark files.
	 * @return A vector of all paths to the landmark files.
	 */
	static vector<path> gather(shared_ptr<ImageSource> imageSource, string fileExtension, GatherMethod gatherMethod);

};

} /* namespace imageio */
#endif /* LANDMARKFILEGATHERER_HPP_ */
