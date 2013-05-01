/*
 * LandmarkFileLoader.hpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#ifndef LANDMARKFILELOADER_HPP_
#define LANDMARKFILELOADER_HPP_

#include "imageio/ImageSource.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <map>
#include <memory>

using boost::filesystem::path;
using std::map;
using std::shared_ptr;

namespace imageio {

class LandmarkCollection;
class FilebasedImageSource;
class LandmarkFormatParser;

/**
 * Provides different means of loading landmark collections
 * from files.
 */
class LandmarkFileLoader {
public:

	LandmarkFileLoader();

	virtual ~LandmarkFileLoader();

	/**
	 * Loads one landmark-file per image. Looks 
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
	 * @param[in] ... ...
	 * @return todo
	 */
	static map<path, LandmarkCollection> loadOnePerImage(shared_ptr<ImageSource> imageSource, shared_ptr<LandmarkFormatParser> landmarkFormatParser);

};

} /* namespace imageio */
#endif /* LANDMARKFILELOADER_HPP_ */
