/*	
 * LandmarkMapper.hpp
 *
 *  Created on: 19.04.2014
 *      Author: Patrik Huber
 */

#ifndef LANDMARKMAPPER_HPP_
#define LANDMARKMAPPER_HPP_

#include "imageio/LandmarkCollection.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"

namespace imageio {

/**
 * Represents a mapping from one kind of landmarks
 * to a different format. Mappings are stored in a
 * file (see libImageIO/share/landmarkMappings for
 * examples).
 */
class LandmarkMapper {
public:

	/**
	* Constructs an empty landmark mapper with no
	* landmark mappings.
	*
	*/
	LandmarkMapper();

	/**
	 * Constructs a new landmark mapper from a mappings-file.
	 *
	 * @param[in] filename A file with landmark mappings.
	 */
	LandmarkMapper(boost::filesystem::path filename);

	/**
	* Constructs a landmark mapper from a mappings-file.
	*
	* @param[in] filename A file with landmark mappings.
	* @return A LandmarkMapper with the given mappings.
	*/
	static LandmarkMapper load(boost::filesystem::path filename);

	/**
	* Converts the given landmark name to the mapped name.
	*
	* @param[in] landmarkName A landmark name to convert.
	* @return The mapped landmark name.
	* @throws out_of_range exception if there is no mapping
	*         for the given landmarkName.
	*/
	std::string convert(std::string landmarkName);


	/**
	* Returns a new landmark with the landmark name replaced
	* with the mapped value.
	*
	* @param[in] landmark A landmark with a given name.
	* @return A Landmark with the new name.
	* @throws out_of_range exception if there is no mapping
	*         for the landmark name of the given landmark.
	*/
	std::shared_ptr<Landmark> convert(std::shared_ptr<Landmark> landmark);

	/**
	* Returns a new LandmarkCollection with all the landmark names
	* replaced with their mapped values.
	* If a mapping for a landmark is not found, it is omitted.
	* Thus, the returned collection might be empty.
	*
	* @param[in] landmarks A collection of landmarks with names to be converted.
	* @return A LandmarkCollection containing all the landmarks
	*         that were successfully converted.
	*/
	LandmarkCollection convert(LandmarkCollection landmarks);

private:
	std::map<std::string, std::string> landmarkMappings;
};

} /* namespace imageio */
#endif /* LANDMARKMAPPER_HPP_ */
