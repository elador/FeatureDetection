/*
 * FaceRecord.hpp
 *
 *  Created on: 14.06.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FACERECORD_HPP_
#define FACERECORD_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/optional.hpp"

#include <string>

namespace facerecognition {

/**
 * Representation for a face record. Can e.g. come from MultiPIE or from PaSC XML.
 * Next, we probably need loader (or createFrom(...)), e.g. for MultiPIE load from the filename, from PaSC from XML.
 */
class FaceRecord {
public:
	std::string identifier; ///< A unique (among the respective database) subject identifier.
	std::string subjectId; ///< The ID of the subject.
	boost::filesystem::path dataPath; ///< The full path to where the image (or whatever data is associated with this record) inside the database can be found

	// The following is additional metadata that may or may not be available, depending on the database:
	boost::optional<float> roll;
	boost::optional<float> pitch;
	boost::optional<float> yaw;
	std::string session{""};
	std::string lighting{""};
	std::string expression{""};
	std::string other{""};

	/**
	* Desc.
	*
	* @param[in] in recordTree
	* @return Todo.
	*/
	static FaceRecord createFrom(boost::property_tree::ptree recordTree);

	/**
	* Desc.
	* Note: The name is odd, i.e. ptree entry = facerecognition::FaceRecord::convertTo(faceRecord);
	* doesn't make so much sense.
	*
	* @param[in] in faceRecord
	* @return Todo.
	*/
	static boost::property_tree::ptree convertTo(FaceRecord faceRecord);
};

} /* namespace facerecognition */
#endif /* FACERECORD_HPP_ */
