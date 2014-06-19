/*
 * utils.hpp
 *
 *  Created on: 15.06.2014
 *      Author: Patrik Huber
 */
#pragma once

#include "facerecognition/FaceRecord.hpp"

#ifndef FACERECOGNITION_UTILS_HPP_
#define FACERECOGNITION_UTILS_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"

#include <vector>

/**
 * The facerecognition::utils namespace contains utility
 * functions for miscellaneous face recognition tasks.
 */
namespace facerecognition {
	namespace utils {

/**
 * Desc.
 *
 * @param[in] in Todo
 * @return Todo.
 */
std::vector<FaceRecord> readSigset(boost::filesystem::path filename);

/**
* Desc.
*
* @param[in] in Todo
* @return Todo.
*/
//std::vector<FaceRecord> transformSigset(boost::filesystem::path filename);

class DataPathTransformation
{
public:
	std::string name; // original | basename: use the original name as given in the record or just the basename
	boost::filesystem::path rootPath;
	std::string prefix;
	std::string suffix;
	std::string replaceExtension;

	static DataPathTransformation read(boost::property_tree::ptree tree) {
		DataPathTransformation transformation;
		transformation.name = tree.get<std::string>("name");
		transformation.rootPath = tree.get<boost::filesystem::path>("rootPath");
		transformation.prefix = tree.get<std::string>("prefix", "");
		transformation.suffix = tree.get<std::string>("suffix", "");
		transformation.replaceExtension = tree.get<std::string>("replaceExtension", "");
		return transformation;
	};
};

/**
* Desc.
*
* @param[in] in Todo
* @return Todo.
*/
boost::filesystem::path transformDataPath(const boost::filesystem::path& originalDataPath, DataPathTransformation transformation);

	} /* namespace utils */
} /* namespace facerecognition */

#endif /* FACERECOGNITION_UTILS_HPP_ */
