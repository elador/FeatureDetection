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

#include "opencv2/core/core.hpp"

#include <vector>

/**
 * The facerecognition::utils namespace contains utility
 * functions for miscellaneous face recognition tasks.
 */
namespace facerecognition {
	namespace utils {

// TODO: This should rather go into libImageIO.
// Caution: This will eat a lot of RAM, 1-2 GB for 600 RGB frames at 720p
std::vector<cv::Mat> getFrames(boost::filesystem::path videoFilename);

/**
 * Reads in a sigset from the PaSC database.
 * It is probably the usual format used by NIST/BEE.
 *
 * @param[in] filename Filename to a PaSC .xml sigset
 * @param[in] stripFilePath Strips the filename from the xml to only contain the base-filename (with ext.). Necessary for training dataset as the sigsets contain a wrong path.
 * @return A vector of all the entries in the sigset..
 * (Todo proper doxygen) Throws a std::runtime_error when the xml can't be read by the parser.
 */
std::vector<FaceRecord> readPascSigset(boost::filesystem::path filename, bool stripFilePath = false);

/**
 * Reads in a record list from a boost ptree file... Todo
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

// TODO: This is not really a "utils", move it to somewhere else
//#include "boost/serialization/utility.hpp" // This should best be here, including the one for vector. But there's some error with it.
class VideoScore
{
public:
	std::vector<std::vector<std::pair<facerecognition::FaceRecord, float>>> scores; // For every frame, have a vector with the scores for all gallery
	// Gallery records?
private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & scores;
	}
};


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
