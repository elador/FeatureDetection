/*
 * pasc.hpp
 *
 *  Created on: 27.09.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef PASC_HPP_
#define PASC_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"
#include "boost/optional.hpp"
#include "boost/serialization/serialization.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/optional.hpp"
#include "boost/serialization/string.hpp"

#include <vector>
#include <iomanip>

namespace facerecognition {

// pascFrameNumber starts with 1. Your counting might start with 0, so add 1 to it before passing it here.
std::string getPascFrameName(boost::filesystem::path videoFilename, int pascFrameNumber)
{
	std::ostringstream ss;
	ss << std::setw(3) << std::setfill('0') << pascFrameNumber;
	return videoFilename.stem().string() + "/" + videoFilename.stem().string() + "-" + ss.str() + ".jpg";
}

/**
 * Todo.
 *
 */
class PascVideoDetection
{
public:
	static PascVideoDetection readFromCsv(std::string line);
private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		// When the class Archive corresponds to an output archive, the
		// & operator is defined similar to <<.  Likewise, when the class Archive
		// is a type of input archive the & operator is defined similar to >>.
		ar & frame_id;
		ar & fcen_x;
		ar & fcen_y;
		ar & fwidth;
		ar & fheight;
		ar & fpose_y;
		ar & re_x;
		ar & re_y;
		ar & le_x;
		ar & le_y;
	};
	
public:
	// The class members have the same name as in the header line in the csv file
	std::string frame_id;
	int fcen_x;
	int fcen_y;
	int fwidth;
	int fheight;
	float fpose_y; // yaw
	boost::optional<int> re_x = boost::none; // eye coordinates may or may not be present
	boost::optional<int> re_y = boost::none; // re = which eye? document here.
	boost::optional<int> le_x = boost::none;
	boost::optional<int> le_y = boost::none;
};

/**
 * Todo
 *
 * @param[in] in Todo
 * @return Todo.
 */
std::vector<PascVideoDetection> readPascVideoDetections(boost::filesystem::path csvFile);

} /* namespace facerecognition */

#endif /* PASC_HPP_ */
