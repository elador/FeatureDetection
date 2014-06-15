/*
 * utils.hpp
 *
 *  Created on: 15.06.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FACERECOGNITION_UTILS_HPP_
#define FACERECOGNITION_UTILS_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"

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
void readSigset(boost::filesystem::path filename);

	} /* namespace utils */
} /* namespace facerecognition */

#endif /* FACERECOGNITION_UTILS_HPP_ */
