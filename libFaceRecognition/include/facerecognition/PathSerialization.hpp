/*
 * PathSerialization.hpp
 *
 *  Created on: 13.10.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef PATHSERIALIZATION_HPP_
#define PATHSERIALIZATION_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"
#include "boost/serialization/serialization.hpp"

#include <string>

/**
 * Todo.
 *
 */
namespace boost {
	namespace serialization {

template<class Archive>
void serialize(Archive& ar, boost::filesystem::path& p, const unsigned int version)
{
	std::string s;
	if (Archive::is_saving::value)
		s = p.string();
	ar & boost::serialization::make_nvp("string", s);
	if (Archive::is_loading::value)
		p = s;
};

	} /* namespace serialization */
} /* namespace boost */

#endif /* PATHSERIALIZATION_HPP_ */
