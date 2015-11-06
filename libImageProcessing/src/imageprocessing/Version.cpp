/*
 * Version.cpp
 *
 *  Created on: 27.10.2015
 *      Author: poschmann
 */

#include "imageprocessing/Version.hpp"

namespace imageprocessing {

int Version::nextInstance = 0;

std::ostream& operator<<(std::ostream& out, const Version& version) {
	return out << version.instance << ":" << version.version;
}

} /* namespace imageprocessing */
