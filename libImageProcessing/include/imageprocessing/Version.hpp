/*
 * Version.hpp
 *
 *  Created on: 27.10.2015
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_VERSION_HPP_
#define IMAGEPROCESSING_VERSION_HPP_

#include <ostream>

namespace imageprocessing {

/**
 * Version number.
 */
class Version {
public:

	Version() : instance(nextInstance++), version(0) {}

	Version(const Version& other) = default;

	Version& operator++() {
		++version;
		return *this;
	}

	bool operator==(const Version& other) const {
		return instance == other.instance && version == other.version;
	}

	bool operator!=(const Version& other) const {
		return !(*this == other);
	}

	friend std::ostream& operator<<(std::ostream& out, const Version& version);

private:

	static int nextInstance;

	int instance;
	int version;
};

} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_VERSION_HPP_ */
