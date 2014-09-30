/*
 * SimpleAnnotationSource.cpp
 *
 *  Created on: 21.01.2014
 *      Author: poschmann
 */

#include "imageio/SimpleAnnotationSource.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

using cv::Rect;
using std::string;
using std::invalid_argument;

namespace imageio {

SimpleAnnotationSource::SimpleAnnotationSource(const string& filename) : annotations(), index(-1) {
	string line;
	char separator;
	std::ifstream file(filename.c_str());
	if (!file.is_open())
		throw invalid_argument("SimpleAnnotationSource: file \"" + filename + "\" cannot be opened");
	Rect annotation;
	while (file.good()) {
		if (!std::getline(file, line))
			break;
		// read values from line
		std::istringstream lineStream(line);
		bool nonSpaceSeparator = !std::all_of(line.begin(), line.end(), [](char ch) {
			std::locale loc;
			return std::isdigit(ch, loc) || std::isspace(ch, loc) || '-' == ch;
		});
		lineStream >> annotation.x;
		if (nonSpaceSeparator)
			lineStream >> separator;
		lineStream >> annotation.y;
		if (nonSpaceSeparator)
			lineStream >> separator;
		lineStream >> annotation.width;
		if (nonSpaceSeparator)
			lineStream >> separator;
		lineStream >> annotation.height;

		annotations.push_back(annotation);
	}
}

void SimpleAnnotationSource::reset() {
	index = -1;
}

bool SimpleAnnotationSource::next() {
	index++;
	return index < static_cast<int>(annotations.size());
}

Rect SimpleAnnotationSource::getAnnotation() const {
	if (index < 0 && index >= static_cast<int>(annotations.size()))
		return Rect();
	return annotations[index];
}

} /* namespace imageio */
