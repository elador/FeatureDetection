/*
 * LandmarksHelper.hpp DEPRECATED
 *
 *  Created on: 23.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef LANDMARKSHELPER_HPP_
#define LANDMARKSHELPER_HPP_

namespace imageio {


class LandmarksHelper {
public:

	/**
	 * Parses a line from an LFW .lst file and returns the facebox.
	 * TODO: I think .lst files are actually from MR. They could also contain
	 *       more landmarks, but at the moment we don't have that use-case.
	 *
	 * @param[in] line The line with the landmark information to parse.
	 * @return A FaceBoxLandmark object.
	 */
	/*static FaceBoxLandmark readFromLstLine(const string& line) {
		int l=0, r=0, b=0, t=0;	// TODO use floats?
		string buf;
		stringstream ss(line);
		ss >> buf;
		ss >> l;
		ss >> t;
		ss >> r;
		ss >> b;
		return FaceBoxLandmark("facebox", Vec3f(l+(r-l)/2.0f, t+(b-t)/2.0f, 0.0f), r-l, b-t);
	}
	*/

};

} /* namespace imageio */
#endif /* LANDMARKSHELPER_HPP_ */
