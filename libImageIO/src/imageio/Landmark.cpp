/*
 * Landmark.cpp
 *
 *  Created on: 22.03.2013
 *      Author: Patrik Huber
 */

#include "imageio/Landmark.hpp"
#include <utility>

using std::make_pair;

namespace imageio {

Landmark::Landmark(string name, Vec3f position, Size2f size, bool visibility) : name(name), position(position), size(size), visibility(visibility)
{
}

Landmark::~Landmark() {}

bool Landmark::isVisible() const
{
	return visibility;
}

string Landmark::getName() const
{
	return name;
}

Vec3f Landmark::getPosition() const
{
	return position;
}

Size2f Landmark::getSize() const
{
	return size;
}

map<string, array<bool, 9>> LandmarkSymbols::symbolMap;

array<bool, 9> LandmarkSymbols::get(string landmarkName)
{
	if (symbolMap.empty()) {
		array<bool, 9> reye_c		= {	false, true, false,
										false, true, true,
										false, false, false };
		symbolMap.insert(make_pair("right.eye.pupil.center", reye_c));	// Use an initializer list as soon as msvc supports it...

		array<bool, 9> leye_c	= {	false, true, false,
									true, true, false,
									false, false, false };
		symbolMap.insert(make_pair("left.eye.pupil.center", leye_c));

		array<bool, 9> nose_tip	= {	false, false, false,
									false, true, false,
									true, false, true };
		symbolMap.insert(make_pair("center.nose.tip", nose_tip));

		array<bool, 9> mouth_rc	= {	false, false, true,
									false, true, false,
									false, false, true };
		symbolMap.insert(make_pair("right.lips.corner", mouth_rc));

		array<bool, 9> mouth_lc	= {	true, false, false,
									false, true, false,
									true, false, false };
		symbolMap.insert(make_pair("left.lips.corner", mouth_lc));

		array<bool, 9> reye_oc	= {	false, true, false,
									false, true, true,
									false, true, false };
		symbolMap.insert(make_pair("right.eye.corner_outer", reye_oc));

		array<bool, 9> leye_oc	= {	false, true, false,
									true, true, false,
									false, true, false };
		symbolMap.insert(make_pair("left.eye.corner_outer", leye_oc));

		array<bool, 9> mouth_ulb	= {	false, false, false,
										true, true, true,
										false, true, false };
		symbolMap.insert(make_pair("center.lips.upper.outer", mouth_ulb));

		array<bool, 9> nosetrill_r	= {	true, false, false,
										true, true, true,
										false, false, false };
		symbolMap.insert(make_pair("right.nose.wing.tip", nosetrill_r));

		array<bool, 9> nosetrill_l	= {	false, false, true,
										true, true, true,
										false, false, false };
		symbolMap.insert(make_pair("left.nose.wing.tip", nosetrill_l));

		array<bool, 9> rear_DONTKNOW	= {	false, true, true,
											false, true, false,
											false, true, true };
		symbolMap.insert(make_pair("right.ear.DONTKNOW", rear_DONTKNOW)); // right.ear.(antihelix.tip | lobule.center | lobule.attachement)

		array<bool, 9> lear_DONTKNOW	= {	true, true, false,
											false, true, false,
											true, true, false };
		symbolMap.insert(make_pair("left.ear.DONTKNOW", lear_DONTKNOW));

	}
	const auto symbol = symbolMap.find(landmarkName);
	if (symbol == symbolMap.end()) {
		array<bool, 9> unknownLmSymbol	= {	true, false, true,
											false, true, false,
											true, false, true };
		return unknownLmSymbol;
	}
	return symbol->second;
}

void LandmarkSymbols::getColor()
{

}

} /* namespace imageio */
