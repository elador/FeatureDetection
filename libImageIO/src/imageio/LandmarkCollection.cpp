/*
 * LandmarkCollection.cpp
 *
 *  Created on: 23.03.2013
 *      Author: Patrik Huber
 */

#include "imageio/LandmarkCollection.hpp"
#include "imageio/Landmark.hpp"
#include <stdexcept>

namespace imageio {

LandmarkCollection::LandmarkCollection()
{
}

LandmarkCollection::~LandmarkCollection()
{
}

void LandmarkCollection::insert(const Landmark& landmark)
{
	if (landmark.getName().empty())
		throw std::runtime_error("landmark must have a name");
	if (hasLandmark(landmark.getName()))
		throw std::runtime_error("landmark name already in use");

	landmarks.push_back(landmark);
	landmarksMap.insert(make_pair(landmark.getName(), landmarks.size()-1 ));
}

bool LandmarkCollection::hasLandmark(const string& name) const
{
	return landmarksMap.find(name) != landmarksMap.end();
}

Landmark LandmarkCollection::getLandmark(const string& name) const
{
	if (!hasLandmark(name))
		throw std::runtime_error("landmark is not in list");
	return landmarks.at(landmarksMap.find(name)->second);
}

const vector<Landmark>& LandmarkCollection::getLandmarksList() const
{
	return landmarks;
}

} /* namespace imageio */
