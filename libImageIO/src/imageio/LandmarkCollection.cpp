/*
 * LandmarkCollection.cpp
 *
 *  Created on: 23.03.2013
 *      Author: Patrik Huber
 */

#include "imageio/LandmarkCollection.hpp"
#include <stdexcept>

using std::move;

namespace imageio {

LandmarkCollection::LandmarkCollection()
{
}

LandmarkCollection::~LandmarkCollection()
{
}

void LandmarkCollection::clear()
{
	landmarks.clear();
	landmarksMap.clear();
}

void LandmarkCollection::insert(shared_ptr<Landmark> landmark)
{
	if (landmark->getName().empty())
		throw std::invalid_argument("landmark must have a name");
	if (hasLandmark(landmark->getName()))
		throw std::invalid_argument("landmark name already in use");

	landmarks.push_back(landmark);
	landmarksMap.insert(make_pair(landmark->getName(), landmarks.size() - 1));
}

bool LandmarkCollection::isEmpty() const
{
	return landmarks.empty();
}

bool LandmarkCollection::hasLandmark(const string& name) const
{
	return landmarksMap.find(name) != landmarksMap.end();
}

const Landmark& LandmarkCollection::getLandmark(const string& name) const
{
	if (!hasLandmark(name))
		throw std::invalid_argument("there is no landmark with name '" + name + "'");
	return *landmarks.at(landmarksMap.find(name)->second);
}

const Landmark& LandmarkCollection::getLandmark() const
{
	if (isEmpty())
		throw std::invalid_argument("there is no landmark within the collection");
	return *landmarks.front();
}

const vector<shared_ptr<Landmark>>& LandmarkCollection::getLandmarks() const
{
	return landmarks;
}

} /* namespace imageio */
