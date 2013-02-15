/*
 * Sample.cpp
 *
 *  Created on: 27.06.2012
 *      Author: poschmann
 */

#include "tracking/Sample.h"

namespace tracking {

Sample::Sample() : x(0), y(0), size(0), weight(1), object(false) {}

Sample::Sample(int x, int y, int size) : x(x), y(y), size(size), weight(0), object(false) {}

Sample::~Sample() {}

} /* namespace tracking */
