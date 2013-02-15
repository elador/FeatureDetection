/*
 * Patch.cpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann & huber
 */

#include "imageprocessing/Patch.hpp"

namespace imageprocessing {

Patch::Patch(int x, int y, double scale, Mat data) : x(x), y(y), scale(scale), data(data) {}

Patch::~Patch() {}

} /* namespace imageprocessing */
