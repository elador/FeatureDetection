/*
 * OpenGlDevice.cpp
 *
 *  Created on: 23.07.2013
 *      Author: Patrik Huber
 */

#include "render2/OpenGlDevice.hpp"

namespace render {

OpenGlDevice::OpenGlDevice(unsigned int screenWidth, unsigned int screenHeight)
{

}

OpenGlDevice::~OpenGlDevice()
{

}

Mat OpenGlDevice::getImage()
{
	return colorBuffer;
}

Mat OpenGlDevice::getDepthBuffer()
{
	return depthBuffer;
}

} /* namespace render */
