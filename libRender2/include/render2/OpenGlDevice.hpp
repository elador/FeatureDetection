/*
 * OpenGlDevice.hpp
 *
 *  Created on: 23.07.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef OPENGLDEVICE_HPP_
#define OPENGLDEVICE_HPP_

#include "render2/RenderDevice.hpp"

#include "opencv2/core/core.hpp"

namespace render {

/**
 * Desc
 */
class OpenGlDevice : public RenderDevice
{

public:
	OpenGlDevice() {};
	OpenGlDevice(unsigned int screenWidth, unsigned int screenHeight);
	~OpenGlDevice();

	cv::Mat getImage();
	cv::Mat getDepthBuffer();

private:
	cv::Mat colorBuffer;
	cv::Mat depthBuffer;

};

} /* namespace render */

#endif /* OPENGLDEVICE_HPP_ */
