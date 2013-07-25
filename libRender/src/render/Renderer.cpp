/*
 * Renderer.cpp
 *
 *  Created on: 23.07.2013
 *      Author: Patrik Huber
 */


#include "render/Renderer.hpp"
#include "render/MatrixUtils.hpp"
#include "render/Texture.hpp"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cmath>
#include <iostream>
#include <array>
#include <tuple>

namespace render {

Renderer::Renderer(shared_ptr<RenderDevice> renderDevice) : renderDevice(renderDevice)
{
	
}

Renderer::~Renderer()
{

}

} /* namespace render */
