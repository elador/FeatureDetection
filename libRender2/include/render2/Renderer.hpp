/*
 * Renderer.hpp
 *
 *  Created on: 25.07.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef RENDERER_HPP_
#define RENDERER_HPP_

#include "render2/RenderDevice.hpp"

#include "opencv2/core/core.hpp"

#include <memory>

using std::shared_ptr;

namespace render {

/**
 * Desc
 */
class Renderer
{
public:
	Renderer(shared_ptr<RenderDevice> renderDevice);
	virtual ~Renderer();

//private:
public:
	shared_ptr<RenderDevice> renderDevice;

};

} /* namespace render */

#endif /* RENDERER_HPP_ */
