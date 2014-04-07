/*
 * FittingWindow.hpp
 *
 *  Created on: 04.04.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FITTINGWINDOW_HPP_
#define FITTINGWINDOW_HPP_

#include "OpenGLWindow.hpp"
#include "morphablemodel/MorphableModel.hpp"

#include <memory>

namespace imageio {
	class LabeledImageSource;
}
namespace render {
	class QOpenGLRenderer;
}


/**
 * Desc
 * 1
 * 2
 */
class FittingWindow : public OpenGLWindow
{
public:
	FittingWindow(std::shared_ptr<imageio::LabeledImageSource> labeledImageSource, morphablemodel::MorphableModel morphableModel);
	~FittingWindow() {
		//delete m_device;
	}

	void initialize(QOpenGLContext* context);
	void render();
	void fit();

private:
	int m_frame;

	render::QOpenGLRenderer* r;

	//QOpenGLPaintDevice *m_device; // for QPainter

	std::shared_ptr<imageio::LabeledImageSource> labeledImageSource; // todo unique_ptr
	morphablemodel::MorphableModel morphableModel;

};

#endif /* FITTINGWINDOW_HPP_ */
