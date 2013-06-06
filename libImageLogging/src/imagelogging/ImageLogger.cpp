/*
 * ImageLogger.cpp
 *
 *  Created on: 04.06.2013
 *      Author: Patrik Huber
 */

#include "imagelogging/ImageLogger.hpp"
#include "imagelogging/Appender.hpp"

namespace imagelogging {

ImageLogger::ImageLogger(string name) : appenders(), name(name), currentImageName() {}

ImageLogger::~ImageLogger() {}

void ImageLogger::log(const loglevel logLevel, Mat image, function<void ()> functionToApply, const string filenameSuffix)
{
	for (shared_ptr<Appender> appender : appenders) {
		appender->log(logLevel, name, currentImageName, image, functionToApply, filenameSuffix);
	}
}

void ImageLogger::addAppender(shared_ptr<Appender> appender)
{
	appenders.push_back(appender);
}

void ImageLogger::trace(Mat image, function<void ()> functionToApply, const string filenameSuffix)
{
	log(loglevel::TRACE, image, functionToApply, filenameSuffix);
}

void ImageLogger::debug(Mat image, function<void ()> functionToApply, const string filenameSuffix)
{
	log(loglevel::DEBUG, image, functionToApply, filenameSuffix);
}

void ImageLogger::info(Mat image, function<void ()> functionToApply, const string filenameSuffix)
{
	log(loglevel::INFO, image, functionToApply, filenameSuffix);
}

void ImageLogger::warn(Mat image, function<void ()> functionToApply, const string filenameSuffix)
{
	log(loglevel::WARN, image, functionToApply, filenameSuffix);
}

void ImageLogger::error(Mat image, function<void ()> functionToApply, const string filenameSuffix)
{
	log(loglevel::ERROR, image, functionToApply, filenameSuffix);
}

void ImageLogger::panic(Mat image, function<void ()> functionToApply, const string filenameSuffix)
{
	log(loglevel::PANIC, image, functionToApply, filenameSuffix);
}

void ImageLogger::setCurrentImageName(string imageName)
{
	currentImageName = imageName;
}

} /* namespace imagelogging */
