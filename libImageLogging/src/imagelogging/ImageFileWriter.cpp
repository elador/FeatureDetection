/*
 * ImageFileWriter.cpp
 *
 *  Created on: 05.06.2013
 *      Author: Patrik Huber
 */

#include "imagelogging/ImageFileWriter.hpp"
#include "imagelogging/imageloglevels.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/filesystem.hpp"
#include <ios>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>

using std::ios_base;
using std::ostringstream;
using std::chrono::system_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

namespace imagelogging {

ImageFileWriter::ImageFileWriter(loglevel logLevel, path directory) : Appender(logLevel), outputDirectory(directory)
{
	//file << getCurrentTime() << " Starting logging at log-level " << loglevelToString(logLevel) << " to file " << filename << std::endl;
	// TODO We should really also output the loggerName here! So... maybe make a member variable that holds a reference to the appenders parent logger?
	if(!boost::filesystem::exists(directory)) {
		try {
			boost::filesystem::create_directory(directory);
			// TODO use text-logging, output dir-creation
		} catch(const boost::filesystem::filesystem_error& e) {
			std::cout << "ImageFileWriter: Error while creating directory: " << e.what() << std::endl; // TODO use text-logging
		}
	}
}

ImageFileWriter::~ImageFileWriter()
{
}

void ImageFileWriter::log(const loglevel logLevel, const string loggerName, const string filename, Mat image, function<void ()> functionToApply, const string filenameSuffix)
{
	if(logLevel <= this->logLevel) {
		functionToApply();
		try {
			string modifiedFilenameSuffix;
			if (!filenameSuffix.empty()) {
				modifiedFilenameSuffix = string("_") + filenameSuffix;
			} else {
				modifiedFilenameSuffix = filenameSuffix;
			}
			imwrite((outputDirectory/filename).string() + modifiedFilenameSuffix + ".png", image);
		} catch(const cv::Exception& e) {
			std::cout << "Exception occurred while trying to write the file " << (outputDirectory/filename) << std::endl;
			//std::cout << e.what() << std::endl; // imwrite already outputs the error, which is not that nice
			exit(EXIT_FAILURE);	// Could also be a "warning" and we could continue. But we most likely REALLY want to write the image.
			// Todo: Use text-logger.
		}
		// TODO logger.out("Wrote image ...");
	}
}

string ImageFileWriter::getCurrentTime()
{
	system_clock::time_point now = system_clock::now();
	duration<int64_t, std::ratio<1>> seconds = duration_cast<duration<int64_t, std::ratio<1>>>(now.time_since_epoch());
	duration<int64_t, std::milli> milliseconds = duration_cast<duration<int64_t, std::milli>>(now.time_since_epoch());
	duration<int64_t, std::milli> msec = milliseconds - seconds;

	std::time_t t_now = system_clock::to_time_t(now);
	struct tm* tm_now = std::localtime(&t_now);
	ostringstream os;
	os.fill('0');
	os << (1900 + tm_now->tm_year) << '-' << std::setw(2) << (1 + tm_now->tm_mon) << '-' << std::setw(2) << tm_now->tm_mday;
	os << ' ' << std::setw(2) << tm_now->tm_hour << ':' << std::setw(2) << tm_now->tm_min << ':' << std::setw(2) << tm_now->tm_sec;
	os << '.' << std::setw(3) << msec.count();
	return os.str();
}

} /* namespace imagelogging */
