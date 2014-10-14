/*
 * EnrolCoutFeedback.hpp
 *
 *  Created on: 14.10.2014
 *      Author: Patrik Huber
 */

#include "facerecognition/EnrolCoutFeedback.hpp"

#include "logging/LoggerFactory.hpp"

#include <fstream>

using logging::LoggerFactory;
using std::cout;
using std::endl;

namespace facerecognition {

std::ostream& operator<<(std::ostream& o, const FRsdk::Position& p)
{
	o << "[" << p.x() << ", " << p.y() << "]";
	return o;
}

void EnrolCoutFeedback::processingImage(const FRsdk::Image& img)
{
	logging::Loggers->getLogger("facerecognition").debug("Enrolment: processing image[" + img.name() + "]");
}

void EnrolCoutFeedback::eyesFound(const FRsdk::Eyes::Location& eyeLoc)
{
	std::stringstream msg;
	msg << "found eyes at [" << eyeLoc.first << " " << eyeLoc.second << "; confidences: " << eyeLoc.firstConfidence << " " << eyeLoc.secondConfidence << "]";
	logging::Loggers->getLogger("facerecognition").debug("Enrolment: " + msg.str());
}

void EnrolCoutFeedback::success(const FRsdk::FIR& fir_)
{
	fir = new FRsdk::FIR(fir_);
	logging::Loggers->getLogger("facerecognition").debug("Enrolment: Successful enrollment");
	if (firFN != std::string("")) {
		std::stringstream msg;
		msg << "FIR[filename,id,size] = [\"" << firFN << "\",\"" << fir->version() << "\"," << fir->size() << "]";
		logging::Loggers->getLogger("facerecognition").debug("Enrolment: " + msg.str());
		// write the fir
		std::ofstream firOut(firFN.c_str(), std::ios::binary | std::ios::out | std::ios::trunc);
		firOut << *fir;
	}
	firvalid = true;
}

const FRsdk::FIR& EnrolCoutFeedback::getFir() const
{
	// call only if success() has been invoked    
	if (!firvalid)
		throw InvalidFIRAccessError();

	return *fir;
}

} /* namespace facerecognition */
