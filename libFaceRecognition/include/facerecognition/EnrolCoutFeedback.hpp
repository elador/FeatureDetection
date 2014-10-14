/*
 * EnrolCoutFeedback.hpp
 *
 *  Created on: 14.10.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef ENROLCOUTFEEDBACK_HPP_
#define ENROLCOUTFEEDBACK_HPP_

#include "logging/LoggerFactory.hpp"

#include "frsdk/enroll.h"

//#include <iostream>
//#include <fstream>

//using logging::LoggerFactory;

namespace facerecognition {

// for the CoutEnrol
std::ostream& operator<<(std::ostream& o, const FRsdk::Position& p);

class EnrolCoutFeedback : public FRsdk::Enrollment::FeedbackBody
{
public:
	EnrolCoutFeedback(const std::string& firFilename)
		: firFN(firFilename), firvalid(false) { }

	void start() {
		firvalid = false;
		logging::Loggers->getLogger("facerecognition").debug("Enrolment: Start");
	}

	void processingImage(const FRsdk::Image& img);

	void eyesFound(const FRsdk::Eyes::Location& eyeLoc);

	void eyesNotFound() {
		logging::Loggers->getLogger("facerecognition").debug("Enrolment: Eyes not found");
	}

	void sampleQualityTooLow() {
		logging::Loggers->getLogger("facerecognition").debug("Enrolment: sampleQualityTooLow");
	}


	void sampleQuality(const float& f) {
		logging::Loggers->getLogger("facerecognition").debug("Enrolment: Sample Quality: " + std::to_string(f));
	}

	void success(const FRsdk::FIR& fir_);

	void failure() {
		logging::Loggers->getLogger("facerecognition").debug("Enrolment: Failure");
	}

	void end() {
		logging::Loggers->getLogger("facerecognition").debug("Enrolment: End");
	}

	const FRsdk::FIR& getFir() const;

	bool firValid() const {
		return firvalid;
	}

private:
	FRsdk::CountedPtr<FRsdk::FIR> fir;
	std::string firFN;
	bool firvalid;
};

class InvalidFIRAccessError : public std::exception
{
public:
	InvalidFIRAccessError() throw() :
		msg("Trying to access invalid FIR") {}
	~InvalidFIRAccessError() throw() { }
	const char* what() const throw() { return msg.c_str(); }
private:
	std::string msg;
};

} /* namespace facerecognition */
#endif /* ENROLCOUTFEEDBACK_HPP_ */
