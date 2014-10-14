/*
 * EnrolCoutFeedback.hpp
 *
 *  Created on: 14.10.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef ENROLCOUTFEEDBACK_HPP_
#define ENROLCOUTFEEDBACK_HPP_

#include "frsdk/enroll.h"

#include <iostream>
#include <fstream>

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
		std::cout << "start" << std::endl;
	}

	void processingImage(const FRsdk::Image& img);

	void eyesFound(const FRsdk::Eyes::Location& eyeLoc);

	void eyesNotFound()
	{
		std::cout << "eyes not found" << std::endl;
	}

	void sampleQualityTooLow() {
		std::cout << "sampleQualityTooLow" << std::endl;
	}


	void sampleQuality(const float& f) {
		std::cout << "Sample Quality: " << f << std::endl;
	}

	void success(const FRsdk::FIR& fir_);

	void failure() { std::cout << "failure" << std::endl; }

	void end() { std::cout << "end" << std::endl; }

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
