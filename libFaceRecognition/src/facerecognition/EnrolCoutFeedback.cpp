/*
 * EnrolCoutFeedback.hpp
 *
 *  Created on: 14.10.2014
 *      Author: Patrik Huber
 */

#include "facerecognition/EnrolCoutFeedback.hpp"

#include "logging/LoggerFactory.hpp"

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
	std::cout << "processing image[" << img.name() << "]" << std::endl;
}

void EnrolCoutFeedback::eyesFound(const FRsdk::Eyes::Location& eyeLoc)
{
	std::cout << "found eyes at [" << eyeLoc.first
		<< " " << eyeLoc.second << "; confidences: "
		<< eyeLoc.firstConfidence << " "
		<< eyeLoc.secondConfidence << "]" << std::endl;
}

void EnrolCoutFeedback::success(const FRsdk::FIR& fir_)
{
	fir = new FRsdk::FIR(fir_);
	std::cout
		<< "successful enrollment";
	if (firFN != std::string("")) {

		std::cout << " FIR[filename,id,size] = [\""
			<< firFN.c_str() << "\",\"" << (fir->version()).c_str() << "\","
			<< fir->size() << "]";
		// write the fir
		std::ofstream firOut(firFN.c_str(),
			std::ios::binary | std::ios::out | std::ios::trunc);
		firOut << *fir;
	}
	firvalid = true;
	std::cout << std::endl;
}

const FRsdk::FIR& EnrolCoutFeedback::getFir() const
{
	// call only if success() has been invoked    
	if (!firvalid)
		throw InvalidFIRAccessError();

	return *fir;
}

} /* namespace facerecognition */
