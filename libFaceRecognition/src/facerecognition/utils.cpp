/*
 * utils.cpp
 *
 *  Created on: 15.06.2014
 *      Author: Patrik Huber
 */
#include "facerecognition/utils.hpp"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"

using boost::property_tree::ptree;
using boost::filesystem::path;
using std::vector;

namespace facerecognition {
	namespace utils {

std::vector<FaceRecord> readSigset(boost::filesystem::path filename)
{
	ptree sigset;
	try {
		read_info(R"("C:\Users\Patrik\Documents\GitHub\FeatureDetection\libFaceRecognition\share\sigset\MultiPIE_example.txt")", sigset);
	}
	catch (const boost::property_tree::ptree_error& error) {
		//appLogger.error(string("Error reading the sigset file: ") + error.what());
	}
	try {
		for (auto&& e : sigset) {
			//cout << e.first << " " << endl;
			//			cout << e.second << " " << endl;
		}
	}
	catch (const boost::property_tree::ptree_error& error) {
		//appLogger.error("Parsing config: " + string(error.what()));
	}

	return vector<FaceRecord>();
}

	} /* namespace utils */
} /* namespace facerecognition */
