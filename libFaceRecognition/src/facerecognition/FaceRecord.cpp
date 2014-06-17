/*
 * FaceRecord.cpp
 *
 *  Created on: 14.06.2014
 *      Author: Patrik Huber
 */

#include "facerecognition/FaceRecord.hpp"

#include "logging/LoggerFactory.hpp"

namespace facerecognition {

FaceRecord FaceRecord::createFrom(boost::property_tree::ptree recordTree)
{
	FaceRecord faceRecord;

	return faceRecord;
}

boost::property_tree::ptree FaceRecord::convertTo(FaceRecord faceRecord)
{
	boost::property_tree::ptree entry;
	entry.put_value(faceRecord.identifier);

	entry.put("subjectId", faceRecord.subjectId);
	entry.put("dataPath", faceRecord.dataPath.string());
	if (faceRecord.roll) {
		entry.put("roll", faceRecord.roll.get());
	}
	if (faceRecord.pitch) {
		entry.put("pitch", faceRecord.pitch.get());
	}
	if (faceRecord.yaw) {
		entry.put("yaw", faceRecord.yaw.get());
	}
	if (!faceRecord.session.empty()) {
		entry.put("session", faceRecord.session);
	}
	if (!faceRecord.lighting.empty()) {
		entry.put("lighting", faceRecord.lighting);
	}
	if (!faceRecord.expression.empty()) {
		entry.put("expression", faceRecord.expression);
	}
	if (!faceRecord.other.empty()) {
		entry.put("other", faceRecord.other);
	}

	return entry;
}

} /* namespace facerecognition */
