/*
 * ImageLogger.hpp
 *
 *  Created on: 22.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef IMAGELOGGER_HPP_
#define IMAGELOGGER_HPP_

#include "opencv2/core/core.hpp"
#include <string>
#include <vector>

using cv::Mat;
using std::string;
using std::vector;

namespace detection {

/**
 * TODO.
 * Note: I put this in libLogging first. Maybe this belongs more into libImageIO? It's not really logging, more like image-helpers.
 *       ==> I put it in libDetection for now, until it is clear what this becomes!
 */
class ImageLogger
{

	/* Img-logging functions */
	/* Use these functions if you write e.g. your own Cascade / ERT! */
/*	void LogImgInputGray(const cv::Mat*, std::string);	// only if writeInputGray
	void LogImgInputRGB(const cv::Mat*, std::string);	// only if writeInputRGB
	void LogImgPyramid(const Pyramid*, std::string, int); // only if writeImgPyramids
	void LogImgDetectorCandidates(const FdImage*, std::vector<FdPatch*>, std::string, std::string="");	// if writeDetectorCandidates || verboseLevelImages>=2
	void LogImgDetectorProbabilityMap(const cv::Mat*, const std::string, const std::string, std::string="");			// if writeDetectorProbabilityMaps || verboseLevelImages>=2
	void LogImgDetectorFinal(const FdImage*, std::vector<FdPatch*>, std::string, std::string="");		// if writeDetectorCandidates || verboseLevelImages>=1. Draws boxes with certainty color.
	void LogImgDetectorFinal(const FdImage*, std::vector<std::pair<std::string, std::vector<FdPatch*> > >, std::string, std::string="");	// if writeDetectorCandidates || verboseLevelImages>=1. Draws symbols for all Ffps.
	void LogImgRegressor(const FdImage*, std::vector<FdPatch*>, std::string, std::string="");			// if writeRegressor || verboseLevelImages>=1 (or maybe 2)
	void LogImgRegressorPyramids(const FdImage*, std::vector<FdPatch*>, std::string, std::string="");	// if writeRegressorPyramids || verboseLevelImages>=2
																										// outputs an image of the regressor for each scale
	void LogImgCircleDetectorCandidates(const FdImage*, cv::vector<cv::Vec3f>, std::string);			// if writeDetectorCandidates || verboseLevelImages>=2
*/
	/* Helper routines to draw stuff into images */
	/* They probably should be "private", but sometimes its nice to have them directly available... (or put them in a Utils-lib?) */
/*	void drawBoxesWithCertainty(cv::Mat, std::vector<FdPatch*>, std::string);		// Draw FD-boxes with certainty in color.
	void drawBoxesWithAngleColor(cv::Mat, std::vector<FdPatch*>, std::string);		// A nice color gradient for fout-values from -90 to +90.
	void drawCenterpointsWithAngleColor(cv::Mat, std::vector<FdPatch*>, std::string, int);	// Draw the centerpoints of the patches from ONE scale in a nice color gradient for fout-values from -90 to +90. 
	void drawFoutAsText(cv::Mat, std::vector<FdPatch*>, std::string);				// Draws the fout-values as a small text besides the FD-box.
	void drawScaleBox(cv::Mat, int, int);											// Draws a box with (w, h) into the original image, to display the size of the detector scales
	void drawAllScaleBoxes(cv::Mat, const PyramidMap*, std::string, int, int);		// Loops through all pyramids and draws a box with (w, h) into the original image, for each pyramid that is used by the given DetectorId.
	void drawFfpSymbols(cv::Mat, std::pair<std::string, std::vector<FdPatch*> >);	// Loops through all patches of one Ffp (specified by the string) and draws their symbol.
	void drawFfpSymbol(cv::Mat, std::string, FdPatch*);								// Draws the Ffp with its respective symbol into the image.

	void drawFfpsSmallSquare(cv::Mat, std::vector<std::pair<std::string, cv::Point2f> >);	// Loops through all given Ffps and draws a box around them
	void drawFfpSmallSquare(cv::Mat, std::pair<std::string, cv::Point2f>);					// Draws a bot around one Ffp


	struct landmarkData {	// Todo: The optimal place for this would be in the config-file of each detector!
		cv::Scalar bgrColor;
		//std::array<bool, 9> symbol;	// This is C++11. Make this an array again in Dec/Jan. !!!
		std::vector<bool> symbol;
		float displacementFactorH;
		float displacementFactorW;
	};
	std::map<std::string, landmarkData> landmarksData;	// All infos regarding all landmarks: name, color and symbol.
*/
};

} /* namespace detection */
#endif /* IMAGELOGGER_HPP_ */
