#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class FdImage;
class Pyramid;
class FdPatch;
typedef std::map<int, Pyramid*> PyramidMap;

#define Logger SLogger::Instance()


class SLogger
{
private:
	SLogger(void);
	~SLogger(void);
	SLogger(const SLogger &);
	SLogger& operator=(const SLogger &);

public:
	static SLogger* Instance(void);

private:
	/* struct
		-for each detector: output each individual patch
	*/
	typedef struct imgparams {
		bool writeImgInputGray;
		bool writeImgInputRGB;
		bool writeImgPyramids;
		bool writeDetectorCandidates;	// Writes each detector's candidates (face-box) into a separate image
		bool writeRegressor;			// Writes an image (face-box with angle color) of the regression output
		bool writeRegressorPyramids;	// Writes an image for each pyramid, drawing the angle as colored centerpoints
		bool drawRegressorFoutAsText;	// Requires writeRegressor to be true (of course). Additionally draws the fout-value as text.
		bool drawScales;				// Draw the scale (box-width) into output images
	};
	typedef struct textparams {
		bool todo;
	};

	typedef struct params {
		textparams text;
		imgparams img;
	};
	
	class FdPatchLogInfo {
	public:
		//wvm_info, svm_info.. then, a map of those for each detector
	private:
		//pointer to patch
	};
	//public... vector<FdPatchLogInfo>;

	int verboseLevelText;
	int verboseLevelImages;

	std::string imgLogBasepath;

public:

	void setVerboseLevelText(int);
	void setVerboseLevelImages(int);

	/* Verbose-level Images:
		0: Don't even write a single image
		1: Write one single output image with the final result
		2: Output each detector's images and stages (eg 5 images for FD-cascade)
		3: Write an output image for each scale, for each detector
	*/
	/* Img-logging functions */
	/* Use these functions if you write e.g. your own Cascade / ERT! */
	void LogImgInputGray(const cv::Mat*, std::string);	// only if writeInputGray
	void LogImgInputRGB(const cv::Mat*, std::string);	// only if writeInputRGB
	void LogImgPyramid(const Pyramid*, std::string, int); // only if writeImgPyramids
	void LogImgDetectorCandidates(const FdImage*, std::vector<FdPatch*>, std::string, std::string="");	// if writeDetectorCandidates || verboseLevelImages>=2
	void LogImgDetectorFinal(const FdImage*, std::vector<FdPatch*>, std::string, std::string="");	// if writeDetectorCandidates || verboseLevelImages>=1
	void LogImgRegressor(const FdImage*, std::vector<FdPatch*>, std::string, std::string="");	// if writeRegressor || verboseLevelImages>=1 (or maybe 2)
	void LogImgRegressorPyramids(const FdImage*, std::vector<FdPatch*>, std::string, std::string="");	// if writeRegressorPyramids || verboseLevelImages>=2
																										// outputs an image of the regressor for each scale

	params global;	// global settings

	std::map<std::string, params> detectorParams;	// detector specific settings. If it exists, use it instead of the global settings. If it doesnt exist, use global settings.

	/* Helper routines to draw stuff into images */
	/* They probably should be "private", but sometimes its nice to have them directly available... (or put them in a Utils-lib?) */
	void drawBoxesWithCertainty(cv::Mat, std::vector<FdPatch*>, std::string);	// Draw FD-boxes with certainty in color.
	void drawBoxesWithAngleColor(cv::Mat, std::vector<FdPatch*>, std::string);	// A nice color gradient for fout-values from -90 to +90.
	void drawCenterpointsWithAngleColor(cv::Mat, std::vector<FdPatch*>, std::string, int);	// Draw the centerpoints of the patches from ONE scale in a nice color gradient for fout-values from -90 to +90. 
	void drawFoutAsText(cv::Mat, std::vector<FdPatch*>, std::string);	// Draws the fout-values as a small text besides the FD-box.
	void drawScaleBox(cv::Mat, int, int);	// Draws a box with (w, h) into the original image, to display the size of the detector scales
	void drawAllScaleBoxes(cv::Mat, const PyramidMap*, std::string, int, int); // Loops through all pyramids and draws a box with (w, h) into the original image, for each pyramid that is used by the given DetectorId.
	
	
	/* Old stuff below! */
	//void drawSinglePatchYawAngleColor(cv::Mat, FdPatch);	// draws a bounding box around a single e.g. 20x20 patch with the yaw angle color.
	//bool useIndividualFolderForEachDetector;
	//bool useSettingsFromMatConfig;

};

