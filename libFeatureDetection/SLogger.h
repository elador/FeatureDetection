#pragma once
#include "FdPatch.h"

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

	//void LogRegressorDetect(Det &this, &myimg, &candidates_wvr, "Hi");	// Do i really need all arguments? (eg Det)

	void drawBoxes(cv::Mat, std::vector<FdPatch*>, std::string, bool=false, std::string="");
	
	void drawYawAngle(cv::Mat, std::vector<FdPatch*>);
	void drawBoxesWithYawAngleColor(cv::Mat, std::vector<FdPatch*>);
	void drawPatchYawAngleColor(cv::Mat, FdPatch);
	void drawCenterpointsWithYawAngleColor(cv::Mat, std::vector<FdPatch*>, int);


	std::string basepath;
	bool useIndividualFolderForEachDetector;
	bool useSettingsFromMatConfig;

	/*
	struct verboselevel_img
		drawXYZ...
	struct verboselevel_console
		...
		*/
};

