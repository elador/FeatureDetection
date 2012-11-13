#include "stdafx.h"
#include "CascadeFacialFeaturePointsSimple.h"

#include "SLogger.h"


CascadeFacialFeaturePointsSimple::CascadeFacialFeaturePointsSimple(void)
{
	// WE STICK TO THE TERMINOLOGY USED THROUGHOUT THE 3DMM. LEFT MOUTH MEANS LEFT FROM THE PERSON IN THE PICTURE!
	// SO ALL CLASSIFIERS MUST BE SWAPPED! ("rm" in MR means "right" from observer's view. "mouth_rc" in 3DMM means "right" from the person in the image!)
	reye = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_re.mat");
	reye->setIdentifier("leye_c");
	leye = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_le.mat");
	leye->setIdentifier("reye_c");
	nosetip = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_nt.mat");
	nosetip->setIdentifier("nose_tip");
	rmouth = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_rm.mat");
	rmouth->setIdentifier("mouth_lc");
	lmouth = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_lm.mat");
	lmouth->setIdentifier("mouth_rc");
}


CascadeFacialFeaturePointsSimple::~CascadeFacialFeaturePointsSimple(void)
{
	delete reye;
	delete leye;
	delete nosetip;
	delete rmouth;
	delete lmouth;
}

int CascadeFacialFeaturePointsSimple::initForImage(FdImage* myimg)
{
	reye->initForImage(myimg);
	leye->initForImage(myimg);
	nosetip->initForImage(myimg);
	rmouth->initForImage(myimg);
	lmouth->initForImage(myimg);
	return 1;
}

std::vector<std::pair<std::string, std::vector<FdPatch*> > > CascadeFacialFeaturePointsSimple::detectOnImage(FdImage* myimg)
{
	std::vector<std::pair<std::string, std::vector<FdPatch*> > > landmarkCandidates;
	reye->detectOnImage(myimg);
	leye->detectOnImage(myimg);
	nosetip->detectOnImage(myimg);
	rmouth->detectOnImage(myimg);
	lmouth->detectOnImage(myimg);

	landmarkCandidates.push_back(std::make_pair(reye->getIdentifier(), reye->candidates));
	landmarkCandidates.push_back(std::make_pair(leye->getIdentifier(), leye->candidates));
	landmarkCandidates.push_back(std::make_pair(nosetip->getIdentifier(), nosetip->candidates));
	landmarkCandidates.push_back(std::make_pair(rmouth->getIdentifier(), rmouth->candidates));
	landmarkCandidates.push_back(std::make_pair(lmouth->getIdentifier(), lmouth->candidates));

	Logger->LogImgDetectorFinal(myimg, landmarkCandidates, this->getIdentifier());

	return landmarkCandidates;
}

void CascadeFacialFeaturePointsSimple::setRoiInImage(Rect roi)
{
	reye->setRoiInImage(roi);
	leye->setRoiInImage(roi);
	nosetip->setRoiInImage(roi);
	rmouth->setRoiInImage(roi);
	lmouth->setRoiInImage(roi);
}
