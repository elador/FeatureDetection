#include "stdafx.h"
#include "CascadeFacialFeaturePointsSimple.h"

#include "SLogger.h"


CascadeFacialFeaturePointsSimple::CascadeFacialFeaturePointsSimple(void)
{
	// WE STICK TO THE TERMINOLOGY USED THROUGHOUT THE 3DMM. LEFT MOUTH MEANS LEFT FROM THE PERSON IN THE PICTURE!
	// SO ALL CLASSIFIERS MUST BE SWAPPED! ("rm" in MR means "right" from observer's view. "mouth_rc" in 3DMM means "right" from the person in the image!)
	CascadeWvmOeSvmOe* reye = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_re.mat");
	reye->setIdentifier("leye_c");
	this->detectors.insert(std::make_pair(reye->getIdentifier(), reye));	// The map handles the memory!
	CascadeWvmOeSvmOe* leye = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_le.mat");
	leye->setIdentifier("reye_c");
	this->detectors.insert(std::make_pair(leye->getIdentifier(), leye));
	CascadeWvmOeSvmOe* nosetip = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_nt.mat");
	nosetip->setIdentifier("nose_tip");
	this->detectors.insert(std::make_pair(nosetip->getIdentifier(), nosetip));
	CascadeWvmOeSvmOe* rmouth = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_rm.mat");
	rmouth->setIdentifier("mouth_lc");
	this->detectors.insert(std::make_pair(rmouth->getIdentifier(), rmouth));
	CascadeWvmOeSvmOe* lmouth = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_lm.mat");
	lmouth->setIdentifier("mouth_rc");
	this->detectors.insert(std::make_pair(lmouth->getIdentifier(), lmouth));

	CascadeWvmOeSvmOe* reyeOc = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_lx.mat");
	reyeOc->setIdentifier("reye_oc");
	this->detectors.insert(std::make_pair(reyeOc->getIdentifier(), reyeOc));
	CascadeWvmOeSvmOe* leyeOc = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_rx.mat");
	leyeOc->setIdentifier("leye_oc");
	this->detectors.insert(std::make_pair(leyeOc->getIdentifier(), leyeOc));
	CascadeWvmOeSvmOe* mouthMidUp = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_ls.mat");
	mouthMidUp->setIdentifier("mouth_ulb");
	this->detectors.insert(std::make_pair(mouthMidUp->getIdentifier(), mouthMidUp));
	CascadeWvmOeSvmOe* rnose = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_lb.mat");
	rnose->setIdentifier("nose_rc");
	this->detectors.insert(std::make_pair(rnose->getIdentifier(), rnose));
	CascadeWvmOeSvmOe* lnose = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_rb.mat");
	lnose->setIdentifier("nose_lc");
	this->detectors.insert(std::make_pair(lnose->getIdentifier(), lnose));
	CascadeWvmOeSvmOe* rear = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_la.mat");
	rear->setIdentifier("rear_c");
	this->detectors.insert(std::make_pair(rear->getIdentifier(), rear));
	CascadeWvmOeSvmOe* lear = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_ra.mat");
	lear->setIdentifier("lear_c");
	this->detectors.insert(std::make_pair(lear->getIdentifier(), lear));

}


CascadeFacialFeaturePointsSimple::~CascadeFacialFeaturePointsSimple(void)
{
	for(std::map<std::string, CascadeWvmOeSvmOe*>::iterator it = detectors.begin(); it != detectors.end();) {
		delete it->second; it->second = NULL;
		detectors.erase(it++);
	}
}

int CascadeFacialFeaturePointsSimple::initForImage(FdImage* myimg)
{
	for(std::map<std::string, CascadeWvmOeSvmOe*>::iterator it = detectors.begin(); it != detectors.end(); ++it) {
		it->second->initForImage(myimg);
	}
	return 1;
}

std::vector<std::pair<std::string, std::vector<FdPatch*> > > CascadeFacialFeaturePointsSimple::detectOnImage(FdImage* myimg)
{
	std::vector<std::pair<std::string, std::vector<FdPatch*> > > landmarkCandidates;

	for(std::map<std::string, CascadeWvmOeSvmOe*>::iterator it = detectors.begin(); it != detectors.end(); ++it) {
		it->second->detectOnImage(myimg);
		landmarkCandidates.push_back(std::make_pair(it->second->getIdentifier(), it->second->candidates));
	}

	Logger->LogImgDetectorFinal(myimg, landmarkCandidates, this->getIdentifier());

	return landmarkCandidates;
}

void CascadeFacialFeaturePointsSimple::setRoiInImage(Rect roi)
{
	for(std::map<std::string, CascadeWvmOeSvmOe*>::iterator it = detectors.begin(); it != detectors.end(); ++it) {
		it->second->setRoiInImage(roi);
	}
}
