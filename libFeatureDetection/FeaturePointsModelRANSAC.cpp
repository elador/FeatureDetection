/*
 * FeaturePointsRANSAC.cpp
 *
 *  Created on: 22.10.2012
 *      Author: Patrik Huber
 */
#include "stdafx.h"
#include "FeaturePointsModelRANSAC.h"

#include "SLogger.h"
#include "FdImage.h"

#include "cpp/H5Cpp.h"
//#include "hdf5.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <string>

#include <ctime>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <opencv2/highgui/highgui.hpp>


FeaturePointsRANSAC::FeaturePointsRANSAC(void)
{

}


FeaturePointsRANSAC::~FeaturePointsRANSAC(void)
{

}

void FeaturePointsRANSAC::init(void)
{
	loadModel("C:\\Users\\Patrik\\Documents\\GitHub\\models\\model2012p.h5");
	loadFeaturePoints("C:\\Users\\Patrik\\Documents\\GitHub\\models\\featurePoints_head.txt");
}

void FeaturePointsRANSAC::runRANSAC(FdImage* img, std::vector<std::pair<std::string, std::vector<FdPatch*> > > landmarkData, float thresholdForDatapointFitsModel, int numIter/*=0*/, int numClosePointsRequiredForGoodFit/*=4*/, int minPointsToFitModel/*=3*/)
{
	if(numIter==0) {
		// Calculate/estimate how many iterations we should run (see Wikipedia)
		numIter = 2500;
	}numIter = 2500;
	minPointsToFitModel = 3;
	bool useSVD = false;
	thresholdForDatapointFitsModel = 30.0f;

	// Note: Doesn't seem to make sense to use more than 3 points for the initial projection. With 4 (and SVD instead of inv()), the projection of the other model-LMs is quite wrong already.
	// So: minPointsToFitModel == 3 unchangeble! (hm, really...?)

	// RANSAC general TODO: Bei Nase beruecksichtigen, dass das Gesicht nicht auf dem Kopf stehen darf? (da sonst das feature nicht gefunden wuerde, nicht symmetrisch top/bottom)

	// TODO Use 1/2 of detected face patch = IED, then discard all model proposals that are smaller than 2/3 IED or so? Make use of face-box...
	// Take face-box as probability of a face there... then also PSM output... then take regions where the FD doesn't have a response, e.g. Skin, Circles etc... and also try PSM there.

	// Build an "iterative global face-probability map" with all those detectors?
	// use additional feature spaces (gabor etc)

	// 
	// Another Idea: Use 3 detected points, project the 3DMM, then, around a possible 4th point, use a detector and search this region!
	//					(I guess I need a dynamic detector map for this :-) Attaching to Image not a good idea... (really?)... use a Master-detector or so? Hm I kind of already have that (Casc.Fac.Feat...))

	// only temporary to print the 3DMM
	std::vector<std::string> modelPointsToRender;
	modelPointsToRender.push_back("nose_tip");
	modelPointsToRender.push_back("mouth_rc");
	modelPointsToRender.push_back("mouth_lc");
	modelPointsToRender.push_back("chin");
	modelPointsToRender.push_back("reye_c");
	modelPointsToRender.push_back("leye_c");
	modelPointsToRender.push_back("rear_c");
	modelPointsToRender.push_back("lear_c");
	modelPointsToRender.push_back("mouth_ulb");
	modelPointsToRender.push_back("mouth_llb");
	modelPointsToRender.push_back("rebr_olc");
	modelPointsToRender.push_back("lebr_olc");
	modelPointsToRender.push_back("rear_ic");
	modelPointsToRender.push_back("lear_ic");
	modelPointsToRender.push_back("rear_lobe");
	modelPointsToRender.push_back("lear_lobe");
	modelPointsToRender.push_back("rear_oc");
	modelPointsToRender.push_back("lear_oc");
	modelPointsToRender.push_back("rear_uc");
	modelPointsToRender.push_back("lear_uc");

	unsigned int iterations = 0;

	boost::random::mt19937 rndGen(std::time(0));

	// Todo: Probably faster if I convert all LMs to cv::Point2i beforehands, so I only do it once.
	// Maybe I want the patch certainty later, then I need to make a std::pair<Point2i, float> or so. But the rest of the Patch I won't need for certain.

	std::vector<int> indexMap;		// This is created so each point of all landmarks together has the same probability of being chosen (eg we have more eyes -> eyes get chosen more often)
	for (unsigned int i=0; i<landmarkData.size(); ++i) {
		for (unsigned int j=0; j<landmarkData[i].second.size(); ++j) {
			indexMap.push_back(i);
		}
	}
	boost::random::uniform_int_distribution<> rndLandmarkDistr(0, indexMap.size()-1);

	// threshold around 1/3 IED?

	// some LMs wrong? nose_tip? No

	std::vector<std::vector<std::pair<std::string, cv::Point2f> >> bestConsensusSets;

	while (iterations < numIter) {
		++iterations;
		// maybe_inliers := #minPointsToFitModel randomly selected values from data
		std::vector<std::pair<std::string, FdPatch*> > maybeInliersPatches;
		unsigned int distinctFfpsChosen = 0;	// we don't need this, we could use size of alreadyChosenFfps?
		std::set<int> alreadyChosenFfps;
		while(distinctFfpsChosen < minPointsToFitModel) {

			int whichFfp = indexMap[rndLandmarkDistr(rndGen)];	// Find out which ffp to take
			boost::random::uniform_int_distribution<> rndCandidateDistr(0, landmarkData[whichFfp].second.size()-1);	// this could be taken out of the loop and only done once for each landmark-type
			int whichPoint = rndCandidateDistr(rndGen);			// Find out which detected point from the selected ffp to take
			std::string thePointLMName = landmarkData[whichFfp].first;	// The LM we took (e.g. reye_c)
			FdPatch* thePointPatch = landmarkData[whichFfp].second[whichPoint];	// The patch (with 2D-coords) we took

			// Check if we didn't already choose this FFP-type
			if(alreadyChosenFfps.find(whichFfp)==alreadyChosenFfps.end()) {
				// We didn't already choose this FFP-type:
				maybeInliersPatches.push_back(std::make_pair(thePointLMName, thePointPatch));
				alreadyChosenFfps.insert(whichFfp);
				++distinctFfpsChosen;
			}
		}
		// We got #minPointsToFitModel distinct maybe_inliers, go on
		// == maybeInliers
		// maybe_model := model parameters fitted to maybeInliers

		std::vector<std::pair<std::string, cv::Point2f> > maybeInliers;
		/*maybeInliersDbg.push_back(std::make_pair("reye_c", cv::Point2f(100.0f, 100.0f)));
		maybeInliersDbg.push_back(std::make_pair("leye_c", cv::Point2f(160.0f, 100.0f)));
		//maybeInliersDbg.push_back(std::make_pair("mouth_rc", cv::Point2f(110.0f, 180.0f)));
		maybeInliersDbg.push_back(std::make_pair("mouth_lc", cv::Point2f(150.0f, 180.0f)));*/
		for (unsigned int i=0; i<maybeInliersPatches.size(); ++i) {
			maybeInliers.push_back(std::make_pair(maybeInliersPatches[i].first, cv::Point2f((float)maybeInliersPatches[i].second->c.x, (float)maybeInliersPatches[i].second->c.y)));
		}

		LandmarksVerticesPair lmVertPairAtOrigin = getCorrespondingLandmarkpointsAtOriginFrom3DMM(maybeInliers);
		// Calculate s and t
		// Todo: Using this method, we don't check if the face is facing us or not. Somehow check this.
		//			E.g. give the LMs color, or text, or some math?
		std::pair<cv::Mat, cv::Mat> projection = calculateProjection(lmVertPairAtOrigin.landmarks, lmVertPairAtOrigin.vertices);
		cv::Mat s = projection.first;
		cv::Mat t = projection.second;

		cv::Mat outimg = img->data_matbgr.clone();
		//drawFull3DMMProjection(outimg, lmVertPairAtOrigin.verticesMean, lmVertPairAtOrigin.landmarksMean, s, t);
		//std::vector<std::pair<std::string, cv::Point2f> > pointsInImage = projectVerticesToImage(modelPointsToRender, lmVertPairAtOrigin.verticesMean, lmVertPairAtOrigin.landmarksMean, s, t);
		//drawAndPrintLandmarksInImage(outimg, pointsInImage);

		Logger->drawFfpsSmallSquare(outimg, maybeInliers);		// TODO: Should not call this directly! Make a LogImg... function!
		drawLandmarksText(outimg, maybeInliers);
		cv::imwrite("out\\a_ransac_initialPoints.png", outimg);

		// consensus_set := maybe_inliers
		std::vector<std::pair<std::string, cv::Point2f> > consensusSet = maybeInliers;

		// (TODO? get the IED of the current projection, to calculate the threshold. Hmm - what if I dont have the eye-LMs... Somehow make thresh dynamic (because of different face-sizes! absolute Pixels not good!))
		// But the FaceDetector knows the face-box size!!!

		// for every point in data not in maybe_inliers
		for (unsigned int i=0; i<indexMap.size(); ++i) {	// This could be made faster, by using i<landmarkData.size(), and then skip the whole landmark
			int whichFfp = indexMap[i];	// Find out which ffp to take
			// Check if we didn't already choose this FFP-type
			if(alreadyChosenFfps.find(whichFfp)==alreadyChosenFfps.end()) {
				// We didn't already choose this FFP-type:
				alreadyChosenFfps.insert(whichFfp);
				++distinctFfpsChosen;
				// Loop through all candidates of this FFP-type. If several fit, use the one with the lowest error.
				std::vector<double> currentFfpDistances; //TODO pre-Alloc with ...
				for (unsigned int j=0; j<landmarkData[whichFfp].second.size(); ++j) {
					// if point fits maybe_model with an error smaller than t
					// Fit the point with current model:
					std::string thePointLMName = landmarkData[whichFfp].first;	// The LM we took (e.g. reye_c)
					FdPatch* thePointPatch = landmarkData[whichFfp].second[j];	// The patch (with 2D-coords) we took
					// get "thePointLMName" vertex from 3DMM, project into 2D with s, t
					std::vector<std::string> theNewLmToProject;
					theNewLmToProject.push_back(thePointLMName);
					std::vector<std::pair<std::string, cv::Point2f> > theNewLmInImage = projectVerticesToImage(theNewLmToProject, lmVertPairAtOrigin.verticesMean, lmVertPairAtOrigin.landmarksMean, s, t);
					// error of theNewLmInImage[0] - thePointPatch < t ?
					cv::Vec2f theNewLm(theNewLmInImage[0].second.x, theNewLmInImage[0].second.y);
					cv::Vec2f theNewPointPatch((float)thePointPatch->c.x, (float)thePointPatch->c.y);
					double distance = cv::norm(theNewLm, theNewPointPatch, cv::NORM_L2);
					currentFfpDistances.push_back(distance);
				}
				std::vector<double>::iterator beforeItr = std::min_element(currentFfpDistances.begin(), currentFfpDistances.end());
				if(*beforeItr > thresholdForDatapointFitsModel) {
					// None of the candidates for the current Ffp fit the model well. Don't add anything to the consensusSet.
				} else {
					// We found a candidate that fits the model, add it to the consensusSet! (and continue with the next Ffp)
					int positionOfMinimumElement = std::distance(currentFfpDistances.begin(), beforeItr);
					FdPatch* thePointPatch = landmarkData[whichFfp].second[positionOfMinimumElement];
					cv::Point2f theNewPointPatch((float)thePointPatch->c.x, (float)thePointPatch->c.y);
					consensusSet.push_back(std::make_pair(landmarkData[whichFfp].first, theNewPointPatch));
				}
			}
		}
		// We went through all Ffp's.
		// if the number of elements in consensus_set is > d (this implies that we may have found a good model, now test how good it is)
		if (consensusSet.size()<numClosePointsRequiredForGoodFit) {
			continue;	// We didn't find a good model, the consensusSet only consists of the points projected (e.g. 3).
		}
		// this_model := model parameters fitted to all points in consensus_set

		LandmarksVerticesPair consensusSetLmVertPairAtOrigin = getCorrespondingLandmarkpointsAtOriginFrom3DMM(consensusSet);	// Todo: It's kind of not clear if this expects centered 2D LMs or not-centered (it's not-centered)
		// Todo: Using this method, we don't check if the face is facing us or not. Somehow check this.
		std::pair<cv::Mat, cv::Mat> projectionC = calculateProjection(consensusSetLmVertPairAtOrigin.landmarks, consensusSetLmVertPairAtOrigin.vertices);
		cv::Mat sc = projectionC.first;
		cv::Mat tc = projectionC.second;

		cv::Mat outimgConsensus = img->data_matbgr.clone();
		//drawFull3DMMProjection(outimgConsensus, consensusSetLmVertPairAtOrigin.verticesMean, consensusSetLmVertPairAtOrigin.landmarksMean, sc, tc);
		//std::vector<std::pair<std::string, cv::Point2f> > pointsInImageC = projectVerticesToImage(modelPointsToRender, consensusSetLmVertPairAtOrigin.verticesMean, consensusSetLmVertPairAtOrigin.landmarksMean, sc, tc);
		//drawAndPrintLandmarksInImage(outimgConsensus, pointsInImageC);

		Logger->drawFfpsSmallSquare(outimgConsensus, consensusSet);		// TODO: Should not call this directly! Make a LogImg... function!
		drawLandmarksText(outimgConsensus, consensusSet);
		cv::imwrite("out\\a_ransac_ConsensusSet.png", outimgConsensus);

		// this_error := a measure of how well this_model fits these points
		// TODO
		// we could check how many of the 3dmm LMs projected into the model with s and t are inside the actual image. If they are far off, the model is not good. (?)
		// also check this for the three initial points?
		// or better use some kind of matrix/projection/regularisation, that those transformations doesn't exist in the first place...

		// Check for number of points AND some kind of error measure?
		if(bestConsensusSets.empty()) {
			bestConsensusSets.push_back(consensusSet);
		} else {
			if(consensusSet.size()==bestConsensusSets[0].size()) {
				bestConsensusSets.push_back(consensusSet);
			} else if (consensusSet.size()>bestConsensusSets[0].size()) {
				bestConsensusSets.clear();
				bestConsensusSets.push_back(consensusSet);
			}
		}
		

		std::cout << "Finished one iteration" << std::endl;

	}

	for (unsigned int i=0; i<bestConsensusSets.size(); ++i) {
		cv::Mat output = img->data_matbgr.clone();
		drawAndPrintLandmarksInImage(output, bestConsensusSets[i]);
		Logger->drawFfpsSmallSquare(output, bestConsensusSets[i]);		// TODO: Should not call this directly! Make a LogImg... function! Make these Logger-functions private?
		//drawLandmarksText(output, bestConsensusSets[i]);
		std::stringstream sstm;
		sstm << "out\\a_ransac_ConsensusSet_" << i << ".png";
		std::string fn = sstm.str();
		cv::imwrite(fn, output);
	}

}


void FeaturePointsRANSAC::loadModel(std::string modelPath)
{
	H5::H5File h5Model;

	try {
		h5Model = H5::H5File(modelPath, H5F_ACC_RDONLY);
	}
	catch (H5::Exception& e) {
		std::string msg( std::string( "Could not open HDF5 file \n" ) + e.getCDetailMsg() );
		throw msg;
	}

	// Load the Shape
	H5::Group modelReconstructive = h5Model.openGroup("/shape/ReconstructiveModel/model");
	H5::DataSet dsMean = modelReconstructive.openDataSet("./mean");
	hsize_t dims[1];
	dsMean.getSpace().getSimpleExtentDims(dims, NULL);
	std::cout << "Dims: " << dims[0] << std::endl;		// TODO: I guess this whole part could be done A LOT better!
	float* testData = new float[dims[0]];
	dsMean.read(testData, H5::PredType::NATIVE_FLOAT);
	this->modelMeanShp.reserve(dims[0]);
	
	for (unsigned int i=0; i < dims[0]; ++i)	{
		modelMeanShp.push_back(testData[i]);
	}
	delete[] testData;
	testData = NULL;

	// // Load the Texture
	H5::Group modelReconstructiveTex = h5Model.openGroup("/color/ReconstructiveModel/model");
	H5::DataSet dsMeanTex = modelReconstructiveTex.openDataSet("./mean");
	hsize_t dimsTex[1];
	dsMeanTex.getSpace().getSimpleExtentDims(dimsTex, NULL);
	std::cout << "Dims: " << dimsTex[0] << std::endl;		// TODO: I guess this whole part could be done A LOT better!
	float* testDataTex = new float[dimsTex[0]];
	dsMeanTex.read(testDataTex, H5::PredType::NATIVE_FLOAT);
	this->modelMeanTex.reserve(dimsTex[0]);

	for (unsigned int i=0; i < dimsTex[0]; ++i)	{
		modelMeanTex.push_back(testDataTex[i]);
	}
	delete[] testDataTex;
	testDataTex = NULL;

}

void FeaturePointsRANSAC::loadFeaturePoints(std::string featurePointsFile)
{
	std::ifstream ffpList;
	ffpList.open(featurePointsFile.c_str(), std::ios::in);
	if (!ffpList.is_open()) {
		std::cout << "[FeaturePointsRANSAC] Error opening feature points file!" << std::endl;
		exit(EXIT_FAILURE);
	}
	std::string line;
	while( ffpList.good() ) {
		std::getline(ffpList, line);
		if(line=="") {
			continue;
		}
		std::string currFfp; // Have a buffer string
		int currVertex = 0;
		std::stringstream ss(line); // Insert the string into a stream
		ss >> currFfp;
		ss >> currVertex;
		//filenames.push_back(buf);
		featurePointsMap.insert(std::map<std::string, int>::value_type(currFfp, currVertex));
		currFfp.clear();
	}
	ffpList.close();
}

void FeaturePointsRANSAC::renderModel(FdImage* img)
{
	/*cv::Mat mymean(this->modelMeanTex);
	mymean = ((mymean/100000)+1)*100;
	//std::cout << mymean << std::endl;
	cv::Mat zbuffer(img->h, img->w, CV_32FC1, cv::Scalar::all(0.0f));
	unsigned int i=0;
	while (i<mymean.rows) {
		int x = (int)mymean.at<float>(i, 0);
		++i;
		int y = (int)mymean.at<float>(i, 0);
		++i;
		int z = (int)mymean.at<float>(i, 0);
		++i;
		if (y<0) {
			y=0;
		}
		if(zbuffer.at<float>(y, x)<=z) {
			cv::rectangle(img->data_matbgr, cv::Point(x, y), cv::Point(x, y), cv::Scalar(this->modelMeanTex[i-1]*255, this->modelMeanTex[i-2]*255, this->modelMeanTex[i-3]*255));
			//img->data_matbgr.at<uchar>(y, x) = 5;
			
		}
	}
	cv::imwrite("out\\aaaaa.png", img->data_matbgr);*/
}

cv::Point2f FeaturePointsRANSAC::calculateCenterpoint( std::vector<std::pair<std::string, cv::Point2f> > points )
{
	cv::Point2f center(0.0f, 0.0f);
	for (unsigned int i=0; i<points.size(); ++i) {
		center.x += points[i].second.x;
		center.y += points[i].second.y;
	}
	center.x /= (float)points.size();
	center.y /= (float)points.size();

	return center;
}

cv::Point3f FeaturePointsRANSAC::calculateCenterpoint( std::vector<std::pair<std::string, cv::Point3f> > points )
{
	cv::Point3f center(0.0f, 0.0f, 0.0f);
	for (unsigned int i=0; i<points.size(); ++i) {
		center.x += points[i].second.x;
		center.y += points[i].second.y;
		center.z += points[i].second.z;
	}
	center.x /= (float)points.size();
	center.y /= (float)points.size();
	center.z /= (float)points.size();

	return center;
}

std::vector<std::pair<std::string, cv::Point2f> > FeaturePointsRANSAC::movePointsToCenter( std::vector<std::pair<std::string, cv::Point2f> > points, cv::Point2f center )
{
	std::vector<std::pair<std::string, cv::Point2f> > centeredPoints;
	for (unsigned int i=0; i<points.size(); ++i) {
		cv::Point2f tmp(points[i].second.x-center.x, points[i].second.y-center.y);
		centeredPoints.push_back(std::make_pair(points[i].first, tmp));
	}

	return centeredPoints;
}

std::vector<std::pair<std::string, cv::Point3f> > FeaturePointsRANSAC::movePointsToCenter( std::vector<std::pair<std::string, cv::Point3f> > points, cv::Point3f center )
{
	std::vector<std::pair<std::string, cv::Point3f> > centeredPoints;
	for (unsigned int i=0; i<points.size(); ++i) {
		cv::Point3f tmp(points[i].second.x-center.x, points[i].second.y-center.y, points[i].second.z-center.z);
		centeredPoints.push_back(std::make_pair(points[i].first, tmp));
	}

	return centeredPoints;
}

void FeaturePointsRANSAC::printPointsConsole( std::string text, std::vector<std::pair<std::string, cv::Point2f> > points )
{
	std::cout << text << std::endl;
	for (unsigned int i=0; i<points.size(); ++i) {
		std::cout << points[i].first << " = " << cv::Point2f(points[i].second.x, points[i].second.y) << std::endl;
	}

}

void FeaturePointsRANSAC::printPointsConsole( std::string text, std::vector<std::pair<std::string, cv::Point3f> > points )
{
	std::cout << text << std::endl;
	for (unsigned int i=0; i<points.size(); ++i) {
		std::cout << points[i].first << " = " << cv::Point3f(points[i].second.x, points[i].second.y, points[i].second.z) << std::endl;
	}
}

std::vector<std::pair<std::string, cv::Point2f> > FeaturePointsRANSAC::projectVerticesToImage( std::vector<std::string> landmarknames, cv::Point3f centerIn3dmm, cv::Point2f centerInImage, cv::Mat s, cv::Mat t )
{
	std::vector<std::pair<std::string, cv::Point2f> > pointsInImage;

	cv::Mat pointsIn3dCentered(landmarknames.size(), 3, CV_32F);	// As many rows as Ffp's, and 3 cols (x, y, z)
	for (unsigned int i=0; i<landmarknames.size(); ++i) {
		int theVertexToGet = this->featurePointsMap[landmarknames[i]];
		pointsIn3dCentered.at<float>(i, 0) = this->modelMeanShp[3*theVertexToGet+0]-centerIn3dmm.x;
		pointsIn3dCentered.at<float>(i, 1) = this->modelMeanShp[3*theVertexToGet+1]-centerIn3dmm.y;
		pointsIn3dCentered.at<float>(i, 2) = this->modelMeanShp[3*theVertexToGet+2]-centerIn3dmm.z;
		float x = pointsIn3dCentered.row(i).t().dot(s) + centerInImage.x;
		float y = pointsIn3dCentered.row(i).t().dot(t) + centerInImage.y;
		pointsInImage.push_back(std::make_pair(landmarknames[i], cv::Point2f(x, y)));
	}

	return pointsInImage;
}

void FeaturePointsRANSAC::drawAndPrintLandmarksInImage( cv::Mat image, std::vector<std::pair<std::string, cv::Point2f> > landmarks )
{
	for (unsigned int i=0; i<landmarks.size(); ++i) {

		std::cout << landmarks[i].first << " from model projected to 2d-coords: x= " << landmarks[i].second.x << ", y= " << landmarks[i].second.y << std::endl;

		cv::Point center(cvRound(landmarks[i].second.x), cvRound(landmarks[i].second.y));
		int radius = cvRound(3);
		cv::circle(image, center, 1, cv::Scalar(0,255,0), 1, 8, 0 );	// draw the circle center
		cv::circle(image, center, radius, cv::Scalar(0,0,255), 1, 8, 0 );	// draw the circle outline

		std::ostringstream text;
		int fontFace = cv::FONT_HERSHEY_PLAIN;
		double fontScale = 0.7;
		int thickness = 1;  
		text << landmarks[i].first << std::ends;
		cv::putText(image, text.str(), center, fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
		text.str("");
	}
}

void FeaturePointsRANSAC::drawLandmarksText( cv::Mat image, std::vector<std::pair<std::string, cv::Point2f> > landmarks )
{
	for (unsigned int i=0; i<landmarks.size(); ++i) {

		cv::Point center(cvRound(landmarks[i].second.x), cvRound(landmarks[i].second.y));

		std::ostringstream text;
		int fontFace = cv::FONT_HERSHEY_PLAIN;
		double fontScale = 0.7;
		int thickness = 1;  
		text << landmarks[i].first << std::ends;
		cv::putText(image, text.str(), center, fontFace, fontScale, cv::Scalar(0, 0, 255), thickness, 8);
		text.str("");
	}
}

// TODO should this go into the Logger class?
void FeaturePointsRANSAC::drawFull3DMMProjection( cv::Mat img, cv::Point3f centerIn3dmm, cv::Point2f centerInImage, cv::Mat s, cv::Mat t )
{
	std::vector<std::pair<std::string, cv::Point2f> > pointsInImage;

	cv::Mat zbuffer(img.rows, img.cols, CV_32FC1, cv::Scalar::all(0.0f));

	cv::Mat pointsIn3dCentered(this->modelMeanShp.size()/3, 3, CV_32F);	// As many rows as Ffp's, and 3 cols (x, y, z)
	for (unsigned int i=0; i<this->modelMeanShp.size()/3; ++i) {
		int theVertexToGet = i;
		pointsIn3dCentered.at<float>(i, 0) = this->modelMeanShp[3*theVertexToGet+0]-centerIn3dmm.x;
		pointsIn3dCentered.at<float>(i, 1) = this->modelMeanShp[3*theVertexToGet+1]-centerIn3dmm.y;
		pointsIn3dCentered.at<float>(i, 2) = this->modelMeanShp[3*theVertexToGet+2]-centerIn3dmm.z;
		float x = pointsIn3dCentered.row(i).t().dot(s) + centerInImage.x;
		float y = pointsIn3dCentered.row(i).t().dot(t) + centerInImage.y;

		if(x>=0 && y>=0 && x<img.cols && y<img.rows) {
			std::stringstream sstm;
			sstm << i;
			std::string result = sstm.str();
			pointsInImage.push_back(std::make_pair(result, cv::Point2f(x, y)));

			img.at<cv::Vec3b>(y,x)[0] = (uchar)(255.0f * this->modelMeanTex[3*theVertexToGet+2]);
			img.at<cv::Vec3b>(y,x)[1] = (uchar)(255.0f * this->modelMeanTex[3*theVertexToGet+1]);
			img.at<cv::Vec3b>(y,x)[2] = (uchar)(255.0f * this->modelMeanTex[3*theVertexToGet+0]);
		}
	}

}

std::vector<std::pair<std::string, cv::Point3f> > FeaturePointsRANSAC::get3dmmLmsFromFfps( std::vector<std::pair<std::string, cv::Point2f> > ffps )
{
	std::vector<std::pair<std::string, cv::Point3f> > mmPoints;
	for (unsigned int i=0; i<ffps.size(); ++i) {
		int theVertexToGet = this->featurePointsMap[ffps[i].first];
		cv::Point3f tmp(this->modelMeanShp[3*theVertexToGet+0], this->modelMeanShp[3*theVertexToGet+1], this->modelMeanShp[3*theVertexToGet+2]); // TODO! Does this start to count at 0 or 1 ? At 0. (99.999% sure)
		mmPoints.push_back(std::make_pair(ffps[i].first, tmp));
	}

	return mmPoints;
}

std::pair<cv::Mat, cv::Mat> FeaturePointsRANSAC::calculateProjection( std::vector<std::pair<std::string, cv::Point2f> > ffpsAtOrigin, std::vector<std::pair<std::string, cv::Point3f> > mmPointsAtOrigin)
{
	// Construct V
	cv::Mat V(mmPointsAtOrigin.size(), 3, CV_32F);	// As many rows as Ffp's, and 3 cols (x, y, z)
	for (unsigned int i=0; i<mmPointsAtOrigin.size(); ++i) {
		V.at<float>(i, 0) = mmPointsAtOrigin[i].second.x;
		V.at<float>(i, 1) = mmPointsAtOrigin[i].second.y;
		V.at<float>(i, 2) = mmPointsAtOrigin[i].second.z;
	}
	// Construct wx, wy
	cv::Mat wx(mmPointsAtOrigin.size(), 1, CV_32F);
	cv::Mat wy(mmPointsAtOrigin.size(), 1, CV_32F);
	for (unsigned int i=0; i<mmPointsAtOrigin.size(); ++i) {
		wx.at<float>(i, 0) = ffpsAtOrigin[i].second.x;
		wy.at<float>(i, 0) = ffpsAtOrigin[i].second.y;
	}

	// TODO: check if matrix square. If not, use another cv::invert(...) instead of .inv().
	bool useSVD = true;
	if(V.rows==V.cols)
		useSVD = false;

	std::cout <<  "wx = " << wx << std::endl;
	std::cout <<  "wy = " << wy << std::endl;
	std::cout <<  "V = " << V << std::endl;

	cv::Mat s;
	cv::Mat t;

	if(!useSVD) {
		std::cout << "det(V) = " << cv::determinant(V) << std::endl;	// Det() of non-symmetric matrix?
		std::cout <<  "V.inv() = " << V.inv() << std::endl;
		s = V.inv() * wx;
		t = V.inv() * wy;
	} else {
		cv::Mat Vinv;
		cv::invert(V, Vinv, cv::DECOMP_SVD);
		std::cout <<  "invert(V) with SVD = " << Vinv << std::endl;
		s = Vinv * wx;
		t = Vinv * wy;
	}

	std::cout <<  "s = " << s << std::endl;
	std::cout <<  "t = " << t << std::endl;

	for(unsigned int i=0; i<V.rows; ++i) {
		std::cout <<  "<v|s> = wx? " << V.row(i).t().dot(s) << " ?= " << wx.row(i) << std::endl;
		std::cout <<  "<v|t> = wy? " << V.row(i).t().dot(t) << " ?= " << wy.row(i) << std::endl;
	}

	return std::make_pair(s, t);
}

FeaturePointsRANSAC::LandmarksVerticesPair FeaturePointsRANSAC::getCorrespondingLandmarkpointsAtOriginFrom3DMM(std::vector<std::pair<std::string, cv::Point2f> > landmarks)
{

	LandmarksVerticesPair out;

	printPointsConsole("Input: FeaturePoints 2D:", landmarks);

	// Calculate the center of the maybeInliers
	out.landmarksMean = calculateCenterpoint(landmarks);

	// Move the Ffps around (0, 0)
	out.landmarks = movePointsToCenter(landmarks, out.landmarksMean);
	printPointsConsole("FeaturePoints 2D origin:", out.landmarks);

	// Get the #minPointsToFitModel feature points from the 3DMM
	std::vector<std::pair<std::string, cv::Point3f> > landmarkVerticesIn3DMM = get3dmmLmsFromFfps(out.landmarks);
	printPointsConsole("3DMM points 3D:", landmarkVerticesIn3DMM);

	out.verticesMean = calculateCenterpoint(landmarkVerticesIn3DMM);	// The center of the initial 3DMM points

	// Move the 3DMM initial points around (0, 0, 0)
	out.vertices = movePointsToCenter(landmarkVerticesIn3DMM, out.verticesMean);
	printPointsConsole("3DMM points 3D origin:", out.vertices);

	return out;
}		
