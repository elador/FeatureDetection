/*
 * FeaturePointsRANSAC.cpp
 *
 *  Created on: 22.10.2012
 *      Author: Patrik Huber
 */
#include "shapemodels/FeaturePointsModelRANSACtmp.hpp"

#include "logging/LoggerFactory.hpp"

#include "H5Cpp.h"

#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int_distribution.hpp"

#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <string>
#include <set>
#include <ctime>

using logging::LoggerFactory;
using cv::Point;
using cv::Scalar;
using std::set;
using std::make_pair;

using std::cout; // TODO REMOVE!
using std::endl;

namespace shapemodels {

void drawFfpSmallSquare(Mat img, pair<string, Point2f> point)
{
	cv::rectangle(img, Point(cvRound(point.second.x-2.0f), cvRound(point.second.y-2.0f)), Point(cvRound(point.second.x+2.0f), cvRound(point.second.y+2.0f)), Scalar(255, 0, 0));
}

void drawFfpsSmallSquare(Mat img, vector<pair<string, Point2f>> points)
{
	for (unsigned int i=0; i<points.size(); ++i) {
		drawFfpSmallSquare(img, points[i]);
	}
}


void FeaturePointsRANSAC::runRANSAC(Mat img, vector<pair<string, vector<shared_ptr<imageprocessing::Patch>>>> landmarkData, MorphableModel mm, float thresholdForDatapointFitsModel, int numIter/*=0*/, int numClosePointsRequiredForGoodFit/*=4*/, int minPointsToFitModel/*=3*/)
{
	if(numIter==0) {
		// Calculate/estimate how many iterations we should run (see Wikipedia)
		numIter = 2500;
	}
	numIter = 50;
	minPointsToFitModel = 3;
	bool useSVD = false;
	thresholdForDatapointFitsModel = 30.0f;

	// Note: Doesn't seem to make sense to use more than 3 points for the initial projection. With 4 (and SVD instead of inv()), the projection of the other model-LMs is quite wrong already.
	// So: minPointsToFitModel == 3 unchangeable! (hm, really...?)

	// RANSAC general TODO: Bei Nase beruecksichtigen, dass das Gesicht nicht auf dem Kopf stehen darf? (da sonst das feature nicht gefunden wuerde, nicht symmetrisch top/bottom)

	// TODO Use 1/2 of detected face patch = IED, then discard all model proposals that are smaller than 2/3 IED or so? Make use of face-box...
	// Take face-box as probability of a face there... then also PSM output... then take regions where the FD doesn't have a response, e.g. Skin, Circles etc... and also try PSM there.

	// Build an "iterative global face-probability map" with all those detectors?
	// use additional feature spaces (gabor etc)

	// 
	// Another Idea: Use 3 detected points, project the 3DMM, then, around a possible 4th point, use a detector and search this region!
	//					(I guess I need a dynamic detector map for this :-) Attaching to Image not a good idea... (really?)... use a Master-detector or so? Hm I kind of already have that (Casc.Fac.Feat...))

	// only temporary to print the 3DMM
	vector<string> modelPointsToRender;
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

	//boost::random::mt19937 rndGen(time(0));	// Pseudo-random number generators should not be constructed (initialized) frequently during program execution, for two reasons... see boost doc. But this is not a problem as long as we don't call runRANSAC too often (e.g. every frame).
	boost::random::mt19937 rndGen(1); // TODO: Use C++11

	// Todo: Probably faster if I convert all LMs to Point2i beforehands, so I only do it once.
	// Maybe I want the patch certainty later, then I need to make a pair<Point2i, float> or so. But the rest of the Patch I won't need for certain.

	vector<int> indexMap;		// This is created so each point of all landmarks together has the same probability of being chosen (eg we have more eyes -> eyes get chosen more often)
	for (unsigned int i=0; i<landmarkData.size(); ++i) {
		for (unsigned int j=0; j<landmarkData[i].second.size(); ++j) {
			indexMap.push_back(i);
		}
	}
	boost::random::uniform_int_distribution<> rndLandmarkDistr(0, indexMap.size()-1);

	// threshold around 1/3 IED?

	// some LMs wrong? nose_tip? No

	vector<vector<pair<string, Point2f>>> bestConsensusSets;
	
	while (iterations < numIter) {
		++iterations;
		// maybe_inliers := #minPointsToFitModel randomly selected values from data
		vector<pair<string, shared_ptr<imageprocessing::Patch> > > maybeInliersPatches; // CHANGED TO PATCH
		unsigned int distinctFfpsChosen = 0;	// we don't need this, we could use size of alreadyChosenFfps?
		set<int> alreadyChosenFfps;
		while(distinctFfpsChosen < minPointsToFitModel) {

			int whichFfp = indexMap[rndLandmarkDistr(rndGen)];	// Find out which ffp to take
			boost::random::uniform_int_distribution<> rndCandidateDistr(0, landmarkData[whichFfp].second.size()-1);	// this could be taken out of the loop and only done once for each landmark-type
			int whichPoint = rndCandidateDistr(rndGen);			// Find out which detected point from the selected ffp to take
			string thePointLMName = landmarkData[whichFfp].first;	// The LM we took (e.g. reye_c)
			shared_ptr<imageprocessing::Patch> thePointPatch = landmarkData[whichFfp].second[whichPoint];	// The patch (with 2D-coords) we took // CHANGED TO PATCH

			// Check if we didn't already choose this FFP-type
			if(alreadyChosenFfps.find(whichFfp)==alreadyChosenFfps.end()) {
				// We didn't already choose this FFP-type:
				maybeInliersPatches.push_back(make_pair(thePointLMName, thePointPatch));
				alreadyChosenFfps.insert(whichFfp);
				++distinctFfpsChosen;
			}
		}
		// We got #minPointsToFitModel distinct maybe_inliers, go on
		// == maybeInliers
		// maybe_model := model parameters fitted to maybeInliers

		vector<pair<string, Point2f> > maybeInliers;
		/*maybeInliersDbg.push_back(make_pair("reye_c", Point2f(100.0f, 100.0f)));
		maybeInliersDbg.push_back(make_pair("leye_c", Point2f(160.0f, 100.0f)));
		//maybeInliersDbg.push_back(make_pair("mouth_rc", Point2f(110.0f, 180.0f)));
		maybeInliersDbg.push_back(make_pair("mouth_lc", Point2f(150.0f, 180.0f)));*/
		for (unsigned int i=0; i<maybeInliersPatches.size(); ++i) {
			maybeInliers.push_back(make_pair(maybeInliersPatches[i].first, Point2f((float)maybeInliersPatches[i].second->getX(), (float)maybeInliersPatches[i].second->getY())));
		}

		LandmarksVerticesPair lmVertPairAtOrigin = getCorrespondingLandmarkpointsAtOriginFrom3DMM(maybeInliers, mm);
		// Calculate s and t
		// Todo: Using this method, we don't check if the face is facing us or not. Somehow check this.
		//			E.g. give the LMs color, or text, or some math?
		pair<Mat, Mat> projection = calculateProjection(lmVertPairAtOrigin.landmarks, lmVertPairAtOrigin.vertices);
		Mat s = projection.first;
		Mat t = projection.second;

		Mat outimg = img.clone(); // CHANGED 2013
		//drawFull3DMMProjection(outimg, lmVertPairAtOrigin.verticesMean, lmVertPairAtOrigin.landmarksMean, s, t);
		//vector<pair<string, Point2f> > pointsInImage = projectVerticesToImage(modelPointsToRender, lmVertPairAtOrigin.verticesMean, lmVertPairAtOrigin.landmarksMean, s, t);
		//drawAndPrintLandmarksInImage(outimg, pointsInImage);

		// TODO COMMENTED 2013
		//Logger->drawFfpsSmallSquare(outimg, maybeInliers);		// TODO: Should not call this directly! Make a LogImg... function!
		//drawLandmarksText(outimg, maybeInliers);
		//imwrite("out\\a_ransac_initialPoints.png", outimg);
		// END

		// consensus_set := maybe_inliers
		vector<pair<string, Point2f> > consensusSet = maybeInliers;

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
				vector<double> currentFfpDistances; //TODO pre-Alloc with ...
				for (unsigned int j=0; j<landmarkData[whichFfp].second.size(); ++j) {
					// if point fits maybe_model with an error smaller than t
					// Fit the point with current model:
					string thePointLMName = landmarkData[whichFfp].first;	// The LM we took (e.g. reye_c)
					shared_ptr<imageprocessing::Patch> thePointPatch = landmarkData[whichFfp].second[j];	// The patch (with 2D-coords) we took // // CHANGED TO PATCH
					// get "thePointLMName" vertex from 3DMM, project into 2D with s, t
					vector<string> theNewLmToProject;
					theNewLmToProject.push_back(thePointLMName);
					vector<pair<string, Point2f> > theNewLmInImage = projectVerticesToImage(theNewLmToProject, lmVertPairAtOrigin.verticesMean, lmVertPairAtOrigin.landmarksMean, s, t, mm);
					// error of theNewLmInImage[0] - thePointPatch < t ?
					cv::Vec2f theNewLm(theNewLmInImage[0].second.x, theNewLmInImage[0].second.y);
					cv::Vec2f theNewPointPatch((float)thePointPatch->getX(), (float)thePointPatch->getY());
					double distance = cv::norm(theNewLm, theNewPointPatch, cv::NORM_L2);
					currentFfpDistances.push_back(distance);
				}
				vector<double>::iterator beforeItr = min_element(currentFfpDistances.begin(), currentFfpDistances.end());
				if(*beforeItr > thresholdForDatapointFitsModel) {
					// None of the candidates for the current Ffp fit the model well. Don't add anything to the consensusSet.
				} else {
					// We found a candidate that fits the model, add it to the consensusSet! (and continue with the next Ffp)
					int positionOfMinimumElement = distance(currentFfpDistances.begin(), beforeItr);
					shared_ptr<imageprocessing::Patch> thePointPatch = landmarkData[whichFfp].second[positionOfMinimumElement]; // CHANGED TO PATCH
					Point2f theNewPointPatch((float)thePointPatch->getX(), (float)thePointPatch->getY());
					consensusSet.push_back(make_pair(landmarkData[whichFfp].first, theNewPointPatch));
				}
			}
		}
		// We went through all Ffp's.
		// if the number of elements in consensus_set is > d (this implies that we may have found a good model, now test how good it is)
		if (consensusSet.size()<numClosePointsRequiredForGoodFit) {
			continue;	// We didn't find a good model, the consensusSet only consists of the points projected (e.g. 3).
		}
		// this_model := model parameters fitted to all points in consensus_set

		LandmarksVerticesPair consensusSetLmVertPairAtOrigin = getCorrespondingLandmarkpointsAtOriginFrom3DMM(consensusSet, mm);	// Todo: It's kind of not clear if this expects centered 2D LMs or not-centered (it's not-centered)
		// Todo: Using this method, we don't check if the face is facing us or not. Somehow check this.
		pair<Mat, Mat> projectionC = calculateProjection(consensusSetLmVertPairAtOrigin.landmarks, consensusSetLmVertPairAtOrigin.vertices);
		Mat sc = projectionC.first;
		Mat tc = projectionC.second;

		Mat outimgConsensus = img.clone(); // TODO 2013 changed
		drawFull3DMMProjection(outimgConsensus, consensusSetLmVertPairAtOrigin.verticesMean, consensusSetLmVertPairAtOrigin.landmarksMean, sc, tc, mm);
		vector<pair<string, Point2f> > pointsInImageC = projectVerticesToImage(modelPointsToRender, consensusSetLmVertPairAtOrigin.verticesMean, consensusSetLmVertPairAtOrigin.landmarksMean, sc, tc, mm);
		drawAndPrintLandmarksInImage(outimgConsensus, pointsInImageC);

		// TODO 2013 CHANGED
		drawFfpsSmallSquare(outimgConsensus, consensusSet);		// TODO: Should not call this directly! Make a LogImg... function!
		drawLandmarksText(outimgConsensus, consensusSet);
		imwrite("out\\a_ransac_ConsensusSet.png", outimgConsensus);
		// END

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
		
		Loggers->getLogger("shapemodels").debug("Finished one iteration.");
	}

	// Hmm, the bestConsensusSets is not unique? Doubled sets? (print them?) Write a comparator for two landmarks (make a landmark class?), then for two LM-sets, etc.
	for (unsigned int i=0; i<bestConsensusSets.size(); ++i) {
		// TODO 2013 CHANGED
		//Mat output = img->data_matbgr.clone();
		//drawAndPrintLandmarksInImage(output, bestConsensusSets[i]);
		//Logger->drawFfpsSmallSquare(output, bestConsensusSets[i]);		// TODO: Should not call this directly! Make a LogImg... function! Make these Logger-functions private?
		//drawLandmarksText(output, bestConsensusSets[i]);
		//stringstream sstm;
		//sstm << "out\\a_ransac_ConsensusSet_" << i << ".png";
		//string fn = sstm.str();
		//imwrite(fn, output);
		// END
	}
	for (unsigned int i=0; i<bestConsensusSets.size(); ++i) {
		for (unsigned int j=0; j<bestConsensusSets[i].size(); ++j) {
			cout << "[" << bestConsensusSets[i][j].first << " " << bestConsensusSets[i][j].second  << "], ";
		}
		cout << endl;
	}

}



void FeaturePointsRANSAC::renderModel(Mat img)
{
	/*Mat mymean(this->modelMeanTex);
	mymean = ((mymean/100000)+1)*100;
	//cout << mymean << endl;
	Mat zbuffer(img->h, img->w, CV_32FC1, Scalar::all(0.0f));
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
			rectangle(img->data_matbgr, Point(x, y), Point(x, y), Scalar(this->modelMeanTex[i-1]*255, this->modelMeanTex[i-2]*255, this->modelMeanTex[i-3]*255));
			//img->data_matbgr.at<uchar>(y, x) = 5;
			
		}
	}
	imwrite("out\\aaaaa.png", img->data_matbgr);*/
}

Point2f FeaturePointsRANSAC::calculateCenterpoint(vector<pair<string, Point2f> > points)
{
	Point2f center(0.0f, 0.0f);
	for (unsigned int i=0; i<points.size(); ++i) {
		center.x += points[i].second.x;
		center.y += points[i].second.y;
	}
	center.x /= (float)points.size();
	center.y /= (float)points.size();

	return center;
}

Point3f FeaturePointsRANSAC::calculateCenterpoint(vector<pair<string, Point3f> > points)
{
	Point3f center(0.0f, 0.0f, 0.0f);
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

vector<pair<string, Point2f> > FeaturePointsRANSAC::movePointsToCenter(vector<pair<string, Point2f> > points, Point2f center)
{
	vector<pair<string, Point2f> > centeredPoints;
	for (unsigned int i=0; i<points.size(); ++i) {
		Point2f tmp(points[i].second.x-center.x, points[i].second.y-center.y);
		centeredPoints.push_back(make_pair(points[i].first, tmp));
	}

	return centeredPoints;
}

vector<pair<string, Point3f> > FeaturePointsRANSAC::movePointsToCenter(vector<pair<string, Point3f> > points, Point3f center)
{
	vector<pair<string, Point3f> > centeredPoints;
	for (unsigned int i=0; i<points.size(); ++i) {
		Point3f tmp(points[i].second.x-center.x, points[i].second.y-center.y, points[i].second.z-center.z);
		centeredPoints.push_back(make_pair(points[i].first, tmp));
	}

	return centeredPoints;
}

void FeaturePointsRANSAC::printPointsConsole(string text, vector<pair<string, Point2f> > points)
{
	cout << text << endl;
	for (unsigned int i=0; i<points.size(); ++i) {
		cout << points[i].first << " = " << Point2f(points[i].second.x, points[i].second.y) << endl;
	}

}

void FeaturePointsRANSAC::printPointsConsole(string text, vector<pair<string, Point3f> > points)
{
	cout << text << endl;
	for (unsigned int i=0; i<points.size(); ++i) {
		cout << points[i].first << " = " << Point3f(points[i].second.x, points[i].second.y, points[i].second.z) << endl;
	}
}

vector<pair<string, Point2f> > FeaturePointsRANSAC::projectVerticesToImage(vector<string> landmarknames, Point3f centerIn3dmm, Point2f centerInImage, Mat s, Mat t, MorphableModel mm)
{
	vector<pair<string, Point2f> > pointsInImage;
/*
	Mat pointsIn3dCentered(landmarknames.size(), 3, CV_32F);	// As many rows as Ffp's, and 3 cols (x, y, z)
	for (unsigned int i=0; i<landmarknames.size(); ++i) {
		int theVertexToGet = mm.getShapeModel().getFeaturePointsMap()[landmarknames[i]];
		pointsIn3dCentered.at<float>(i, 0) = mm.getShapeModel().getMean()[3*theVertexToGet+0]-centerIn3dmm.x;
		pointsIn3dCentered.at<float>(i, 1) = mm.getShapeModel().getMean()[3*theVertexToGet+1]-centerIn3dmm.y;
		pointsIn3dCentered.at<float>(i, 2) = mm.getShapeModel().getMean()[3*theVertexToGet+2]-centerIn3dmm.z;
		float x = pointsIn3dCentered.row(i).t().dot(s) + centerInImage.x;
		float y = pointsIn3dCentered.row(i).t().dot(t) + centerInImage.y;
		pointsInImage.push_back(make_pair(landmarknames[i], Point2f(x, y)));
	}
*/
	return pointsInImage;
}

void FeaturePointsRANSAC::drawAndPrintLandmarksInImage(Mat image, vector<pair<string, Point2f> > landmarks)
{
	for (unsigned int i=0; i<landmarks.size(); ++i) {

		cout << landmarks[i].first << " from model projected to 2d-coords: x= " << landmarks[i].second.x << ", y= " << landmarks[i].second.y << endl;

		Point center(cvRound(landmarks[i].second.x), cvRound(landmarks[i].second.y));
		int radius = cvRound(3);
		circle(image, center, 1, Scalar(0,255,0), 1, 8, 0 );	// draw the circle center
		circle(image, center, radius, Scalar(0,0,255), 1, 8, 0 );	// draw the circle outline

		std::ostringstream text;
		int fontFace = cv::FONT_HERSHEY_PLAIN;
		double fontScale = 0.7;
		int thickness = 1;  
		text << landmarks[i].first << std::ends;
		putText(image, text.str(), center, fontFace, fontScale, Scalar::all(0), thickness, 8);
		text.str("");
	}
}

void FeaturePointsRANSAC::drawLandmarksText(Mat image, vector<pair<string, Point2f> > landmarks)
{
	for (unsigned int i=0; i<landmarks.size(); ++i) {

		Point center(cvRound(landmarks[i].second.x), cvRound(landmarks[i].second.y));

		std::ostringstream text;
		int fontFace = cv::FONT_HERSHEY_PLAIN;
		double fontScale = 0.7;
		int thickness = 1;  
		text << landmarks[i].first << std::ends;
		putText(image, text.str(), center, fontFace, fontScale, Scalar(0, 0, 255), thickness, 8);
		text.str("");
	}
}

// TODO should this go into the Logger class?
void FeaturePointsRANSAC::drawFull3DMMProjection(Mat img, Point3f centerIn3dmm, Point2f centerInImage, Mat s, Mat t, MorphableModel mm)
{
	vector<pair<string, Point2f> > pointsInImage;
/*
	Mat zbuffer(img.rows, img.cols, CV_32FC1, Scalar::all(0.0f));
	
	Mat pointsIn3dCentered(mm.getShapeModel().getMean().size()/3, 3, CV_32F);	// As many rows as Ffp's, and 3 cols (x, y, z)
	for (unsigned int i=0; i < mm.getShapeModel().getMean().size()/3; ++i) {
		int theVertexToGet = i;
		pointsIn3dCentered.at<float>(i, 0) = mm.getShapeModel().getMean()[3*theVertexToGet+0]-centerIn3dmm.x;
		pointsIn3dCentered.at<float>(i, 1) = mm.getShapeModel().getMean()[3*theVertexToGet+1]-centerIn3dmm.y;
		pointsIn3dCentered.at<float>(i, 2) = mm.getShapeModel().getMean()[3*theVertexToGet+2]-centerIn3dmm.z;
		float x = pointsIn3dCentered.row(i).t().dot(s) + centerInImage.x;
		float y = pointsIn3dCentered.row(i).t().dot(t) + centerInImage.y;

		if(x>=0 && y>=0 && x<img.cols && y<img.rows) {
			std::stringstream sstm;
			sstm << i;
			string result = sstm.str();
			pointsInImage.push_back(make_pair(result, Point2f(x, y)));

			img.at<cv::Vec3b>(y,x)[0] = (uchar)(255.0f * mm.getShapeModel().getMean()[3*theVertexToGet+2]);
			img.at<cv::Vec3b>(y,x)[1] = (uchar)(255.0f * mm.getShapeModel().getMean()[3*theVertexToGet+1]);
			img.at<cv::Vec3b>(y,x)[2] = (uchar)(255.0f * mm.getShapeModel().getMean()[3*theVertexToGet+0]);
		}
	}
*/	
}

vector<pair<string, Point3f> > FeaturePointsRANSAC::get3dmmLmsFromFfps(vector<pair<string, Point2f>> ffps, MorphableModel mm)
{
	vector<pair<string, Point3f> > mmPoints;
/*	
	for (unsigned int i=0; i<ffps.size(); ++i) {
		int theVertexToGet = mm.getShapeModel().getFeaturePointsMap()[ffps[i].first];
		Point3f tmp(mm.getShapeModel().getMean()[3*theVertexToGet+0], mm.getShapeModel().getMean()[3*theVertexToGet+1], mm.getShapeModel().getMean()[3*theVertexToGet+2]); // TODO! Does this start to count at 0 or 1 ? At 0. (99.999% sure)
		mmPoints.push_back(make_pair(ffps[i].first, tmp));
	}
*/
	return mmPoints;
}

pair<Mat, Mat> FeaturePointsRANSAC::calculateProjection(vector<pair<string, Point2f> > ffpsAtOrigin, vector<pair<string, Point3f> > mmPointsAtOrigin)
{
	// Construct V
	Mat V(mmPointsAtOrigin.size(), 3, CV_32F);	// As many rows as Ffp's, and 3 cols (x, y, z)
	for (unsigned int i=0; i<mmPointsAtOrigin.size(); ++i) {
		V.at<float>(i, 0) = mmPointsAtOrigin[i].second.x;
		V.at<float>(i, 1) = mmPointsAtOrigin[i].second.y;
		V.at<float>(i, 2) = mmPointsAtOrigin[i].second.z;
	}
	// Construct wx, wy
	Mat wx(mmPointsAtOrigin.size(), 1, CV_32F);
	Mat wy(mmPointsAtOrigin.size(), 1, CV_32F);
	for (unsigned int i=0; i<mmPointsAtOrigin.size(); ++i) {
		wx.at<float>(i, 0) = ffpsAtOrigin[i].second.x;
		wy.at<float>(i, 0) = ffpsAtOrigin[i].second.y;
	}

	// TODO: check if matrix square. If not, use another invert(...) instead of .inv().
	bool useSVD = true;
	if(V.rows==V.cols)
		useSVD = false;

	cout <<  "wx = " << wx << endl;
	cout <<  "wy = " << wy << endl;
	cout <<  "V = " << V << endl;

	Mat s;
	Mat t;

	if(!useSVD) {
		cout << "det(V) = " << determinant(V) << endl;	// Det() of non-symmetric matrix?
		cout <<  "V.inv() = " << V.inv() << endl;
		s = V.inv() * wx;
		t = V.inv() * wy;
	} else {
		Mat Vinv;
		invert(V, Vinv, cv::DECOMP_SVD);
		cout <<  "invert(V) with SVD = " << Vinv << endl;
		s = Vinv * wx;
		t = Vinv * wy;
	}

	cout <<  "s = " << s << endl;
	cout <<  "t = " << t << endl;

	for(unsigned int i=0; i<V.rows; ++i) {
		cout <<  "<v|s> = wx? " << V.row(i).t().dot(s) << " ?= " << wx.row(i) << endl;
		cout <<  "<v|t> = wy? " << V.row(i).t().dot(t) << " ?= " << wy.row(i) << endl;
	}

	return make_pair(s, t);
}

FeaturePointsRANSAC::LandmarksVerticesPair FeaturePointsRANSAC::getCorrespondingLandmarkpointsAtOriginFrom3DMM(vector<pair<string, Point2f> > landmarks, MorphableModel mm)
{

	LandmarksVerticesPair out;

	printPointsConsole("Input: FeaturePoints 2D:", landmarks);

	// Calculate the center of the maybeInliers
	out.landmarksMean = calculateCenterpoint(landmarks);

	// Move the Ffps around (0, 0)
	out.landmarks = movePointsToCenter(landmarks, out.landmarksMean);
	printPointsConsole("FeaturePoints 2D origin:", out.landmarks);

	// Get the #minPointsToFitModel feature points from the 3DMM
	vector<pair<string, Point3f> > landmarkVerticesIn3DMM = get3dmmLmsFromFfps(out.landmarks, mm);
	printPointsConsole("3DMM points 3D:", landmarkVerticesIn3DMM);

	out.verticesMean = calculateCenterpoint(landmarkVerticesIn3DMM);	// The center of the initial 3DMM points

	// Move the 3DMM initial points around (0, 0, 0)
	out.vertices = movePointsToCenter(landmarkVerticesIn3DMM, out.verticesMean);
	printPointsConsole("3DMM points 3D origin:", out.vertices);

	return out;
}		

}