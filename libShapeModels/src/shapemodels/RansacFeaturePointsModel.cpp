/*
 * RansacPositFeaturePointsModel.cpp
 *
 *  Created on: 03.10.2013
 *      Author: Patrik Huber
 */
#include "shapemodels/RansacFeaturePointsModel.hpp"

#include "logging/LoggerFactory.hpp"

#include <random>
#include <set>
#include <iostream>

using logging::LoggerFactory;
using cv::Point2f;
using std::pair;
using std::string;
using std::set;
using std::make_pair;

namespace shapemodels {

map<string, shared_ptr<imageprocessing::Patch>> RansacFeaturePointsModel::run(Mat img, float thresholdForDatapointFitsModel/*=30.0f*/, int numIter/*=0*/, int numClosePointsRequiredForGoodFit/*=4*/, int minPointsToFitModel/*=3*/)
{
	//The whole algorithm here. evaluator only evaluates 1 combo. All logic here. Selector provides functions to get random items (1 or several) etc.

	//map<string, vector<shared_ptr<imageprocessing::Patch>>> featureSelection;
	//do {
	//featureSelection.clear(); // Not necessary?
	//	featureSelection = selector->select();
	//} while (evaluator->evaluate(featureSelection));
	
	numIter = 50;
	minPointsToFitModel = 4;
	bool useSVD = false; // => to eval.
	thresholdForDatapointFitsModel = 30.0f; // same ?

	unsigned int iterations = 0;
	vector<vector<pair<string, Point2f>>> bestConsensusSets;

	while (iterations < numIter) {
		++iterations;

		//set<string> alreadyChosenFfps;
		map<string, vector<shared_ptr<imageprocessing::Patch>>> landmarks = selector->getAllLandmarks();

		// maybe_inliers := #minPointsToFitModel randomly selected values from data
		map<string, shared_ptr<imageprocessing::Patch>> maybeInliersPatches = selector->getDistinctRandomPoints(minPointsToFitModel);

		for (const auto& l : maybeInliersPatches) {
			landmarks.erase(l.first);
		}

		// We got #minPointsToFitModel distinct maybe_inliers, go on
		// == maybeInliers
		// maybe_model := model parameters fitted to maybeInliers
		// Use the 3DMM and POSIT:
		pair<Mat, Mat> transRot = evaluator->evaluate(maybeInliersPatches, img);

		vector<pair<string, Point2f>> maybeInliers;

		for (const auto& p : maybeInliersPatches) {
			maybeInliers.push_back(make_pair(p.first, Point2f((float)p.second->getX(), (float)p.second->getY())));
			//alreadyChosenFfps.insert(p.first);
		}

		// output the points, draw the model?

		// consensus_set := maybe_inliers
		vector<pair<string, Point2f>> consensusSet = maybeInliers;

		// (TODO? get the IED of the current projection, to calculate the threshold. Hmm - what if I dont have the eye-LMs... Somehow make thresh dynamic (because of different face-sizes! absolute Pixels not good!))
		// But the FaceDetector knows the face-box size!!!

		

		// for every point in data not in maybe_inliers
		for (const auto& l : landmarks) { // better: choose randomly

			// Loop through all candidates of this FFP-type. If several fit, use the one with the lowest error.
			vector<double> currentFfpDistances; //TODO pre-Alloc with ...
			for (const auto& p : l.second) {
				// if point fits maybe_model with an error smaller than t
				// Fit the point with current model:
				string thePointLMName = l.first;	// The LM we took (e.g. reye_c)
				shared_ptr<imageprocessing::Patch> thePointPatch = p;	// The patch (with 2D-coords) we took // // CHANGED TO PATCH
				// get the new vertex from 3DMM, project into 2D with trans and rot matrix calculated by cvPOSIT
				// vector<pair<string, Point2f> > theNewLmInImage = projectVerticesToImage(theNewLmToProject, lmVertPairAtOrigin.verticesMean, lmVertPairAtOrigin.landmarksMean, s, t, mm);
				// Do we accept the point or not? error of theNewLmInImage[0] - thePointPatch < t ?
				// if point fits maybe_model with an error smaller than t, add point to consensus_set
/*				cv::Vec2f theNewLm(theNewLmInImage[0].second.x, theNewLmInImage[0].second.y);
				cv::Vec2f theNewPointPatch((float)thePointPatch->getX(), (float)thePointPatch->getY());
				double distance = cv::norm(theNewLm, theNewPointPatch, cv::NORM_L2);
				currentFfpDistances.push_back(distance);*/
			}
			vector<double>::iterator beforeItr = min_element(currentFfpDistances.begin(), currentFfpDistances.end());
			if(*beforeItr > thresholdForDatapointFitsModel) {
				// None of the candidates for the current Ffp fit the model well. Don't add anything to the consensusSet.
			} else {
				// We found a candidate that fits the model, add it to the consensusSet! (and continue with the next Ffp)
/*				int positionOfMinimumElement = distance(currentFfpDistances.begin(), beforeItr);
				shared_ptr<imageprocessing::Patch> thePointPatch = landmarkData[whichFfp].second[positionOfMinimumElement]; // CHANGED TO PATCH
				Point2f theNewPointPatch((float)thePointPatch->getX(), (float)thePointPatch->getY());
				consensusSet.push_back(make_pair(landmarkData[whichFfp].first, theNewPointPatch)); */
			}
			
		}
		// We went through all Ffp's.
		// if the number of elements in consensus_set is > d (this implies that we may have found a good model, now test how good it is)
		if (consensusSet.size()<numClosePointsRequiredForGoodFit) {
			continue;	// We didn't find a good model, the consensusSet only consists of the points projected (e.g. 3).
		}
		// this_model := model parameters fitted to all points in consensus_set

/*		LandmarksVerticesPair consensusSetLmVertPairAtOrigin = getCorrespondingLandmarkpointsAtOriginFrom3DMM(consensusSet, mm);	// Todo: It's kind of not clear if this expects centered 2D LMs or not-centered (it's not-centered)
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
*/
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
			std::cout << "[" << bestConsensusSets[i][j].first << " " << bestConsensusSets[i][j].second  << "], ";
		}
		std::cout << std::endl;
	}

	return map<string, shared_ptr<imageprocessing::Patch>>();
}

} /* namespace shapemodels */
