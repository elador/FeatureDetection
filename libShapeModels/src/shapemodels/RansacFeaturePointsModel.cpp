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
#include <limits>

using logging::LoggerFactory;
using cv::Point2f;
using std::pair;
using std::string;
using std::set;
using std::make_pair;

namespace shapemodels {

// Idea: in evaluateNewPoint, we could use the pixel-data in the patch as an additional indicator if it's a good fit

map<string, shared_ptr<imageprocessing::Patch>> RansacFeaturePointsModel::run(Mat img, float thresholdForDatapointFitsModel/*=30.0f*/, int numIter/*=0*/, int numClosePointsRequiredForGoodFit/*=4*/, int minPointsToFitModel/*=3*/)
{
	//The whole algorithm here. evaluator only evaluates 1 combo. All logic here. Selector provides functions to get random items (1 or several) etc.

	//map<string, vector<shared_ptr<imageprocessing::Patch>>> featureSelection;
	//do {
	//featureSelection.clear(); // Not necessary?
	//	featureSelection = selector->select();
	//} while (evaluator->evaluate(featureSelection));
	
	numIter = 7;
	minPointsToFitModel = 4;
	bool useSVD = false; // => to eval.
	thresholdForDatapointFitsModel = 30.0f; // same ?

	unsigned int iterations = 0;
	vector<vector<pair<string, Point2f>>> bestConsensusSets;
	float best_error = std::numeric_limits<float>::max();

	while (iterations < numIter) {
		++iterations;

		//set<string> alreadyChosenFfps;
		map<string, vector<shared_ptr<imageprocessing::Patch>>> landmarks = selector->getAllLandmarks();

		// maybe_inliers := #minPointsToFitModel randomly selected values from data
		map<string, shared_ptr<imageprocessing::Patch>> maybeInliersPatches = selector->getDistinctRandomPointsTEST(minPointsToFitModel);

		for (const auto& l : maybeInliersPatches) {
			landmarks.erase(l.first);
		}

		// We got #minPointsToFitModel distinct maybe_inliers, go on
		// == maybeInliers
		// maybe_model := model parameters fitted to maybeInliers
		// Use the 3DMM and POSIT:
		//pair<Mat, Mat> transRot = evaluator->doPosit(maybeInliersPatches, img);
		Mat firstStepImg = img.clone();
		pair<Mat, Mat> transRot = evaluator->doPnP(maybeInliersPatches, firstStepImg);

		vector<pair<string, Point2f>> maybeInliers;

		for (const auto& p : maybeInliersPatches) {
			maybeInliers.push_back(make_pair(p.first, Point2f((float)p.second->getX(), (float)p.second->getY())));
			//alreadyChosenFfps.insert(p.first);
		}

		// output the points, draw the model?

		// consensus_set := maybe_inliers
		vector<pair<string, Point2f>> consensusSet = maybeInliers;
		// Todo: We could do a fast rejection here, see how good our points are and
		// if bad, directly reject them? That's not possible when we work with 3 points
		// as then the combination is always good!

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
				
				Mat newPointImg = img.clone();
				float error = evaluator->evaluateNewPointPnP(l.first, p, transRot.first, transRot.second, newPointImg);
				currentFfpDistances.push_back(error);
			}
			// Do we accept the point or not?
			// if point fits maybe_model with an error smaller than t, add point to consensus_set
			vector<double>::iterator beforeItr = min_element(currentFfpDistances.begin(), currentFfpDistances.end());
			if(*beforeItr > thresholdForDatapointFitsModel) {
				// None of the candidates for the current Ffp fit the model well. Don't add anything to the consensusSet.
			} else {
				// We found a candidate that fits the model, add it to the consensusSet! (and continue with the next Ffp)
				int positionOfMinimumElement = distance(currentFfpDistances.begin(), beforeItr);
				shared_ptr<imageprocessing::Patch> thePointPatch = l.second[positionOfMinimumElement];
				Point2f theNewPointPatch((float)thePointPatch->getX(), (float)thePointPatch->getY());
				consensusSet.push_back(make_pair(l.first, theNewPointPatch));
				maybeInliersPatches.insert(make_pair(l.first, thePointPatch));
			}
			//landmarks.erase(l.first); // we're finished with this landmark, delete it from the list. Produces an iterator error. But we don't have to do this anyway, just loop through the map. We could use erase if we looped through it randomly.
		} // end for all landmarks
		// We went through all Ffp's.
		// if the number of elements in consensus_set is > d (this implies that we may have found a good model, now test how good it is)
		if (consensusSet.size()<numClosePointsRequiredForGoodFit) {
			continue;	// We didn't find a good model, the consensusSet only consists of the points projected (e.g. 3).
		}
		// (this implies that we may have found a good model, now test how good it is)
		// this_model := model parameters fitted to all points in consensus_set
		// Todo: We don't check if the face is facing us or not. Somehow check this?

		// Do POSIT with ALL the points in the consensus set!
		Mat fullSetImg = img.clone();
		transRot = evaluator->doPnP(maybeInliersPatches, fullSetImg);

		// this_error := a measure of how well this_model fits these points
		// The line above contains the bug. This_error should be replaced by a score that is either the size of the consensus set, or the robust error norm computed on ALL samples (not just the consensus set).
		float this_error = 0.0f;
		for (const auto& p : maybeInliersPatches) {
			this_error += evaluator->evaluateNewPointPnP(p.first, p.second, transRot.first, transRot.second, fullSetImg);
		}

		// if this_error < best_error // = (we have found a model which is better than any of the previous ones; keep it until a better one is found)
		// best_model := this_model
		// best_consensus_set := consensus_set
		// best_error := this_error
		// increment iterations

		// Check for number of points AND the error measure?
		// At the moment we a) store only the best (one), and b) numPoints take precedence over the error
		if(bestConsensusSets.empty()) {
			bestConsensusSets.push_back(consensusSet); // first consensusSet, add it
			best_error = this_error;
		} else {
			if(consensusSet.size()==bestConsensusSets[0].size()) { // we already have a consensusSet with equal amount of points
				if (this_error < best_error) {
					bestConsensusSets.clear();
					bestConsensusSets.push_back(consensusSet);
					best_error = this_error;
				}
			} else if (consensusSet.size()>bestConsensusSets[0].size()) {
				bestConsensusSets.clear();
				bestConsensusSets.push_back(consensusSet);
				best_error = this_error;
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

	map<string, shared_ptr<imageprocessing::Patch>> bestSet;
	if (bestConsensusSets.size() > 0) {
		for (const auto& l : bestConsensusSets[0]) {
			bestSet.insert(make_pair(l.first, make_shared<imageprocessing::Patch>(l.second.x, l.second.y, 33, 33, Mat()))); // TODO return the correct patches, not empty ones (x/y are correct)
		}
	}
	return bestSet;
}

} /* namespace shapemodels */
