/*
 * OverlapElimination.hpp
 *
 *  Created on: 26.02.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef OVERLAPELIMINATION_HPP_
#define OVERLAPELIMINATION_HPP_

#include "opencv2/core/core.hpp"
#include <memory>
#include <vector>

using cv::Mat;
using std::vector;
using std::shared_ptr;

namespace detection {

class ClassifiedPatch;

/**
 * An overlap elimination, that takes a list of image patches and eliminates clusters of similar patches to leave only the best one(s).
 * At the moment, it only works with probabilistic detectors, as ClassifiedPatch contains a probability.
 *
 * Maybe we will add an interface for this too in the future (e.g. for OE's or in general for clustering algorithms that use a list of points), but at the moment we only have this one OE.
 */
class OverlapElimination {
public:

	/**
	 * Constructs a new OverlapElimination with the values TODO OR default values TODO.
	 *
	 * Note: MR used different "default" values. 5.0/0.0 does eliminate a bit but not too
	 *       much, while 0.6/0.65 is a very strong setting.
	 * @param[in] dist If greater than one, the distance in pixels at which two patches are not anymore
	 *                 considered to be in the same cluster. 
	 *                 If between 0 and 1, two patches are not anymore considered to be in the same cluster when
	                   the distance between them is greater than the given dist times the width of the patch.
	 * @param[in] ratio The ratio of the widths of two patches (between 0 and 1), at which they are not anymore
	 *                  considered to be in the same cluster. A value of 0.0 means that this criterion is ignored.
	 */
	OverlapElimination(float dist=5.0f, float ratio=0.0f);

	~OverlapElimination() {}

	/**
	 * Run the overlap elimination on a list of patches with their corresponding detector output.
	 *
	 * @param[in] classifiedPatches A list of classified patches where we want to eliminate some overlapping ones.
	 * @return The reduced list of classified patches.
	 */
	vector<shared_ptr<ClassifiedPatch>> eliminate(vector<shared_ptr<ClassifiedPatch>> &classifiedPatches);

private:	
	float dist;	//Clustering: maximal distance that detections belong to the same cluster;		// See the matlab configs, value FD.distOverlapElimination.#0
	//Values: float (>0.0, <=1.0) relative to feature width or int (>1) in pixel; Default: 0.6
	//pp_oe_percent[0];  //if more overlap (smaller dist as [0] and ratio [1](not for SVMoe)), than the all detections with the lower likelihood will be deleted 
	// Note: If >1, then we mean pixel in the original image. Pixels in the pyramids doesn't make sense.

	float ratio;	//Clustering: maximal ratio of size that detections belong to the same cluster (only fist WRVM-stage, for fullSVM=1.0);		// See the matlab configs, value FD.distOverlapElimination.#1
	//Values: float (0.0: off; >0.0, <=1.0: on), e.g. smallest/larges feature < maxratio => same cluster; Default: 0.65
	//pp_oe_percent[1];

	// Note: Another useful variable would be to be able to specify how many patches to reduce each cluster into. This was 
	// the "FD.doesPPOverlapElimination" in the old implementation. (e.g. keep only the best patch per cluster, or the best 3).
};

} /* namespace detection */
#endif /* OVERLAPELIMINATION_HPP_ */
