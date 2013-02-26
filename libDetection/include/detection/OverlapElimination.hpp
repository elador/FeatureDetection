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

using cv::Mat;

namespace detection {

/**
 * An overlap elimination, that takes a list of image patches and eliminates clusters of similar patches to leave only the best one(s).
 * TODO Do we only work on probabilistic detector outputs or on both? Test this.
 * Maybe we will add an interface for this too in the future (e.g. for OE's or in general for clustering algorithms that use a list of points), but at the moment we only have this one OE.
 */
class OverlapElimination {
public:

	/**
	 * Constructs a new OverlapElimination.
	 */
	OverlapElimination();

	~OverlapElimination() {}

	/**
	 * Run the overlap elimination on a list of patches with their corresponding detector output.
	 *
	 * @param[in] image The image that the detector should run on.
	 * @return Something probably.
	 */
	void eliminate(const Mat& image);

	std::vector<FdPatch*> expNumFpElimination(std::vector<FdPatch*>&, std::string);

	int load(const std::string);

	int doOE;	//Reduce detections per cluster; Values: int (0:only after last stage to one (the best) per cluster, 
	// n: reduce to n best detections after the WRVM- and best after fullSVM-stage per cluster); Default: 3
	// See the matlab configs, value FD.doesPPOverlapElimination

protected:	
	float dist;	//Clustering: maximal distance that detections belong to the same cluster;		// See the matlab configs, value FD.distOverlapElimination.#0
	//Values: float (>0.0, <=1.0) relative to feature width or int (>1) in pixel; Default: 0.6
	//pp_oe_percent[0];  //if more overlap (smaller dist as [0] and ratio [1](not for SVMoe)), than the all detections with the lower likelihood will be deleted 

	float ratio;	//Clustering: maximal ratio of size that detections belong to the same cluster (only fist WRVM-stage, for fullSVM=1.0);		// See the matlab configs, value FD.distOverlapElimination.#1
	//Values: float (0.0: off; >0.0, <=1.0: on), e.g. smallest/larges feature < maxratio => same cluster; Default: 0.65
	//pp_oe_percent[1];

	int maxNumFaces;
};

} /* namespace detection */
#endif /* OVERLAPELIMINATION_HPP_ */
