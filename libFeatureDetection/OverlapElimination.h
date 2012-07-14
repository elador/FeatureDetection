#pragma once

#include "MatlabReader.h"
#include <iterator>

class OverlapElimination
{
public:
	OverlapElimination(void);
	~OverlapElimination(void);

	std::vector<FdPatch*> eliminate(std::vector<FdPatch*>&, std::string);
	std::vector<FdPatch*> exp_num_fp_elimination(std::vector<FdPatch*>&, std::string);

	int load(const std::string);

	std::string getIdentifier();
	void setIdentifier(std::string);

	int doOE;	//Reduce detections per cluster; Values: int (0:only after last stage to one (the best) per cluster, 
				//n: reduce to n best detections after the WRVM- and best after fullSVM-stage per cluster); Default: 3

protected:	
	float dist;	//Clustering: maximal distance that detections belong to the same cluster; 
				//Values: float (>0.0, <=1.0) relative to feature width or int (>1) in pixel; Default: 0.6
				//pp_oe_percent[0];  //if more overlap (smaller dist as [0] and ratio [1](not for SVMoe)), than the all detections with the lower likelihood will be deleted 
	
	float ratio;	//Clustering: maximal ratio of size that detections belong to the same cluster (only fist WRVM-stage, for fullSVM=1.0); 
					//Values: float (0.0: off; >0.0, <=1.0: on), e.g. smallest/larges feature < maxratio => same cluster; Default: 0.65
					//pp_oe_percent[1];

	char outputPath[_MAX_PATH];
	int expected_num_faces[2];

	std::string identifier;

};

