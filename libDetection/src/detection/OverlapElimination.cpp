/*
 * OverlapElimination.cpp
 *
 *  Created on: 26.02.2013
 *      Author: Patrik Huber
 */

#include "detection/OverlapElimination.hpp"
#include "detection/ClassifiedPatch.hpp"
#include "logging/LoggerFactory.hpp"

#include "boost/iterator/indirect_iterator.hpp"
#include "boost/lexical_cast.hpp"
#include <iostream>	// TODO remove the cout's here and replace with logger/exceptions.
#include <string>
#include <functional>

using logging::Logger;
using logging::LoggerFactory;
using boost::make_indirect_iterator;
using boost::lexical_cast;
using std::string;
using std::sort;
using std::greater;
using std::min;
using std::max;
using std::abs;

namespace detection {

OverlapElimination::OverlapElimination(float dist, float ratio) : dist(dist), ratio(ratio)
{
}


///////////////////////////////////////////////////////////////////////////////////
// pp_overlap_elimination
// simple clustering function, for each cluster only the best (with highest certainty)
// will be kept, the others deleted.
// Objects are at the same cluster when 
//   The max. distance of the center coord. is smaller as thresholds[0] and 
//	 the ratio of the obj. width is smaller as thresholds[1].
//	 The distance is measured in pixel if thresholds[0]>1 else rel. to patch width.
vector<shared_ptr<ClassifiedPatch>> OverlapElimination::eliminate(vector<shared_ptr<ClassifiedPatch>> &classifiedPatches)
{

	vector<shared_ptr<ClassifiedPatch>> candidates = classifiedPatches;
	if (candidates.size() == 0)
		return candidates;

	Logger log = Loggers->getLogger("detection");

	float dist = this->dist;
	float ratio = ((this->ratio > 0.0f) && (this->ratio <= 1.0f))? this->ratio : 0.0f;
	float d;

	// Note: This is the simplified, slightly different OE from Andreas.
	//       It produces a little bit different results than the old OE from
	//       MR, but it's okay. A reimplementation of the MR OE is below - see
	//       the notes there.
	
	sort(make_indirect_iterator(candidates.begin()), make_indirect_iterator(candidates.end()), greater<ClassifiedPatch>());
	 
	// Commented-out is the code from Andreas. It might be a nice extension in the future.
	//int K = 9999; // how many to keep, 1-X
	//int L = 1; // (level_span>0) ? level_span : 1;
	//float R = 1; // (radius>0) ? radius*radius : 25.f;

     //for (int acc=0; acc<candidates.size() && acc < K; ++acc)
     for (vector<shared_ptr<ClassifiedPatch>>::iterator accepted = candidates.begin(); accepted != candidates.end(); accepted++)
     {
         //for (int pro=acc+1; pro<candidates.size(); )
         for (vector<shared_ptr<ClassifiedPatch>>::iterator proband = accepted+1; proband!= candidates.end(); )
         {
			 if (dist <= 1.0) {
				 d = dist*max((*accepted)->getPatch()->getWidth(), (*proband)->getPatch()->getWidth());
			 } else {
				 d = dist;
			 }
             //if ( abs(candidates[acc].s-candidates[pro].s)<L && sq_dist(candidates[acc],candidates[pro])<R )
             if ( (abs((*accepted)->getPatch()->getX() - (*proband)->getPatch()->getX()) < d)
			   && (abs((*accepted)->getPatch()->getY() - (*proband)->getPatch()->getY()) < d)
			   && ( ((float)min((*accepted)->getPatch()->getWidth(), (*proband)->getPatch()->getWidth()) / (float)max((*accepted)->getPatch()->getWidth(), (*proband)->getPatch()->getWidth()) ) > ratio) )
             {
                 //candidates.erase((candidates.begin()+pro));
                 proband = candidates.erase(proband);
             } else {
                 //++pro;
                 proband++;
             }
         }
         //if (!(--K>0))
         //{
         //    candidates.erase(++accepted,candidates.end());
         //    break;
         //}
     }
     //if (K<candidates.size())
     //{
     //    candidates.erase((candidates.begin()+K),candidates.end());
     //}

	 log.debug("OverlapElimination reduced the candidate patches from " + lexical_cast<string>(classifiedPatches.size()) + " to " + lexical_cast<string>(candidates.size()) + ".");

	 return candidates;

	// The following is a reimplementation of the OE code from MR. It was compiling before
	// the "refact" and should be working. It can be tested in the last software version
	// before the refact. However, there might have been some problem with it (or maybe just
	// similar results), so I chose to use the simpler OE above.
	/*	
	std::vector<FdPatch*>::iterator it2;
	std::vector<FdPatch*>::iterator it;
	bool dontIncIt = false;
	bool dontIncIt2 = false;
	it = candidates.begin();
	//int outer=-1;
	//int inner=-1;
	while(it != candidates.end()-1) {
		//outer++;
		//std::cout << "o:" << outer << std::endl;
		//std::cout << *(candidates.end()-1)  << std::endl;
		it2 = it+1;
		while(it2 != candidates.end()) {
			//inner++;
			//std::cout << "i:" << inner << std::endl;
				if (dist<=1.0) d=dist*max((*it)->w_inFullImg,(*it2)->w_inFullImg); else d=dist;
				//printf("it:(%d,%d,%d,%1.2f), it2:(%d,%d,%d,%1.2f), d:(%d,%d) (<%1.1f), r:%1.2f (<%1.2f)",
				//it->box().left,it->box().top,it->box().right,it->certainty, it2->box().left,it2->box().top,it2->box().right,it2->certainty, 
				//abs(it->c.x-it2->c.x),abs(it->c.y-it2->c.y),d,((float)min(it->w,it2->w)/(float)max(it->w,it2->w)),ratio);
				if ( (abs((*it)->c.x-(*it2)->c.x) < d) && (abs((*it)->c.y-(*it2)->c.y) < d) && (((float)min((*it)->w_inFullImg,(*it2)->w_inFullImg)/(float)max((*it)->w_inFullImg,(*it2)->w_inFullImg)) > ratio) ) {
				// there is some overlapping between it2 and it -> delete not better one
					if ((*it2)->certainty < (*it)->certainty) {
						// it is better than it2 -> kill it2
						//printf("=> deleted it2\n");
						it2 = candidates.erase(it2);
						dontIncIt2 = true;
					} else {
						// kill it
						it = candidates.erase(it);
						//printf("=> deleted it\n");
						dontIncIt = true;
						break;
					}
				} //else printf("=> not deleted\n");
				if(dontIncIt2==true) {
					dontIncIt2 = false;
				} else {
					++it2;
				}
		}
		if(dontIncIt==true) {
			dontIncIt = false;
		} else {
			if(it == candidates.end()-1)	// if i'm the second-last element, and the last element just got deleted in the inner loop => bad
				break;
			else
			++it;
		}
	}
	// The candidates of this OE are now in "candidates".
	*/
}

} /* namespace detection */
