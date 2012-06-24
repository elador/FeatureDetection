#include "StdAfx.h"
#include "OverlapElimination.h"


OverlapElimination::OverlapElimination(void)
{
	strcpy(outputPath, "");
	identifier = "OverlapElimination";
	expected_num_faces[0]=0; expected_num_faces[1]=1;

	doOE = 1;	// only before SVM
	dist = 5.0f;	//5px
	ratio = -1.0f;	//no ratio
}


OverlapElimination::~OverlapElimination(void)
{
}

std::string OverlapElimination::getIdentifier()
{
	return this->identifier;
}

void OverlapElimination::setIdentifier(std::string identifier)
{
	this->identifier = identifier;
}

int OverlapElimination::load(const char* filename)
{
	//char* configFile = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_fd.mat";
	std::cout << "[OverlapElimination] Loading " << filename << std::endl;

	MatlabReader *configReader = new MatlabReader(filename);
	int id;
	char buff[255], key[255], pos[255];

	if(!configReader->getKey("FD.ffp", buff))	// which feature point does this detector detect?
		std::cout << "Warning: Key in Config nicht gefunden, key:'" << "FD.ffp" << "'" << std::endl;
	else
		std::cout << "[OverlapElimination] ffp: " << atoi(buff) << std::endl;

	if (!configReader->getKey("ALLGINFO.outputdir", this->outputPath)) // Output folder of this detector
		std::cout << "Warning: Key in Config nicht gefunden, key:'" << "ALLGINFO.outputdir" << "'" << std::endl;
	std::cout << "[OverlapElimination] outputdir: " << this->outputPath << std::endl;

	//min. und max. erwartete Anzahl Gesichter im Bild (vorerst null bis eins);											  
	sprintf(pos,"FD.expected_number_faces.#%d",0);																		  
	if (!configReader->getKey(pos,buff))																						  
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %d\n", pos,this->expected_num_faces[0]);
	else
		this->expected_num_faces[0]=atoi(buff);
	sprintf(pos,"FD.expected_number_faces.#%d",1);
	if (!configReader->getKey(pos,buff))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %d\n", pos,this->expected_num_faces[1]);
	else
		this->expected_num_faces[1]=atoi(buff);

	std::cout << "[OverlapElimination] expected_num_faces: " << this->expected_num_faces[0] << ", " << this->expected_num_faces[1] << std::endl;


	//Does overlap elimination 
	if (!configReader->getKey("FD.doesPPOverlapElimination",buff))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %d\n",
		"FD.doesPPOverlapElimination",this->doOE);
	else this->doOE=atoi(buff);

	//Dist overlap elimination 
	if (!configReader->getKey("FD.distOverlapElimination.#0",buff))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %g\n",
		"FD.distOverlapElimination.#0",this->dist);
	else this->dist=(float)atof(buff);
	if (!configReader->getKey("FD.distOverlapElimination.#1",buff))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %g\n",
		"FD.distOverlapElimination.#1",this->ratio);
	else this->ratio=(float)atof(buff);

	delete configReader;

	std::cout << "[OverlapElimination] Done reading OverlapElimination parameters!" << std::endl;
	return 1;

}


///////////////////////////////////////////////////////////////////////////////////
// pp_overlap_elimination
// simple clustering function, for each cluster only the best (with highest certainty)
// will be keept, the others deleted.
// Objects are at the same cluster when 
//   The max. distance of the centre coord. is smaller as thresholds[0] and 
//	 the ratio of the obj. width is smaller as thresholds[1].
//	 The distance is messured in pixel if thresholds[0]>1 else rel. to patch width.
std::vector<FdPatch*> OverlapElimination::eliminate(std::vector<FdPatch*> &patchvec, std::string detectorIdForSorting)
{

	std::cout << "[OverlapElimination] Running OverlapElimination..." << std::endl;
	if(this->doOE > 1) {
		std::cout << "[OverlapElimination] I am sorry, FD.doesPPOverlapElimination > 1 is not yet implemented. I'm going to set it to 1 and continue with this." << std::endl;
		this->doOE = 1;
	}

	//std::vector<FdPatch*> newcand;
	std::vector<FdPatch*> candidates = patchvec;
	if (candidates.size() == 0)
		return candidates;

//	std::vector<FdPatch*>::iterator it2;
//	std::vector<FdPatch*>::iterator it;

	//CFdPatchMap::iterator tmp;
	float dist=this->dist;
	float ratio=((this->ratio>0.0f) && (this->ratio<=1.0f))? this->ratio:0.0f;
	float d;

	//std::vector<FdPatch*>::iterator itr;
	//for (itr = patchvec.begin(); itr != patchvec.end(); ++itr ) {
	//	std::cout << "Hi!";
	//}
/*	bool dontIncIt = false;
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

	std::vector<FdPatch*> candidates_of_OE1 = candidates;
	candidates.clear();
	candidates = patchvec;
*/	
/*============================================================================*/

		//FdPatch::SortByCertainty bla;
		//bla.detectorType = "DetectorWVM";
	
	std::sort(candidates.begin(), candidates.end(), FdPatch::SortByCertainty(detectorIdForSorting));
	 
     //int K = 9999;// how many to keep, 1-X
     //int L = 1; //(level_span>0) ? level_span : 1;
     //float R = 1; //(radius>0) ? radius*radius : 25.f;

     //for (int acc=0; acc<candidates.size() && acc < K; ++acc)
     for (std::vector<FdPatch*>::iterator accepted = candidates.begin(); accepted != candidates.end(); accepted++)
     {
         //for (int pro=acc+1; pro<candidates.size(); )
         for (std::vector<FdPatch*>::iterator proband = accepted+1; proband!= candidates.end(); )
         {
			 if (dist<=1.0) d=dist*max((*accepted)->w_inFullImg,(*proband)->w_inFullImg); else d=dist;
             //if ( abs(candidates[acc].s-candidates[pro].s)<L && sq_dist(candidates[acc],candidates[pro])<R )
             if (  (abs((*accepted)->c.x-(*proband)->c.x) < d) && (abs((*accepted)->c.y-(*proband)->c.y) < d) && (((float)min((*accepted)->w_inFullImg,(*proband)->w_inFullImg)/(float)max((*accepted)->w_inFullImg,(*proband)->w_inFullImg)) > ratio) )
             {
                 //candidates.erase((candidates.begin()+pro));
                 proband = candidates.erase(proband);
             }
             else
             {
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

     //std::cout << "\t\tto:" << candidates.size() << "\n" << std::flush; }

	return candidates; //candidates_of_OE1;// candidates;
}



std::vector<FdPatch*> OverlapElimination::exp_num_fp_elimination(std::vector<FdPatch*> &patchvec, std::string detectorIdForSorting)
{
	std::cout << "[OverlapElimination] Running exp_num_fp_elimination, eliminating to the " << this->expected_num_faces[1] << " patches with highest probability." << std::endl;
	if(patchvec.size() > this->expected_num_faces[1])
	{
		std::vector<FdPatch*> candidates;
		
		std::sort(patchvec.begin(), patchvec.end(), FdPatch::SortByCertainty(detectorIdForSorting));
		
		for(int i=0; i < this->expected_num_faces[1]; i++) {
			candidates.push_back(patchvec[i]);
		}
		return candidates;

	} else {
		return patchvec;
	}

}



/*
std::vector<unsigned int> CFdDetectImg::pp_overlap_elimination(CFdPatchMap& faces, float thresholds[2], int cache, bool collect_removed=true) const {

	std::vector<unsigned int> removedFaces;
	
	if (faces.size() == 0)
		return removedFaces;

	CFdPatchMap::iterator it2;
	CFdPatchMap::iterator it = faces.begin();
	CFdPatchMap::iterator end = faces.end();
	CFdPatchMap::iterator end_it = faces.end();
	--end_it;
	bool reset_cache;
	int c, cc;
	CFdPatchMap::iterator *ca;
	ca = new CFdPatchMap::iterator[cache+1];

	for (c=0;c<=cache;c++) ca[c]=NULL;
	reset_cache=false;
	float dist=thresholds[0], ratio=((thresholds[1]>0.0f) && (thresholds[1]<=1.0f))? thresholds[1]:0.0f, d;
	cache--;
	for (;it < end_it; ++it) {
		//it2 = it; ++it2;
		for (it2 = it+1; it2 < end; ++it2) {
			if (dist<=1.0) d=dist*max(it->w,it2->w); else d=dist;
			if ( (abs(it->c.x-it2->c.x) < d) && (abs(it->c.y-it2->c.y) < d) && (((float)min(it->w,it2->w)/(float)max(it->w,it2->w)) > ratio) ) {
				//printf("ca(%d): ",cache); for (cc=0;cc<=cache;cc++) if(ca[cc]==NULL)  printf("[%d].(N,0,0,0,0)) ",cc); else { box3=ca[cc]->box(); printf("[%d].(%d,%d,%d,%1.2f) ",cc,box3.left,box3.top,box3.right,ca[cc]->certainty);} printf("\n");
				//printf("it:(%d,%d,%d,%1.2f), it2:(%d,%d,%d,%1.2f), inters:%1.1f\n",
				//	box.left,box.top,box.right,it->certainty,box2.left,box2.top,box2.right,it2->certainty,(float)intersect_area/min(box_area,box2_area));

				// there is some overlapping between it2 and it
				//look if better than another in the cache
				c=0; while ( (c<cache) && (ca[c]!=NULL) && (it2->certainty <= ca[c]->certainty) ) c++;
				if (c>=cache) {
					// cache full and it_cur worth then last -> if worth than it_current too
					if (it2->certainty <= it->certainty) {
						//...kill it						   //cache full, it2 worth than last and worth than it
						if (collect_removed) removedFaces.push_back(it2->sampleID);
						faces.erase(it2);
					}  else {
						//kill it_current and reset_cache		   //cache full, it2 worth than last but better than it
						if (collect_removed) removedFaces.push_back(it->sampleID);
						faces.erase(it);
						reset_cache=true;
						break;
					}
				}  else {
					// cache not full
					if (ca[c]==NULL) {
						//cache not full and worth then last -> put at the end of the cache
						ca[c]=it2;
					} else {
						//cache was not full and it_cur better then last -> push all one down and sort in,
						cc=cache; while (cc>c)  { if (ca[cc-1]!=NULL) ca[cc]=ca[cc-1]; cc--;  }
						ca[c]=it2;
					}

					//...but if cache now full kill last or it_current
					if (ca[cache]!=NULL)	{
						if (ca[cache]->certainty <= it->certainty) {
							//kill last							   //cache full, it2 better than last and last worth than it
							if (collect_removed) removedFaces.push_back(ca[cache]->sampleID);
							faces.erase(ca[cache]);
							ca[cache]=NULL;
						} else {
							//kill it_current and reset_cache	   //cache full, it2 better than last and last better than it
							if (collect_removed) removedFaces.push_back(it->sampleID);
							faces.erase(it);
							reset_cache=true;
							break;
						}
					}
				}
			}
		}
		if ( (it2==end) || reset_cache ) {
			//if (it2==end) printf("was last it2\n");
			//if (reset_cache) printf("kill it and reset_cache\n");
			//reset_cache
			
			reset_cache=false;
			for (c=0;c<=cache;c++)	ca[c]=NULL;
		}
	}
	delete[] ca;
	return removedFaces;
}
*/