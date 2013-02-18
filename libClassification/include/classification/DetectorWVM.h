#pragma once
#include "VDetectorVectorMachine.h"

class IImg;

class DetectorWVM : public VDetectorVectorMachine
{
public:
	DetectorWVM(void);
	~DetectorWVM(void);

	bool classify(FdPatch*);
	

	int load(const std::string);
	int initForImage(FdImage*);

	void setCalculateProbabilityOfAllPatches(bool);


protected:



	bool calculateProbabilityOfAllPatches; // Default = false. Calculate the probability of patches that don't live until the last wvm vector. If false, set the prob. to zero.
											// Warning: The probabilities are not really correct for all stages not equal to the last stage. 
};

