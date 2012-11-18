#pragma once
#include "VDetectorVectorMachine.h"

class IImg;

class DetectorWVM : public VDetectorVectorMachine
{
public:
	DetectorWVM(void);
	~DetectorWVM(void);

	bool classify(FdPatch*);
	float linEvalWvmHisteq64(int, int, int, int, float*, float*, const IImg*, const IImg*) const;

	int load(const std::string);
	int initForImage(FdImage*);

	void setCalculateProbabilityOfAllPatches(bool);
	int getNumUsedFilters(void);
	void setNumUsedFilters(int);			// Change the number of currently used wavelet-vectors
	float getLimitReliabilityFilter(void);
	void setLimitReliabilityFilter(float);	// Rewrites the hierarchicalThresholds vector with the new thresholds

protected:

	float**  linFilters;      // points to the filter array (this is support_hk%d in the .mat-file). These are the actual vectors.
	float**  hkWeights;       // weights[i] contains the weights of the kernels of the i hierarchical kernel.


	int numUsedFilters;			// Read from the matlab config. 0=use all, >0 && <numLinFilters ==> don't use all, only numUsedFilters.
	int numLinFilters;			// Read from detector .mat. Used in the WVM loop (run over all vectors). (e.g. 280)
	
	int	numFiltersPerLevel;		// number of filters per level (e.g. 14)
	int	numLevels;			// number of levels with filters (e.g. 20)
	

	float* lin_thresholds;	// arrays of the thresholds (the SVM's b). All values of this array (e.g. 280) are set to nonlin_threshold read from the detector .mat on load().
							// This is then used inside WvmEvalHistEq64(). The same for all vectors. This could be just a float. Actually it can only be a float because it comes from param_nonlin1 in the .mat.

	std::vector<float> hierarchicalThresholds;	// a pixel whose correlation with filter i is > hierarchicalThresholds[i] 
										// is retained for further investigation, if it is lower, it is classified as being not a face 
	std::vector<float> hierarchicalThresholdsFromFile;	// This is the same as hierarchicalThresholds. Checked on 17.11.12 - this is really not needed.
																	// hierarchicalThresholdsFromFile is only used for reading from the config, then not used anymore.
	
	float limitReliabilityFilter;	// This is added to hierarchicalThresholds on startup (read from the config, FD.limitReliabilityFilter), then not used anymore.


	typedef struct _rec {
		int x1,x2,y1,y2,uull,uur,dll,dr;
	} TRec, *PRec;

	class Area
	{
	public:
		Area(void);
		Area(int, int*);
		~Area(void);
		
		void dump(char*);
		int		cntval;
		double	*val;
		TRec	**rec;
		int		*cntrec;
		int		cntallrec;
	};

	Area** area;	// rectangles and gray values of the appr. rsv
	double	*app_rsv_convol;	// convolution of the appr. rsv (pp)

	float posterior_wrvm[2];	// probabilistic wrvm output: p(ffp|t) = 1 / (1 + exp(p[0]*t +p[1]))

	float *filter_output;		// temporary output of each filter level
	float *u_kernel_eval;		// temporary cache, size=numFiltersPerLevel (or numLevels?)

	bool calculateProbabilityOfAllPatches; // Default = false. Calculate the probability of patches that don't live until the last wvm vector. If false, set the prob. to zero.
											// Warning: The probabilities are not really correct for all stages not equal to the last stage. 
};

