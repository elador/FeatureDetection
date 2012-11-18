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

protected:

	float**  linFilters;      // points to the filter array (this is support_hk%d in the .mat-file). These are the actual vectors.
	float**  hkWeights;       // weights[i] contains the weights of the kernels of the i hierarchical kernel.


	int numUsedFilter;			// Read from the matlab config. 0=use all, >0 && <numLinFilters ==> don't use all, only numUsedFilter.
	int numLinFilters;			// Read from detector .mat. Used in the WVM loop (run over all vectors). (e.g. 280)
	
	int	numFiltersPerLevel;		// number of filters per level (e.g. 14)
	int	numLevels;			// number of levels with filters (e.g. 20)
	

	float*   lin_thresholds;   // arrays of the thresholds (the SVM's b). All values of this array (e.g. 280) are set to nonlin_threshold read from the detector .mat on load().
								// This is then used inside WvmEvalHistEq64()

	float*   lin_hierar_thresh;// a pixel whose correlation with filter i is > lin_hierar_thresh[i] 
	                           // is retained for further investigation, if it is lower, it is classified as being not a face 
	std::vector< std::pair<int, float> > hierarchical_thresholds;	// This is the same as lin_hierar_thresh. Checked on 17.11.12 - this is really not needed.
																	// hierarchical_thresholds is only used for reading from the config, then not used anymore.
	
	float limit_reliability_filter;	// This is added to lin_hierar_thresh on startup (read from the config, FD.limit_reliability_filter), then not used anymore.


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

