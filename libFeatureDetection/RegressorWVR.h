#pragma once
#include "VDetectorVectorMachine.h"

class IImg;

class RegressorWVR : public VDetectorVectorMachine
{
public:
	RegressorWVR(void);
	~RegressorWVR(void);

	bool classify(FdPatch*);

	int load(const std::string);
	int init_for_image(FdImage*);

protected:

	float lin_eval_wvm_histeq64(int, int, int, int, float*, float*, const IImg*, const IImg*) const;

	int numUsedFilter;
	//float limit_reliability_filter;	//We don't need this for Regression
	int nLinFilters;
	float**  lin_filters;      // points to the filter array (this is support_hk%d in the .mat-file)
	float*   lin_thresholds;   // arrays of the thresholds (the SVM's b)
	float*   lin_hierar_thresh;// a pixel whose correletion with filter i is > lin_hierar_thresh[i] 
	                           // is retained for futher investigation,
	                           // if it is lower, it is classified as being not a face 
	float**  hk_weights;       // weights[i] contains the weights of the kernels of the i hierarchical kernel.

	int	nLinFilters_wvm;		// number of filters per level
	int	nLevels_wvm;			// number of levels with filters

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

	std::vector< std::pair<int, float> > hierarchical_thresholds;

	//float posterior_wrvm[2];	// probabilistic wrvm output: p(ffp|t) = 1 / (1 + exp(p[0]*t +p[1]))

	float *filter_output;
	float *u_kernel_eval;

};

