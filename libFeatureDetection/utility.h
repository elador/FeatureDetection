#pragma once
//#include "Pyramid.h"
#include "FdPatch.h"


//class Pyramid;	// Forward-declaration, so that I don't need to include Pyramid.h here


#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#ifndef _MAX_PATH
#define _MAX_PATH (255)
#endif

//typedef std::map<int, Pyramid*> PyramidMap;

#define NOTCOMPUTED -1e5
