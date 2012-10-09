// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>


// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#ifdef _DEBUG
   #ifndef DBG_NEW
      #define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
      #define new DBG_NEW
   #endif
#endif  // _DEBUG

// TODO: reference additional headers your program requires here
#include <iostream>

#include "opencv2/core/core.hpp" // for FdImage.h
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "mat.h"

//#include "DetectorSVM.h"
//#include "DetectorWVM.h"
//#include "CascadeWvmSvm.h"
//#include "RegressorSVR.h"

#include "SLogger.h"

#include "CascadeERT.h"
#include "FdImage.h"