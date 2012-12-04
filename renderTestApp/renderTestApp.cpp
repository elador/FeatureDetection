// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

#ifdef WIN32
	#include <crtdbg.h>
#endif

#ifdef _DEBUG
   #ifndef DBG_NEW
      #define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
      #define new DBG_NEW
   #endif
#endif  // _DEBUG

#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp" // for FdImage.h
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include <boost/program_options.hpp>


namespace po = boost::program_options;
using namespace std;

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
    copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
    return os;
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(3759128);
	#endif

	std::string filename; // Create vector to hold the filenames
	
	try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("input-file,i", po::value<string>(), "input image")
        ;

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).
                  options(desc).run(), vm);
        po::notify(vm);
    
        if (vm.count("help")) {
            cout << "[ffpDetectApp] Usage: options_description [options]\n";
            cout << desc;
            return 0;
        }
        if (vm.count("input-file"))
        {
            cout << "[ffpDetectApp] Using input images: " << vm["input-file"].as< vector<string> >() << "\n";
			filename = vm["input-file"].as<string>();
        }
    }
    catch(std::exception& e) {
        cout << e.what() << "\n";
        return 1;
    }





	return 0;
}
