// -*- c++ -*-
// Copyright @ 2002-2009, Cognitec Systems AG 
// All rights reserved.
//
// $Revision: 1.29 $
//

#include <exception>
#include <list>
#include <cstdlib>

#include <frsdk/config.h>
#include <frsdk/image.h>
#include <frsdk/enroll.h>

#include "cmdline.h"
#include "edialog.h"

using namespace std;

int usage() 
{
  cerr << "usage:\n"
       << "enroll -cfg <config file> -fir <fir> "
          "-imgs <image0,image1,image2,...>\n\n" 
       << "\tconfig file ... the frsdk config file\n"
       << "\tfir         ... a filename for a FIR\n"
       << "\tjpeg image  ... one or more jpeg images for enrollment.\n" << endl;
  return EXIT_FAILURE;
}


// main ----------------------------------------------------------------------
int main( int argc, const char* argv[] )
{
  try {
    
    FRsdk::CmdLine cmd( argc, argv );
    if ( cmd.hasflag("-h")) return usage();
    if ( !cmd.getspaceflag("-cfg")) return usage();
    if ( !cmd.getspaceflag("-fir")) return usage();
    if ( !cmd.getspaceflag("-imgs")) return usage();

    // read the configuration file
    FRsdk::Configuration cfg( cmd.getspaceflag("-cfg"));

    // get the file names of the enrollment images
    string delimImgFileNames = cmd.getspaceflag( "-imgs" );
    list<string> imgFileNames = 
      FRsdk::parseDelimitedText< list<string> >( delimImgFileNames );
    
    if ( imgFileNames.size() == 0 )
    {
      cerr << "There are no input images specified!" << endl;
      usage();
    }

    cout << "Loading the input images..." << endl;
    FRsdk::SampleSet enrollmentImages;    
    list<string>::const_iterator it = imgFileNames.begin();
    while ( it != imgFileNames.end() )
    {
      const string& imgFileName = *it;
      cout  << "  \"" << imgFileName << "\"" << endl;    
      FRsdk::Image img( FRsdk::ImageIO::load( imgFileName ) );
      enrollmentImages.push_back( FRsdk::Sample( img ) );
      ++it;
    }
    cout << "...Done.\n"
         << enrollmentImages.size() << " image(s) loaded." << endl;
    if ( enrollmentImages.size() == 0 )
    {
      cerr << "There are no samples to process!" << endl;
      return EXIT_FAILURE;
    }

    cout << "Start processing ... " << flush;
    
    // create an enrollment processor
    FRsdk::Enrollment::Processor proc( cfg);
    
    // create the needed interaction instances
    FRsdk::Enrollment::Feedback
      feedback( new EnrolCoutFeedback( cmd.getspaceflag("-fir")));

    // do the enrollment    
    proc.process( enrollmentImages.begin(),
                  enrollmentImages.end(), feedback);
    cout << "...Done." << endl;
    
  }
  catch( const FRsdk::FeatureDisabled& e) {
    cout << "Feature not enabled: " << e.what() << endl;
    return EXIT_FAILURE;
  }  
  catch( const FRsdk::LicenseSignatureMismatch& e) {
    cout << "License violation: " << e.what() << endl;
    return EXIT_FAILURE;
  }
  catch( exception& e) {
    cout << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
