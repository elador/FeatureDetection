// -*- c++ -*-
// Copyright @ 2002-2009, Cognitec Systems AG 
// All rights reserved.
//
// $Revision: 1.20 $
//


#include <frsdk/config.h>
#include <frsdk/match.h>
#include "cmdline.h"
#include "boost/filesystem.hpp"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <exception>
#include <list>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>

using namespace std;
using boost::filesystem::path;

int usage() 
{
  cerr << "usage:" << endl
       << "match -cfg <config file> -probe <fir> -gallery <file list> -out <scores output-file>" << endl
       << endl << endl
       << "\tconfig file ... the frsdk config file" << endl
       << "\tfir         ... the fir to test against" << endl
       << "\tgallery  ... a file-list with one or more FIR files for the population" << endl
	   << "\tout  ... the output file to write the scores to"
       << endl << endl;
  return 1;
}

// main -----------------------------------------------------------------------
int main( int argc, const char* argv[] )
{
  try {

    FRsdk::CmdLine cmd( argc, argv);
    if( cmd.hasflag("-h")) return usage();
    if( !cmd.getspaceflag("-cfg")) return usage();
    if( !cmd.getspaceflag("-probe")) return usage();
    if( !cmd.getspaceflag("-gallery")) return usage();
	if( !cmd.getspaceflag("-out")) return usage();
	std::cout << "cmd stuff done" << std::endl;

    // initialize and resource allocation
    FRsdk::Configuration cfg( cmd.getspaceflag("-cfg"));
	std::cout << "cfg init." << std::endl;
	string outputFilename = cmd.getspaceflag("-out");
	std::cout << "cfg init. (2)" << std::endl;

    // load the fir
    ifstream firStream( cmd.getspaceflag("-probe"), ios::in|ios::binary);
	std::cout << "firStream done" << std::endl;
    FRsdk::FIRBuilder firBuilder( cfg);
	std::cout << "FIRBuilder done" << std::endl;
    FRsdk::FIR fir = firBuilder.build( firStream);
	std::cout << "FIR built" << std::endl;

    // load the fir population for identification
	std::vector<std::string> galleryNames;

    std::string galleryListFile = cmd.getspaceflag("-gallery");
	vector<string> galleryFilepaths;
	std::ifstream listfileStream;
	listfileStream.open(galleryListFile.c_str(), std::ios::in);
	if (!listfileStream.is_open()) {
		throw runtime_error("FileListImageSource: Error opening file list!");
	}
	string line;
	while (listfileStream.good()) {
		getline(listfileStream, line);
		if(line=="") {
			continue;
		}
		string buf;
		std::stringstream ss(line);
		ss >> buf;	
		galleryFilepaths.push_back(buf);	// Insert the image filename, just ignore the rest of the line
	}
	listfileStream.close();

	std::cout << "gallery paths loaded" << std::endl;

	/*
    size_t pos = 0;
    size_t fpos = 0;
    FRsdk::Population population( cfg);
    while( (pos = firs.find(',', fpos)) != std::string::npos){
      std::string firn = firs.substr( fpos, pos-fpos);
      fpos = pos +1;
      //cout  << "[" << firn << "]" << endl;
      ifstream firIn( firn.c_str(), ios::in|ios::binary);
      population.append( firBuilder.build( firIn), firn.c_str());
	  galleryNames.push_back(firn);
    } 
    if( fpos < firs.size()){
      std::string firn = firs.substr( fpos, pos-fpos);
      //cout  << "[" << firn << "]" << endl;
      ifstream firIn( firn.c_str(), ios::in|ios::binary);
      population.append( firBuilder.build( firIn), firn.c_str());
	  galleryNames.push_back(firn);
    }
	*/

	// Note Patrik: This could get problematic if someone could not be enrolled
	//		 into the gallery, but we don't have that with mpie-frontal atm.
	FRsdk::Population population( cfg);
	for(vector<string>::const_iterator fiter = galleryFilepaths.begin(); fiter != galleryFilepaths.end(); ++fiter) {
		ifstream firIn( fiter->c_str(), ios::in|ios::binary);
		population.append( firBuilder.build( firIn), fiter->c_str());
		galleryNames.push_back(*fiter);
	}
	std::cout << "Population built" << std::endl;

    // initialize matching facility
    FRsdk::FacialMatchingEngine me( cfg);
	std::cout << "MatchingEngine initialized" << std::endl;

    //compare() does not care about the configured number of Threads
    //for the comparison algorithm. It uses always one thrad to
    //compare all inorder to preserve the order of the scores
    //according to the order in the population (orer of adding FIRs to
    //the population)
    FRsdk::CountedPtr<FRsdk::Scores> scores = me.compare( fir, population);
	std::cout << "FIR loaded" << std::endl;

    // print the results
    unsigned int n = 0;
	cout << "Writing scores to: " << outputFilename << endl;
	ofstream scoresFile;
	scoresFile.open(outputFilename, ios::out | ios::trunc);
	for( FRsdk::Scores::const_iterator siter = scores->begin(); siter != scores->end(); siter++) {
		boost::filesystem::path imgName(galleryNames[n]);
		scoresFile << imgName.filename().string() << " " << float( *siter) << endl;
		//cout << "[ #" << n << " " << galleryNames[n] << "] \t:" << float( *siter) << endl; 
		n++;
    }
	scoresFile.close();
    
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
