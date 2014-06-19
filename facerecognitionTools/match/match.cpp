// -*- c++ -*-
// Copyright @ 2002-2009, Cognitec Systems AG 
// All rights reserved.
//
// $Revision: 1.20 $
//


#include <frsdk/config.h>
#include <frsdk/match.h>
#include "cmdline.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <exception>
#include <list>
#include <cstdlib>

using namespace std;

int usage() 
{
  cerr << "usage:" << endl
       << "match -cfg <config file> -probe <fir> -gallery <fir0,fir1,fir2,...> [-thr <threshold> | -far <requestedFAR> | -frr <requestedFRR>] [-maxmatch <number of matches>]" << endl
       << endl << endl
       << "\tconfig file ... the frsdk config file" << endl
       << "\tfir         ... the fir to test against" << endl
       << "\tfir0,fir1,...  ... one or more FIR files for the population"
       << endl 
       << "\tthreshold   ... threshold for successfull matches (default is score for FAR of 0.001)" << endl
       << "\trequestedFAR ... request score for this FAR" << endl
       << "\trequestedFRR ... request score for this FRR" << endl
       << "\tnumber of matches ... maximal number of matches in returneed match list" 
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


    // initialize and resource allocation
    FRsdk::Configuration cfg( cmd.getspaceflag("-cfg"));

    // load the fir
    ifstream firStream( cmd.getspaceflag("-probe"), ios::in|ios::binary);
    FRsdk::FIRBuilder firBuilder( cfg);
    FRsdk::FIR fir = firBuilder.build( firStream);
    
    // load the fir population for identification
    std::string firs = cmd.getspaceflag("-gallery");
    size_t pos = 0;
    size_t fpos = 0;
    FRsdk::Population population( cfg);
    while( (pos = firs.find(',', fpos)) != std::string::npos){
      std::string firn = firs.substr( fpos, pos-fpos);
      fpos = pos +1;
      cout  << "[" << firn << "]" << endl;
      ifstream firIn( firn.c_str(), ios::in|ios::binary);
      population.append( firBuilder.build( firIn), firn.c_str());
    } 
    if( fpos < firs.size()){
      std::string firn = firs.substr( fpos, pos-fpos);
      cout  << "[" << firn << "]" << endl;
      ifstream firIn( firn.c_str(), ios::in|ios::binary);
      population.append( firBuilder.build( firIn), firn.c_str());
    }

    // request Score match list size
    FRsdk::ScoreMappings sm( cfg);
    FRsdk::Score score = sm.requestFAR( 0.001f);
    if( cmd.getspaceflag("-far")){
      if( cmd.getspaceflag("-frr") || cmd.getspaceflag("-score")) return usage(); 
    }
    if( cmd.getspaceflag("-frr")){
      if( cmd.getspaceflag("-score")) return usage();
    }
    
    if( cmd.getspaceflag("-frr")){
      score = sm.requestFRR( atof( cmd.getspaceflag("-frr")));
    }
    if( cmd.getspaceflag("-far")){
      score = sm.requestFAR( atof( cmd.getspaceflag("-far")));
    }
    if( cmd.getspaceflag("-score")){
      score = FRsdk::Score( atof( cmd.getspaceflag("-score")));
    }
    cout << "used matching threshold: " << score << endl;

    unsigned int numofmatches = 3;
    if( cmd.getspaceflag("-maxmatch")){
      numofmatches = atoi( cmd.getspaceflag("-maxmatch"));
    }
    cout << "maximal matchlist size:" << numofmatches << endl;

    // initialize matching facility
    FRsdk::FacialMatchingEngine me( cfg);

    // bestMatches() takes care about the configured number of threads
    // to be used in comparison algorithm. 
    FRsdk::CountedPtr<FRsdk::Matches> matches =
      me.bestMatches( fir, population, FRsdk::Score( score), numofmatches);

    // print the match results
    for( FRsdk::Matches::const_iterator iter = matches->begin();
         iter != matches->end(); iter++) {
      FRsdk::Match match = *iter;
      cout << "[" << match.first << "] \t:" << match.second << endl; 
    }

    //compare() does not care about the configured number of Threads
    //for the comparison algorithm. It uses always one thrad to
    //compare all inorder to preserve the order of the scores
    //according to the order in the population (orer of adding FIRs to
    //the population)
    FRsdk::CountedPtr<FRsdk::Scores> scores = me.compare( fir, population);

    // print the results
    unsigned int n = 0;
    for( FRsdk::Scores::const_iterator siter = scores->begin();
         siter != scores->end(); siter++) {
      cout << "[ #" << n++ << " ] \t:" << float( *siter) << endl; 
    }
    

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
