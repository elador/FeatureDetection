// -*- c++ -*-
// Copyright (c) 2002 Cognitec Systems GmbH
//
// $Revision: 1.16 $
//

/** \file 
    \brief 
*/ 

#ifndef EDIALOG_H
#define EDIALOG_H

#include <frsdk/enroll.h>
#include <frsdk/cptr.h>
#include <fstream>
#include <iostream>

// small helper for tracing purpose
std::ostream&
operator<<( std::ostream& o, const FRsdk::Position& p)
{
  o << "[" << p.x() << ", " << p.y() << "]";
  return o;
}

namespace {
  class InvalidFIRAccessError: public std::exception
  {
  public:
    InvalidFIRAccessError() throw():
      msg("Trying to access invalid FIR") {}
    ~InvalidFIRAccessError() throw() { }
    const char* what() const throw() { return msg.c_str(); }
  private:
    std::string msg;
  };
}

// the concrete feedback which prints to stdout 
class EnrolCoutFeedback : public FRsdk::Enrollment::FeedbackBody
{
public:
  EnrolCoutFeedback( const std::string& firFilename)
    : firFN( firFilename), firvalid(false) { }
  ~EnrolCoutFeedback() {}

  // the feedback interface
  void start() {
    firvalid = false;
    std::cout << "start" << std::endl;
  }

  void processingImage( const FRsdk::Image& img) 
  {
    std::cout << "processing image[" << img.name() << "]" << std::endl;
  }
  
  void eyesFound( const FRsdk::Eyes::Location& eyeLoc) 
  {
    std::cout << "found eyes at ["<< eyeLoc.first 
              << " " << eyeLoc.second << "; confidences: " 
              << eyeLoc.firstConfidence << " " 
              << eyeLoc.secondConfidence << "]" << std::endl;
  }
  
  void eyesNotFound() 
  {    
    std::cout << "eyes not found" << std::endl;
  }

  void sampleQualityTooLow() {
    std::cout << "sampleQualityTooLow" << std::endl;
  }


  void sampleQuality( const float& f) {
    std::cout << "Sample Quality: " << f << std::endl;
  }  
  
  void success( const FRsdk::FIR& fir_) 
  {
    fir = new FRsdk::FIR(fir_);
    std::cout 
      << "successful enrollment";
    if(firFN != std::string("")) {
      
      std::cout << " FIR[filename,id,size] = [\"" 
      << firFN.c_str() << "\",\"" << (fir->version()).c_str() << "\"," 
      << fir->size() << "]";
      // write the fir
      std::ofstream firOut( firFN.c_str(), 
                    std::ios::binary|std::ios::out|std::ios::trunc);
      firOut << *fir;            
    }
    firvalid = true;
    std::cout << std::endl;
  }
  
  void failure() { std::cout << "failure" << std::endl; }
  
  void end() { std::cout << "end" << std::endl; }
  
  const FRsdk::FIR& getFir() const {
    // call only if success() has been invoked    
    if(!firvalid)
      throw InvalidFIRAccessError();
    
    return *fir;
  }

  bool firValid() const {
    return firvalid;
  }
    
private:
  FRsdk::CountedPtr<FRsdk::FIR> fir;
  std::string firFN;
  bool firvalid;
};

#endif
