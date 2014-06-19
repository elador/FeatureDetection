// -*- c++ -*-
// Copyright (c) 2002 Cognitec Systems GmbH
//
// $Revision: 1.7 $
//

/** \file cmdline.h
    \brief minimal command line parser 

    this is a really minimal command line parser to look for certain
    command line flags and flag options.
 */ 

#ifndef FRSDK_CMDLINE_H
#define FRSDK_CMDLINE_H

#include <string>
#include <algorithm>
#include <functional>
#include <cctype>

#include <frsdk/platform.h>

namespace FRsdk
{
  class CmdLine
  {
  public: 
  /** Creates a CmdLine. The arguments should in the same form as
      passed to main() */
    CmdLine( int argc, char const* const* argv )
     : argc_( argc ), argv_( argv ) {}

    /// returns true if the given string appears in the command line option.
    bool hasflag( const char* flag ) const
    {
      if ( argc_ < 2 || flag == 0)
        return false;
      
      for ( int i = 1; i < argc_; i++ )
      {
        std::string currArg( argv_[i] );
        if ( currArg == flag )
          return true;
      }
      return false;
    } // hasflag

    /** if the given flag appears in the command line option,
        the succeeding argument will be returned, otherwise returns 0 */
    const char* getspaceflag( const char* flag ) const
    {
      if ( argc_ < 2 || flag == 0 )
        return 0;
      
      for ( int i = 1; i < argc_; i++ )
      {
        std::string currArg( argv_[i] );
        if ( currArg == flag )
        {
          if ( i == argc_ - 1 )
            return 0;
          else
            return( argv_[i + 1] );
        }
      }
      return 0;
    } // getspaceflag

  private:
    const int argc_;
    char const* const* argv_;
  };
  
  template<typename Container>
  Container parseDelimitedText( const std::string& text, 
                               char delimiter = ',',
                               bool stripLeadingSpaces = true )
   {
     Container container;

     std::string::const_iterator curPos = text.begin();
     while ( true )
     {
       std::string::const_iterator nextDelimiter = std::find( curPos,
                                                              text.end(),
                                                              delimiter );

       if ( stripLeadingSpaces )
           curPos = std::find_if( curPos,
                                  nextDelimiter,
                                  std::not1( std::ptr_fun( ::isspace ) ) );

       container.push_back( std::string( curPos, nextDelimiter ) );
       
       if ( nextDelimiter == text.end() )
         break;

       curPos = ++nextDelimiter;
     }
     
     return container;
   }

} // namespace FRsdk

#endif
