/*
 * stacktrace.hpp
 *
 *  Created on: 09.10.2015
 *      Author: poschmann
 */

#ifndef STACKTRACE_HPP_
#define STACKTRACE_HPP_

#include <cxxabi.h>
#include <execinfo.h>
#include <iostream>
#include <signal.h>
#include <string>
#include <string.h>
#include <unistd.h>

// created using the information from:
// https://medium.com/@hij1nx/how-to-make-debugging-c-errors-easier-7824922a8cc6
// http://stackoverflow.com/questions/77005/how-to-generate-a-stacktrace-when-my-gcc-c-app-crashes
// http://panthema.net/2008/0901-stacktrace-demangled/
// possible alternative: https://github.com/bombela/backward-cpp

void error_handler(int sig_num, siginfo_t * info, void * ucontext) {
	std::cout << "Error: " << strsignal(sig_num) << " (" << sig_num << ')' << std::endl;

	// get callstack and symbols
	void* callstack[50];
	int size = backtrace(callstack, 50);
	char** symbols = backtrace_symbols(callstack, size);

	// print stack frames, skip first (points here) and second (libc)
  for (int i = 2; i < size && symbols != nullptr; ++i) {
    std::string symbol = symbols[i];
    int pos_after_mangled = symbol.find_last_of('+');
    int pos_after_filename = symbol.find_last_of('(', pos_after_mangled);
    std::string filename = symbol.substr(0, pos_after_filename);
    std::string mangled_name = symbol.substr(pos_after_filename + 1, pos_after_mangled - pos_after_filename - 1);

    int status;
    char* real_name = abi::__cxa_demangle(mangled_name.c_str(), nullptr, nullptr, &status);
    if (status == 0) {
      std::cout << "  in " << filename << " at " << real_name << std::endl;
    } else {
    	std::cout << "  in " << filename << " at " << mangled_name << std::endl;
    }
    free(real_name);
  }
	free(symbols);
	exit(EXIT_FAILURE);
}

struct Trace {

	struct sigaction sa;

	Trace() {
		sa.sa_sigaction = error_handler;
    sa.sa_flags = SA_RESTART | SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);
	}
} trace;

#endif /* STACKTRACE_HPP_ */
