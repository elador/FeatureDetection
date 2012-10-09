#pragma once

#include "mat.h"

#include <string>

class MatlabReader
{
public:
	MatlabReader(void);
	MatlabReader(const std::string);
	~MatlabReader(void);

	int getKey(const char*, char*); // liest einen Key oder eine Section
	int getInt(const char*, int*);	// liest einen Integer

private:
	MATFile *matFile;

};

