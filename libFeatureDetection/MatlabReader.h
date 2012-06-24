#pragma once

class MatlabReader
{
public:
	MatlabReader(void);
	MatlabReader(const char*);
	~MatlabReader(void);

	int getKey(const char*, char*); // liest einen Key oder eine Section
	int getInt(const char*, int*);	// liest einen Integer

private:
	MATFile *matFile;

};

