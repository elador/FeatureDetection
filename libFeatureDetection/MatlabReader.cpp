#include "stdafx.h"
#include "MatlabReader.h"

#include "SLogger.h"

#include <iostream>
#include <cstring>

MatlabReader::MatlabReader(void)
{
	this->matFile = NULL;
}

MatlabReader::MatlabReader(const std::string filename)
{
	this->matFile = matOpen(filename.c_str(), "r");
	if (this->matFile == 0) {
		std::cout << "[MatlabReader] Error: Unable to open file " << filename << std::endl;
		exit(EXIT_FAILURE);
		return;
	}
	if(Logger->getVerboseLevelText()>=3) {
		std::cout << "[MatlabReader] opened " << filename << std::endl;
	}
}

MatlabReader::~MatlabReader(void)
{
	if(this->matFile!=NULL) {
		if (matClose(this->matFile) != 0) {
			std::cout << "[MatlabReader] Error closing file." << std::endl;
		}
	}
}


int MatlabReader::getKey(const char *key, char *buffer) // liest einen Key oder eine Section
{
	if (this->matFile==NULL) {
		std::cout << "[MatlabReader] Error: file not opened." << std::endl;
		return false;
	} else {
		char sc[255]="",ky[255]="";
		int id=-1;

		//"ALLGINFO.dumping.#9"
		int i=0,len=strlen(key); *sc=*ky=0;
		while ((key[i]!='.') && (i<len)) sc[i]=key[i++];
		if (i==0) {
			std::cout << "getKey() Error in pos, no Section; '"<< key << "'\n" << std::endl;
			return false;
		}
		if (i>=len) {
			std::cout << "getKey() Error in pos, no '.'; '"<< key << "'\n" << std::endl;
			return false;
		}
		int j=0; sc[i]=0;  i++; 
 		while ((key[i]!='.') && (i<len)) ky[j++]=key[i++];
		ky[j]=0;
		if (!strncmp(&key[i],".#",2) && (key[i+2]!=0)) {
			id=atoi(&key[i+2]);
		}
		//fprintf(stdout,"key:'%s' => '%s.%s.#%d'\n",key,sc,ky,id);

					
		mxArray* mat = matGetVariable(this->matFile, sc);  
		if (mat == 0 || !mxIsStruct(mat)) {
 			std::cout << "getKey() Error in pos, Section is not found or no struct'.'; '"<< sc << "'\n" << std::endl;
			return false;
		}

		mxArray* mk=mxGetField(mat,0,ky);
  		if (mk == 0) {
 			std::cout << "getKey() Error in pos '"<< key << "', key '"<< ky << "' not found in section '"<< sc << "'" << std::endl;
			return false;
		}
			
		if (id>-1) {
 			double *b=(double*)mxGetPr(mk);
  			if (mk == 0) {
 				std::cout << "getKey() Error in pos '"<< key << "', id " << id << " not found\n" << std::endl;
				return false;
			}
			sprintf(buffer,"%1.16f",b[id]);
		} else
			if (mxGetString(mk,buffer,255))   {
				double *b=(double*)mxGetPr(mk);
  				if (mk == 0) {
 					std::cout << "getKey() Error in pos '"<< key << "', id " << id << " not found" << std::endl;
					return false;
				}
				sprintf(buffer,"%1.16f",b[0]);
			}
		len=strlen(buffer);
		for (i=0;i<len;i++) 
			#ifdef WIN32
				if (buffer[i]=='/')  buffer[i]='\\';
			#else
				if (buffer[i]=='\\')  buffer[i]='/';
			#endif //WIN32

		mxDestroyArray(mat);

	}

	return true;
}

int MatlabReader::getInt(const char *k, int *i)	// liest einen Integer
{
	char buffer[255];
	int rc;

	if ((rc=getKey(k, buffer))!=0)
		*i=atoi(buffer);
	else 
		*i=0;

	return rc;
}

