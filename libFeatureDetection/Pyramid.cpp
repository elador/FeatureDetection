#include "StdAfx.h"
#include "Pyramid.h"


Pyramid::Pyramid(void)
{
}


Pyramid::~Pyramid(void)
{
	FdPatchSet::iterator it = patches.begin();
	FdPatchSet::iterator tmp;		// see http://stackoverflow.com/questions/2874441/deleting-elements-from-stl-set-while-iterating-new-solution
	for(; it != patches.end(); ) {
		tmp = it;
		++tmp;
		delete (*it);
		it = tmp;
	}
	/*FdPatchSet::iterator it = patches.begin();
	for(; it != patches.end(); it++) {
		delete it->second;
		it->second = NULL;
	}*/
}
