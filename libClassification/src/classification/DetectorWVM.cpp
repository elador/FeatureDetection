
/*
Returns a bool if the patch passes the cascade (always true for SVMs, for WVM, only true if passes last filter)
*/
bool DetectorWVM::classify(FdPatch* fp)
{
	//std::cout << "[DetWVM] Classifying!\n";

	// Check if the patch has already been classified by this detector! If yes, don't do it again.
	// OK. We can't do this here! Because we do not know/save "filter_level". So we don't know when
	// the patch dropped out. Only the fout-value is not sufficient. So we can't know if we should
	// return true or false.
	// Possible solution: Store the filter_level somewhere.

	// So: The fout value is not already computed. Go ahead.

	// patch II of fp already calc'ed?
	if(fp->iimg_x == NULL) {
		fp->iimg_x = new IImg(this->filter_size_x, this->filter_size_y, 8);
		fp->iimg_x->calIImgPatch(fp->data, false);
	}
	if(fp->iimg_xx == NULL) {
		fp->iimg_xx = new IImg(this->filter_size_x, this->filter_size_y, 8);
		fp->iimg_xx->calIImgPatch(fp->data, true);
	}

	for (int n=0;n<this->numFiltersPerLevel;n++) {
		u_kernel_eval[n]=0.0f;
	}
	int filter_level=-1;
	float fout = 0.0;
	do {
		filter_level++;
		fout = this->linEvalWvmHisteq64(filter_level, (filter_level%this->numFiltersPerLevel), fp->c.x_py, fp->c.y_py, filter_output, u_kernel_eval, fp->iimg_x, fp->iimg_xx);
	//} while (fout >= this->hierarchicalThresholds[filter_level] && filter_level+1 < this->numLinFilters); //280
	} while (fout >= this->hierarchicalThresholds[filter_level] && filter_level+1 < this->numUsedFilters); //280
	
	//fp->fout = fout;
	std::pair<FoutMap::iterator, bool> fout_insert = fp->fout.insert(FoutMap::value_type(this->identifier, fout));
	if(fout_insert.second == false) {
		if(Logger->getVerboseLevelText()>=4) {
			std::cout << "[DetectorWVM] An element 'fout' already exists for this detector, you classified the same patch twice. We can't circumvent this for now." << std::endl;
		}
	}
	// fout = final result now!


	// TODO: filter statistics, nDropedOutAsNonFace[filter_level]++;
	// We ran till the REAL LAST filter (not just the numUsedFilters one), save the certainty
	if(filter_level+1 == this->numLinFilters && fout >= this->hierarchicalThresholds[filter_level]) {
		return true;	// Positive patch!
	}


}



