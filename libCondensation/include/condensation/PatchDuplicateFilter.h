/*
 * PatchDuplicateFilter.h
 *
 *  Created on: 30.08.2012
 *      Author: poschmann
 */

#ifndef PATCHDUPLICATEFILTER_H_
#define PATCHDUPLICATEFILTER_H_

#include <vector>
#include <string>

class FdPatch;

using std::vector;
using std::string;

namespace tracking {

/**
 * Mix-in for filtering duplicates of patches.
 */
class PatchDuplicateFilter {
public:

	virtual ~PatchDuplicateFilter() {}

protected:

	/**
	 * Takes the best distinct patches according to their certainty. The patches will be checked
	 * for equality by comparing the pointers.
	 *
	 * @param[in] patches The patches.
	 * @param[in] count The amount of distinct patches that should be taken.
	 * @param[in] detectorId The identifier of the detector used for computing the certainties.
	 * @return A new vector containing at most count different patches that have a higher certainty than the
	 *         ones that were not taken.
	 */
	vector<FdPatch*> takeDistinctBest(vector<FdPatch*> patches, unsigned int count, string detectorId);

	/**
	 * Takes the worst distinct patches according to their certainty. The patches will be checked
	 * for equality by comparing the pointers.
	 *
	 * @param[in] patches The patches.
	 * @param[in] count The amount of distinct patches that should be taken.
	 * @param[in] detectorId The identifier of the detector used for computing the certainties.
	 * @return A new vector containing at most count different patches that have a lower certainty than the
	 *         ones that were not taken.
	 */
	vector<FdPatch*> takeDistinctWorst(vector<FdPatch*> patches, unsigned int count, string detectorId);

	/**
	 * Takes the first n distinct patches. The patches will be checked for equality by comparing the pointers.
	 *
	 * @param[in] patches The patches.
	 * @param[in] count The amount of distinct patches that should be taken.
	 * @return A new vector containing at most count different patches.
	 */
	vector<FdPatch*> takeDistinct(const vector<FdPatch*>& patches, unsigned int count);
};

} /* namespace tracking */
#endif /* PATCHDUPLICATEFILTER_H_ */
