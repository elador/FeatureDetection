/*
 * PatchDuplicateFilter.cpp
 *
 *  Created on: 30.08.2012
 *      Author: poschmann
 */

#include "tracking/PatchDuplicateFilter.h"
#include "FdPatch.h"
#include <algorithm>

namespace tracking {

std::vector<FdPatch*> PatchDuplicateFilter::takeDistinctBest(std::vector<FdPatch*> patches,
		unsigned int count, std::string detectorId) {
	std::sort(patches.begin(), patches.end(), FdPatch::SortByCertainty(detectorId));
	return takeDistinct(patches, count);
}

std::vector<FdPatch*> PatchDuplicateFilter::takeDistinctWorst(std::vector<FdPatch*> patches,
		unsigned int count, std::string detectorId) {
	std::sort(patches.begin(), patches.end(), FdPatch::SortByCertainty(detectorId));
	std::reverse(patches.begin(), patches.end());
	return takeDistinct(patches, count);
}

std::vector<FdPatch*> PatchDuplicateFilter::takeDistinct(const std::vector<FdPatch*>& patches,
		unsigned int count) {
	if (patches.empty())
		return patches;
	std::vector<FdPatch*> remainingPatches;
	remainingPatches.reserve(count);
	std::vector<FdPatch*>::const_iterator pit = patches.begin();
	remainingPatches.push_back(*pit);
	++pit;
	for (; remainingPatches.size() < count && pit < patches.end(); ++pit) {
		if (std::find(remainingPatches.begin(), remainingPatches.end(), *pit) == remainingPatches.end())
			remainingPatches.push_back(*pit);
	}
	return remainingPatches;
}

} /* namespace tracking */
