/*
 * FilterTree.cpp
 *
 *  Created on: 07.10.2015
 *      Author: poschmann
 */

#include "imageprocessing/filtering/FilterTree.hpp"
#include <stdexcept>

using cv::Mat;
using imageprocessing::filtering::detail::FilterNode;
using std::invalid_argument;
using std::shared_ptr;
using std::vector;

namespace imageprocessing {
namespace filtering {

Mat FilterTree::applyTo(const Mat& image, Mat& result) const {
	vector<Mat> results;
	for (const FilterNode& node : nodes) {
		vector<Mat> nodeResults = node.applyTo(image);
		results.insert(results.end(), nodeResults.begin(), nodeResults.end());
	}
	if (results.empty())
		return Mat();
	cv::merge(results, result);
	return result;
}

void FilterTree::addChain(const vector<shared_ptr<ImageFilter>>& filters) {
	if (filters.empty())
		throw invalid_argument("FilterTree: filter chain must not be empty");
	getNode(filters.front()).addChain(filters.begin() + 1, filters.end());
}

FilterNode& FilterTree::getNode(const shared_ptr<ImageFilter>& filter) {
	for (FilterNode& node : nodes) {
		if (node.hasFilter(filter))
			return node;
	}
	nodes.emplace_back(filter);
	return nodes.back();
}

namespace detail {

FilterNode::FilterNode(const shared_ptr<ImageFilter>& filter) :
		returnFilteredImage(false), filter(filter), nodes() {}

vector<Mat> FilterNode::applyTo(const Mat& image) const {
	vector<Mat> results;
	Mat filteredImage = filter->applyTo(image);
	if (returnFilteredImage)
		results.push_back(filteredImage);
	for (const FilterNode& node : nodes) {
		vector<Mat> nodeResults = node.applyTo(filteredImage);
		results.insert(results.end(), nodeResults.begin(), nodeResults.end());
	}
	return results;
}

void FilterNode::addChain(
		vector<shared_ptr<ImageFilter>>::const_iterator begin,
		vector<shared_ptr<ImageFilter>>::const_iterator end) {
	if (begin == end) // this is the last node of the chain
		returnFilteredImage = true; // so we need to collect the filter output
	else // there are more filters to apply to the chain
		getNode(*begin).addChain(begin + 1, end);
}

FilterNode& FilterNode::getNode(const shared_ptr<ImageFilter>& filter) {
	for (FilterNode& node : nodes) {
		if (node.hasFilter(filter))
			return node;
	}
	nodes.emplace_back(filter);
	return nodes.back();
}

} /* namespace detail */

} /* namespace filtering */
} /* namespace imageprocessing */
