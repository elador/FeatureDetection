/*
 * FilterTree.hpp
 *
 *  Created on: 07.10.2015
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_FILTERTREE_HPP_
#define IMAGEPROCESSING_FILTERING_FILTERTREE_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include <memory>
#include <vector>

namespace imageprocessing {
namespace filtering {

namespace detail {

/**
 * Image filter that passes the filtered image to additional nodes and returns the collected filtered images.
 */
class FilterNode {
public:

	/**
	 * Constructs a new filter node.
	 *
	 * @param[in] filter Filter that is applied to the images before passing it to the filter chains.
	 */
	FilterNode(const std::shared_ptr<ImageFilter>& filter);

	/**
	 * Applies the filter to an image and the previously added filter chains to the filtered image.
	 *
	 * @param[in] image Image that should be put through the filter chains.
	 * @return Result of each of the filter chains.
	 */
	std::vector<cv::Mat> applyTo(const cv::Mat& image) const;

	/**
	 * Adds a filter chain that is applied to images after the filter of this node.
	 *
	 * @param[in] begin Iterator pointing to the first filter of the chain.
	 * @param[in] end Iterator pointing behind the last filter of the chain.
	 */
	void addChain(
			std::vector<std::shared_ptr<ImageFilter>>::const_iterator begin,
			std::vector<std::shared_ptr<ImageFilter>>::const_iterator end);

	/**
	 * Returns the child node that applies the given filter. Will create that node if it does not exist yet.
	 *
	 * @param[in] filter Image filter that should be applied by a child node.
	 * @return Reference to the child node that applies the filter.
	 */
	FilterNode& getNode(const std::shared_ptr<ImageFilter>& filter);

	/**
	 * Determines whether this node applies the given filter.
	 *
	 * @param[in] filter Filter that is tested against this node.
	 * @return True if the given filter is the same as the filter that is applied by this node, false otherwise.
	 */
	bool hasFilter(const std::shared_ptr<ImageFilter>& filter) const {
		return this->filter == filter;
	}

private:

	bool returnFilteredImage; ///< Flag that indicates whether the filter output of this node should be returned with the results.
	std::shared_ptr<ImageFilter> filter; ///< Filter that is applied to the images.
	std::vector<FilterNode> nodes; ///< Nodes the filtered image is passed down to.
};

} /* namespace detail */

/**
 * Image filter that applies several filter chains to an image and returns the merged filtered images.
 *
 * If filter chains start with the same filters, then they will be applied only once and the chains will share
 * the resulting filtered image, so there are no duplicate filter operations. To accomplish this goal, a tree
 * structure is created, where each node applies one filter, passes the filtered image down to other nodes and
 * collects and returns their filter results.
 */
class FilterTree : public ImageFilter {
public:

	using ImageFilter::applyTo;

	/**
	 * Applies the previously added filter chains to an image.
	 *
	 * @param[in] image Image that should be put through the filter chains.
	 * @param[out] result Image for writing the merged results of the filter chains into.
	 * @return Image containing the merged results of the filter chains.
	 */
	cv::Mat applyTo(const cv::Mat& image, cv::Mat& result) const;

	/**
	 * Adds a filter chain that is applied to images.
	 *
	 * @param[in] filters Filters that should be applied in the given order.
	 */
	void addChain(const std::vector<std::shared_ptr<ImageFilter>>& filters);

	/**
	 * Returns the child node that applies the given filter. Will create that node if it does not exist yet.
	 *
	 * @param[in] filter Image filter that should be applied by a child node.
	 * @return Reference to the child node that applies the filter.
	 */
	detail::FilterNode& getNode(const std::shared_ptr<ImageFilter>& filter);

private:

	std::vector<detail::FilterNode> nodes; ///< Nodes the image is passed down to.
};

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_FILTERTREE_HPP_ */
