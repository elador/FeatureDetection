/*
 * OpticalFlowTransitionModel.hpp
 *
 *  Created on: 14.06.2013
 *      Author: poschmann
 */

#ifndef OPTICALFLOWTRANSITIONMODEL_HPP_
#define OPTICALFLOWTRANSITIONMODEL_HPP_

#include "condensation/TransitionModel.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/variate_generator.hpp"
#include <memory>

namespace condensation {

/**
 * Optical flow based transition model.
 */
class OpticalFlowTransitionModel : public TransitionModel {
public:

	/**
	 * Constructs a new optical flow transition model.
	 *
	 * @param[in] fallback Fallback model in case the optical flow cannot be used.
	 * @param[in] positionDeviation Standard deviation of the translation noise.
	 * @param[in] sizeDeviation Standard deviation of the scale change noise.
	 * @param[in] gridSize The size of the point grid (amount of points).
	 * @param[in] circle Flag that indicates whether the point grid should be a circle (points in the corners are discarded).
	 * @param[in] windowSize Size of the search window for the optical flow calculation.
	 * @param[in] maxLevel 0-based maximal pyramid level number of the optical flow calculation.
	 */
	explicit OpticalFlowTransitionModel(std::shared_ptr<TransitionModel> fallback, double positionDeviation, double sizeDeviation,
			cv::Size gridSize = cv::Size(10, 10), bool circle = true, cv::Size windowSize = cv::Size(15, 15), int maxLevel = 2);

	void init(const cv::Mat& image);

	void predict(std::vector<std::shared_ptr<Sample>>& samples, const cv::Mat& image, const std::shared_ptr<Sample> target);

	/**
	 * Draws the flow of the chosen points into an image.
	 *
	 * @param[in] image The image.
	 * @param[in] thickness The thickness of the flow lines or radius of points. If negative, only the current correct points will be drawn.
	 * @param[in] color The color of the flow lines or points.
	 * @param[in] badColor The color of the bad (ignored) flow lines.
	 */
	void drawFlow(cv::Mat& image, int thickness, cv::Scalar color, cv::Scalar badColor = cv::Scalar(-1, -1, -1)) const;

	/**
	 * @return The standard deviation of the translation noise.
	 */
	double getPositionDeviation() const {
		return positionDeviation;
	}

	/**
	 * @param[in] deviation The new standard deviation of the translation noise.
	 */
	void setPositionDeviation(double deviation) {
		this->positionDeviation = deviation;
	}

	/**
	 * @return The standard deviation of the scale change noise.
	 */
	double getSizeDeviation() const {
		return sizeDeviation;
	}

	/**
	 * @param[in] deviation The new standard deviation of the scale change noise.
	 */
	void setSizeDeviation(double deviation) {
		this->sizeDeviation = deviation;
	}

private:

	/**
	 * Converts the given image to grayscale, if necessary.
	 *
	 * @param[in] image The image.
	 * @return A grayscale image.
	 */
	const cv::Mat makeGrayscale(const cv::Mat& image) const;

	std::shared_ptr<TransitionModel> fallback; ///< Fallback model in case the optical flow cannot be used.
	std::vector<cv::Point2f> templatePoints; ///< Grid of points that will be scaled and moved to the target position to then get tracked.
	cv::Size gridSize;   ///< The size of the point grid.
	cv::Size windowSize; ///< Size of the search window for the optical flow calculation.
	int maxLevel;        ///< 0-based maximal pyramid level number of the optical flow calculation.
	std::vector<cv::Mat> previousPyramid; ///< Image pyramid of the previous image for the optical flow calculation.
	std::vector<cv::Mat> currentPyramid;  ///< Image pyramid of the current image for the optical flow calculation.
	std::vector<cv::Point2f> points;         ///< Points in the previous image where the optical flow should be calculated.
	std::vector<cv::Point2f> forwardPoints;  ///< Resulting points of the optical flow calculation in the current frame.
	std::vector<cv::Point2f> backwardPoints; ///< Resulting points of the backwards optical flow calculation in the previous frame.
	std::vector<uchar> forwardStatus;    ///< Status of each point of the forward calculation.
	std::vector<uchar> backwardStatus;   ///< Status of each point of the backward calculation.
	std::vector<float> error;            ///< Error values of the points.
	std::vector<std::pair<float, int>> squaredDistances; ///< Forward-backward-error paired with the point index, ordered by the error.
	unsigned int correctFlowCount; ///< The amount of flows that are considered correct and are used for median flow.
	double positionDeviation; ///< Standard deviation of the translation noise.
	double sizeDeviation;     ///< Standard deviation of the scale change noise.
	boost::variate_generator<boost::mt19937, boost::normal_distribution<>> generator; ///< Random number generator.
};

} /* namespace condensation */
#endif /* OPTICALFLOWTRANSITIONMODEL_HPP_ */
