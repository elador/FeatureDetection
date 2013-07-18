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
using cv::Size;
using cv::Point2f;
using cv::Scalar;
using std::shared_ptr;
using std::pair;

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
	 * @param[in] scatter The scatter that controls the diffusion of the position.
	 * @param[in] gridSize The size of the point grid (amount of points).
	 * @param[in] circle Flag that indicates whether the point grid should be a circle (points in the corners are discarded).
	 * @param[in] windowSize Size of the search window for the optical flow calculation.
	 * @param[in] maxLevel 0-based maximal pyramid level number of the optical flow calculation.
	 */
	explicit OpticalFlowTransitionModel(shared_ptr<TransitionModel> fallback, double scatter = 0.02,
			Size gridSize = Size(10, 10), bool circle = true, Size windowSize = Size(15, 15), int maxLevel = 2);

	~OpticalFlowTransitionModel();

	void init(const Mat& image);

	void predict(vector<Sample>& samples, const Mat& image, const optional<Sample>& target);

	/**
	 * Draws the flow of the chosen points into an image.
	 *
	 * @param[in] image The image.
	 * @param[in] thickness The thickness of the flow lines or radius of points. If negative, only the current correct points will be drawn.
	 * @param[in] color The color of the flow lines or points.
	 * @param[in] badColor The color of the bad (ignored) flow lines.
	 */
	void drawFlow(Mat& image, int thickness, Scalar color, Scalar badColor = Scalar(-1, -1, -1)) const;

	/**
	 * @return The scatter that controls the diffusion of the position.
	 */
	double getScatter() {
		return scatter;
	}

	/**
	 * @param[in] scatter The new scatter that controls the diffusion of the position.
	 */
	void setScatter(double scatter) {
		this->scatter = scatter;
	}

private:

	/**
	 * Converts the given image to grayscale, if necessary.
	 *
	 * @param[in] image The image.
	 * @return A grayscale image.
	 */
	const Mat makeGrayscale(const Mat& image) const;

	shared_ptr<TransitionModel> fallback; ///< Fallback model in case the optical flow cannot be used.
	vector<Point2f> templatePoints; ///< Grid of points that will be scaled and moved to the target position to then get tracked.
	Size gridSize;   ///< The size of the point grid.
	Size windowSize; ///< Size of the search window for the optical flow calculation.
	int maxLevel;    ///< 0-based maximal pyramid level number of the optical flow calculation.
	vector<Mat> previousPyramid; ///< Image pyramid of the previous image for the optical flow calculation.
	vector<Mat> currentPyramid;  ///< Image pyramid of the current image for the optical flow calculation.
	vector<Point2f> points;         ///< Points in the previous image where the optical flow should be calculated.
	vector<Point2f> forwardPoints;  ///< Resulting points of the optical flow calculation in the current frame.
	vector<Point2f> backwardPoints; ///< Resulting points of the backwards optical flow calculation in the previous frame.
	vector<uchar> forwardStatus;    ///< Status of each point of the forward calculation.
	vector<uchar> backwardStatus;   ///< Status of each point of the backward calculation.
	vector<float> error;            ///< Error values of the points.
	vector<pair<float, int>> squaredDistances; ///< Forward-backward-error paired with the point index, ordered by the error.
	unsigned int correctFlowCount;             ///< The amount of flows that are considered correct and are used for median flow.
	double scatter; ///< The scatter that controls the diffusion of the position.
	boost::variate_generator<boost::mt19937, boost::normal_distribution<>> generator; ///< Random number generator.
};

} /* namespace condensation */
#endif /* OPTICALFLOWTRANSITIONMODEL_HPP_ */
