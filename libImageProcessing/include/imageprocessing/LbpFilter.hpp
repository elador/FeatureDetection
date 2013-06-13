/*
 * LbpFilter.hpp
 *
 *  Created on: 05.06.2013
 *      Author: poschmann
 */

#ifndef LBPFILTER_HPP_
#define LBPFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <array>

using cv::BaseFilter;
using std::array;

namespace imageprocessing {

/**
 * Image filter that computes the (original) local binary pattern codes for each pixel. If all non-uniform patterns should
 * be put into one bin, the resulting amount of bins is 59. Otherwise, the amount of bins is 256. The input image has to
 * have a single channel.
 *
 * The code for computing the LBP codes was taken from http://www.bytefish.de/blog/local_binary_patterns/, where can be found
 * the code for the extended LBP, too.
 */
class LbpFilter : public ImageFilter {
public:

	/**
	 * Constructs a new LBP filter.
	 *
	 * @param[in] uniform Flag that indicates whether all non-uniform patterns should fall into the same bin.
	 */
	explicit LbpFilter(bool uniform);

	~LbpFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered);

	void applyInPlace(Mat& image);

private:

	/**
	 * Determines whether the given pattern is uniform.
	 *
	 * @param[in] code The local binary pattern code.
	 * @return True if the pattern is uniform, false otherwise.
	 */
	bool isUniform(uchar code);

	bool uniform;          ///< Flag that indicates whether all non-uniform patterns should fall into the same bin.
	array<uchar, 256> map; ///< Maps values from LBP code to bin indices, putting all the non-uniform patterns into one bin.

	/**
	 * The actual filter.
	 */
	template <typename T>
	class Filter : public BaseFilter {
	public:

		Filter() {
			anchor = cv::Point(1, 1);
			ksize = cv::Size(3, 3);
		}

		~Filter() {}

		void operator()(const uchar** src, uchar* dst, int dststep, int height, int width, int cn) {
			for (int y = 0; y < height; ++y, dst += dststep) {
				const T* prevRow = reinterpret_cast<const T*>(src[y]);
				const T* currRow = reinterpret_cast<const T*>(src[y + 1]);
				const T* nextRow = reinterpret_cast<const T*>(src[y + 2]);
				for (int x = 0; x < width; ++x) {
					T center = currRow[x + 1];
					uchar code = 0;
					code |= (prevRow[x] > center) << 7;
					code |= (prevRow[x + 1] > center) << 6;
					code |= (prevRow[x + 2] > center) << 5;
					code |= (currRow[x + 2] > center) << 4;
					code |= (nextRow[x + 2] > center) << 3;
					code |= (nextRow[x + 1] > center) << 2;
					code |= (nextRow[x] > center) << 1;
					code |= (currRow[x] > center) << 0;
					dst[x] = code;
				}
			}
		}
	};
};

} /* namespace imageprocessing */
#endif /* LBPFILTER_HPP_ */
