/*
 * LbpFilter.hpp
 *
 *  Created on: 05.06.2013
 *      Author: poschmann
 */

#ifndef LBPFILTER_HPP_
#define LBPFILTER_HPP_

#include "imageprocessing/HistogramFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <array>
#include <stdexcept>

using cv::Ptr;
using cv::Size;
using cv::Point;
using cv::BaseFilter;
using std::array;
using std::invalid_argument;

namespace imageprocessing {

/**
 * Image filter that computes the (original) local binary pattern codes for each pixel.
 *
 * The code for computing the LBP codes was taken from http://www.bytefish.de/blog/local_binary_patterns/, where the
 * code for the extended LBP can be found, too.
 */
class LbpFilter : public HistogramFilter {
public:

	/**
	 * The local binary pattern type.
	 */
	enum class Type {
		LBP8, ///< Original local binary patterns using the 8-neighborhood, resulting in 256 bins.
		LBP8_UNIFORM, ///< LBP8 with all non-uniform patterns falling into the same bin, resulting in 59 bins.
		LBP4, ///< Local binary patterns using the 4-neighborhood (the pixel to the left, right, top and bottom), resulting in 16 bins.
		LBP4_ROTATED ///< Similar to LBP4, but using the diagonal neighbors (top left, top right, bottom left, bottom right), resulting in 16 bins.
	};

	/**
	 * Constructs a new LBP filter.
	 *
	 * @param[in] uniform Flag that indicates whether all non-uniform patterns should fall into the same bin.
	 */
	explicit LbpFilter(Type type = Type::LBP8);

	~LbpFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered) const;

	void applyInPlace(Mat& image) const;

	unsigned int getBinCount() const;

private:

	/**
	 * Determines whether the given pattern is uniform.
	 *
	 * @param[in] code The local binary pattern code.
	 * @return True if the pattern is uniform, false otherwise.
	 */
	bool isUniform(uchar code) const;

	/**
	 * Creates the base filter.
	 *
	 * @param[in] imageType The type of the image data.
	 */
	template <template <typename A> class T>
	Ptr<BaseFilter> createBaseFilter(int imageType) const {
		switch (imageType) {
			case CV_8U:  return Ptr<BaseFilter>(new T<uchar>());
			case CV_8S:  return Ptr<BaseFilter>(new T<char>());
			case CV_16U: return Ptr<BaseFilter>(new T<ushort>());
			case CV_16S: return Ptr<BaseFilter>(new T<short>());
			case CV_32S: return Ptr<BaseFilter>(new T<int>());
			case CV_32F: return Ptr<BaseFilter>(new T<float>());
			case CV_64F: return Ptr<BaseFilter>(new T<double>());
			default: throw invalid_argument("LbpFilter: unsupported image type " + imageType);
		}
	}

	Type type;             ///< The local binary pattern type.
	array<uchar, 256> map; ///< Maps values from LBP code to bin indices, putting all the non-uniform patterns into one bin.

	/**
	 * The LBP-8 filter.
	 */
	template <typename T>
	class Lbp8Filter : public BaseFilter {
	public:

		Lbp8Filter() {
			anchor = Point(1, 1);
			ksize = Size(3, 3);
		}

		~Lbp8Filter() {}

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

	/**
	 * The LBP-4 filter.
	 */
	template <typename T>
	class Lbp4Filter : public BaseFilter {
	public:

		Lbp4Filter() {
			anchor = Point(1, 1);
			ksize = Size(3, 3);
		}

		~Lbp4Filter() {}

		void operator()(const uchar** src, uchar* dst, int dststep, int height, int width, int cn) {
			for (int y = 0; y < height; ++y, dst += dststep) {
				const T* prevRow = reinterpret_cast<const T*>(src[y]);
				const T* currRow = reinterpret_cast<const T*>(src[y + 1]);
				const T* nextRow = reinterpret_cast<const T*>(src[y + 2]);
				for (int x = 0; x < width; ++x) {
					T center = currRow[x + 1];
					uchar code = 0;
					code |= (prevRow[x + 1] > center) << 3;
					code |= (currRow[x + 2] > center) << 2;
					code |= (nextRow[x + 1] > center) << 1;
					code |= (currRow[x] > center) << 0;
					dst[x] = code;
				}
			}
		}
	};

	/**
	 * The rotated LBP-4 filter.
	 */
	template <typename T>
	class RotatedLbp4Filter : public BaseFilter {
	public:

		RotatedLbp4Filter() {
			anchor = Point(1, 1);
			ksize = Size(3, 3);
		}

		~RotatedLbp4Filter() {}

		void operator()(const uchar** src, uchar* dst, int dststep, int height, int width, int cn) {
			for (int y = 0; y < height; ++y, dst += dststep) {
				const T* prevRow = reinterpret_cast<const T*>(src[y]);
				const T* currRow = reinterpret_cast<const T*>(src[y + 1]);
				const T* nextRow = reinterpret_cast<const T*>(src[y + 2]);
				for (int x = 0; x < width; ++x) {
					T center = currRow[x + 1];
					uchar code = 0;
					code |= (prevRow[x] > center) << 3;
					code |= (prevRow[x + 2] > center) << 2;
					code |= (nextRow[x + 2] > center) << 1;
					code |= (nextRow[x] > center) << 0;
					dst[x] = code;
				}
			}
		}
	};
};

} /* namespace imageprocessing */
#endif /* LBPFILTER_HPP_ */
