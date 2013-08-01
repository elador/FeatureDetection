/*!
 * \file MatrixUtils.hpp
 *
 * \author Patrik Huber
 * \date December 6, 2012
 *
 * [comment here]
 */
#pragma once
#ifndef MATRIXUTILS_HPP_
#define MATRIXUTILS_HPP_

#include "opencv2/core/core.hpp"

// Todo: Class with static methods? Or just functions? I don't know which method is better.

namespace render {

	namespace utils {

		class MatrixUtils
		{
		public:
			static cv::Mat createRotationMatrixX(float angle);
			static cv::Mat createRotationMatrixY(float angle);
			static cv::Mat createRotationMatrixZ(float angle);
			static cv::Mat createScalingMatrix(float sx, float sy, float sz);
			static cv::Mat createTranslationMatrix(float tx, float ty, float tz);

			static unsigned char getMaxPossibleMipmapsNum(ushort width, ushort height)	// TODO: This belongs more in a ImageUtils, TextureUtils, or whatever...
			{
				unsigned char mipmapsNum = 1;
				ushort size = std::max(width, height);

				if (size == 1)
					return 1;

				do {
					size >>= 1;
					mipmapsNum++;
				} while (size != 1);

				return mipmapsNum;
			}

		};

	}

}
#endif /* MATRIXUTILS_HPP_ */
