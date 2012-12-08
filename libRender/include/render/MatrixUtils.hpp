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

#include <opencv2/core/core.hpp>

// Todo: Class with static methods? Or just functions? I don't know which method is better.

namespace render {

	namespace utils {

		class MatrixUtils
		{
		public:
			static cv::Mat createRotationMatrixX(float angle)
			{
				cv::Mat rotX = (cv::Mat_<float>(4,4) << 
						1.0f,				0.0f,				0.0f,				0.0f,
						0.0f,				std::cosf(angle),	std::sin(angle),	0.0f,
						0.0f,				-std::sin(angle),	std::cos(angle),	0.0f,
						0.0f,				0.0f,				0.0f,				1.0f);
				return rotX;
			};

			static cv::Mat createRotationMatrixY(float angle)
			{
				cv::Mat rotY = (cv::Mat_<float>(4,4) << 
						std::cos(angle),	0.0f,				-std::sin(angle),	0.0f,
						0.0f,				1.0f,				0.0f,				0.0f,
						std::sin(angle),	0.0f,				std::cos(angle),	0.0f,
						0.0f,				0.0f,				0.0f,				1.0f);
				return rotY;
			};

			static cv::Mat createRotationMatrixZ(float angle)
			{
				cv::Mat rotZ = (cv::Mat_<float>(4,4) << 
						std::cos(angle),	std::sin(angle),	0.0f,				0.0f,
						-std::sin(angle),	std::cos(angle),	0.0f,				0.0f,
						0.0f,				0.0f,				1.0f,				0.0f,
						0.0f,				0.0f,				0.0f,				1.0f);
				return rotZ;
			}

			static cv::Mat createScalingMatrix(float sx, float sy, float sz)
			{
				cv::Mat scaling = (cv::Mat_<float>(4,4) << 
						sx,					0.0f,				0.0f,				0.0f,
						0.0f,				sy,					0.0f,				0.0f,
						0.0f,				0.0f,				sz,					0.0f,
						0.0f,				0.0f,				0.0f,				1.0f);
				return scaling;
			}

			static cv::Mat createTranslationMatrix(float tx, float ty, float tz)
			{
				cv::Mat translation = (cv::Mat_<float>(4,4) << 
						1.0f,				0.0f,				0.0f,				0.0f,
						0.0f,				1.0f,				0.0f,				0.0f,
						0.0f,				0.0f,				1.0f,				0.0f,
						tx,					ty,					tz,					1.0f);
				return translation;
			}

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
