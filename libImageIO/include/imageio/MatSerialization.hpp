/*
 * MatSerialization.hpp
 *
 *  Created on: 08.10.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef MATSERIALIZATION_HPP_
#define MATSERIALIZATION_HPP_

#include "opencv2/core/core.hpp"
//#include "boost/serialization/vector.hpp" // Todo: Some includes are most likely missing

/**
 * Serialization for the OpenCV cv::Mat class.
 *
 * Based on answer from: http://stackoverflow.com/questions/4170745/serializing-opencv-mat-vec3f 
 * Different method and tests: http://cheind.wordpress.com/2011/12/06/serialization-of-cvmat-objects-using-boost/
 *
 * Todos:
 *  - Add the unit tests from above blog.
 *  - Add XML support (need to add make_nvp stuff?)
 */
namespace boost {
	namespace serialization {

		/**
		 * Serialize a cv::Mat using boost::serialization.
		 *
		 * Supports all types of matrices as well as non-contiguous ones.
		 */
		template<class Archive>
		void serialize(Archive &ar, cv::Mat& mat, const unsigned int)
		{
			int rows, cols, type;
			bool continuous;

			if (Archive::is_saving::value) {
				rows = mat.rows; cols = mat.cols; type = mat.type();
				continuous = mat.isContinuous();
			}

			ar & rows & cols & type & continuous;

			if (Archive::is_loading::value)
				mat.create(rows, cols, type);

			if (continuous) {
				const unsigned int data_size = rows * cols * mat.elemSize();
				ar & boost::serialization::make_array(mat.ptr(), data_size);
			}
			else {
				const unsigned int row_size = cols * mat.elemSize();
				for (int i = 0; i < rows; i++) {
					ar & boost::serialization::make_array(mat.ptr(i), row_size);
				}
			}
		};

	} /* namespace serialization */
} /* namespace boost */

#endif /* MATSERIALIZATION_HPP_ */
