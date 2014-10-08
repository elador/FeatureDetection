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
#include "boost/serialization/split_free.hpp"
#include "boost/serialization/vector.hpp"

/**
 * Serialization for the OpenCV cv::Mat class.
 *
 * Based on http://cheind.wordpress.com/2011/12/06/serialization-of-cvmat-objects-using-boost/
 * Original post with some further hints: http://stackoverflow.com/questions/4170745/serializing-opencv-mat-vec3f
 *
 * Note: Only works for contiguous matrices so far (i.e. use cv::Mat::clone()
 * or cv::Mat::isContiguous() to be sure)
 *
 * Possible improvements:
 * - http://www.boost.org/doc/libs/1_56_0/libs/serialization/doc/serialization.html#splitting
 *   "Note that although the functionality to split the serialize function into save/load has
 *    been provided, the usage of the serialize function with the corresponding & operator is
 *    preferred. The key to the serialization implementation is that objects are saved and
 *    loaded in exactly the same sequence. Using the & operator and serialize function guarantees
 *    that this is always the case and will minimize the occurrence of hard to find errors related
 *    to synchronization of save and load functions."
 *
 * Todos:
 *  - Add the unit tests from above blog.
 *  - Add XML support (need to add make_nvp stuff?)
 */
BOOST_SERIALIZATION_SPLIT_FREE(::cv::Mat)
namespace boost {
	namespace serialization {

		/**
		 * Serialize a cv::Mat. Save method.
		 * Note: Has to be contiguous.
		 */
		template<class Archive>
		void save(Archive & ar, const ::cv::Mat& m, const unsigned int version)
		{
			int elem_type = m.type();

			ar & m.rows;
			ar & m.cols;
			ar & elem_type;

			const size_t data_size = m.rows * m.cols * m.elemSize();
			ar & boost::serialization::make_array(m.ptr(), data_size);
		}

		/**
		 * Serialize a cv::Mat. Load method.
		 * Note: Has to be contiguous.
		 */
		template<class Archive>
		void load(Archive & ar, ::cv::Mat& m, const unsigned int version)
		{
			int rows, cols;
			int elem_type;

			ar & rows;
			ar & cols;
			ar & elem_type;

			m.create(rows, cols, elem_type);
			size_t data_size = m.cols * m.rows * m.elemSize();
			ar & boost::serialization::make_array(m.ptr(), data_size);
		}

	}
}

#endif /* MATSERIALIZATION_HPP_ */
