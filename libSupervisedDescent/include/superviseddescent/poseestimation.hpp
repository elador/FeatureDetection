/*
 * poseestimation.hpp
 *
 *  Created on: 05.12.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef POSEESTIMATION_HPP_
#define POSEESTIMATION_HPP_

#include "superviseddescent.hpp"
#include "matserialisation.hpp"
#include "logging/LoggerFactory.hpp"

#include "opencv2/core/core.hpp"
#include "Eigen/Dense"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/serialization/vector.hpp"

#include <memory>
#include <chrono>

namespace superviseddescent {
	namespace v2 {

/**
 * T.
 *
 * @param[in] a B.
 * @return c.
 */
class ModelFitting
{
public:
	ModelFitting()
	{

	};
	ModelFitting(float rx, float ry, float rz, float tx, float ty, boost::optional<float> tz, std::vector<float> shapeCoeffs, std::vector<float> albedoCoeffs, float focalLength) :
		rotationX(rx), rotationY(ry), rotationZ(rz), tx(tx), ty(ty), tz(tz), shapeCoeffs(shapeCoeffs), albedoCoeffs(albedoCoeffs), focalLength(focalLength)
	{

	};
	// Note: Better would be to internally store the rotation as a quaternion 'rotation'
	// Model orientation:
	float rotationX;
	float rotationY;
	float rotationZ;

	float tx;
	float ty;
	boost::optional<float> tz;
	
	std::vector<float> shapeCoeffs;
	std::vector<float> albedoCoeffs; // Rename to ModelState and move to libMorphableModel?
	
	// Camera parameters:
	float focalLength;

	// We need this too for the window transform / fov calculation?
	// Maybe separate stuff into "ModelInstance" or "ModelFitting", and "ImageFittingResult" (bad name...) or something?
	// int imageWidth, imageHeight

	// AngleConvention (not needed when stored as quaternion?)
	// ProjectionType?
	// Texture information? Model information?

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & rotationX;
		ar & rotationY;
		ar & rotationZ;
		ar & tx;
		ar & ty;
		ar & tz;
		ar & shapeCoeffs;
		ar & albedoCoeffs;
	};
};

// Todo Move to its own file!
// 6DOF esti? Document etc.
// We could further abstract one layer, by defining this 'h' separately, and
// then make a 6DOFSupervisedDescentPoseEsti class that combines the two classes. But...
template<class RegressorType>
class SupervisedDescentPoseEstimation
{
public:
	SupervisedDescentPoseEstimation()
	{
	};

	SupervisedDescentPoseEstimation(std::vector<RegressorType> regressors) : optimiser(regressors)
	{
	};

	static cv::Mat packParameters(ModelFitting modelFitting)
	{
		Mat params(1, 5, CV_32FC1);
		params.at<float>(0) = modelFitting.rotationX;
		params.at<float>(1) = modelFitting.rotationY;
		params.at<float>(2) = modelFitting.rotationZ;
		params.at<float>(3) = modelFitting.tx;
		params.at<float>(4) = modelFitting.ty;
		return params;
	}

	static ModelFitting unpackParameters(cv::Mat parameters)
	{
		ModelFitting modelFitting;
		modelFitting.rotationX = parameters.at<float>(0);
		modelFitting.rotationY = parameters.at<float>(1);
		modelFitting.rotationZ = parameters.at<float>(2);
		modelFitting.tx = parameters.at<float>(3);
		modelFitting.ty = parameters.at<float>(4);
		return modelFitting;
	}

	class h_proj
	{
	public:
		h_proj() // maybe change outer member to unique_ptr instead
		{
		};

		// additional stuff needed by this specific 'h'
		// M = 3d model, 3d points.
		h_proj(cv::Mat model) : model(model)
		{

		};

		// the generic params of 'h', i.e. exampleId etc.
		// Here: R, t. (, f)
		// Returns the transformed 2D coords
		cv::Mat operator()(cv::Mat parameters)
		{
			using cv::Mat;
			//project the 3D model points using the current params
			ModelFitting unrolledParameters = unpackParameters(parameters);
			Mat rotPitchX = render::matrixutils::createRotationMatrixX(render::utils::degreesToRadians(unrolledParameters.rotationX));
			Mat rotYawY = render::matrixutils::createRotationMatrixY(render::utils::degreesToRadians(unrolledParameters.rotationY));
			Mat rotRollZ = render::matrixutils::createRotationMatrixZ(render::utils::degreesToRadians(unrolledParameters.rotationZ));
			Mat translation = render::matrixutils::createTranslationMatrix(unrolledParameters.tx, unrolledParameters.ty, -1900.0f);
			Mat modelMatrix = translation * rotYawY * rotPitchX * rotRollZ;
			const float aspect = static_cast<float>(640) / static_cast<float>(480);
			float fovY = render::utils::focalLengthToFovy(1500.0f, 480);
			Mat projectionMatrix = render::matrixutils::createPerspectiveProjectionMatrix(fovY, aspect, 0.1f, 5000.0f);

			int numLandmarks = model.cols;
			Mat new2dProjections(1, numLandmarks * 2, CV_32FC1);
			for (int lm = 0; lm < numLandmarks; ++lm) {
				cv::Vec3f vtx2d = render::utils::projectVertex(cv::Vec4f(model.col(lm)), projectionMatrix * modelMatrix, 640, 480);
				new2dProjections.at<float>(lm) = vtx2d[0]; // the x coord
				new2dProjections.at<float>(lm + numLandmarks) = vtx2d[1]; // y coord
			}

			// Todo/Note: Write a function in libFitting: render(ModelFitting) ?
			return new2dProjections;
		};

		cv::Mat model;

		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			ar & model;
		};
	};

	// By only forwarding those functions (known y), we
	// prevent misuse by using the wrong ones.
	template<class OnTrainingEpochCallback>
	void train(cv::Mat x, cv::Mat y, cv::Mat x0, OnTrainingEpochCallback onTrainingEpochCallback)
	{
		return optimiser.train(x, y, x0, h, onTrainingEpochCallback);
	};

	//template<>
	void train(cv::Mat x, cv::Mat y, cv::Mat x0)
	{
		return optimiser.train(x, y, x0, h);
	};

	// Returns the final prediction
	template<class OnRegressorIterationCallback>
	cv::Mat test(cv::Mat y, cv::Mat x0, OnRegressorIterationCallback onRegressorIterationCallback)
	{
		return optimiser.test(y, x0, h, onRegressorIterationCallback);
	};

	//template<>
	cv::Mat test(cv::Mat y, cv::Mat x0)
	{
		return optimiser.test(y, x0, h);
	};

	//template<>
	cv::Mat predict(cv::Mat x0, cv::Mat template_y)
	{
		return optimiser.predict(x0, template_y, h);
	};

	// Move to private
	// Better store the landmark identifiers? Or names?
	// Well, it's kind of model dependent, so this is ok?
	std::vector<int> vertexIds;
	
	h_proj h; // getFunction() (const&?)

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & optimiser;
		ar & vertexIds;
		ar & h;
	};

private:
	SupervisedDescentOptimiser<LinearRegressor> optimiser;
};

	} /* namespace v2 */
} /* namespace superviseddescent */
#endif /* POSEESTIMATION_HPP_ */
