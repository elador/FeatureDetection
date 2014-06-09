/*
 * MatrixUtils.cpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */

#include "render/MatrixUtils.hpp"
#include "boost/math/constants/constants.hpp"

namespace render {
	namespace utils {

cv::Mat MatrixUtils::createRotationMatrixX(float angle) // angle is in radians!
{
	cv::Mat rotX = (cv::Mat_<float>(4,4) << 
		1.0f,				0.0f,				0.0f,				0.0f,
		0.0f,				std::cos(angle),	-std::sin(angle),	0.0f,
		0.0f,				std::sin(angle),	std::cos(angle),	0.0f,
		0.0f,				0.0f,				0.0f,				1.0f);
	return rotX;
}

cv::Mat MatrixUtils::createRotationMatrixY(float angle) // angle is in radians!
{
	cv::Mat rotY = (cv::Mat_<float>(4,4) << 
		std::cos(angle),	0.0f,				std::sin(angle),	0.0f,
		0.0f,				1.0f,				0.0f,				0.0f,
		-std::sin(angle),	0.0f,				std::cos(angle),	0.0f,
		0.0f,				0.0f,				0.0f,				1.0f);
	return rotY;
}

cv::Mat MatrixUtils::createRotationMatrixZ(float angle) // angle is in radians!
{
	cv::Mat rotZ = (cv::Mat_<float>(4,4) << 
		std::cos(angle),	-std::sin(angle),	0.0f,				0.0f,
		std::sin(angle),	std::cos(angle),	0.0f,				0.0f,
		0.0f,				0.0f,				1.0f,				0.0f,
		0.0f,				0.0f,				0.0f,				1.0f);
	return rotZ;
}

cv::Mat MatrixUtils::createScalingMatrix(float sx, float sy, float sz)
{
	cv::Mat scaling = (cv::Mat_<float>(4,4) << 
		sx,					0.0f,				0.0f,				0.0f,
		0.0f,				sy,					0.0f,				0.0f,
		0.0f,				0.0f,				sz,					0.0f,
		0.0f,				0.0f,				0.0f,				1.0f);
	return scaling;
}

cv::Mat MatrixUtils::createTranslationMatrix(float tx, float ty, float tz)
{
	cv::Mat translation = (cv::Mat_<float>(4,4) << 
		1.0f,				0.0f,				0.0f,				tx,
		0.0f,				1.0f,				0.0f,				ty,
		0.0f,				0.0f,				1.0f,				tz,
		0.0f,				0.0f,				0.0f,				1.0f);
	return translation;
}

// identical to OGL, Qt
/*orthographic
projection for a window with lower - left corner(\a left, \a bottom),
upper - right corner(\a right, \a top), and the specified \a nearPlane
and \a farPlane clipping planes.*/
cv::Mat MatrixUtils::createOrthogonalProjectionMatrix(float l, float r, float b, float t, float n, float f)
{
	// OpenGL format (http://www.songho.ca/opengl/gl_projectionmatrix.html) (and Qt):
	cv::Mat orthogonal = (cv::Mat_<float>(4, 4) <<
		2.0f / (r - l), 0.0f, 0.0f, -(r + l) / (r - l),
		0.0f, 2.0f / (t - b), 0.0f, -(t + b) / (t - b),
		0.0f, 0.0f, -2.0f / (f - n), -(f + n) / (f - n), // CG book has denominator (n-f) ? I had (f-n) before. When n and f are neg and here is n-f, then it's the same as n and f pos and f-n here.
		0.0f, 0.0f, 0.0f, 1.0f);
	return orthogonal;

	/* My renderer last working:
		cv::Mat orthogonal = (cv::Mat_<float>(4, 4) <<
		2.0f / (r - l), 0.0f, 0.0f, -(r + l) / (r - l),
		0.0f, 2.0f / (t - b), 0.0f, -(t + b) / (t - b),
		0.0f, 0.0f, 2.0f / (n - f), -(n + f) / (n - f), // CG book has denominator (n-f) ? I had (f-n) before. When n and f are neg and here is n-f, then it's the same as n and f pos and f-n here.
		0.0f, 0.0f, 0.0f, 1.0f);
	*/
}

cv::Mat MatrixUtils::createPerspectiveProjectionMatrix(float l, float r, float b, float t, float n, float f)
{
	// OpenGL format (http://www.songho.ca/opengl/gl_projectionmatrix.html) (and Qt):
	cv::Mat perspective = (cv::Mat_<float>(4, 4) <<
		2.0f * n / (r - l), 0.0f, (l + r) / (r - l), 0.0f,
		0.0f, 2.0f * n / (t - b), (t + b) / (t - b), 0.0f,
		0.0f, 0.0f, -(n + f) / (f - n), (-2.0f * n * f) / (f - n),
		0.0f, 0.0f, -1.0f, 0.0f);
	return perspective;
}

// angle in degrees
cv::Mat MatrixUtils::createPerspectiveProjectionMatrix(float verticalAngle, float aspectRatio, float n, float f)
{
	// OpenGL format (http://www.songho.ca/opengl/gl_projectionmatrix.html) (and Qt):
	float radians = (verticalAngle / 2.0f) * boost::math::constants::pi<float>() / 180.0f;
	float sine = std::sin(radians);
	// if sinr == 0.0f, return, something wrong
	float cotan = std::cos(radians) / sine;
	cv::Mat perspective = (cv::Mat_<float>(4, 4) <<
		cotan / aspectRatio, 0.0f, 0.0f, 0.0f,
		0.0f, cotan, 0.0f, 0.0f,
		0.0f, 0.0f, -(n + f) / (f - n), (-2.0f * n * f) / (f - n),
		0.0f, 0.0f, -1.0f, 0.0f);
	return perspective;
}

unsigned int getMaxPossibleMipmapsNum(unsigned int width, unsigned int height)	// TODO: This belongs more in a ImageUtils, TextureUtils, or whatever... => render::utils::texturing
{
	unsigned int mipmapsNum = 1;
	unsigned int size = std::max(width, height);

	if (size == 1)
		return 1;

	do {
		size >>= 1;
		mipmapsNum++;
	} while (size != 1);

	return mipmapsNum;
}

	} /* namespace utils */
} /* namespace render */
