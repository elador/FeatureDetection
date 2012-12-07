/*!
 * \file Triangle.h
 *
 * \author Patrik Huber
 * \date December 4, 2012
 *
 * [comment here]
 */
#pragma once

#include <array>

#include "render/Vertex.hpp"

namespace render {

class Triangle
{
public:
	Triangle(void);
	Triangle(Vertex v0, Vertex v1, Vertex v2);
	~Triangle(void);

	std::array<Vertex, 3> vertices;
};

class plane
{
public:
	plane() {}

	plane(float a, float b, float c, float d)
	{
		this->a = a;
		this->b = b;
		this->c = c;
		this->d = d;
	}

	plane(const cv::Vec3f& normal, float d = 0.0f)
	{
		this->a = normal[0];
		this->b = normal[1];
		this->c = normal[2];
		this->d = d;
	}

	plane(const cv::Vec3f& point, const cv::Vec3f& normal)
	{
		a = normal[0];
		b = normal[1];
		c = normal[2];
		d = -(point.dot(normal));
	}

	plane(const cv::Vec3f& point1, const cv::Vec3f& point2, const cv::Vec3f& point3)
	{
		cv::Vec3f v1 = point2 - point1;
		cv::Vec3f v2 = point3 - point1;
		cv::Vec3f normal = (v1.cross(v2));
		normal /= cv::norm(normal, cv::NORM_L2);

		a = normal[0];
		b = normal[1];
		c = normal[2];
		d = -(point1.dot(normal));
	}

	void normalize()
	{
		float length = sqrtf(a*a + b*b + c*c);

		a /= length;
		b /= length;
		c /= length;
	}

	float getSignedDistanceFromPoint(const cv::Vec3f& point) const
	{
		return a*point[0] + b*point[1] + c*point[2] + d;
	}

	float getSignedDistanceFromPoint(const cv::Vec4f& point) const
	{
		return a*point[0] + b*point[1] + c*point[2] + d;
	}

	void transform(cv::Mat transform);

public:
	float a, b, c;
	float d;
};

class TriangleToRasterize
{
	Vertex v0;
	Vertex v1;
	Vertex v2;
	//const CTexture* texture;
	float one_over_z0;
	float one_over_z1;
	float one_over_z2;
	int minX;
	int maxX;
	int minY;
	int maxY;
	float one_over_v0ToLine12;
	float one_over_v1ToLine20;
	float one_over_v2ToLine01;
	plane alphaPlane;
	plane betaPlane;
	plane gammaPlane;
	float one_over_alpha_c;
	float one_over_beta_c;
	float one_over_gamma_c;
	float alpha_ffx;
	float beta_ffx;
	float gamma_ffx;
	float alpha_ffy;
	float beta_ffy;
	float gamma_ffy;
	int tileMinX;
	int tileMaxX;
	int tileMinY;
	int tileMaxY;
	int padding[1];

	bool coversTile(int tileX, int tileY)
	{
		return ( (tileX >= tileMinX) && (tileX <= tileMaxX) && (tileY >= tileMinY) && (tileY <= tileMaxY) );
	}
};

}
