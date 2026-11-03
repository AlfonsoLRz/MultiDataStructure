#pragma once

#include <cmath>
#include <cstddef>

#if defined(__CUDACC__)
#define MDS_HOST_DEVICE __host__ __device__
#else
#define MDS_HOST_DEVICE
#endif

// The normal of a point neighbourhood: the eigenvector of the smallest eigenvalue of its 3x3
// covariance (principal component analysis), the usual estimate for point-cloud normals. The same
// closed form runs on the CPU and the GPU: trigonometric eigenvalues of the symmetric matrix, then
// the longest cross product of two rows of A - lambda I. It is solved in double, once per
// neighbourhood; the sign of the normal is arbitrary.
namespace PointNormals
{
	struct Estimate
	{
		float	_normal[3] = { 0.0f, 0.0f, 1.0f };
		float	_eigen[3] = { 0.0f, 0.0f, 0.0f };	// ascending
	};

	// c = { xx, xy, xz, yy, yz, zz }, a covariance at any positive scale.
	MDS_HOST_DEVICE inline Estimate fromCovariance(const float* c)
	{
		Estimate estimate;
		double scale = 0.0;
		for (int i = 0; i < 6; ++i)
			scale = fmax(scale, fabs(static_cast<double>(c[i])));
		if (!(scale > 0.0))
			return estimate;
		const double a00 = c[0] / scale, a01 = c[1] / scale, a02 = c[2] / scale;
		const double a11 = c[3] / scale, a12 = c[4] / scale, a22 = c[5] / scale;

		double l0, l1, l2;	// ascending
		const double p1 = a01 * a01 + a02 * a02 + a12 * a12;
		const double q = (a00 + a11 + a22) / 3.0;
		const double p2 = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) + 2.0 * p1;
		const double p = sqrt(p2 / 6.0);
		if (p <= 1e-15)
		{
			// Isotropic: every direction is an eigenvector.
			for (int i = 0; i < 3; ++i)
				estimate._eigen[i] = static_cast<float>(q * scale);
			return estimate;
		}
		const double b00 = (a00 - q) / p, b01 = a01 / p, b02 = a02 / p, b11 = (a11 - q) / p, b12 = a12 / p, b22 = (a22 - q) / p;
		double r = 0.5 * (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02));
		r = fmin(1.0, fmax(-1.0, r));
		const double phi = acos(r) / 3.0;
		l2 = q + 2.0 * p * cos(phi);
		l0 = q + 2.0 * p * cos(phi + 2.0943951023931957);	// + 2 pi / 3
		l1 = 3.0 * q - l0 - l2;

		const double r0[3] = { a00 - l0, a01, a02 };
		const double r1[3] = { a01, a11 - l0, a12 };
		const double r2[3] = { a02, a12, a22 - l0 };
		const auto cross = [](const double* a, const double* b, double* out) {
			out[0] = a[1] * b[2] - a[2] * b[1];
			out[1] = a[2] * b[0] - a[0] * b[2];
			out[2] = a[0] * b[1] - a[1] * b[0];
		};
		double c01[3], c02[3], c12[3];
		cross(r0, r1, c01);
		cross(r0, r2, c02);
		cross(r1, r2, c12);
		const double n01 = c01[0] * c01[0] + c01[1] * c01[1] + c01[2] * c01[2];
		const double n02 = c02[0] * c02[0] + c02[1] * c02[1] + c02[2] * c02[2];
		const double n12 = c12[0] * c12[0] + c12[1] * c12[1] + c12[2] * c12[2];
		const double* best = n01 >= n02 && n01 >= n12 ? c01 : (n02 >= n12 ? c02 : c12);
		const double length = sqrt(fmax(n01, fmax(n02, n12)));
		if (length > 1e-30)
			for (int i = 0; i < 3; ++i)
				estimate._normal[i] = static_cast<float>(best[i] / length);
		estimate._eigen[0] = static_cast<float>(l0 * scale);
		estimate._eigen[1] = static_cast<float>(l1 * scale);
		estimate._eigen[2] = static_cast<float>(l2 * scale);
		return estimate;
	}

	// From `count` positions given by position(i, xyz): the mean, then the centred covariance.
	template <typename Position>
	MDS_HOST_DEVICE inline Estimate fromPoints(size_t count, const Position& position)
	{
		if (count == 0)
			return Estimate{};
		float mean[3] = { 0.0f, 0.0f, 0.0f }, p[3];
		for (size_t i = 0; i < count; ++i)
		{
			position(i, p);
			for (int axis = 0; axis < 3; ++axis)
				mean[axis] += p[axis];
		}
		for (int axis = 0; axis < 3; ++axis)
			mean[axis] /= static_cast<float>(count);
		float c[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		for (size_t i = 0; i < count; ++i)
		{
			position(i, p);
			const float dx = p[0] - mean[0], dy = p[1] - mean[1], dz = p[2] - mean[2];
			c[0] += dx * dx; c[1] += dx * dy; c[2] += dx * dz;
			c[3] += dy * dy; c[4] += dy * dz; c[5] += dz * dz;
		}
		return fromCovariance(c);
	}

	// Whether a normal computed elsewhere (another summation order, another device) is the same
	// direction as `reference`, sign ignored. Neighbourhoods whose two smallest eigenvalues are
	// nearly equal (a line, a sphere) have no stable normal and always agree; so do those with
	// fewer than three points.
	inline bool agrees(const Estimate& reference, size_t count, const float* normal, double toleranceRadians = 1e-2)
	{
		if (count < 3)
			return true;
		const double gap = static_cast<double>(reference._eigen[1]) - reference._eigen[0];
		if (!(gap > 1e-3 * fabs(static_cast<double>(reference._eigen[2]))))
			return true;
		const double dot = fabs(static_cast<double>(reference._normal[0]) * normal[0] +
								static_cast<double>(reference._normal[1]) * normal[1] +
								static_cast<double>(reference._normal[2]) * normal[2]);
		return dot >= cos(toleranceRadians);
	}
}
