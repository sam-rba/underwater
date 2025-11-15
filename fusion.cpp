#include <cassert>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "underwater.hpp"

// Gaussian lowpass filter.
static cv::Mat
filter(const cv::Mat &img) {
	cv::Mat g;
	cv::GaussianBlur(img, g, GAUSS_KSIZE, 0);
	return g;
}

// Decimate by a factor of 2 in both directions.
static cv::Mat
decimate(const cv::Mat &img) {
	cv::Mat d;
	cv::resize(img, d, cv::Size(img.cols/2, img.rows/2), 0, 0, cv::INTER_AREA);
	return d;
}

// Upsample to a larger resolution.
static cv::Mat
upsample(const cv::Mat &img, int rows, int cols) {
	cv::Mat u;
	cv::resize(img, u, cv::Size(cols, rows), 0, 0, cv::INTER_CUBIC);
	return u;
}

// Element-wise multiply each channel of a 3-channel matrix by a 1-channel matrix of the same size.
static cv::Mat
mul(const cv::Mat &c3, const cv::Mat &c1) {
	cv::Mat chans[3];
	cv::split(c3, chans);

	cv::multiply(chans[0], c1, chans[0]);
	cv::multiply(chans[1], c1, chans[1]);
	cv::multiply(chans[2], c1, chans[2]);

	cv::Mat m;
	cv::merge(chans, 3, m);
	return m;
}

// Recursively fuse each level of the pyramid.
// i1, i2 -- Ll{Ik(x)} + Gl{Ik(x)} for each image k.
// w1, w2 -- Gl{Wk(x)} for each weight map k.
// n -- number of levels remaining.
static cv::Mat
fuseLevel(const cv::Mat &i1, const cv::Mat &i2, const cv::Mat &w1, const cv::Mat &w2, int nlevel) {
	assert(i1.type() == CV_64FC3);
	assert(i2.type() == CV_64FC3);
	assert(w1.type() == CV_64F);
	assert(w2.type() == CV_64F);
	assert(nlevel >= 0);
	assert(i1.size == i2.size);
	assert(i1.rows == w1.rows && i1.cols == w1.cols);
	assert(i1.rows == w2.rows && i1.cols == w2.cols);

	cv::Mat gi1, gi2; // image Gaussian-filtered and decimated l times -- Gl{Ik(x)}
	cv::Mat li1, li2; // Laplacian of image at level l -- Ll{Ik(x)}
	cv::Mat gw1, gw2; // weight map Gaussian-filtered and decimated l times -- Gl{Wk(x)}
	cv::Mat r; // fused image

	// Build this level of Laplacian and Gaussian pyramids
	#pragma omp parallel sections
	{
		// Laplacian pyramid of images
		#pragma omp section
		{
			gi1 = decimate(filter(i1)); // Gl{I1(x)}
			li1 = decimate(i1) - gi1; // Ll{I1(x)}
		}
		#pragma omp section
		{
			gi2 = decimate(filter(i2)); // Gl{I2(x)}
			li2 = decimate(i2) - gi2; // Ll{I2(x)}
		}

		// Gaussian pyramid of weight maps
		#pragma omp section
		{
			gw1 = decimate(filter(w1));
		}
		#pragma omp section
		{
			gw2 = decimate(filter(w2));
		}
	}

	// Fuse this level
	r = mul(li1, gw1) + mul(li2, gw2);

	// Collapse levels of pyramid recursively
	if (nlevel > 0) {
		// Recurse
		r += fuseLevel(li1+gi1, li2+gi2, gw1, gw2, nlevel-1);
	}
	return upsample(r, i1.rows, i1.cols);
}

cv::Mat
fuse(const cv::Mat &i1, const cv::Mat &i2, const cv::Mat &w1, const cv::Mat &w2, int nlevel) {
	assert(i1.type() == CV_64FC3);
	assert(i2.type() == CV_64FC3);
	assert(w1.type() == CV_64F);
	assert(w2.type() == CV_64F);
	assert(nlevel > 0);
	assert(i1.size == i2.size);
	assert(i1.rows == w1.rows && i1.cols == w1.cols);
	assert(i1.rows == w2.rows && i1.cols == w2.cols);

	return fuseLevel(i1, i2, w1, w2, nlevel+1);
}
