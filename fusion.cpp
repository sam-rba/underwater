#include <cassert>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "underwater.hpp"

using namespace std;

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

cv::Mat
fuse(const cv::Mat &i1, const cv::Mat &i2, const cv::Mat &w1, const cv::Mat &w2, int n) {
	assert(i1.type() == CV_64FC3);
	assert(i2.type() == CV_64FC3);
	assert(w1.type() == CV_64F);
	assert(w2.type() == CV_64F);
	assert(n > 0);
	assert(i1.size == i2.size);
	assert(i1.rows == w1.rows && i1.cols == w1.cols);
	assert(i1.rows == w2.rows && i1.cols == w2.cols);

	cv::Mat r; // fused image
	cv::Mat rl; // fused image at level l
	cv::Mat il1, il2; // input image at level l -- filtered decimated image of previous level
	cv::Mat gi1, gi2; // image Gaussian-filtered and decimated l times -- Gl{Ik(x)}
	cv::Mat gw1, gw2; // weight map Gaussian-filtered and decimated l times -- Gl{Wk(x)}
	cv::Mat li1, li2; // Laplacian of image at level l -- Ll{Ik(x)}

	// Initialize
	r = cv::Mat::zeros(i1.rows, i1.cols, CV_64FC3);
	il1 = i1;
	il2 = i2;
	gw1 = w1;
	gw2 = w2;

	// For each level
	for (int i = 0; i < n; i++) {
		// Gaussian and Laplacian of images
		gi1 = decimate(filter(il1)); // Gl{I(x)}
		gi2 = decimate(filter(il2));
		li1 = il1 - upsample(gi1, il1.rows, il1.cols); // Ll{I(x)}
		li2 = il2 - upsample(gi2, il2.rows, il2.cols);

		// Gaussian of weight maps
		gw1 = decimate(filter(gw1));
		gw2 = decimate(filter(gw2));

		// Fuse level l
		cerr << li1.size << endl;
		cerr << gw1.size << endl;
		rl = mul(li1, upsample(gw1, li1.rows, li2.cols)) +
			mul(li2, upsample(gw2, li2.rows, li2.cols)); // compute this level

		// Add to running sum -- R = sum(Rl)
		r += upsample(rl, r.rows, r.cols);

		// Start next level
		il1 = gi1; // use decimated low-pass image as input to next iteration
		il2 = gi2;
	}

	// Return fused image
	// R = sum(Rl)
	return r;
}
