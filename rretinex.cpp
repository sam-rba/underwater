/* Implementation of "Structure-Revealing Low-Light Image Enhancement
 * Via Robust Retinex Model", Li et al. 2018.
 */

#include <cassert>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

// Tuning parameters:
#define BETA 0.05
#define OMEGA 0.01
#define DELTA 1.0
#define LAMBDA 10.0
#define SIGMA 10.0
#define MU0 1.0
#define RHO 1.5
#define GAMMA 2.2
#define EPSILON 0.005

#define THRESH 1e-3 // iteration threshold: stop once |R^k - R^(k+1)| < THRESH
#define MAXITER 100 // maximum number of iterations

typedef enum {
	OK,
	FAIL,
} Status;

// Read an image from disk and convert it to HSV, double precision, normalized to [0,1].
static Status
readImage(const string &path, cv::Mat &img) {
	// Read from disk
	cv::Mat bgrImg = cv::imread(path, cv::IMREAD_COLOR_BGR);
	if (bgrImg.empty()) {
		return FAIL;
	}

	cv::cvtColor(bgrImg, img, cv::COLOR_BGR2HSV); // convert to HSV colorspace
	img.convertTo(img, CV_64FC3); // convert to double
	cv::normalize(img, img, 1.0, 0.0, cv::NORM_MINMAX); // normalize to [0,1]

	return OK;
}

// Write to disk an HSV double precision image normalized to [0,1].
static void
writeImage(const string &path, const cv::Mat &img) {
	cv::Mat tmp;
	cv::normalize(img, tmp, 255, 0, cv::NORM_MINMAX); // normalize to [0,255]
	tmp.convertTo(tmp, CV_8U); // convert to uint8
	cv::cvtColor(tmp, tmp, cv::COLOR_HSV2BGR); // convert to BGR colorspace
	cv::imwrite(path, tmp); // write to disk
}

static double
frobeniusNormSquared(const cv::Mat &img) {
	assert(img.channels() == 1);

	cv::Mat tmp;
	cv::pow(img, 2, tmp); // square each element
	return cv::sum(tmp)[0]; // accumulate
}

static cv::Mat
flatten(const cv::Mat &m) {
	return m.reshape(1, m.total());
}

// Create a diagonal matrix with the elements of the given matrix as its entries.
static cv::Mat
diagonal(const cv::Mat &m) {
	assert(m.channels() == 1);

	cout << m.total() << endl;
	const cv::Mat v = m.reshape(1, m.total());
	cout << v.rows << " " << v.cols << endl;
	return cv::Mat::diag(v);
}

// Gram matrix.
// Called f(x) in the paper.
static cv::Mat
gram(const cv::Mat &x) {
	cv::Mat xt;
	cv::transpose(x, xt);
	return xt * x;
}

// Discrete gradient operator: nxn matrix D.
static cv::Mat
discreteGradientOperator(int n) {
	const cv::Mat negOnesDiag = cv::Mat::diag(cv::Mat(1, n, CV_64FC1, cv::Scalar(-1))); // -1s on diagonal

	const cv::Mat onesDiag = cv::Mat::diag(cv::Mat(1, n, CV_64FC1, cv::Scalar(1))); // 1s on diagonal
	const cv::Mat m = (cv::Mat_<double>(2, 3) <<
		1, 0, 1,
		0, 1, 0);
	cv::Mat ones;
	cv::warpAffine(onesDiag, ones, m, onesDiag.size()); // shift diagonal 1s one place to the right

	return negOnesDiag + ones; // -1s on the diagonal, with a 1 to the right of each
}

// Calculate value of R for next iteration: eq. 14.
static cv::Mat
updateR(const cv::Mat &r, const cv::Mat &l, const cv::Mat &n, const cv::Mat &i, const cv::Mat &g) {
	assert(r.channels() == 1);
	assert(l.channels() == 1);


}

static cv::Mat
updateL(const cv::Mat &l) {
	assert(l.channels() == 1);

	// TODO
	assert(0);
}

static cv::Mat
updateN(const cv::Mat &n) {
	assert(n.channels() == 1);

	// TODO
	assert(0);
}

static cv::Mat
updateT(const cv::Mat &t) {
	assert(t.channels() == 1);

	// TODO
	assert(0);
}

static cv::Mat
updateZ(const cv::Mat &z) {
	assert(z.channels() == 1);

	// TODO
	assert(0);
}

static bool
isConverged(const cv::Mat &r, const cv::Mat &r1, const cv::Mat &l, const cv::Mat &l1) {
	assert(r.channels() == 1);
	assert(r1.channels() == 1);
	assert(l.channels() == 1);
	assert(l1.channels() == 1);

	return (cv::norm(r1 - r) < THRESH) || (cv::norm(l1 - l) < THRESH);
}

// Magnitude of the gradient
static cv::Mat
gradientImage(const cv::Mat &i) {
	cv::Mat gx, gy;
	cv::Sobel(i, gx, CV_64F, 1, 0);
	cv::Sobel(i, gy, CV_64F, 0, 1);
	cv::pow(gx, 2, gx);
	cv::pow(gy, 2, gy);
	cv::Mat g;
	cv::sqrt(gx+gy, g);
	return g;
}

// Compute the adjusted gradient G.
// Takes the V channel of the original image.
static cv::Mat
adjustedGradient(const cv::Mat &i) {
	assert(i.channels() == 1);

	// Modified gradient of image
	const cv::Mat grad = gradientImage(i); // compute (magnitude of) gradient
	const cv::Mat absgrad = cv::abs(grad); // take absolute value
	cv::Mat sign;
	cv::divide(grad, absgrad, sign); // sign of each element
	(void)cv::threshold(absgrad, absgrad, EPSILON, 0, cv::THRESH_TOZERO); // filter out small elements
	cv::Mat gradhat; // modified gradient
	cv::multiply(absgrad, sign, gradhat); // add sign back in to get modified gradient
	const cv::Mat absgradhat = cv::abs(gradhat); // absolute value of modified gradient

	// K factor
	cv::Mat exp;
	cv::exp(-absgradhat / SIGMA, exp);
	const cv::Mat k = LAMBDA * exp + 1;

	// Final G matrix
	cv::Mat g;
	cv::multiply(k, gradhat, g);
	return g;
}

// Decompose the input image (the V channel) into its constituent
// reflectance, illumination, and noise components.
// This is Algorithm 1 in the paper.
static void
decompose(const cv::Mat &i, cv::Mat &r, cv::Mat &l, cv::Mat &n) {
	assert(i.channels() == 1);

	cv::Mat r1, l1, t, z;
	l = i;
	l1 = i;
	int k = 0;

	const cv::Mat g = adjustedGradient(i);

	do {
		r = r1;
		l = l1;

		r1 = updateR(r, l, n, i, g);
		l1 = updateL(l);
		n = updateN(n);
		t = updateT(t);
		z = updateZ(z);
	} while ((k++ < MAXITER) && (!isConverged(r, r1, l, l1)));

	// TODO
	assert(0);
}

static cv::Mat
enhance(const cv::Mat &img) {
	assert(img.channels() == 3);

	// Split channels
	assert(img.channels() == 3);
	cv::Mat channels[3];
	cv::split(img, channels);
	cv::Mat &v = channels[2];

	// Decompose
	cv::Mat r, l, n; // reflectance, illumination, noise
	decompose(v, r, l, n);

	// Adjust illumination
	cv::pow(l, 1.0/GAMMA, l);
	cv::multiply(r, l, v);

	// Recombine channels
	cv::Mat enhanced;
	cv::merge(channels, 3, enhanced);
	return enhanced;
}

int
main(int argc, const char *argv[]) {
	// Read input image
	if (argc < 2) {
		cerr << "Not enough arguments\n";
		return 1;
	}
	const char *path = argv[1];
	cv::Mat img;
	Status status = readImage(path, img);
	if (status != OK) {
		cerr << "Error reading image " << path << endl;
		return 1;
	}

	const cv::Mat enhanced = enhance(img);
	writeImage("enhanced.png", enhanced);

	return 0;
}
