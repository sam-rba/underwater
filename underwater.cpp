/* Implementation of "Color Balance and Fusion for Underwater Image Enhancement"
 * Ancuti et al. 2018
 */

#include <cassert>
#include <iostream>

#include <stdint.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xphoto/white_balance.hpp>

using namespace std;

#define ALPHA 1.0 // red compensation factor
#define GAMMA 0.8 // gamma correction
#define GAUSS_KERN_SIZE cv::Size(5, 5) // Gaussian blur filter size

typedef enum {
	OK,
	FAIL,
} Status;

// Read an image from disk and convert it CV_64FC3 BGR.
static Status
readImage(const string &path, cv::Mat &img) {
	// Read image
	img = cv::imread(path, cv::IMREAD_COLOR_BGR);
	if (img.empty()) {
		return FAIL;
	}

	// Convert to float64
	img.convertTo(img, CV_64FC3);
	cv::normalize(img, img, 1.0, 0.0, cv::NORM_MINMAX);

	return OK;
}

// Write a CV_64FC3 BGR image to disk.
static void
writeImage(const string &path, const cv::Mat &img) {
	cv::Mat tmp;
	cv::normalize(img, tmp, 255, 0, cv::NORM_MINMAX);
	tmp.convertTo(tmp, CV_8UC3);
	imwrite(path, tmp);
}

// Compensate the red channel of the image using (4).
// Image is CV_64FC3 BGR.
static void
compensateRed(cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	// Split BGR channels
	cv::Mat channels[3];
	cv::split(img, channels);
	cv::Mat &g = channels[1];
	cv::Mat &r = channels[2];

	// Compensate red channel
	cv::Mat tmp;
	cv::multiply(ALPHA * (cv::mean(g)[0] - cv::mean(r)[0]) * (1.0 - r), g, tmp);
	r = r + tmp;

	// Merge channels
	cv::merge(channels, 3, img);
}

// White-balance the image using Gray-World.
// Image is CV_64FC3 BGR
static void
whiteBalance(cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	// Convert to uint16
	cv::normalize(img, img, UINT16_MAX, 0, cv::NORM_MINMAX);
	img.convertTo(img, CV_16UC3);

	cv::Ptr<cv::xphoto::GrayworldWB> wb = cv::xphoto::createGrayworldWB();
	wb->balanceWhite(img, img);

	// Convert back to float64
	img.convertTo(img, CV_64FC3);
	cv::normalize(img, img, 1.0, 0.0, cv::NORM_MINMAX);

	writeImage("whitebalanced.png", img);
}

// Gamma-correct the white-balanced image to get the "first input" of the multiscale fusion.
// Input and output are CV_64FC3 BGR.
static cv::Mat
gammaCorrect(const cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	// Convert to uint8
	cv::Mat g = img.clone();
	cv::normalize(g, g, 255, 0, cv::NORM_MINMAX);
	g.convertTo(g, CV_8UC3);

	// Gamma correction
	cv::Mat lookup(1, 256, CV_8U);
	uchar *p = lookup.ptr();
	for (int k = 0; k < 256; k++) {
		p[k] = cv::saturate_cast<uchar>(pow(k / 255.0, GAMMA) * 255.0);
	}
	cv::LUT(g, lookup, g);

	// Convert back to float64
	g.convertTo(g, CV_64FC3);
	cv::normalize(g, g, 1.0, 0.0, cv::NORM_MINMAX);

	return g;
}

// Sharpen the white-balanced image using (6) to get the "second input" of the multiscale fusion.
// Input and output are CV_64FC3 BGR.
static cv::Mat
sharpen(const cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	cv::Mat blur, norm, s;

	// Sharpen
	cv::GaussianBlur(img, blur, GAUSS_KERN_SIZE, 0);
	cv::normalize(img - blur, norm, 1.0, 0.0, cv::NORM_MINMAX);
	s = (img + norm) / 2.0;

	return s;
}

// Enhance the image using color balance and fusion.
// Image is CV_64FC3 BGR.
static void
enhance(cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	// Compensate red channel.
	compensateRed(img);

	// White balance with Gray-World
	whiteBalance(img);

	// Gamma correction and sharpening
	cv::Mat g = gammaCorrect(img);
	cv::Mat s = sharpen(img);
	writeImage("gamma_corrected.png", g);
	writeImage("sharpened.png", s);

	// TODO

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

	// Enhance image
	enhance(img);

	// Write output
	writeImage("enhanced.png", img);
}
