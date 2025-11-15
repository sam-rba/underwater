#include <opencv2/core.hpp>
#include <opencv2/xphoto/white_balance.hpp>

#include "underwater.hpp"

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

	double gmean = cv::mean(g)[0];
	double rmean = cv::mean(r)[0];

	// Compensate red channel
	r += ALPHA * (gmean - rmean) * (1.0 - r).mul(g);

	// Merge channels
	cv::merge(channels, 3, img);
}

void
whiteBalance(cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	// Compensate red channel
	compensateRed(img);
	writeImage("compensated.png", img);

	// Convert to uint16
	cv::normalize(img, img, UINT16_MAX, 0, cv::NORM_MINMAX);
	img.convertTo(img, CV_16UC3);

	// White balance with Gray-World.
	cv::Ptr<cv::xphoto::GrayworldWB> wb = cv::xphoto::createGrayworldWB();
	wb->balanceWhite(img, img);

	// Convert back to float64
	img.convertTo(img, CV_64FC3);
	cv::normalize(img, img, 1.0, 0.0, cv::NORM_MINMAX);
}
