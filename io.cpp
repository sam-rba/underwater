#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "underwater.hpp"

using namespace std;

Status
readImage(const std::string &path, cv::Mat &img) {
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

Status
writeImage(const std::string &path, const cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	cv::Mat tmp;
	cv::normalize(img, tmp, 255, 0, cv::NORM_MINMAX);
	tmp.convertTo(tmp, CV_8UC3);
	try {
		imwrite(path, tmp);
	} catch (const cv::Exception &ex) {
		cerr << ex.what() << endl;
		return FAIL;
	}
	return OK;
}

Status
writeImage1c(const std::string &path, const cv::Mat &img) {
	assert(img.type() == CV_64F);

	cv::Mat tmp;
	cv::normalize(img, tmp, 255, 0, cv::NORM_MINMAX);
	tmp.convertTo(tmp, CV_8U);
	try {
		imwrite(path, tmp);
	} catch (const cv::Exception &ex) {
		cerr << ex.what() << endl;
		return FAIL;
	}
	return OK;
}
