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

#include "underwater.hpp"

using namespace std;

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
	cv::GaussianBlur(img, blur, GAUSS_KSIZE, 0);
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
	cv::Mat g, s;
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			g = gammaCorrect(img);
			writeImage("gamma_corrected.png", g);
		}
		#pragma omp section
		{
			s = sharpen(img);
			writeImage("sharpened.png", s);
		}
	}

	// Weight maps
	cv::Mat wl1, wl2; // Laplacian weights
	cv::Mat wsal1, wsal2; // saliency weights
	cv::Mat wsat1, wsat2; // saturation weights
	#pragma omp parallel sections
	{
		// Laplacian weights
		#pragma omp section
		{
			wl1 = laplacianWeight(g); // W_L of gamma-corrected image
			write1dImage("wl1.png", wl1);
		}
		#pragma omp section
		{
			wl2 = laplacianWeight(s); // W_L of sharpened image
			write1dImage("wl2.png", wl2);
		}

		// Saliency weights
		#pragma omp section
		{
			wsal1 = saliencyWeight(g); // W_S of gamma-corrected image
			write1dImage("wsal1.png", wsal1);
		}
		#pragma omp section
		{
			wsal2 = saliencyWeight(s); // W_S of sharpened image
			write1dImage("wsal2.png", wsal2);
		}

		// Saturation weights
		#pragma omp section
		{
			wsat1 = saturationWeight(g); // W_Sat of gamma-corrected image
			write1dImage("wsat1.png", wsat1);
		}
		#pragma omp section
		{
			wsat2 = saturationWeight(s); // W_Sat of sharpened image
			write1dImage("wsat2.png", wsat2);
		}
	}

	// Merge weight maps
	cv::Mat w1, w2; // aggregated weight map for each input
	mergeWeightMaps(wl1, wl2, wsal1, wsal2, wsat1, wsat2, w1, w2);
	write1dImage("w1.png", w1);
	write1dImage("w2.png", w2);

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
