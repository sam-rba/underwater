/* Implementation of "Color Balance and Fusion for Underwater Image Enhancement",
 * Ancuti et al. 2018.
 */

#include <cassert>
#include <cstring>
#include <iostream>

#include <stdint.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "underwater.hpp"

using namespace std;

const char usage[] = "Underwater image enhancement\n"
	"usage: underwater -i infile outfile";

// Gamma-correct the white-balanced image to get the "first input" of the multiscale fusion.
// Input and output are CV_64FC3 BGR.
static cv::Mat
gammaCorrect(const cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	cv::Mat corr;
	cv::pow(img, GAMMA, corr);
	return corr;
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
static cv::Mat
enhance(cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	// White balance with Gray-World
	whiteBalance(img);
	writeImage("whitebalanced.png", img);

	// Gamma correction and sharpening
	cv::Mat i1, i2;
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			i1 = gammaCorrect(img);
			writeImage("i1.png", i1);
		}
		#pragma omp section
		{
			i2 = sharpen(img);
			writeImage("i2.png", i2);
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
			wl1 = laplacianWeight(i1); // W_L of gamma-corrected image
			writeImage1c("wl1.png", wl1);
		}
		#pragma omp section
		{
			wl2 = laplacianWeight(i2); // W_L of sharpened image
			writeImage1c("wl2.png", wl2);
		}

		// Saliency weights
		#pragma omp section
		{
			wsal1 = saliencyWeight(i1); // W_S of gamma-corrected image
			writeImage1c("wsal1.png", wsal1);
		}
		#pragma omp section
		{
			wsal2 = saliencyWeight(i2); // W_S of sharpened image
			writeImage1c("wsal2.png", wsal2);
		}

		// Saturation weights
		#pragma omp section
		{
			wsat1 = saturationWeight(i1); // W_Sat of gamma-corrected image
			writeImage1c("wsat1.png", wsat1);
		}
		#pragma omp section
		{
			wsat2 = saturationWeight(i2); // W_Sat of sharpened image
			writeImage1c("wsat2.png", wsat2);
		}
	}

	// Merge weight maps
	cv::Mat w1, w2; // aggregated weight map for each input
	mergeWeightMaps(wl1, wl2, wsal1, wsal2, wsat1, wsat2, w1, w2);
	writeImage1c("w1.png", w1);
	writeImage1c("w2.png", w2);

	// Multi-scale fusion
	// Set number of levels s.t. image is ~10x10 at last level
	int n = log2((img.rows + img.cols) / 2 / 10);
	n = max(1, n); // n >= 1
	cv::Mat r = fuse(i1, i2, w1, w2, n);
	return r;
}

int
main(int argc, const char *argv[]) {
	// Parse command line args
	if (argc < 4 || strcmp(argv[1], "-i") != 0) {
		cerr << usage << endl;
		return 1;
	}
	const string &infile = argv[2];
	const string &outfile = argv[3];

	// Read input image
	cv::Mat img;
	Status status = readImage(infile, img);
	if (status != OK) {
		cerr << "Error reading file '" << infile << "'\n";
		return 1;
	}	

	// Enhance image
	cv::Mat r = enhance(img);

	// Write output
	status = writeImage(outfile, r);
	if (status != OK) {
		cerr << "Error writing file '" << outfile << "'\n";
		return 1;
	}

	return 0;
}
