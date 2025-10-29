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
#define GAUSS_KSIZE cv::Size(5, 5) // Gaussian blur kernel size
#define LAPLACE_KSIZE 3 // Laplacian kernel size

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
	assert(img.type() == CV_64FC3);

	cv::Mat tmp;
	cv::normalize(img, tmp, 255, 0, cv::NORM_MINMAX);
	tmp.convertTo(tmp, CV_8UC3);
	imwrite(path, tmp);
}

// Write a CV_64F image to disk.
static void
write1dImage(const string &path, const cv::Mat &img) {
	assert(img.type() == CV_64F);

	cv::Mat tmp;
	cv::normalize(img, tmp, 255, 0, cv::NORM_MINMAX);
	tmp.convertTo(tmp, CV_8U);
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
	cv::GaussianBlur(img, blur, GAUSS_KSIZE, 0);
	cv::normalize(img - blur, norm, 1.0, 0.0, cv::NORM_MINMAX);
	s = (img + norm) / 2.0;

	return s;
}

// Laplacian contrast weight.
// Input is CV_64FC3 BGR.
// Output is CV_64F.
static cv::Mat
laplacianWeight(const cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	// Convert to L*a*b colorspace to get luminance channel
	cv::Mat lab = img.clone();
	lab.convertTo(lab, CV_32FC3); // BGR2Lab only works with float32
	cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);
	cv::Mat channels[3];
	cv::split(lab, channels);
	cv::Mat &lum = channels[0]; // luminance

	// Take Laplacian of luminance
	cv::Laplacian(lum, lum, CV_32F, LAPLACE_KSIZE);
	lum = cv::abs(lum);

	lum.convertTo(lum, CV_64F);

	return lum;
}

// Saliency weight.
// Input is CV_64FC3 BGR.
// Output is CV_64F.
static cv::Mat
saliencyWeight(const cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	// Convert to L*a*b colorspace
	cv::Mat lab = img.clone();
	lab.convertTo(lab, CV_32FC3); // BGR2Lab only works with float32
	cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);
	lab.convertTo(lab, CV_64FC3); // convert back to float64

	// Mean
	cv::Scalar mu = cv::mean(lab);

	// Blur with separable 
	cv::Mat kern = (cv::Mat_<double>(1,5) << 1, 4, 6, 4, 1) / 16.0; // separable binomial kernel
	cv::Mat blur;
	cv::sepFilter2D(lab, blur, CV_64FC3, kern, kern);

	// Mean - Blur
	cv::Mat chans[3];
	cv::split(mu - blur, chans);

	// Element-wise L2 norm across channels
	// I.e. for each element i, compute norm(<Li, ai, bi>)
	cv::Mat s;
	cv::sqrt(
		chans[0].mul(chans[0]) + chans[1].mul(chans[1]) + chans[2].mul(chans[2]),
		s);

	return s;
}

// Saturation weight.
// Input is CV_64FC3 BGR.
// Output is CV_64F.
static cv::Mat
saturationWeight(const cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	// Split BGR channels
	cv::Mat bgrChans[3];
	cv::split(img, bgrChans);
	cv::Mat &b = bgrChans[0];
	cv::Mat &g = bgrChans[1];
	cv::Mat &r = bgrChans[2];

	// Split L*a*b channels to get luminance
	cv::Mat lab = img.clone();
	lab.convertTo(lab, CV_32FC3); // BGR2Lab only works with float32
	cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);
	lab.convertTo(lab, CV_64FC3); // convert back to float64
	cv::Mat labChans[3];
	cv::split(lab, labChans);
	cv::Mat &l = labChans[0]; // luminance

	// Weight
	cv::Mat rsubl2, gsubl2, bsubl2, w;
	cv::pow(r - l, 2, rsubl2);
	cv::pow(g - l, 2, gsubl2);
	cv::pow(b - l, 2, bsubl2);
	cv::sqrt((rsubl2 + gsubl2 + bsubl2) / 3.0, w);
	return w;
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

	// Laplacian weights
	cv::Mat wl1 = laplacianWeight(g); // W_L of gamma-corrected image
	cv::Mat wl2 = laplacianWeight(s); // W_L of sharpened image
	write1dImage("wl1.png", wl1);
	write1dImage("wl2.png", wl2);

	// Saliency weights
	cv::Mat wsal1 = saliencyWeight(g); // W_S of gamma-corrected image
	cv::Mat wsal2 = saliencyWeight(s); // W_S of sharpened image
	write1dImage("wsal1.png", wsal1);
	write1dImage("wsal2.png", wsal2);

	// Saturation weights
	cv::Mat wsat1 = saturationWeight(g); // W_Sat of gamma-corrected image
	cv::Mat wsat2 = saturationWeight(s); // W_Sat of sharpened image
	write1dImage("wsat1.png", wsat1);
	write1dImage("wsat2.png", wsat2);

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
