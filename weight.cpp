#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "underwater.hpp" 

cv::Mat
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

cv::Mat
saliencyWeight(const cv::Mat &img) {
	assert(img.type() == CV_64FC3);

	// Convert to L*a*b colorspace
	cv::Mat lab = img.clone();
	lab.convertTo(lab, CV_32FC3); // BGR2Lab only works with float32
	cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);
	lab.convertTo(lab, CV_64FC3); // convert back to float64

	cv::Scalar mu;
	cv::Mat blur;
	#pragma omp parallel sections
	{
		// Mean
		#pragma omp section
		{
			mu = cv::mean(lab);
		}

		// Blur with separable binomial filter
		#pragma omp section
		{
			cv::Mat kern = (cv::Mat_<double>(1,5) << 1, 4, 6, 4, 1) / 16.0;
			cv::sepFilter2D(lab, blur, CV_64FC3, kern, kern);
		}
	}

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

cv::Mat
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
	#pragma omp parallel sections
	{
		#pragma omp section
			cv::pow(r - l, 2, rsubl2);
		#pragma omp section
			cv::pow(g - l, 2, gsubl2);
		#pragma omp section
			cv::pow(b - l, 2, bsubl2);
	}
	cv::sqrt((rsubl2 + gsubl2 + bsubl2) / 3.0, w);
	return w;
}
