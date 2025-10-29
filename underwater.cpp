/* Implementation of "Color Balance and Fusion for Underwater Image Enhancement"
 * Ancuti et al. 2018
 */

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

#define ALPHA 1.0 // red compensation factor

typedef enum {
	OK,
	FAIL,
} Status;

// Read an image from disk and convert it to float64 normalized to [0, 1].
static Status
readImage(const string &path, Mat &img) {
	img = imread(path, IMREAD_COLOR_BGR);
	if (img.empty()) {
		return FAIL;
	}
	img.convertTo(img, CV_64FC3); // convert to float64
	normalize(img, img, 1.0, 0.0, NORM_MINMAX); // normalize to [0, 1]
	return OK;
}

// Write to disk a float64 image normalized to [0, 1]
static void
writeImage(const string &path, const Mat &img) {
	Mat tmp;
	normalize(img, tmp, 255, 0, NORM_MINMAX); // normalize to [0,255]
	tmp.convertTo(tmp, CV_8UC3);
	imwrite(path, tmp);
}

// Compensate the red channel of the image using (4).
// Expects img in CV_64FC3 BGR colorspace.
// Returns the compensated red channel.
static Mat
compensateRed(const Mat &img) {
	// Split BGR channels
	Mat channels[3];
	split(img, channels);
	Mat &g = channels[1];
	Mat &r = channels[2];

	// Mean of green channel
	Scalar gmeans = mean(g);
	assert((255*(int)gmeans[1] | 255*(int)gmeans[2] | 255*(int)gmeans[3]) == 0); // should have only one channel
	double gmean = gmeans[0];

	// Mean of red channel
	Scalar rmeans = mean(r);
	assert((255*(int)rmeans[1] | 255*(int)rmeans[2] | 255*(int)rmeans[3]) == 0); // should have only one channel
	double rmean = rmeans[0];

	// Compensate red channel
	Mat tmp;
	multiply(ALPHA * (gmean - rmean) * (1.0 - r), g, tmp);
	r = r + tmp;

	return r;
}

static Mat
enhance(const Mat &img) {
	Mat r = compensateRed(img);

	// TODO
	Mat enhanced;
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
	Mat img;
	Status status = readImage(path, img);
	if (status != OK) {
		cerr << "Error reading image " << path << endl;
		return 1;
	}

	Mat enhanced = enhance(img);

	writeImage("enhanced.png", enhanced);
}
