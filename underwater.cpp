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

	writeImage("out.png", img); // TODO: remove
}
