#define ALPHA 1.0 // red compensation factor
#define GAMMA 0.8 // gamma correction
#define GAUSS_KSIZE cv::Size(5, 5) // Gaussian blur kernel size
#define LAPLACE_KSIZE 3 // Laplacian kernel size
#define DELTA 0.1 // regularization term for weight map merge

typedef enum {
	OK,
	FAIL,
} Status;

// Read an image from disk and convert it CV_64FC3 BGR.
Status readImage(const char *path, cv::Mat &img);

// Write a CV_64FC3 BGR image to disk.
void writeImage(const char *path, const cv::Mat &img);

// Write a CV_64F image to disk.
void write1dImage(const char *path, const cv::Mat &img);

// Laplacian contrast weight.
// Input is CV_64FC3 BGR.
// Output is CV_64F.
cv::Mat laplacianWeight(const cv::Mat &img);

// Saliency weight.
// Input is CV_64FC3 BGR.
// Output is CV_64F.
cv::Mat saliencyWeight(const cv::Mat &img);

// Saturation weight.
// Input is CV_64FC3 BGR.
// Output is CV_64F.
cv::Mat saturationWeight(const cv::Mat &img);

// Merge the weight maps of the two inputs into two aggregated weight maps.
// wl[12] are the Laplacian weight maps.
// wsal[12] are the saliency weight maps.
// wsat[[12] are the saturation weight maps.
// w[12] are the aggregated weight maps.
// All Mats are CV_64FC3 BGR.
void
mergeWeightMaps(const cv::Mat &wl1, const cv::Mat &wl2,
	const cv::Mat &wsal1, const cv::Mat &wsal2,
	const cv::Mat &wsat1, const cv::Mat &wsat2,
	cv::Mat &w1, cv::Mat &w2);