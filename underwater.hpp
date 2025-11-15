#define ALPHA 2.5 // red compensation factor
#define GAMMA 2.2 // gamma correction
#define GAUSS_KSIZE cv::Size(5, 5) // Gaussian blur kernel size
#define LAPLACE_KSIZE 5 // Laplacian kernel size
#define DELTA 0.1 // regularization term for weight map merge

typedef enum {
	OK,
	FAIL,
} Status;

// Read an image from disk and convert it CV_64FC3 BGR.
Status readImage(const std::string &path, cv::Mat &img);

// Write a CV_64FC3 BGR image to disk.
Status writeImage(const std::string &path, const cv::Mat &img);

// Write a single-channel CV_64F image to disk.
Status writeImage1c(const std::string &path, const cv::Mat &img);

// White-balance the image using red-channel compensation and Gray-World.
// Image is CV_64FC3 BGR
void whiteBalance(cv::Mat &img);

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

// Perform multi-scale fusion on the two input images and weight maps.
// i1 and i2 are the gamma-enhanced and sharpened images; they are CV_646FC3 BGR.
// w1 and w2 are the weight maps; they are CV_64F.
// nlevel is the number of pyramid levels -- must be at least 1.
// Returns the fused image R.
cv::Mat
fuse(const cv::Mat &i1, const cv::Mat &i2, const cv::Mat &w1, const cv::Mat &w2, int nlevel);
