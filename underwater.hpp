#define ALPHA 1.0 // red compensation factor
#define GAMMA 0.8 // gamma correction
#define GAUSS_KSIZE cv::Size(5, 5) // Gaussian blur kernel size
#define LAPLACE_KSIZE 3 // Laplacian kernel size

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
