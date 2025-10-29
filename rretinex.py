import argparse
import cv2 as cv
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator


BETA = 0.05
OMEGA = 0.01
DELTA = 1.0
LAMBDA = 10.0
SIGMA = 10.0
MU0 = 1.0
RHO = 1.5
GAMMA = 2.2
EPSILON = 0.005

THRESH = 1e-3 # iteration threshold: stop once |R^k - R^(k+1)| < THRESH
MAXITER = 100 # maximum number of iterations

CGMAXITER = 100 # maximum number of Conjugate Gradient iterations


Mat = np.typing.NDArray[np.float64]


# Read an image from disk and convert it to HSV, double precision, normalized to [0,1].
def readImage(path: str) -> Mat:
	# Read image file
	img: Mat = cv.imread(path, cv.IMREAD_COLOR_BGR)
	if img is None:
		raise IOError("Error image file: " + path)

	hsvImg: Mat = cv.cvtColor(img, cv.COLOR_BGR2HSV) # convert to HSV colorspace
	hsvImg = cv.normalize(hsvImg.astype(np.float64), None, 1.0, 0.0, cv.NORM_MINMAX);

	return hsvImg


def updateR(r: Mat, l: Mat, n: Mat, i: Mat, g: Mat) -> Mat:
	print("Updating R...")

	lv: Mat = l.flatten() # vectorize L
	print("lv shape", lv.shape)

	# Solve Ax = b, where x is the vectorized version of r
	def A(rv: Mat) -> Mat:
		left = cv.multiply(lv**2, rv).flatten()
		right = OMEGA * cv.Laplacian(rv.reshape(r.shape), cv.CV_64F).flatten()
		return left + right
	h, w = r.shape
	Ax = LinearOperator((h*w, h*w), matvec=A, dtype=np.float64)

	b: Mat = (cv.multiply(l, i-n) + OMEGA * gradientImage(g)).flatten()
	print("b shape:", b.shape)

	r1v, status = cg(Ax, b, maxiter=CGMAXITER) # solve using Conjugate Gradient method
	if status != 0:
		print(f"Warning: Conjugate Gradient did not converge after {CGMAXITER} iterations")
	return np.reshape(r1v, r.shape)


def isConverged(r: Mat, r1: Mat, l: Mat, l1: Mat) -> bool:
	return cv.norm(r1-r) < THRESH or cv.norm(l1-l) < THRESH


# Magnitude of the gradient
def gradientImage(i: Mat) -> Mat:
	gx: Mat = cv.Sobel(i, cv.CV_64F, 1, 0)
	gy: Mat = cv.Sobel(i, cv.CV_64F, 0, 1)
	gx = cv.pow(gx, 2)
	gy = cv.pow(gy, 2)
	g: Mat = cv.sqrt(gx+gy)
	return g


# The adjust gradient G.
# Takes the V channel of the input image.
def adjustedGradient(i: Mat) -> Mat:
	# Modified gradient of image
	grad: Mat = gradientImage(i) # compute magnitude of gradient
	absgrad: Mat = np.abs(grad) # take absolute value
	sign: Mat = cv.divide(grad, absgrad) # take sign of each element
	_, absgrad = cv.threshold(absgrad, EPSILON, 0, cv.THRESH_TOZERO) # filter out small elements
	gradhat: Mat = cv.multiply(absgrad, sign) # add sign back in to get modified gradient

	# K factor
	k: Mat = LAMBDA * cv.exp(-np.abs(gradhat) / SIGMA) + 1

	# Final G matrix
	return cv.multiply(k, gradhat)


# Decompose the input image's V channel into its constituent
# reflectance, illumination, and noise components.
# This is Algorithm 1 in the paper.
def decompose(i: Mat) -> (Mat, Mat, Mat):
	r: Mat = np.zeros_like(i) # reflectance
	l: Mat = np.zeros_like(i) # illumination
	n: Mat = np.zeros_like(i) # noise
	r1: Mat = np.zeros_like(i) # R^(k+1)
	l1: Mat = np.zeros_like(i) # R^(k+1)
	t: Mat = np.zeros_like(i) # gradient of L
	z: Mat = np.zeros_like(i) # Lagrange multiplier
	g: Mat = np.zeros_like(i) # adjusted gradient of I

	l = i
	l1 = i

	g = adjustedGradient(i)

	k: int = 0
	while True:
		r = r1
		l = l1

		r1 = updateR(r, l, n, i, g)
		# TODO: update l, n, t, z

		k = k+1
		if k > MAXITER or isConverged(r, r1, l, l1):
			break

	# TODO
	raise NotImplementedError

	return (r, l, n)


def enhance(img: Mat) -> Mat:
	h, s, v = cv.split(img)

	r, l, n = decompose(v)

	# TODO
	raise NotImplementedError


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("filename")
	args = argparser.parse_args()

	img: Mat = readImage(args.filename)

	enhanced: Mat = enhance(img)

	cv.imshow("Enhanced image", enhanced)
	cv.waitKey(0)
