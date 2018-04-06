import sys
import matplotlib.pyplot as py
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average

def main():
    # index = 7
    # data = np.load('{0}.npy'.format('../noPooling_3ch/results/results'))
    # gt = np.load('{0}.npy'.format('../data/IRR_crop_registered'))
    # file1 = gt[index]['data']
    # file2 = data[index][0]
    # read images as 2D arrays (convert to grayscale for simplicity)
    img1 = to_grayscale(imread('../data/images/IRR_crop_registered/IRR_crop_registered_6.png').astype(float))
    img2 = to_grayscale(imread('../data/images/noPoolingx3/noPoolingx3_6.png').astype(float))
    # compare
    n_m, n_0 = compare_images(img1, img2)
    print("Manhattan norm:", n_m, "/ per pixel:", n_m/img1.size)
    print ("Zero norm:", n_0, "/ per pixel:", n_0*1.0/img1.size)

def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    py.imshow(img2)
    py.show()
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

if __name__ == "__main__":
    main()