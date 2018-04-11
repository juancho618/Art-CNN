import cv2
import numpy as np
from matplotlib import pyplot as plt

def save_histogram(name, title, img_type):
    if img_type == 'gray':
        gray_img = cv2.imread('../data/{0}.tif'.format(name), cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('Robe',gray_img) to show the image
        hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
        plt.hist(gray_img.ravel(),256,[0,256])
        plt.title(title)
        # plt.show()
       
        # while True:
        #     k = cv2.waitKey(0) & 0xFF     
        #     if k == 27: break             # ESC key to exit 
        # cv2.destroyAllWindows()
    if img_type == 'color':
        from matplotlib import pyplot as plt

        img = cv2.imread('../data/{0}.tif'.format(name), -1)
        # cv2.imshow('GoldenGate',img)

        color = ('b','g','r')
        for channel,col in enumerate(color):
            histr = cv2.calcHist([img],[channel],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.title(title)
        # plt.show()
    plt.savefig('../data/histograms/{0}.png'.format(name))