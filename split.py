from PIL import Image
import h5py
import math
# import pymongo
import csv
# from pymongo import MongoClient
# client = MongoClient('localhost', 27017)

img = Image.open('../../data/registered/VIS_crop_registered.tif')
(imageWidth, imageHeight) = img.size
gridx = 64
gridy = 64
rangex=3264/gridx
rangey=2330/gridy
print(rangex*rangey)
imgs = []
image_array = []



for x in range(math.floor(rangex)):
    for y in range(math.floor(rangey)):
        bbox=(x*gridx, y*gridy, x*gridx+gridx, y*gridy+gridy)
        slice_bit=img.crop(bbox)
        image_array.append(slice_bit)
        #slice_bit.save('irr_train/xmap_'+str(x)+'_'+str(y)+'.tif', optimize=True, bits=6)
        imgs.append(['xmap_'+str(x)+'_'+str(y)+'.tif'])
# data = {name : 'Visual',
#         data: image_array }
h5f = h5py.File('dataVIS.h5', 'w')
h5f.create_dataset('data' , data=image_array)   
h5f.close()

data = imgs
# print(imgs)

'''
To save csv file with the image barch name
'''
# with open("irr_train/data.csv", "w") as f:a
#     wr = csv.writer(f)
#     for i in range(len(data)):
#         wr.writerow(data[i])

