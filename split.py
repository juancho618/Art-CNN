from PIL import Image
import math
import csv

img = Image.open('../../data/registered/IRR_crop_registered.tif')
(imageWidth, imageHeight) = img.size
gridx = 64
gridy = 64
rangex=3264/gridx
rangey=2330/gridy
print(rangex*rangey)
imgs = []

for x in range(math.floor(rangex)):
    for y in range(math.floor(rangey)):
        bbox=(x*gridx, y*gridy, x*gridx+gridx, y*gridy+gridy)
        slice_bit=img.crop(bbox)
        slice_bit.save('irr_train/xmap_'+str(x)+'_'+str(y)+'.tif', optimize=True, bits=6)
        imgs.append(['xmap_'+str(x)+'_'+str(y)+'.tif'])

data = imgs
# print(imgs)


with open("irr_train/data.csv", "w") as f:
    wr = csv.writer(f)
    for i in range(len(data)):
        wr.writerow(data[i])

print(imageWidth)