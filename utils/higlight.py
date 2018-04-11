import numpy as np
from PIL import Image




def getImage(file, index):
    
    #load the image
    data = np.asarray(Image.open('{0}_{1}.png'.format(file, index)))
    new_image =[]
    for i in range(data.shape[0]):
        row = []
        for j in range(data.shape[1]):
            pixel = data.item(i, j)
            if pixel <  238 and pixel > 20:
                pixel= 0
            else:
                pixel = 255
            row.append(pixel)
        new_image.append(row)
    new_image = np.asarray(new_image)
    # print('shape', data.shape)
    
    #print(data)
    # print('shape', new_image.shape)
    # print(new_image)

    original = Image.fromarray(data)
    result = Image.fromarray(new_image)

    # original.show()
    # result.show()
    return new_image


# getImage('../data/images/acrh_diff_denoised_both/acrh_diff_denoised_both', 0)