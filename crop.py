from PIL import Image
im = Image.open('../../data/registered/' + IRR_image)
left = 100
top = 804
width = 300
height = 200
box = (left, top, left+width, top+height)
img = im.crop(box)
img.show()
