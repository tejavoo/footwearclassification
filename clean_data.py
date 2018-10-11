import os
from PIL import Image
from utils import image_utils

folders = os.listdir('./data')

for index, folder in enumerate(folders):
	images = os.listdir('./data/' + folder + '/')
	i = 0
	for image in images:
		try:
			img = Image.open('./data/' + folder + '/' + image) # open the image file
			img.verify() # verify that it is, in fact an image
			img = Image.open('./data/' + folder + '/' + image)
			img = image_utils.resize_img(img, [300, 300])
			if os.path.splitext(image)[1] in (".jpg", ".jpeg", ".png"):
			# print('./data/'+folder+'/'+str(i).zfill(5) + os.path.splitext(image)[1])
				img.save('./data/'+folder+'/'+str(i).zfill(5) + os.path.splitext(image)[1])
				i+=1
		except (IOError, SyntaxError) as e:
			pass
		os.remove('./data/'+folder+'/'+image)