from __future__ import (absolute_import, division, print_function,
						unicode_literals)

import os

import numpy as np
import requests
import validators
from keras import backend as K
from keras.applications import imagenet_utils
from keras.preprocessing.image import pil_image
from PIL import Image

def resize_img(im, image_size, color='black', paste=True):
	ow, oh = im.size
	tw, th = image_size
	match_width = True
	if tw > th:
		if ow / oh < tw / th:
			match_width = False
	else:
		if oh / ow > th / tw:
			match_width = False
	if match_width:
		nw = tw
		nh = nw * oh // ow
	else:
		nh = th
		nw = nh * ow // oh
	im = im.resize((nw, nh), Image.ANTIALIAS)
	if paste:
		bg = Image.new('RGB', (tw, th), color)
		bg.paste(im, ((tw - nw) // 2, (th - nh) // 2))
		return bg
	return im