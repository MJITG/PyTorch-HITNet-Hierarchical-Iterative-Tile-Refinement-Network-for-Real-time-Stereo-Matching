import numpy as np
import re
import sys
from utils.experiment import tensor2numpy


def write_pfm_tensor(file, image, scale = 1):
  image = tensor2numpy(image)
  image = np.array(image, dtype='float32')
  file = open(file, 'wb')

  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write(b'PF\n' if color else b'Pf\n')
  file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write(b'%f\n' % scale)

  image = np.flipud(image)

  image.tofile(file)