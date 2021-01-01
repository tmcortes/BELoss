import numpy as np
from PIL import Image

class CustomCenterCrop(object):
    """Crops the given PIL Image at the center selecting the final crop size
        randomly as a factor of the original image size

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, minScaleFactor, maxScaleFactor=1.0):
        
        print("Created customCenterCrop with param {}".format(minScaleFactor))
        self.minScaleFactor = minScaleFactor
        self.maxScaleFactor = maxScaleFactor

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        
        s = np.random.uniform(self.minScaleFactor, self.maxScaleFactor, 1)
        ow, oh = img.size
        nw, nh = float(ow*s), float(oh*s)
        
        y = int(round((oh - nh) / 2.))
        x = int(round((ow - nw) / 2.))

        return img.crop((x, y, x + nw, y + nh))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    

class CustomResize(object):
    
    def __init__(self, imsize):
        
        self.imsize = imsize
        
    def __call__(self, img):
        
        img.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        return img
    
    
#class CustomResize(object):
#    
#    def __init__(self, imsize):
#        
#        self.imsize = imsize
#        
#    def __call__(self, img):
#        
#        W,H = img.size
#        ratio = min(self.imsize/W, self.imsize/H)
#        img = img.resize((int(W*ratio), int(H*ratio)), Image.BICUBIC)
#
#        return img