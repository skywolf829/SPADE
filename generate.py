import imageio
import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import numpy as np
from skimage import transform, io
import torch

opt = TestOptions().parse()

model = Pix2PixModel(opt)
model.eval()

folder_path = os.path.dirname(os.path.abspath(__file__))

def create_generated_image(seg_map):   
    seg = create_seg_map_tensor(seg_map)
    data = {}
    data['label'] = seg
    data['image'] = seg
    data['instance'] = torch.zeros(1)-1
    #data['path'] = None
    generated = model(data, mode='inference')
    generated = generated.cpu().detach().numpy()[0]    
    return generated

def create_seg_map_tensor(seg_map):
    seg_map = transform.resize(seg_map, (256, 256), preserve_range=True, anti_aliasing=False)
    seg_map = torch.from_numpy(seg_map)
    #seg_map = seg_map.type(torch.FloatTensor)
    t = torch.zeros([1, 1, 256, 256]) - 1
    t[0,0]=seg_map
    return t

def generated_to_savable_image(gen_image):
    gen_image = gen_image.swapaxes(0,2).swapaxes(0,1)
    gen_image = 255*((gen_image+1)/2)
    gen_image = np.uint8(gen_image)
    return gen_image

def save_image(im, path):    
    imageio.imsave(path, im)

seg = imageio.imread(os.path.join(folder_path, "TestFolder", "SegMaps", "ADE_train_00000002.png"))
gen = create_generated_image(seg)
gen_image = generated_to_savable_image(gen)
save_image(gen_image, "TestResults/image2.png")