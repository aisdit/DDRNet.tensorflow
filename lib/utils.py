import os
import numpy as np
from PIL import Image

def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


class Map16(object):
    def __init__(self):
        self.names = ("road", "other", "people", "box",
                "fence", "pole", "traffic light", "traffic sign",
                "vegetation", "terrain", "sky", "person", "rider", "car", "truck",
                "bus", "train", "motorcycle", "bicycle")
        self.colors  = np.array([[255,255,255], # road
                    [0,0,0], # sidewalk
                    [255,0,0], # people
                    [0,0,255], # box
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128], # bed
                    [128,128,128]], dtype=np.uint8)
                
    
    def visualize_result(self, data, pred, dir, img_name=None):
        if not os.path.exists(dir):
            os.makedirs(dir)

        # img = data
        Image.fromarray(data).save(
            os.path.join(dir, img_name+'.jpg'))

        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        for idx in np.argsort(counts)[::-1]:
            name = self.names[uniques[idx]]
            ratio = counts[idx] / pixs * 100
            if ratio > 0.1:
                print("  {}: {:.2f}%".format(name, ratio))

        # colorize prediction
        pred_color = colorEncode(pred, self.colors).astype(np.uint8)

        im_vis = pred_color
        im_vis = im_vis.astype(np.uint8)
        Image.fromarray(im_vis).save(
            os.path.join(dir, img_name+'_label.jpg'))


        return im_vis