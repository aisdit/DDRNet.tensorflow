import argparse

import numpy as np
import cv2

from lib.config import config
from lib.config import update_config
from lib.utils import Map16

import tensorflow as tf

map16 = Map16()

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="./config/giant/ddrnet23_slim.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def inference(model, image):
        size = image.shape
        pred = model(**{'inputx': image})
        pred = pred['outputy']
        pred=tf.transpose(pred,perm=[0,2,3,1])
        pred = tf.image.resize(
            images=pred, size=size[2:4],
            method='bilinear'
        )

        return tf.math.exp(pred)

def multi_scale_inference(model, image):
        ori_height, ori_width, _ = image.shape        

        new_img = multi_scale_aug(image=image)
        height, width = new_img.shape[:-1]

        new_img = new_img.transpose((2, 0, 1))
        new_img = np.expand_dims(new_img, axis=0)
        new_img = tf.convert_to_tensor(new_img, dtype=tf.float32)

        preds = inference(model, new_img)
        preds = preds[:, 0:height, 0:width, :]
        
        preds = tf.image.resize(
            images=preds, size=(ori_height, ori_width),
            method='bilinear'
            )      

        return preds

def multi_scale_aug(image):
        long_size = config.BASE_SIZE
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        
        return image

def input_transform(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= mean
        image /= std
        return image

def main():
    args = parse_args()

    model = tf.saved_model.load(config.MODEL_FILE)

    # rtsp_ip = 'rtsp://root:a1s2d3f4@210.61.163.32:8889/live.sdp'
    rtsp_ip = 'D:/giant_factory/1.mp4'
    cam = cv2.VideoCapture(rtsp_ip)
    frame_id=0
    while cam.isOpened():
        ret , frame = cam.read()
        if not ret:
            break
        # vis = frame.copy()
        # cv2.imshow('cam', vis)
        # key = cv2.waitKey(5)
        # if 0xFF & key == 27:  # Esc
        #     break

        # frame=cv2.imread('./data/giant/dataset2/images/30.jpg')
        frame=cv2.imread('./data/0.jpg')
        image = input_transform(frame)
        pred = multi_scale_inference(            
            model,
            image
            )

        pred=tf.transpose(pred,perm=[0,3,1,2])

        # Visualization
        pred = tf.math.argmax(pred,1)
        pred = np.squeeze(pred.cpu().numpy(),0)
        map16.visualize_result(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), pred, config.OUTPUT_DIR+'giant', str(frame_id))        
        frame_id+=1


if __name__ == '__main__':
    main()
