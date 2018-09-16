import argparse
import os
import random
import shutil
import time
import warnings

from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd300 import SSD300
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np
from coco import COCO
from blazeit.data.video_data import get_video_data
from swag.video_capture import VideoCapture
import cv2
"""
This program runs a pytorch mobilenet SSD on video data and produces labels.
This is not going to be as accurate as the maskRCNN labels.
What am I doing, it's largely unclear.
Also my laptop is so warm OMG
Note that the video is too large to load into memory - so you should label the frames as you read them individually into memory.
"""

def dboxes300_coco():
    """
    Taken from MLPerf code
    """
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def parse_args():
    parser = argparse.ArgumentParser(description = 'Pytorch MobileNet SSD program')
    parser.add_argument('--video', '-v', type=str, default="/lfs/1/deeptir/data/svideo/jackson-town-square/2017-12-14",
                        help='path to produce video for labels')
    parser.add_argument('--video_base', '-b', type=str, default="jackson-town-square",
                        help="base name for video class (used for decoding)")
    parser.add_argument("--index_fname", '-i', type=str, default="/lfs/1/deeptir/data/svideo/jackson-town-square/2017-12-14.json",
                        help="Index filename for this video")
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--checkpoint', '-c', type=str, default="/lfs/1/deeptir/reference/single_stage_detector/ssd/models/iter_180000.pt", help='pretrained SSD model checkpoint')
    parser.add_argument('--gpu', default=None, type=int,
                                help='GPU id to use.')
    parser.add_argument('--coco', '-d', type=str, default='/lfs/1/deeptir/reference/single_stage_detector/coco/',
                        help='path to test and training data files')
    #parser.add_argument('--objects', required=True, help='Object to draw')
    # because we can't load the entire video into memory
    parser.add_argument('--num_frames', type=int, default=-1, help='Number of frames to label')
    parser.add_argument('--start_frame', type=int, default=0, help='Frame to start labeling')
    return parser.parse_args()

def load_empty_model(args):
    dboxes = dboxes300_coco()
    train_trans = SSDTransformer(dboxes, (300, 300), val=False)
    train_annotate = os.path.join(args.coco, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.coco, "train2017")
    train_coco = COCODetection(train_coco_root, train_annotate, train_trans)
    
    # the point of this function is to create an empty model - but the SSD300 class needs a number of classes by default or something
    ssd300 = SSD300(train_coco.labelnum) # initialize an empty model - the parameter represents the number of objects things are classified into
    return ssd300 

def load_model(args):
    # create new model?
    ssd300 = load_empty_model(args)
    # load the weights from the file
    if os.path.isfile(args.checkpoint):
        print("=> loading model from checkpoint file'{}'".format(args.checkpoint))
        od = torch.load(args.checkpoint)
        ssd300.load_state_dict(od["model"])
        print("loaded model!")
    return ssd300

"""
    dboxes = dboxes300_coco()
Loads video into a format for the model to run (???)
Basically taken from daniel's python swag codebase
Returns a numpy array that has the data
"""
def load_and_process_video(ssd, video_file, video_base_name, index_fname, start_frame, num_frames, use_cuda):
   # for SSD300, the resolution is 300 (harcoded)
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    ssd.eval()
    resol = 300
    crop_and_exclude = True
    video_data = get_video_data(video_base_name) # this makes sure the video capture object has the correct cropping for this video
    cap = VideoCapture(video_file, index_fname)

    ret, frame = cap.read() # unclear why this first read is necessary
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) # move pointer to the start frame that we want
    print("Labeling {} frames, starting at frame {}".format(num_frames, start_frame))
    for i in range(num_frames):
        data = np.zeros((1,resol, resol, 3), dtype=np.float32)
        ret, frame = cap.read()
        if not ret:
            print('Something wrong while loading the video')
            sys.exit(1)
        if crop_and_exclude: # supposedly will do the cropping for you
            frame = video_data.process_frame(frame)	
        data[0] = cv2.resize(frame, (resol, resol))
        # based on the mean and STD of imagenet (need to recenter the frame)
        #data /= 255.
        data[...,:] -= [0.485, 0.456, 0.406]
        data[...,:] /= [0.229, 0.224, 0.225]

        # labeling code:
        img = data[0]
        with torch.no_grad():
            inp = torch.from_numpy(img).unsqueeze(0)
        resized = torch.from_numpy(inp.numpy().reshape(1, 3, resol, resol))
        print('Size of thing: {}'.format(resized.size()))
        ploc, plabel = ssd(resized)
        #try:
        print("About to call decode function")
        result = encoder.decode_batch(ploc, plabel, .5, 200)[0]
        # do we use decode single or decode batch?
        print("GOT RESULT")
        #except:
        #    print("No object detected in idx: {}".format(i + start_frame))
        #    print("")
           # continue
        print("Result: {}".format(result))
        loc, label, prob = [r.cpu().numpy() for r in result]
        for loc_, label_, prob_ in zip(loc, label, prob):
            print("In result loop")
            # see if the probability is high?
            # check if the object is a car
            # write the output to a CSV file for comparing with the other thing
            if prob < .0015: # low probability label, continue
                continue
            # get actual location from the crop
            print("Got to a label?")
            print(loc_)
            print(label_)
            print(prob_)
            exit(1)
        #except:
        #    print("")
        #    print("No object detected in idx: {}".format(i + start_frame))

def try_coco(ssd300, args):
    # try labeling the coco val set to see if I've done things correctly
    use_cuda = (args.no_cuda == False)
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)

    val_annotate = os.path.join(args.coco, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.coco, "val2017")

    cocoGt = COCO(annotation_file=val_annotate)
    coco = COCODetection(val_coco_root, val_annotate, val_trans)
    inv_map = {v:k for k,v in coco.label_map.items()}
    ret = []
    ssd300.eval()
    if use_cuda:
        ssd300.cuda()
    start = time.time()
    for idx, image_id in enumerate(coco.img_keys):
        img, (htot, wtot), _, _ = coco[idx]

        with torch.no_grad():
            print("Parsing image: {}/{}".format(idx+1, len(coco)), end='\r')
            inp = img.unsqueeze(0)
            if use_cuda:
                inp = inp.cuda()
            print("Size of inP: {}".format(inp.size()))
            ploc, plabel = ssd300(inp)

            try:
                result = encoder.decode_batch(ploc, plabel, 0.50, 200)[0]
            except:
                #raise
                print("")
                print("No object detected in idx: {}".format(idx))
                continue
            print("GOT AN OBJECT")
            loc, label, prob = [r.cpu().numpy() for r in result]
            for loc_, label_, prob_ in zip(loc, label, prob):
                ret.append([image_id, loc_[0]*wtot, \
                                      loc_[1]*htot,
                                      (loc_[2] - loc_[0])*wtot,
                                      (loc_[3] - loc_[1])*htot,
                                      prob_,
                                      inv_map[label_]])
    print("")
    print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))

    cocoDt = cocoGt.loadRes(np.array(ret))

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))
    return (E.stats[0] >= threshold) #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]
    

def main():
    args = parse_args()
    ssd300 = load_model(args)
    #try_coco(ssd300, args)
    # loads the cropped video dataload_and_process_video(ssd300. args.video, args.video_base, args.index_fname, args.start_frame, args.num_frames)
    load_and_process_video(ssd300, args.video, args.video_base, args.index_fname, args.start_frame, args.num_frames, (args.no_cuda == False ))
    # produce the labels and the bounding boxes for comparison to the other thingy?
    # frame number, 'car', 

    # need to:
    # go through frame by frame and generate a bunch of labels per frame (classes data) and write it to a JSON file
    # write now we're only labeling 'car' (later we can generalize to other objects as well)
    

if __name__ == '__main__':
    main()

