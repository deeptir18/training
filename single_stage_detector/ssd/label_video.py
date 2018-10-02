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
import feather
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from train import dboxes300_coco

"""
This program runs a pytorch mobilenet SSD, trained on the MSCOCO dataset, on individual frames of a video clip to produce labels.
Used to produce less accurate labels than maskRCNN to see if assertions can be useful.
We use blazeit and swag-python to decode the video.
"""

def parse_args():
    parser = argparse.ArgumentParser(description = 'Pytorch MobileNet SSD program')
    parser.add_argument('--video', '-v', type=str,
                        default="/lfs/1/deeptir/data/svideo/jackson-town-square/2017-12-14",
                        help='path to video frames to label')
    parser.add_argument('--video_base', '-b', type=str, default="jackson-town-square",
                        help="base name for video class (used for decoding)")
    parser.add_argument("--index_fname", '-i', type=str,
                        default="/lfs/1/deeptir/data/svideo/jackson-town-square/2017-12-14.json",
                        help="Index filename for this video")
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--checkpoint', '-c', type=str, 
                        default="/lfs/1/deeptir/reference/single_stage_detector/ssd/models/iter_180000.pt",
                        help='pretrained SSD model checkpoint')
    parser.add_argument('--gpu',
                        default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--coco', '-d', type=str,
                        default='/lfs/1/deeptir/reference/single_stage_detector/coco/',
                        help='path to test and training data files')
    parser.add_argument('--num_frames', type=int, default=-1, help='Number of frames to label')
    parser.add_argument('--start_frame', type=int, default=0, help='Frame to start labeling')
    parser.add_argument('--label_file', type=str,
                        default="/lfs/1/deeptir/reference/single_stage_detector/ssd/labels.txt",
                        help="label to class map")
    parser.add_argument('--feather_fname', required=True, type=str,
                        help="Write labels to this output dataframe file")
    parser.add_argument('--draw_single_frame',action='store_true')
    parser.add_argument('--parse_feather', action='store_true')

    return parser.parse_args()

def get_df(all_rows):
    """
    Takes raw array of row information and returns a dataframe object to record.
    Arguments:
        all_rows: array of individual labels per row
    Returns:
        dataframe object
    """
    df = pd.DataFrame(all_rows,
                      columns=['frame', 'object_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax', 'index']) # index is not used initially
    f32 = ['confidence', 'xmin', 'ymin', 'xmax', 'ymax']
    for f in f32:
        df[f] = df[f].astype('float32')
    df = df.sort_values(by=['frame', 'confidence', 'object_name'], ascending=[True, False, True])
    return df

def read_label_map(args):
    """
    Function to return map from label numbers to string labels in the coco dataset.
    Args:
        args: Argparse args that contains a path to the cocodataset.
    Returns:
        Map from label numbers to string labels.
    """
    ret = {}
    with open(args.label_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for line in content:
        key = int(line.split(": ")[0])
        classname = line.split(" ")[1]
        ret[key] = classname
    return ret

def get_inv_map(args):
    """
    Function to invert label numbers in coco validation data to the global labels.
    Args:
        args: Argparse args that contains a path to the cocodataset.
    Returns:
        Map from label numbers produced by the ssd to the real labels that correspond to some text.
    """
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)
    val_annotate = os.path.join(args.coco, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.coco, "val2017")
    coco = COCODetection(val_coco_root, val_annotate, val_trans)
    inv_map = {v:k for k,v in coco.label_map.items()}
    return inv_map


def load_empty_model(args):
    """
    Function to load an empty ssd300 model.
    Args:
        args: Argparse args that contain a path to the cocodataset.
    Returns:
        ssd300 object (without any weights)
    """
    dboxes = dboxes300_coco()
    train_trans = SSDTransformer(dboxes, (300, 300), val=False)
    train_annotate = os.path.join(args.coco, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.coco, "train2017")
    train_coco = COCODetection(train_coco_root, train_annotate, train_trans)
    
    # the point of this function is to create an empty model - but the SSD300 class needs a number of classes by default or something
    ssd300 = SSD300(train_coco.labelnum) # initialize an empty model - the parameter represents the number of objects things are classified into
    return ssd300 

def load_model(args):
    """
from blazeit.vis.utils import draw_label
    Function to load weights
    Args:
        args: Argparse args that contains a path to the saved pytorch model for the SSD300.
    Returns:
        ssd300 object, with weights loaded if the file is passed in.
    """
    # create new model
    ssd300 = load_empty_model(args)
    # load the weights from the file
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if os.path.isfile(args.checkpoint):
        print("=> loading model from checkpoint file'{}'".format(args.checkpoint))
        od = torch.load(args.checkpoint)
        ssd300.load_state_dict(od["model"])
        print("loaded model!")
    else:
        print("Warning: ssd300 is not trained!!!")
    return ssd300

def get_crop_dict(video_base_name):
    """
    For the video data we have, returns ***hardcoded*** cropping parameters.
    Args:
        video_base_name: string identifier for specific video being labeled.
    Returns:
        crop array for video
    """
    res = {}
    res['jackson-town-square'] = [0, 540, 1750, 1080]
    res['venice-rialto'] = [440, 660, 1675, 1050]
    res['venice-grand-canal'] = [0, 490, 1300, 935]
    res['amsterdam'] = [575, 390, 1250, 720]
    res['archie-day'] = [2170, 800, 3840, 2160]
    if video_base_name not in res:
        print("Could not find video base name for cropping parameters: {}, options are: {}".format(video_base_name, res.keys()))
    return res[video_base_name]


def draw_label(xmin, xmax, ymin, ymax, image, confidence, object_name, color=(0, 255, 0), outfile = "foo.png"):
    print("OpenCV xmin, xmax, ymin, ymax: [{},{},{},{}]".format(xmin, xmax, ymin, ymax))
    x, y = image.shape[0:2]
    tl = (int(xmin), int(ymin))
    br = (int(xmax), int(ymax))
    print("TL: {}, BR: {}".format(tl, br))
    cv2.rectangle(image, tl, br, color, 2)
    area = (tl[0] - br[0]) * (tl[1] - br[1])
    text = '%s %2.1f' % (object_name, confidence * 100)
    cv2.putText(image, text,
                tl, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.imwrite(outfile, image)
    return image

def label_single_frame(ssd, inv_map, label_map, video_file, video_base_name, index_fname, start_frame, use_cuda = False):
    """
    For debugging purposes, labels a single frame and writes the image to a file.
    video_file and frame: creates path to image
    out_image: where to write the labeled image to
    """
    threshold = .45
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    ssd.eval()
    resol = 300
    crop_and_exclude = True
    video_data = get_video_data(video_base_name) # object to process frames/crop data
    cap = VideoCapture(video_file, index_fname) # load specific video frames
    crop = get_crop_dict(video_base_name) # crop dictionary
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) # move pointer for video capture to the frame we are interested in
    ret, original_frame = cap.read()
    print("Labeling frame {}".format(start_frame))
    
    if not ret:
        print("Something wrong with labeling frame {}".format(frame))

    data = np.zeros((1, resol, resol, 3), dtype = np.float32)
    print("Original frame shape: {}".format(original_frame.shape))
    print("Crop: {}".format(crop))

    # access wtot and htot from original frame
    wtot = original_frame.shape[1]
    htot = original_frame.shape[0]
    cv2.imwrite("frame{}_original.jpg".format(start_frame), original_frame)
    if crop_and_exclude:
        frame = video_data.process_frame(original_frame) # apply cropping if necessary, from video_base_name
        wtot = frame.shape[1]
        htot = frame.shape[0]
        cv2.imwrite("frame{}_cropped.jpg".format(start_frame), frame)
        print("Cropped frame shape: {}".format(frame.shape))
    data[0] = cv2.resize(frame, (resol, resol))

    #recentering for imagenet mean and stddev
    data /= 255
    data[...,:] -= [0.485, 0.456, 0.406]
    data[...,:] /= [0.229, 0.224, 0.225]
    transposed_frame = (data[0].transpose(2,0,1)).astype('float32')

    with torch.no_grad():
        inp = torch.from_numpy(transposed_frame).unsqueeze(0)
        ploc, plabel = ssd(inp)
        try:
            result = encoder.decode_batch(ploc, plabel, 0.5, 200)[0]
        except:
            print("")
            print("no object detected in idx: {}".format(start_frame))
        loc, label, prob = [r.cpu().numpy() for r in result]
        for loc_, label_, prob_ in zip(loc, label, prob):
            if prob_ > threshold:
                # calculate real location
                print("local coords: {}".format(loc_))
                (xmin, xmax, ymin, ymax) = get_real_location(loc_, wtot, htot, crop)
                print ("Frame {}, detected {} with prob {} at location [{}, {}, {}, {}: xmin,ymin, xmax, ymax]".format(start_frame, label_map[inv_map[label_]], prob_, xmin, ymin, xmax, ymax))
                object_name = label_map[inv_map[label_]]
                # draw into the frame's
                original_frame = draw_label(xmin, xmax, ymin, ymax, original_frame, prob_, object_name, (255, 255, 0), "frame{}.jpg".format(start_frame))
    cap.release()

def load_and_process_video(ssd, inv_map, label_map, video_file, video_base_name, index_fname, start_frame, num_frames, feather_file, threshold = .25, debug = False, use_cuda = False):
    """
    Functions that loads the video frames from the file specified,
    and processes the specified frames to label all objects with confidence > THRESHOLD.
    Writes label info and bounding box info to a feather file; later can be used to visualize the labels.
    Args:
        ssd: ssd300 model, ideally trained
        inv_map: apparently the labels produced need to be mapped to the real labels for some reason.
        label_map: maps the object number to the object name
        video_file: folder that contains video frames we are interested in
        video_base_name: class of video files (each video has specific cropping parameters defined in @get_crop_dict)
        index_fname: path to index file with metadata about the video
        start_frame: frame to start labeling
        num_frames: number of frames to label
        feather_file: File to write final labels into (meant for further parsing later)
        threshold: threshold probability for writing objects and labels into the feather file
        debug: print out debug statements or not
        use_cuda: use cuda - lets you select a specific GPU for storing specific tensors
    """
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    ssd.eval()
    resol = 300
    crop_and_exclude = True
    video_data = get_video_data(video_base_name) # object to process frames/crop data
    cap = VideoCapture(video_file, index_fname) # load specific video frames
    crop = get_crop_dict(video_base_name) # crop dictionary

    all_rows = [] # write into feather file at end

    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) # move pointer for video capture to the frame we are interested in
    
    print("Labeling {} frames, starting at frame {}".format(num_frames, start_frame))
    for i in range(num_frames):
        ret, frame = cap.read() # get this specific frame
        if not ret:
            print("Something wrong while loading the video at frame {}".format(start_frame + i))
            exit(1)
        data = np.zeros((1, resol, resol, 3), dtype = np.float32)
        if crop_and_exclude:
            frame = video_data.process_frame(frame) # apply cropping if necessary, from video_base_name
        
        # (already cropped if necessary) frame width and height
        wtot = frame.shape[1]
        htot = frame.shape[0] # the video coordinates are HWC
        data[0] = cv2.resize(frame, (resol, resol))

        #recentering for imagenet mean and stddev
        data /= 255
        data[...,:] -= [0.485, 0.456, 0.406]
        data[...,:] /= [0.229, 0.224, 0.225]

        # change to CHW
        transposed_frame = (data[0].transpose(2,0,1)).astype('float32')

        with torch.no_grad():
            inp = torch.from_numpy(transposed_frame).unsqueeze(0)
            ploc, plabel = ssd(inp)
            try:
                result = encoder.decode_batch(ploc, plabel, 0.5, 200)[0]
            except:
                print("")
                print("no object detected in idx: {}".format(start_frame + i))
            loc, label, prob = [r.cpu().numpy() for r in result]
            for loc_, label_, prob_ in zip(loc, label, prob):
                if prob_ > threshold:
                    # calculate real location
                    (xmin, xmax, ymin, ymax) = get_real_location(loc_, wtot, htot, crop) # no crop for now
                    # add row to dataframe
                    # frame, object_name, confidence, xmin, ymin, xmax, ymax, index
                    # for here, index can be 0
                    row = [start_frame + i, label_map[inv_map[label_]], prob_,  xmin, ymin, xmax, ymax, 0]
                    # debug print
                    if debug:
                        print ("Frame {}, detected {} with prob {} at location [xmin: {}, xmax: {}, ymin: {}, ymax{}]".format(start_frame + i, label_map[inv_map[label_]], prob_, xmin, xmax, ymin, ymax))
                    all_rows.append(row)
    print("Finished labeling selected frames, writing into file {}".format(feather_file))
    df = get_df(all_rows)
    feather.write_dataframe(df, feather_file)
    cap.release()

def get_real_location(loc, wtot, htot, crop = [0,0,0,0]):
    """
    Given ssd output location, this function calculates the corresponding bounding box,
    using the scaling and cropping factors.
    Args:
        loc: array output of ssd model
        wtot: frame width
        htot: frame height
        crop: how video was cropped initially.
    Returns:
        Tuple containing (xmin, xmax, ymin, ymax) coordinates of bounding box.
    """
    # unlike frames, crop coords are (xmin, ymin, xmax, ymax) 
    # wtot = (xmax - xmin) ; htot = (ymax - min)
    xmin = loc[0] * wtot + crop[0]
    xmax = loc[2] * wtot + crop[0]
    ymin = loc[1] * htot + crop[1]
    ymax = loc[3] * htot + crop[1]
    return (xmin, xmax, ymin, ymax)

def parse_feather(feather_file):
    """
    Parses the feather file into a csv to do interesting things.
    """
    # write confident cars into a csv
    threshold = .2
    #columns=['frame', 'object_name', 'prob', 'xmin', 'ymin', 'xmax', 'ymax'])
    df = pd.read_feather(feather_file)
    filtered = df[(df["object_name"] == 'car')]
    print(filtered)
    # we want to write to a csv file, where we have the rows: 
    # frame, object_name, confidence, xmin, ymin, xmax, ymax, ind
    new_rows = []
    last_frame = 0
    ind = 0
    for index, row in filtered.iterrows():
        if row["frame"] > last_frame + 40:
            ind += 1
        new_rows.append([row["frame"], row["object_name"], row["confidence"], row["xmin"], row["ymin"], row["xmax"], row["ymax"], ind])
        last_frame = row["frame"]
    filtered_df = get_df(new_rows)
    filtered_df.to_csv("data2.csv", sep=",", index=False, header=True)
    
def get_inv_map(args):
    """
    Returns inversion map from ssd output to labels corresponding to coco objects.
    """
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)

    val_annotate = os.path.join(args.coco, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.coco, "val2017")

    cocoGt = COCO(annotation_file=val_annotate)
    coco = COCODetection(val_coco_root, val_annotate, val_trans)
    inv_map = {v:k for k,v in coco.label_map.items()}
    return inv_map

def main():
    threshold = .1
    debug = True
    args = parse_args()
    if args.parse_feather:
        parse_feather(args.feather_fname)
        return
    label_map = read_label_map(args)
    inv_map = get_inv_map(args)
    ssd300 = load_model(args)
    if args.draw_single_frame:
        # just load and draw a single frame
        label_single_frame(ssd300, inv_map, label_map, args.video, args.video_base,
                            args.index_fname, args.start_frame)
    else:
        load_and_process_video(ssd300, inv_map, label_map, args.video, args.video_base,
                            args.index_fname, args.start_frame, args.num_frames, args.feather_fname,
                            threshold, debug, (args.no_cuda == False))

if __name__ == '__main__':
    main()

