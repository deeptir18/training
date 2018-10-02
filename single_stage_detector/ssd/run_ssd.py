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


"""
This program runs a pytorch mobilenet SSD on video data and produces labels.
This is not going to be as accurate as the maskRCNN labels.
What am I doing, it's largely unclear.
Also my laptop is so warm OMG
Note that the video is too large to load into memory - so you should label the frames as you read them individually into memory.
"""
def get_df(all_rows):
    df = pd.DataFrame(all_rows,
                      columns=['imgID', 'classname', 'prob', 'xmin', 'ymin', 'xdist', 'ydist'])
    f32 = ['prob', 'xmin', 'ymin', 'xdist', 'ydist']
    for f in f32:
        df[f] = df[f].astype('float32')
    df = df.sort_values(by=['imgID', 'prob', 'classname'], ascending=[True, False, True])
    print(df)
    return df

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
    parser.add_argument('--label_file', type=str, default="/lfs/1/deeptir/reference/single_stage_detector/ssd/labels.txt", help="label to class map")
    parser.add_argument('--feather_fname', required=True, type=str)
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
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    train_trans = SSDTransformer(dboxes, (300, 300), val=False)
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)

    val_annotate = os.path.join(args.coco, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.coco, "val2017")
    train_annotate = os.path.join(args.coco, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.coco, "train2017")

    cocoGt = COCO(annotation_file=val_annotate)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    train_coco = COCODetection(train_coco_root, train_annotate, train_trans)
    if os.path.isfile(args.checkpoint):
        print("=> loading model from checkpoint file'{}'".format(args.checkpoint))
        od = torch.load(args.checkpoint)
        ssd300.load_state_dict(od["model"])
        print("loaded model!")
    return ssd300

# need crop dict for uncropping images
def get_crop_dict():
    #HARD CODDED!!!!!!!!!!!!!!!!
    res = {}
    res['jackson-town-square'] = [0, 540, 1750, 1080]
    res['venice-rialto'] = [440, 660, 1675, 1050]
    res['venice-grand-canal'] = [0, 490, 1300, 935]
    res['amsterdam'] = [575, 390, 1250, 720]
    res['archie-day'] = [2170, 800, 3840, 2160]
    return res

"""
    dboxes = dboxes300_coco()
Loads video into a format for the model to run (???)
Basically taken from daniel's python swag codebase
Returns a numpy array that has the data
"""
def load_and_process_video(ssd, video_file, video_base_name, index_fname, start_frame, num_frames, use_cuda, inv_map, class_map):
   # for SSD300, the resolution is 300 (harcoded)
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    ssd.eval()
    resol = 300
    crop_and_exclude = True
    video_data = get_video_data(video_base_name) # this makes sure the video capture object has the correct cropping for this video
    cap = VideoCapture(video_file, index_fname)
    all_rows = [] # write the array to this feather thing at the end for post processing
    crop = get_crop_dict()[video_base_name]

    ret, frame = cap.read() # unclear why this first read is necessary
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) # move pointer to the start frame that we want
    print("Labeling {} frames, starting at frame {}".format(num_frames, start_frame))
    for i    in range(num_frames):
        data = np.zeros((1,resol, resol, 3), dtype=np.float32)
        ret, frame = cap.read()
        if not ret:
            print('Something wrong while loading the video')
            sys.exit(1)
        if crop_and_exclude: # supposedly will do the cropping for you
            frame = video_data.process_frame(frame)
        print(frame.shape)
        wtot = frame.shape[0]
        htot = frame.shape[1]
        data[0] = cv2.resize(frame, (resol, resol))
        #data /= 255
        data[...,:] -= [0.485, 0.456, 0.406]
        data[...,:] /= [0.229, 0.224, 0.225]
        #uncropped = [loc_[0] + crop[0], loc_[1] + crop[1], loc_[2] + crop[0], loc_[3] + crop[1]]
        # labeling code:
        img = data[0]
        with torch.no_grad():
            inp = torch.from_numpy(img).unsqueeze(0)
            resized = torch.from_numpy(inp.numpy().reshape(1, 3, resol, resol))
            ploc, plabel = ssd(resized)
            try:
                result = encoder.decode_batch(ploc, plabel, .5, 200)[0]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    print("In result loop")
                    # check if the object is a car
                    if prob_ < .0010: # low probability label, continue
                        continue
                    # write the output to a CSV file for comparing with the other thing
                    uncropped = loc_
                    #uncropped = [loc_[0] + crop[0], loc_[1] + crop[1], loc_[2] + crop[0], loc_[3] + crop[1]]
                    #uncropped = [loc_[0]*frame.shape[0] + crop[0],
                    #             loc_[1]*frame.shape[1]]
                    xmin = loc_[0] * wtot + crop[0]
                    ymin = loc_[1] * htot + crop[1]
    # because we can't load the entire video into memory
                    xmax = loc_[2] * wtot + crop[2]
                    ymax = loc_[3] * htot + crop[3]
                    real = [xmin, xmax, ymin, ymax]
                    print("frame: {}, loc: {}, label: {}, prob: {}, label: {}, class: {}".format(start_frame + i, uncropped, label_, prob_, inv_map[label_], class_map[label_]))
            except:
                print("")
                print("No object detected in idx: {}".format(i + start_frame))

        # TODO: write to a feather file
        cap.release()


# for COCO image - to understand what its doing
def draw_label(xmin, xmax, ymin, ymax, image, confidence, object_name, color=(0, 255, 0), outfile = "foo.png"):
    x, y = image.shape[0:2]
    tl = (int(xmin), int(ymin))
    br = (int(xmax), int(ymax))

    cv2.rectangle(image, tl, br, color, 2)
    area = (tl[0] - br[0]) * (tl[1] - br[1])
    text = '%s %2.1f' % (object_name, confidence * 100)
    cv2.putText(image, text,
                tl, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.imwrite(outfile, image)
    return image

def run_coco_eval(ssd300, args, class_map):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)

    val_annotate = os.path.join(args.coco, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.coco, "val2017")

    cocoGt = COCO(annotation_file=val_annotate)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    
    inv_map = get_inv_map(args)
    coco_eval(ssd300, args.feather_fname, val_coco, cocoGt, encoder, inv_map, .212, use_cuda, class_map)
# run coco eval

def coco_eval(model, feather_file, coco, cocoGt, encoder, inv_map, threshold, use_cuda=True, class_map = {}):
    from pycocotools.cocoeval import COCOeval
    print("")
    model.eval()
    if use_cuda:
        model.cuda()
    ret = []
    start = time.time()
    all_rows = []
    for idx, image_id in enumerate(coco.img_keys):
        img, (htot, wtot), _, _ = coco[idx]

        with torch.no_grad():
            print("Parsing image: {}/{}".format(idx+1, len(coco)), end="\r")
            inp = img.unsqueeze(0)
            if use_cuda:
                inp = inp.cuda()
            ploc, plabel = model(inp)

            try:
                result = encoder.decode_batch(ploc, plabel, 0.50, 200)[0]
            except:
                #raise
                print("")
                print("No object detected in idx: {}".format(idx))
                continue

            loc, label, prob = [r.cpu().numpy() for r in result]
            for loc_, label_, prob_ in zip(loc, label, prob):
                ret.append([image_id, loc_[0]*wtot, \
                                      loc_[1]*htot,
                                      (loc_[2] - loc_[0])*wtot,
                                      (loc_[3] - loc_[1])*htot,
                                      prob_,
                                      inv_map[label_]])
                row = [image_id, class_map[inv_map[label_]], prob_, loc_[0]*wtot, loc_[1]*htot, (loc_[2] - loc_[0])*wtot, (loc_[3]-loc_[1])*htot]
                all_rows.append(row)

    print("")
    print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))
    df = get_df(all_rows)
    feather.write_dataframe(df, feather_file)
    cocoDt = cocoGt.loadRes(np.array(ret))


    cocoDt = cocoGt.loadRes(np.array(ret))

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))
    return (E.stats[0] >= threshold) #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]


def get_label_map(args):
    ret = {}
    with open(args.label_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for line in content:
        key = int(line.split(": ")[0])
        classname = line.split(" ")[1]
        ret[key] = classname
    return ret

def parse_feather():
    df = pd.read_feather('feather.csv')
    #print(df[df["imgID"] == 139])

def classify_bear(args, ssd300 = None, class_map = None): # image # 285
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)

    val_annotate = os.path.join(args.coco, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.coco, "val2017")

    cocoGt = COCO(annotation_file=val_annotate)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    inv_map = get_inv_map(args)
    # load da bear
    #bear_path = "/lfs/1/deeptir/reference/single_stage_detector/coco/val2017/000000000285.jpg"
    bear_path = "/lfs/1/deeptir/reference/single_stage_detector/coco/val2017/000000000139.jpg"
    bear = cv2.imread(bear_path)
    data = np.zeros((300, 300, 3), dtype=np.float32)

    # transformations: resize, to tensor, normalize
    print("BEAR shape: {}".format(bear.shape))
    # htot and wtot 
    htot = bear.shape[1]
    wtot = bear.shape[0]
    print("htot: {}, wtot: {}".format(htot, wtot))
    # now need to resize
    arr = np.zeros((1,300,300, 3), dtype=np.float32)
    arr[0] = cv2.resize(bear, (300, 300))
    #pil_image = Image.fromarray(transposed.astype('uint8'))
    arr /= 255
    arr[...,:] -= [0.485, 0.456, 0.406]
    arr[...,:] /= [0.229, 0.224, 0.225]
    transposed = (arr[0].transpose(2,0,1)).astype('float32')
    print("transposed shape: {}".format(transposed.shape))
    ssd300.eval()
    #img, (htot, wtot), _,_ = val_coco[170]
    inv_map = get_inv_map(args)
    with torch.no_grad():
        inp = torch.from_numpy(transposed).unsqueeze(0)
        print(inp.shape)
        #inp = transposed.unsqueeze(0)
        ploc, plabel = ssd300(inp)
        try:
            result = encoder.decode_batch(ploc, plabel, 0.50, 200)[0]
        except:
            #raise
            print("")
            print("No object detected in idx: {}".format(idx))
        loc, label, prob = [r.cpu().numpy() for r in result]
        for loc_, label_, prob_ in zip(loc, label, prob):
            if prob_ > .4:
                row = [170, class_map[inv_map[label_]], prob_, loc_[0]*htot, loc_[1]*wtot, (loc_[2])*htot, (loc_[3])*wtot]
                print("ROW: {}".format(row))
                xmin = loc_[0]*htot
                xmax = (loc_[2])*htot
                ymin = loc_[1]*wtot
                ymax = (loc_[3])*wtot
                object_name = class_map[inv_map[label_]]

                # NOTE: xmax is first (openCV coord system)
            # NOTE: xmin and xmax multipled by htot, ymin and ymax by wtot
                bear = draw_label(xmax,xmin,ymin,ymax,bear, prob_, object_name, (0, 255, 0), "foo.jpg")



     

# TODO: is this really necessary - YES it is
def get_inv_map(args):
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)

    val_annotate = os.path.join(args.coco, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.coco, "val2017")

    cocoGt = COCO(annotation_file=val_annotate)
    coco = COCODetection(val_coco_root, val_annotate, val_trans)
    inv_map = {v:k for k,v in coco.label_map.items()}
    print(inv_map)
    return inv_map

def main():
    parse_feather()
    args = parse_args()
    class_map = get_label_map(args)
    ssd300 = load_model(args)

    # classify a SINGLE image
    
    #classify_bear(args, ssd300, class_map)
    
    #run_coco_eval(ssd300, args, class_map)
    #label_map = get_inv_map(args)
    #try_first_image(ssd300, args, label_map, class_map)
    #load_and_process_video(ssd300, args.video, args.video_base, args.index_fname, args.start_frame, args.num_frames, (args.no_cuda == False ), label_map, class_map)
    

if __name__ == '__main__':
    main()

