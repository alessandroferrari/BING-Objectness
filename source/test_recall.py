'''
Created on Jan 31, 2015

@author: alessandro
'''
import os
import cv2
import sys
import json
import warnings
import numpy as np
from bing import Bing
from dataset import Dataset
from common import bounding_box_overlap

class EvaluateRecall(object):
    
    def __init__(self, w_1st, sizes_idx, w_2nd, num_bbs_per_size_1st_stage= 130, num_bbs_final = 1500):
    
        self.w_1st = w_1st
        self.sizes_idx = sizes_idx
        self.w_2nd = w_2nd      
        self.bing = Bing(w_1st,sizes_idx,w_2nd, num_bbs_per_size_1st_stage = num_bbs_per_size_1st_stage, num_bbs_final = num_bbs_final)  
    
    def evaluate_test_set(self, test_annotations):
        
        tot_num_gt_bbs = 0
        
        print "Getting ground truth and predicted bounding boxes from testing images."
        images_bbs_dict = dict()
        for key in test_annotations.keys():
            ann_dict = test_annotations[key]
            fn = os.path.join(ann_dict["path"],ann_dict["basename"])
            img = cv2.imread(fn)
            if img is None:
                warnings.warn("The image %s does not exist in the filesystem."%fn)
            #calculating features for each ground truth bounding box
            bbs = ann_dict["bbs"]
            predicted_bbs, _ = self.bing.predict(img)
            tot_num_gt_bbs = tot_num_gt_bbs + len(bbs)        
            images_bbs_dict[ann_dict["basename"]] = (bbs, predicted_bbs)
        
        print "Calculate the recall of predicted bounding boxes that overlap at least the 50% with ground truth bounding boxes."
        overlaps_array = np.zeros(tot_num_gt_bbs)
        gt_bbs_idx = 0
        for img_bn in images_bbs_dict.keys():
            gt_bbs, predicted_bbs = images_bbs_dict[img_bn]
            for i, gt_bb in enumerate(gt_bbs):
                overlaps_array[gt_bbs_idx+i] = bounding_box_overlap(predicted_bbs, bb_query = gt_bb["bb"])
            gt_bbs_idx = gt_bbs_idx + len(gt_bbs)      
        detected = (overlaps_array>0.5).astype(float)
        
        recall = np.sum(detected)/len(detected)
        
        return recall
        
def parse_cmdline_inputs():
    """
    Example parameters:
    
    {
        "basepath": "/opt/Datasets/VOC2007",
        "training_set_fn": "/opt/Datasets/VOC2007/ImageSets/Main/train.txt",
        "test_set_fn": "/opt/Datasets/VOC2007/ImageSets/Main/test.txt",
        "annotations_path": "/opt/Datasets/VOC2007/Annotations",
        "images_path": "/opt/Datasets/VOC2007/JPEGImages",
        "results_dir": "/opt/Datasets/VOC2007/BING_Results",
        "1st_stage_weights_fn":"/opt/Datasets/VOC2007/BING_Results/weights.txt",
        "2nd_stage_weights_fn": "/opt/Datasets/VOC2007/BING_Results/2nd_stage_weights.json",
        "sizes_indeces_fn": "/opt/Datasets/VOC2007/BING_Results/sizes.txt",
        "num_win_psz": 130,
        "num_bbs": 1500
    }
    """
    if len(sys.argv) != 2:
        print "Example of usage: ' python train_bing.py /path/to/dataset/parameters.json '"
        sys.exit(2)
    params_file = sys.argv[1]
    if not os.path.exists(params_file):
        print "Specified file for parameters %s does not exist."%params_file
        sys.exit(2)
    try:
        f = open(params_file, "r")
        params_str = f.read()
        f.close()
    except Exception as e:
        print "Error while reading parameters file %s. Exception: %s."%(params_file,e)
        sys.exit(2)
    try:
        params = json.loads(params_str)
    except Exception as e:
        print "Error while parsing parameters json file %s. Exception: %s."%(params_file,e)
        sys.exit(2)
    
    if not params.has_key("num_win_psz"):
        params["num_win_psz"] = 130
    if not params.has_key("num_bbs"):
        params["num_bbs"] = 1500
    
    return params    
    
if __name__=='__main__':
    
    params = parse_cmdline_inputs()
    basepath = params["basepath"]
    training_set_fn = params["training_set_fn"]
    test_set_fn = params["test_set_fn"]
    annotations_path = params["annotations_path"]
    images_path = params["images_path"]
    results_dir = params["results_dir"]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
   
    if not os.path.exists(params["1st_stage_weights_fn"]):
        print "The weights for the first stage does not exist!"
        sys.exit(2)
    w_1st = np.genfromtxt(params["1st_stage_weights_fn"], delimiter=",").astype(np.float32)
    
    if not os.path.exists(params["sizes_indeces_fn"]):
        print "The sizes indices file does not exist!"
        sys.exit(2)
    sizes = np.genfromtxt(params["sizes_indeces_fn"], delimiter=",").astype(np.int32)
    
    if not os.path.exists(params["2nd_stage_weights_fn"]):
        print "The weights for the second stage does not exist!"
        sys.exit(2)
    f = open(params["2nd_stage_weights_fn"])
    w_str = f.read()
    f.close()
    w_2nd = json.loads(w_str)
    
    ds = Dataset(basepath = basepath, training_set_fn = training_set_fn, test_set_fn = test_set_fn, annotations_path = annotations_path, images_path = images_path)
       
    test_annotations = ds.load_annotations( mode = Dataset.TEST )
    print "Evaluating recall..."
    eval_recall = EvaluateRecall(w_1st = w_1st, 
                                 sizes_idx = sizes, 
                                 w_2nd = w_2nd, 
                                 num_bbs_per_size_1st_stage = params["num_win_psz"], 
                                 num_bbs_final = params["num_bbs"])
    recall = eval_recall.evaluate_test_set(test_annotations)
    print "Recall obetained with {0} windows per size index and {1} total final windows: {2:.5f}".format(params["num_win_psz"],params["num_bbs"],recall)
