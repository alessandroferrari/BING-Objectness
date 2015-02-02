'''
Created on Jan 2, 2015

@author: alessandro
'''
import os
import cv2
import sys
import json
import random
import warnings
import numpy as np
from threading import Lock
from dataset import Dataset
from multiprocessing import cpu_count
from test_recall import EvaluateRecall
from multiprocessing.dummy import Pool as ThreadsPool
from common import bounding_box_overlapping, bounding_box_overlap_on_ground_truths
from bing import get_features, EDGE, BASE_LOG, MIN_EDGE_LOG, MAX_EDGE_LOG, EDGE_LOG_RANGE, FirstStagePrediction, NUM_WIN_PSZ
from svm_container import fit_svm, shuffle_dataset, model_selection, CUSTOM_LIBLINEAR_WRAPPER, SKLEARN_LIBLINEAR_WRAPPER

NUM_NEGATIVE_BOUNDING_BOXES = 100
POSITIVE_BB = 1
NEGATIVE_BB = -1

class TrainBing(object):
   
    def __init__(self, base_log = BASE_LOG, min_edge_log = MIN_EDGE_LOG, max_edge_log = MAX_EDGE_LOG, edge_log_range = EDGE_LOG_RANGE, gradient_edge = EDGE, num_negatives_bbs_for_image = NUM_NEGATIVE_BOUNDING_BOXES, results_dir = None):
        
        self.base_log = base_log
        self.min_edge_log = min_edge_log
        self.max_edge_log = max_edge_log
        self.edge_log_range = edge_log_range
        self.gradient_edge = gradient_edge
        self.num_negatives_bbs_for_image = num_negatives_bbs_for_image
        self.features_labels_lock = Lock()
        self.num_cpus = cpu_count()
        self.results_dir = results_dir
        if not self.results_dir is None and not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)                                                               
    
    
    #get the index of the sampled bounding box size, useful for keeping track of the sizes with a counter
    def get_sampling_size_idx(self, log_w,log_h):
        
        tmp_w = log_w - self.min_edge_log
        tmp_h = log_h - self.min_edge_log
        #the reason of the +1 is not really clear to me, it was like that in the original version of the code.
        #return tmp_h * self.edge_log_range + tmp_w + 1
        return tmp_h * self.edge_log_range + tmp_w
    
    #starting from the bounding box drawed as ground truth, 
    #bounding boxes with edges lengths of the power of base_log are generated
    def ground_truth_bounding_box_sampling(self, bb):
        
        w_gt = bb[2] - bb[0] + 1
        h_gt = bb[3] - bb[1] + 1
        #determing the range of exponent to apply to base_log for performing bounding boxes sampling.
        log_w_gt = np.log(w_gt) / np.log(self.base_log)
        log_h_gt = np.log(h_gt) / np.log(self.base_log)
        lower_bound_w_log = max(self.min_edge_log, int(np.floor(log_w_gt - 0.5)))
        upper_bound_w_log = min(self.max_edge_log, int(np.floor(log_w_gt + 1.5)))
        lower_bound_h_log = max(self.min_edge_log, int(np.floor(log_h_gt - 0.5)))
        upper_bound_h_log = min(self.max_edge_log, int(np.floor(log_h_gt + 1.5)))
        #among the exponents in the range, keep only those that leads to a bounding box with 
        #at least 50% overlapping with the original one.        
        sampled_bbs = []
        sizes_record = []
        for w_log_iter in xrange(lower_bound_w_log, upper_bound_w_log + 1):
            for h_log_iter in xrange(lower_bound_h_log, upper_bound_h_log + 1):
                w = self.base_log ** w_log_iter - 1
                h = self.base_log ** h_log_iter - 1
                sampled_bb = (bb[0], bb[1], bb[0] + w, bb[1] + h)
                #if the sampled bounding box shares enough with the original one, add it
                overlapping = bounding_box_overlapping(sampled_bb, bb)
                if overlapping>=0.5:
                    sampled_bbs.append(sampled_bb)
                    #useful for keeping track of which bounding boxes sizes as ground truth candidates
                    sizes_record.append(self.get_sampling_size_idx(w_log_iter,h_log_iter))
        return sampled_bbs, sizes_record   
    
    
    def extract_features_per_image(self, ann_dict):
        """This function extract features of correct and wrong bounding boxes for an image.
        """
        fn = os.path.join(ann_dict["path"],ann_dict["basename"])
        img = cv2.imread(fn)
        if img is None:
            warnings.warn("The image %s does not exist in the filesystem."%fn)
        img_h, img_w, nch = img.shape
        
        #calculating features for each ground truth bounding box
        bbs = ann_dict["bbs"]
        for k,obj in enumerate(bbs):
            bb = obj["bb"]
            ann_dict["pos_bbs"], img_sizes_records = self.ground_truth_bounding_box_sampling(bb)
            for idx, sampled_bb in enumerate(ann_dict["pos_bbs"]):
                grad = get_features(img, bb = sampled_bb, w = self.gradient_edge, h = self.gradient_edge)
                feat = np.reshape(grad,(1,self.gradient_edge**2))
                flipped_grad = cv2.flip(grad,1) #1 to flip horizontally
                flipped_feat = np.reshape(flipped_grad,(1,self.gradient_edge**2))
                #shared memory among different threads, synchonization needed for avoiding race conditions                
                with self.features_labels_lock:
                    self.features.append(feat.astype(float))
                    self.features.append(flipped_feat.astype(float))
                    self.labels.append(POSITIVE_BB)
                    self.labels.append(POSITIVE_BB)
                size_idx = img_sizes_records[idx]
                self.pos_sizes_records.append(size_idx)
        
        #generating negative samples by randomly extracting bounding boxes, that do not overlap enough with 
        #the ground truths ones.       
        neg_counter = 0
        counter = 0
        ann_dict["neg_bbs"] = []
        max_iter = 100
        while neg_counter<self.num_negatives_bbs_for_image:
            counter = counter + 1
            if counter>max_iter:
                break
            x1 = random.randint(1,img_w)
            y1 = random.randint(1,img_h)
            x2 = random.randint(1,img_w)
            y2 = random.randint(1,img_h)
            neg_bb_candidate = (min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2))
            if bounding_box_overlap_on_ground_truths(ground_truth_bbs = bbs, bb = neg_bb_candidate)<0.5:
                ann_dict["neg_bbs"].append(neg_bb_candidate)
                grad = get_features(img, bb = neg_bb_candidate, w = self.gradient_edge, h = self.gradient_edge)
                feat = np.reshape(grad,(1,self.gradient_edge**2))
                #shared memory among different threads, synchonization needed for avoiding race condition
                with self.features_labels_lock:
                    self.features.append(feat)
                    self.labels.append(NEGATIVE_BB)
                    neg_counter = neg_counter + 1
    
    
    def build_dataset(self, annotations, num_negatives_bbs_for_image = NUM_NEGATIVE_BOUNDING_BOXES, gradient_edge = EDGE, sizes_fn = None ):
        
        self.pos_sizes_records = []
        self.features = []
        self.labels = []
        self.annotations = annotations
        
        #too many threads will not help, because there are many accesses on disks.
        num_threads = min(self.num_cpus-1, 4)
        
        print "Starting features extraction on multiple threads. Number of threads: %s."%num_threads
        #threads pool (multiprocessing.dummy), so shared memory among them        
        pool = ThreadsPool(num_threads)
        pool.map(self.extract_features_per_image, annotations.values())
        
        #determing counter of training samples for each size 
        num_sizes = self.edge_log_range * self.edge_log_range + 1
        arr_sizes = np.array(self.pos_sizes_records, dtype=int)
        sizes_count, bin_edges = np.histogram(arr_sizes, bins = num_sizes, range = (0, num_sizes))
        #if the number of instances for a certain size is less than 50, just do not consider it.        
        active_sizes = sizes_count > 50
        idxs = np.arange(num_sizes, dtype = np.int32)
        #define the indeces of the considered sizes
        self.scale_space_sizes = idxs[active_sizes]
        
        #pack features and labels array as np.float32 numpy arrays
        self.features_array = np.vstack(tuple(self.features))
        if self.features_array.dtype != np.float32:
            self.features_array = self.features_array.astype(np.float32)
        self.labels_array = np.array(self.labels, dtype=np.float32)
        
        print "Features extraction is over! %s  samples extracted."%self.features_array.shape[0]
        
        if not self.results_dir is None:
            if not os.path.exists(self.results_dir):
                raise Exception("The destination path %s suggested to save datasets does not exist!"%self.results_dir)
            print "Saving extracted features to %s."%self.results_dir
            np.savetxt(os.path.join(self.results_dir,"features.txt"),self.features_array,fmt='%d', delimiter=',', newline='\n')
            np.savetxt(os.path.join(self.results_dir,"labels.txt"),self.labels_array,fmt='%d', delimiter=',', newline='\n')
            
        if not sizes_fn is None:
            basedir = os.path.dirname(sizes_fn)
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            np.savetxt(sizes_fn,self.scale_space_sizes,fmt='%d', delimiter=',', newline='\n')
        
        return self.features_array.astype(np.float32), self.labels_array.astype(np.float32), self.scale_space_sizes
    
    
    def split_positive_and_negative_features(self,X,y):
        """Split the training set in positive and negative samples.
	    """
        pos_mask = y == POSITIVE_BB
        neg_mask = y == NEGATIVE_BB
        
        X_pos = X[pos_mask,:]        
        X_neg = X[neg_mask,:]
        
        return X_pos, X_neg
    
    def pack_positive_negative_features(self, Xp, Xn):
        
        len_p = Xp.shape[0]
        len_n = Xn.shape[0]
        X = np.vstack((Xp,Xn))
        y = np.zeros(len_p+len_n)
        y[:len_p] = 1
        
        return X, y
        
    def training_set_average(self, X, y, repr_edge = 400):
        """Perform an averaging of all the training set, thw whole, the positives and the negatives.
    	The averaging is normalized so that it has minimum 0 and maximum 255, and it is quantized to np.uint8, for being 
    	visualized as image.
    	A visualization of the dataset is saved if the destination results directory is specified in the constructor.
    	"""        
        X_pos, X_neg = self.split_positive_and_negative_features(X, y)
        
        avg_pos = np.average(X_pos, axis=0)
        avg_pos = np.reshape(avg_pos,(self.gradient_edge,self.gradient_edge))
        avg_neg = np.average(X_neg, axis=0)
        avg_neg = np.reshape(avg_neg,(self.gradient_edge,self.gradient_edge))
        
        avg = np.zeros((self.gradient_edge,2*self.gradient_edge))
        avg[:,:self.gradient_edge] = avg_pos
        avg[:,self.gradient_edge:] = avg_neg
        
        min_avg = np.min(avg)
        max_avg = np.max(avg)
        avg = ((avg - min_avg )/( max_avg - min_avg ))*255
        avg = cv2.resize(avg, (2*repr_edge, repr_edge), interpolation = cv2.INTER_NEAREST)
        avg = avg.astype(np.uint8)
        
        min_avg_pos = np.min(avg_pos)
        max_avg_pos = np.max(avg_pos)
        avg_pos = (( avg_pos - min_avg_pos )/( max_avg_pos - min_avg_pos )) * 255
        avg_pos = cv2.resize(avg_pos, (repr_edge,repr_edge), interpolation = cv2.INTER_NEAREST)
        avg_pos = avg_pos.astype(np.uint8)
        
        min_avg_neg = np.min(avg_neg)
        max_avg_neg = np.max(avg_neg)
        avg_neg = (( avg_neg - min_avg_neg )/( max_avg_neg - min_avg_neg )) * 255
        avg_neg = cv2.resize(avg_neg, (repr_edge,repr_edge), interpolation = cv2.INTER_NEAREST)
        avg_neg = avg_neg.astype(np.uint8)
        
        if not self.results_dir is None:
            if not os.path.exists(self.results_dir):
                raise Exception("The destination path %s suggested to save the averaged dataset does not exist!"%self.results_dir)
            cv2.imwrite(os.path.join(self.results_dir, "avg_pos.png"),  avg_pos)
            cv2.imwrite(os.path.join(self.results_dir, "avg_neg.png"), avg_neg)
            cv2.imwrite(os.path.join(self.results_dir, "avg.png"), avg)
    
        return avg, avg_pos, avg_neg
       
    
    def reduce_dataset(self, X, y, nr):
        """
        This function reduces to nr the number of samples. It mantains all the positive instances of the training set, while
        discarding negative ones.
        """
        old_nr = X.shape[0]
        
        if old_nr <= nr:
            return X, y
 
        #shuffling is important for avoiding biases in the dataset
        X, y = shuffle_dataset(X,y)
    
        #trick for keeping all the positive instances
        X_pos, X_neg = self.split_positive_and_negative_features(X, y)   
        nr_pos = X_pos.shape[0]
        new_nr_neg = nr - nr_pos
        new_X_neg = X_neg[:new_nr_neg,:]
        new_X = np.vstack((X_pos, new_X_neg))
        new_y = np.zeros(nr,dtype=np.float32)
        new_y[:nr_pos] = POSITIVE_BB
        new_y[nr_pos:] = NEGATIVE_BB
        
        return new_X, new_y        
        
    
    def first_stage_training(self, X, y, C_list = None, wrapper_type = CUSTOM_LIBLINEAR_WRAPPER, repr_edge = 400, weights_fn = None):
        
        print "First stage training started..."
        
        weights, bias = model_selection(X,y, C_list = C_list, wrapper_type = wrapper_type)
        self.first_stage_weights = weights
        
        if self.first_stage_weights.dtype != np.float32:
            self.first_stage_weights = self.first_stage_weights.astype(np.float32)
        
        if not weights_fn is None:
            basedir = os.path.dirname(weights_fn)
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            np.savetxt(weights_fn, self.first_stage_weights, fmt='%10.5f', delimiter=',', newline='\n')
        
        #if a results directory specified, a image representation of the weights is saved. For visualizations purposes
        #the range is normalized within 0 and 255, and the values are quantized to uint8.
        if not self.results_dir is None:
            if not os.path.exists(self.results_dir):
                raise Exception("The destination path %s suggested to save the first stage learning weights does not exist!"%self.results_dir)
                        
            repr_weights = np.reshape(self.first_stage_weights,(self.gradient_edge, self.gradient_edge))
            repr_weights = cv2.resize(repr_weights, (repr_edge,repr_edge), interpolation = cv2.INTER_NEAREST)
            repr_min = np.min(repr_weights)
            repr_max = np.max(repr_weights)
            repr_weights = ( repr_weights - repr_min ) * 255.0  / ( repr_max - repr_min )
            repr_weights = repr_weights.astype(np.uint8)
            cv2.imwrite(os.path.join(self.results_dir,"weights.png"),repr_weights)
        
        print "First stage_training finished."
        
        return weights, bias
    
    def second_stage_training(self, wrapper_type = CUSTOM_LIBLINEAR_WRAPPER, weights_fn = None):
        
        print "Starting second stage training."
        fstp = FirstStagePrediction(self.first_stage_weights, self.scale_space_sizes, edge = self.gradient_edge, base_log = self.base_log, min_edge_log = self.min_edge_log, edge_log_range = self.edge_log_range, num_win_psz = NUM_WIN_PSZ)
        
        SCORE = 0
        LABEL = 1
        sizes_dict = dict()
        for sz in self.scale_space_sizes:
            #each key of sizes_dict correspond to a value that is a score, label tuple
            sizes_dict["%s"%sz] = (list(),list())
        
        #first stage detection are performed on all the training set, then are marked as correct or
        #not, according to their overlapping with the ground-truth bounding boxes.
        #for each size, 1D features array and labels array are prepared.
        #second stage prediction will be 2nd_stage_score = w1_s * 1st_stage_score + w0_s
        #w1_s and w2_s are size dependent. 
        print "Building second stage training training set..."
        for ann_key in self.annotations.keys():
            ann_dict = self.annotations[ann_key]
            fn = os.path.join(ann_dict["path"],ann_dict["basename"])
            img = cv2.imread(fn)
            if img is None:
                warnings.warn("The image %s does not exist in the filesystem."%fn)
            img_h, img_w, nch = img.shape
            ground_truth_bbs_objs = ann_dict["bbs"]
            predictions = fstp.predict(img, nss = 2)
            for pred_bbs, score, size in predictions:
                y = 1 if bounding_box_overlap_on_ground_truths(ground_truth_bbs_objs, pred_bbs) > 0.5 else -1
                sizes_dict["%s"%size][SCORE].append(score)
                sizes_dict["%s"%size][LABEL].append(y)
        
        #in this loop, the training is performed for each size, w1_s and w0_s are determined.
        print "Learning size-specific coefficients."
        to_delete = []      
        for size_key in sorted(sizes_dict.keys(), key=lambda x:int(x)):
            item = sizes_dict[size_key]
            sizes_dict[size_key] = dict()
            #features array is 1-D, the only feature is the first-stage bounding-box detection score
            train = np.reshape(np.array(item[SCORE]),(-1,1))
            labels = np.array(item[LABEL])
            #really weird thing, liblinear takes as positive label the first labels it encounters
            Xp, Xn = self.split_positive_and_negative_features(train, labels)
            train, labels = self.pack_positive_negative_features(Xp, Xn)
            num_pos = np.sum((labels==1).astype(int))
            print "Size %s: number training samples %s, number positive samples %s."%(size_key, train.shape[0], num_pos)
            if train.shape[0]==0:
                to_delete.append(size_key)
                print "Training set for size %s is missing. Skip this size." % size_key
                continue
            weight, bias = fit_svm(train, labels, C = 100, wrapper_type = wrapper_type)
            #weight is 1 element array!
            sizes_dict[size_key]["weight"] = float(weight[0])
            sizes_dict[size_key]["bias"] = float(bias)
        
        #remove the item that do not have enough candidates for a training.
        for key in to_delete:
            del sizes_dict[key]
    
        if not weights_fn is None:
            basedir = os.path.dirname(weights_fn)
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            if weights_fn[-5:]!=".json":
                warnings.warn("The filename for saving second stage weights does not have a json extension!")
            print "Saving second stage learning coefficients to file."
            sizes_json = json.dumps(sizes_dict, sort_keys=True, indent=4, separators=(',', ': '))
            f = open(weights_fn,"w")
            f.write(sizes_json)
            f.close()
    
        print "Storing second stage learning coefficients in numpy array."
        num_sizes = len(sizes_dict.keys())
        coeffs = np.zeros((num_sizes,2))
        for i, size_key in enumerate(sizes_dict.keys()):
            sz_dict = sizes_dict[size_key]
            coeffs[i,0] = sz_dict["weight"]
            coeffs[i,1] = sz_dict["bias"]
           
        self.coeffs = coeffs
        
        return sizes_dict, coeffs

def parse_cmdline_inputs():
    """
    Example parameters:
    
    {
        "basepath": "/opt/Datasets/VOC2007",
        "training_set_fn": "/opt/Datasets/VOC2007/ImageSets/Main/train.txt",
        "test_set_fn": "/opt/Datasets/VOC2007/ImageSets/Main/test.txt",
        "annotations_path": "/opt/Datasets/VOC2007/Annotations",
        "images_path": "/opt/Datasets/VOC2007/JPEGImages",
        "results_dir": "/opt/Datasets/VOC2007/BING_Results"
        "1st_stage_weights_fn":"/opt/Datasets/VOC2007/BING_Results/weights.txt",
        "2nd_stage_weights_fn": "/opt/Datasets/VOC2007/BING_Results/2nd_stage_weights.json",
        "sizes_indeces_fn": "/opt/Datasets/VOC2007/BING_Results/sizes.txt",
        "num_win_psz": 130
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
    
    ds = Dataset(basepath = basepath, training_set_fn = training_set_fn, test_set_fn = test_set_fn, annotations_path = annotations_path, images_path = images_path)
    annotations = ds.load_annotations( mode = Dataset.TRAINING )
    tb = TrainBing(results_dir = results_dir, num_negatives_bbs_for_image = 50)
    X,y, sizes=tb.build_dataset(annotations, sizes_fn = params["sizes_indeces_fn"])
    X, y = tb.reduce_dataset(X, y, nr = 100000)
    weights, bias = tb.first_stage_training(X,y,C_list=[10],wrapper_type = SKLEARN_LIBLINEAR_WRAPPER, weights_fn = params["1st_stage_weights_fn"])
    tb.training_set_average(X, y)
    w_2nd_dict, coeffs = tb.second_stage_training(wrapper_type = SKLEARN_LIBLINEAR_WRAPPER, weights_fn = params["2nd_stage_weights_fn"])
    
    test_annotations = ds.load_annotations( mode = Dataset.TEST )
    eval_recall = EvaluateRecall(w_1st = weights, sizes_idx = sizes, 
                                 w_2nd = w_2nd_dict, 
                                 num_bbs_per_size_1st_stage = params["num_win_psz"], 
                                 num_bbs_final = params["num_bbs"])
    recall = eval_recall.evaluate_test_set(test_annotations)
    print "Recall obetained with {0} windows per size index and {1} total final windows: {2:.5f}".format(params["num_win_psz"],params["num_bbs"],recall)
    
    