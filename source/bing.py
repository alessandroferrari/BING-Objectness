'''
Created on Jan 2, 2015

@author: alessandro
'''
import os
import cv2
import sys
import json
import getopt
import random
import numpy as np
from filter_tig import FilterTIG

EDGE = 8
BASE_LOG = 2
MIN_EDGE_LOG = int(np.ceil(np.log(10.)/np.log(BASE_LOG)))
MAX_EDGE_LOG = int(np.ceil(np.log(500.)/np.log(BASE_LOG)))
EDGE_LOG_RANGE = MAX_EDGE_LOG - MIN_EDGE_LOG + 1
NUM_WIN_PSZ = 130

def magnitude(x,y):
    #return np.sqrt(np.square(x)+np.square(y))
    return x + y
    
def sobel_gradient(img, ksize):
    gray = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=ksize)
    y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=ksize)
    mag = magnitude(x,y)
    return mag

def sobel_gradient_8u(img, ksize):
    grad = sobel_gradient(img, ksize)
    grad[grad<0] = 0
    grad[grad>255] = 255
    return grad.astype(np.uint8)

def rgb_gradient(img):
    
    img = img.astype(float)
    
    h,w, nch = img.shape
    
    gradientX = np.zeros((h,w))
    gradientY = np.zeros((h,w))
    
    d1 = np.abs(img[:,1,:] - img[:,0,:])
    gradientX[:,0] = np.max( d1, axis = 1) * 2
    d2 = np.abs(img[:,-1,:] - img[:,-2,:])
    gradientX[:,-1] = np.max( d2 , axis = 1) * 2
    d3 = np.abs(img[:,2:w,:] - img[:,0:w-2,:])
    gradientX[:,1:w-1] = np.max( d3 , axis = 2 )
    
    d1 = np.abs(img[1,:,:] - img[0,:,:])
    gradientY[0,:] = np.max( d1, axis = 1) * 2
    d2 = np.abs(img[-1,:,:] - img[-2,:,:])
    gradientY[-1,:] = np.max( d2 , axis = 1) * 2
    d3 = np.abs(img[2:h,:,:] - img[0:h-2,:,:])
    gradientY[1:h-1,:] = np.max( d3 , axis = 2 )
    
    mag = magnitude(gradientX,gradientY)
    
    mag[mag<0] = 0
    mag[mag>255] = 255
    return mag.astype(np.uint8) 
    

def get_features(img,bb, w = EDGE,h = EDGE, ksize=3, idx = None):
    crop_img = img[bb[1]-1:bb[3], bb[0]-1:bb[2],:]
    if not idx is None:
        cv2.imwrite("/tmp/%s.png"%idx,crop_img)
    sub_img = cv2.resize(crop_img,(w,h))
    grad = rgb_gradient(sub_img)
    return grad


class FirstStagePrediction(object):
    
    def __init__(self, weights_1st_stage, scale_space_sizes_idxs, num_win_psz = 130, edge = EDGE, base_log = BASE_LOG, min_edge_log = MIN_EDGE_LOG, edge_log_range = EDGE_LOG_RANGE):
        
        self.filter_tig = FilterTIG()
        self.weights_1st_stage = weights_1st_stage
        self.filter_tig.update(self.weights_1st_stage)
        self.filter_tig.reconstruct(self.weights_1st_stage) 
        self.scale_space_sizes_idxs = scale_space_sizes_idxs
        self.base_log = base_log
        self.min_edge_log = min_edge_log   
        self.edge_log_range = edge_log_range 
        self.edge = edge
        self.num_win_psz = num_win_psz
        
    def predict(self, image, nss = 2):

        bbs = []
        img_h,img_w,nch = image.shape
        for size_idx in self.scale_space_sizes_idxs:
            w = round(pow(self.base_log, size_idx % self.edge_log_range +  self.min_edge_log))
            h = round(pow(self.base_log, size_idx // self.edge_log_range + self.min_edge_log))
            if (h > img_h * self.base_log) or (w > img_w * self.base_log):
                continue
            h = min(h, img_h)
            w = min(w, img_w)
            new_w = int(round(float(self.edge)*img_w/w))
            new_h = int(round(float(self.edge)*img_h/h))
            img_resized = cv2.resize(image,(new_w,new_h))
            grad = rgb_gradient(img_resized)
            match_map = self.filter_tig.match_template(grad)
            points = self.filter_tig.non_maxima_suppression(match_map, nss, self.num_win_psz, False)
            ratio_x = w / self.edge
            ratio_y = h / self.edge
            i_max = min(len(points), self.num_win_psz)
            for i in xrange(i_max):
                point, score = points[i]      
                x0 = int(round(point[0] * ratio_x))
                y0 = int(round(point[1] * ratio_y))
                x1 = min(img_w, int(x0+w))
                y1 = min(img_h, int(y0+h))
                x0 = x0 + 1
                y0 = y0 + 1
                bbs.append(((x0,y0,x1,y1), score, size_idx))
        return bbs

class SecondStagePrediction(object):
    
    def __init__(self, second_stage_weights):
        self.second_stage_weights = second_stage_weights
        
    def predict(self, bbs):
        normalized_bbs = []
        for bb, score, size_idx in bbs:
            try:
                weights = self.second_stage_weights["%s"%size_idx]
            except:
                #if a size_idx is missing, it means that training error for it was empty, so just skip it!
                continue
            #normalize the score with respect with the size
            normalized_score = weights["weight"] * score + weights["bias"] 
            normalized_bbs.append((normalized_score,bb))
        return normalized_bbs
            
class Bing(object):
    
    def __init__(self, weights_1st_stage, sizes_idx, weights_2nd_stage, num_bbs_per_size_1st_stage= NUM_WIN_PSZ, num_bbs_final = 1500, edge = EDGE, base_log = BASE_LOG, min_edge_log = MIN_EDGE_LOG, edge_log_range = EDGE_LOG_RANGE):
        
        self.first_stage_prediction = FirstStagePrediction(weights_1st_stage, sizes_idx, num_win_psz = num_bbs_per_size_1st_stage, edge = edge, base_log = base_log, min_edge_log = min_edge_log, edge_log_range = edge_log_range)
        self.second_stage_prediction = SecondStagePrediction(weights_2nd_stage)
        self.num_bbs_final = num_bbs_final
        
    def predict(self, image):		
    
        bbs_1st = self.first_stage_prediction.predict(image)
        bbs = self.second_stage_prediction.predict(bbs_1st)
        sorted_bbs = sorted(bbs, key = lambda x:x[0], reverse = True)
        results = [(bb[0],bb[1]) for bb in sorted_bbs[:self.num_bbs_final]]
        score_bbs, results_bbs = zip(*results)
        return results_bbs, score_bbs

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
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "num_bbs_per_size=",
                                               "num_bbs=" ])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        sys.exit(2)
    
    params_file = sys.argv[-2]
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
    
    for o, a in opts:
            
        if o == "--help" or o =="-h":
            print "python bing.py --num_bbs_per_size 130 --num_bbs 1500 /path/to/dataset/parameters.json /path/to/image.jpg"
            sys.exit(0)
        elif o == "--num_bbs_per_size":
            try:
                params["num_win_psz"] = int(a)
            except Exception as e:
                print "Error while converting parameter --num_bb_per_size %s to int. Exception: %s."%(a,e)
                sys.exit(2)
        elif o == "--num_bbs":
            try:
                params["num_bbs"] = int(a)
            except Exception as e:
                print "Error while converting parameter --num_bbs %s to int. Exception: %s."%(a,e)
                sys.exit(2)
        else:
            print "Invalid parameter %s. Type 'python bing -h' "%o
            sys.exit(2)
          
    if not params.has_key("num_bbs"):
        params["num_bbs"] = 1500
    if not params.has_key("num_win_psz"):
        params["num_win_psz"] = 130
                
    params["image_file"] = sys.argv[-1]
    if not os.path.exists(params["image_file"]):
        print "Specified file for image %s does not exist."%params["image_file"]
        sys.exit(2)
    image = cv2.imread(params["image_file"])
    
    return params, image

if __name__=="__main__":
    
    params, image = parse_cmdline_inputs()
    results_dir = params["results_dir"]
    if not os.path.exists(results_dir):
        print "The results directory that should contains weights and sizes indeces does not exist. Be sure to have already performed training. "
        sys.exit(2)
    
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
    
    b = Bing(w_1st,sizes,w_2nd, num_bbs_per_size_1st_stage= params["num_win_psz"], num_bbs_final = params["num_bbs"])
    bbs, scores = b.predict(image)
    
    for bb in bbs:
        cv2.rectangle(image,(bb[0],bb[1]),(bb[2],bb[3]),color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    
    cv2.imwrite(os.path.join(results_dir, "bounding_box_image.png"), image)
    
    f = open(os.path.join(results_dir,"bbs.csv"),"w")
    f.write("filename,xmin,ymin,xmax,ymax\n")
    for bb in bbs:
        f.write("%s,%s,%s,%s,%s\n"%(params["image_file"],bb[0],bb[1],bb[2],bb[3]))
    f.close()