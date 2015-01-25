'''
Created on Jan 1, 2015

@author: alessandro
'''
import os
import xmltodict

class Dataset(object):

    TRAINING = 0
    TEST = 1

    def __init__(self, basepath, training_set_fn, test_set_fn, annotations_path, images_path):

        self.basepath = basepath
        self.training_set_fn = training_set_fn
        self.test_set_fn = test_set_fn
        self.annotations_path = annotations_path
        self.images_path = images_path
        
        f = open(training_set_fn,"r")
        train_idxs = f.readlines()
        f.close()
        self.train_idxs = [item.replace("\n","").replace("\r","") for item in train_idxs]
        
        f = open(test_set_fn,"r")
        test_idxs = f.readlines()
        f.close()
        self.test_idxs = [item.replace("\n","").replace("\r","") for item in test_idxs]

    def parse_xml_annotations(self, idx):
        
        #if classes is None, control over the classes is disabled. Otherwise class existence is checked
        xml_fn = os.path.join(self.annotations_path,"%s.xml"%idx)
        if not os.path.exists(xml_fn):
            raise Exception("The specified xml %s file does not exist!"%xml_fn)
        f = open(xml_fn,"r")
        xml = f.read()
        f.close()
        d = xmltodict.parse(xml)
        objects_list = d["annotation"]["object"]
        if isinstance(objects_list,dict):
            objects_list = [objects_list] 
        annotation_dict = dict() 
        annotation_dict["basename"] = d["annotation"]["filename"]
        annotation_dict["path"] = self.images_path
        bbs = []
        for obj in objects_list:
            bb_dict = dict()
            xmin = int(obj["bndbox"]["xmin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymin = int(obj["bndbox"]["ymin"])
            ymax = int(obj["bndbox"]["ymax"])
            bb_dict["bb"] = (xmin,ymin,xmax,ymax)
            bb_dict["difficult"] = int(obj["difficult"])
            bb_dict["class"] = obj["name"]
            bbs.append(bb_dict)
        annotation_dict["bbs"] = bbs
        return annotation_dict
    
    def load_annotations(self, mode = TRAINING):
        if mode == self.TRAINING:
            idxs = self.train_idxs
        else:
            idxs = self.test_idxs
        if not os.path.exists(self.basepath):
            raise Exception("The specified annotations basepath %s does not exists!"%self.basepath)
        annotations_dict = dict()
        print "Loading annotations..."
        for idx in idxs:
            try:
                ann_dict = self.parse_xml_annotations(idx)
            except Exception as e:
                print e
                continue            
            self.set_image_annotations(annotations_dict,idx,ann_dict)
        if not annotations_dict:
            raise Exception("The bounding boxes dict is empty!")
        print "Annotations loaded."
        return annotations_dict
    
    def set_image_annotations(self, bbs_dict,idx, bbs):
        key = "%s"%idx
        bbs_dict[key] = bbs
        
    def get_image_annotations(self, bbs_dict,idx):
        key = "%s"%idx
        return bbs_dict[key]


 