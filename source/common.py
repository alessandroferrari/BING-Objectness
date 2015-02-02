'''
Created on Jan 31, 2015

@author: alessandro
'''
#determine the overlapping among two bounding boxes
def bounding_box_overlapping(bb1, bb2):
        
        x0_1, y0_1, x1_1, y1_1 = bb1
        w_1 = x1_1 - x0_1 + 1
        h_1 = y1_1 - y0_1 + 1
        x0_2, y0_2, x1_2, y1_2 = bb2
        w_2 = x1_2 - x0_2 + 1
        h_2 = y1_2 - y0_2 + 1
        x0 = max(x0_1,x0_2)
        x1 = min(x1_1,x1_2)
        y0 = max(y0_1,y0_2)
        y1 = min(y1_1,y1_2)
        xover = x1-x0+1
        yover = y1-y0+1
        if xover<=0 or yover<=0:
            return .0
        over = float(xover*yover)
        not_over = w_1*h_1 + w_2 * h_2 - over
        ratio = over / not_over
        return ratio
    
#determine the overlapping among a query bounding box (bb) and the ground truth 
#bounding boxes of an image   
def bounding_box_overlap_on_ground_truths(ground_truth_bbs, bb):
    
    max_overlap = 0.0
    for obj in ground_truth_bbs:
        gt_bb = obj["bb"]
        over = bounding_box_overlapping(bb, gt_bb)
        if over > max_overlap:
            max_overlap = over
    return max_overlap

#more generic compared to bounding_box_overlap_on_ground_truths that follows 
#the specific structure of annotations
def bounding_box_overlap(bbs, bb_query):
    
    max_overlap = 0.0
    for bb in bbs:
        over = bounding_box_overlapping(bb_query, bb)
        if over > max_overlap:
            max_overlap = over
    return max_overlap