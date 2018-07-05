from __future__ import division
import os, time, pickle
import numpy as np
import torch
import torch.nn as nn
from math import ceil, floor
from pycocotools.coco import COCO
from PIL import Image
from sklearn.metrics import average_precision_score
import logging
import gc
from scipy.stats import rankdata
import xarray as xr
from multiprocessing.dummy import Pool as ThreadPool
import time



all_cats = ['bear', 'bird', 'cat', 'cow', 'dog', 'elephant','giraffe',
            'horse','sheep','zebra','airplane','bicycle','boat','bus',
            'car','motorcycle','train','truck', 'background']




# IoU
def iou(rect1, rect2): # rect = [x, y, w, h]
    x1, y1, w1, h1 = rect1
    X1, Y1 = x1+w1, y1 + h1
    x2, y2, w2, h2 = rect2
    X2, Y2 = x2+w2, y2 + h2
    a1 = (X1 - x1 + 1) * (Y1 - y1 + 1)
    a2 = (X2 - x2 + 1) * (Y2 - y2 + 1)
    x_int = max(x1, x2)
    X_int = min(X1, X2)
    y_int = max(y1, y2)
    Y_int = min(Y1, Y2)
    a_int = (X_int - x_int + 1) * (Y_int - y_int + 1) * 1.0
    if x_int > X_int or y_int > Y_int:
        a_int = 0.0
    return a_int / (a1 + a2 - a_int)


# nearest neighbor in 1-based indexing
def _nnb_1(x):
    x1 = int(floor((x + 8) / 16.0))
    x1 = max(1, min(x1, 13))
    return x1


def project_onto_feature_space(rect, image_dims):
    # project bounding box onto conv net
    # @param rect: (x, y, w, h)
    # @param image_dims: (imgx, imgy), the size of the image
    # output bbox: (x, y, x'+1, y'+1) where the box is x:x', y:y'

    # For conv 5, center of receptive field of i is i_0 = 16 i for 1-based indexing
    imgx, imgy = image_dims
    x, y, w, h = rect
    # scale to 224 x 224, standard input size.
    x1, y1 = ceil((x + w) * 224 / imgx), ceil((y + h) * 224 / imgy)
    x, y = floor(x * 224 / imgx), floor(y * 224 / imgy)
    px = _nnb_1(x + 1) - 1 # inclusive
    py = _nnb_1(y + 1) - 1 # inclusive
    px1 = _nnb_1(x1 + 1) # exclusive
    py1 = _nnb_1(y1 + 1) # exclusive

    return [px, py, px1, py1]


class Featurizer:
    dim = 11776  # for small features
    # dim = 1472

    def __init__(self):
        # pyramidal pooling of sizes 1, 3, 6
        self.pool1 = nn.AdaptiveMaxPool2d(1)
        self.pool3 = nn.AdaptiveMaxPool2d(3)
        self.pool6 = nn.AdaptiveMaxPool2d(6)
        self.lst = [self.pool1, self.pool3, self.pool6]

    def featurize(self, projected_bbox, image_features):
        # projected_bbox: bbox projected onto final layer
        # image_features: C x W x H tensor : output of conv net
        full_image_features = torch.from_numpy(image_features)
        x, y, x1, y1 = projected_bbox
        crop = full_image_features[:, x:x1, y:y1]
        #         return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),
        #                           self.pool6(crop).view(-1)], dim=0) # returns torch Variable
        return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),
                          self.pool6(crop).view(-1)], dim=0).data.numpy()  # returns numpy array


class LSH:
    def __init__(self, dataDir, dataType, ann_file, feats):
        self.dataDir = dataDir
        self.dataType = dataType
        self.ann_file = ann_file
        self.load_from_coco(dataDir, dataType, ann_file, feats)

    def load_from_coco(self, dataDir, dataType, ann_file, feats):
        self.coco = COCO(ann_file)
        self.cats = self.coco.loadCats(self.coco.getCatIds()) # categories
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.cats}  # category id to name mapping
        #[self.img_list, self.feats] = pickle.load(open(os.path.join(dataDir, 'features_small', '{}.p'.format(dataType)),'rb'),
        #                                 encoding='latin1')
        self.feats = feats
        [self.img_list, self.pre_bboxes] = pickle.load(open(os.path.join(dataDir, 'bboxes_retrieval', '{}_bboxes_retrieval.p'.format(dataType)),'rb'),
                                            encoding='latin1')
        print(len(self.pre_bboxes))
        self.img_size = len(self.img_list)

    def generate_random_vector(self, k, dim):
        return np.random.randn(dim, k) / np.sqrt(dim)

    def hash_function(self, u, p):
        return np.floor(p.dot(u) / (2 * self.R))

    def construct(self, L, k, R):
        self.R = R
        self.L = L
        np.random.seed(0)
        self.hash_matrixes = []
        for i in range(L):
            self.hash_matrixes.append(self.generate_random_vector(k, Featurizer.dim))

        t1 = time.time()
        self.hash_tables = []
        for i in range(L):
            self.hash_tables.append({})

        for i in range(self.img_size):
            print("image %d out of %d" % (i, self.img_size))
            img_id = self.img_list[i]
            img = self.coco.loadImgs([img_id])[0]
            img_pil = Image.open('%s/%s/%s' % (self.dataDir, self.dataType, img['file_name']))

            # annIds = self.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            # anns = self.coco.loadAnns(annIds)

            bboxes = self.pre_bboxes[i]
            if bboxes is None:  ### OpenCV has thrown an error. Discard image.
                print('discard image from consideration.')
                continue

            img_feats = self.feats[i]
            featurizer = Featurizer()
            train_dt = []

            for j in range(len(bboxes)):
                projected_bbox = project_onto_feature_space(bboxes[j], img_pil.size)
                bbox_feats = featurizer.featurize(projected_bbox, img_feats)
                train_dt.append(bbox_feats)

            train_dt = np.array(train_dt)

            for j in range(len(self.hash_matrixes)):
                m = self.hash_matrixes[j]
                table = self.hash_tables[j]
                res = self.hash_function(m, train_dt)
                for patch_idx, value in enumerate(res):
                    value_temp = tuple(value)
                    if value_temp not in table:
                        table[value_temp] = []
                    # i is the img index and data_idx is the patch index of that image
                    table[value_temp].append((i, patch_idx))


        print("construct time: %f" % (time.time() - t1))
        # return {'matrix': self.hash_matrixes, 'hash_table': self.hash_tables}

    def euclidean_distance(self, x, y):
        x = x.reshape([-1,1])
        y = y.reshape([-1,1])
        return np.sqrt(np.einsum('ij,ij->j', x-y, x-y))[0]

    def query(self, bbox_feat, num_neighbor, c, subsample=1.0):
        self.c = c
        candidate_set = set()
        num_dist_compute = 0
        img_set = set()
        
        distance = []
        for i in range(len(self.hash_matrixes)):
            m = self.hash_matrixes[i]
            table = self.hash_tables[i]
            res = self.hash_function(m, bbox_feat)
            value = tuple(res)
            if value in table:
                candidates = table[value]
                for candidate in candidates:
                    
                    #subsample 50% to brute force search
                    if np.random.uniform(size=1) > subsample:
                        continue
                    
                    img_idx = candidate[0]
                    if img_idx in img_set:
                        continue
                    else:
                        img_set.add(img_idx)
                    patch_idx = candidate[1]
                    img_id = self.img_list[img_idx]
                    img = self.coco.loadImgs([img_id])[0]
                    img_pil = Image.open('%s/%s/%s' % (self.dataDir, self.dataType, img['file_name']))

                    annIds = self.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
                    anns = self.coco.loadAnns(annIds)

                    # get the candidate image patch
                    train_bbox = self.pre_bboxes[img_idx][patch_idx]

                    img_feats = self.feats[img_idx]
                    featurizer = Featurizer()
                    projected_bbox = project_onto_feature_space(train_bbox, img_pil.size)
                    train_bbox_feats = featurizer.featurize(projected_bbox, img_feats)

                    distance += [self.euclidean_distance(train_bbox_feats, bbox_feat)]
                    num_dist_compute += 1
                    
                    
                    if distance[-1] > self.c * self.R:
                        distance.pop()
                        continue

                    cat_to_true_bboxes = {}
                    for ann in anns:
                        cat_name = self.cat_id_to_name[ann['category_id']]
                        if cat_name not in all_cats:
                            continue
                        if cat_name not in cat_to_true_bboxes:
                            cat_to_true_bboxes[cat_name] = []
                        cat_to_true_bboxes[cat_name] += [ann['bbox']]

                    label = 'background'
                    for cat in cat_to_true_bboxes:
                        true_bboxes = cat_to_true_bboxes[cat]
                        max_iou = 0
                        for true_rect in true_bboxes:
                            cur_iou = iou(true_rect, train_bbox)
                            if cur_iou > max_iou:
                                max_iou = cur_iou
                        if max_iou > 0.5:
                            label = cat
                            break
                    
                    # append the distance and the category
                    candidate_list = list(candidate)
                    candidate_list.extend([distance[-1], label])
                    candidate_set.add(tuple(candidate_list))
                    
                    
                    if len(candidate_set) == 5*num_neighbor:
                        break
        
        if distance == []:
            #print('Nothing in the buckets! Try larger R!')
            return candidate_set, num_dist_compute            
        
        
        #brute force search knn in candidate set
        distance = np.array(distance)
        knn_dis_thres = distance[np.argwhere(rankdata(distance,method='ordinal')<=num_neighbor)].max()
        
        candidate_set = set([_ for _ in candidate_set if _[2] <= knn_dis_thres])
                    
        return candidate_set, num_dist_compute

    def query_one_train(self, num_neighbor, idx, c, subsample=1.0):
        img_id = self.img_list[idx]
        img = self.coco.loadImgs([img_id])[0]
        img_pil = Image.open('%s/%s/%s' % (self.dataDir, self.dataType, img['file_name']))

        bboxes = self.pre_bboxes[idx]
        n = len(bboxes)
        print(n)
        j = 0
        img_feats = self.feats[idx]
        featurizer = Featurizer()

        projected_bbox = project_onto_feature_space(bboxes[j], img_pil.size)
        bbox_feats = featurizer.featurize(projected_bbox, img_feats)

        annIds = self.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        cat_to_true_bboxes = {}
        for ann in anns:
            cat_name = self.cat_id_to_name[ann['category_id']]
            if cat_name not in all_cats:
                print(cat_name)
                continue
            if cat_name not in cat_to_true_bboxes:
                cat_to_true_bboxes[cat_name] = []
            cat_to_true_bboxes[cat_name] += [ann['bbox']]

        true_label = 'background'
        for cat in cat_to_true_bboxes:
            true_bboxes = cat_to_true_bboxes[cat]
            max_iou = 0
            for true_rect in true_bboxes:
                cur_iou = iou(true_rect, bboxes[j])
                if cur_iou > max_iou:
                    max_iou = cur_iou
            if max_iou > 0.5:
                true_label = cat
                break
                # append the distance and the category
        return self.query(bbox_feats, num_neighbor, c, subsample), true_label

    def query_test(self, test_loader, num_neighbor, c, subsample=1.0):
        assert isinstance(test_loader, LSH)
        num_patch = 0
        avg_dist = 0
        avg_dist_compute = 0
        test_score = []
        test_lable = []
        for idx in range(test_loader.img_size):
            print("query image %d out of %d" % (idx, test_loader.img_size))
            img_id = test_loader.img_list[idx]
            img = test_loader.coco.loadImgs([img_id])[0]
            img_pil = Image.open('%s/%s/%s' % (test_loader.dataDir, test_loader.dataType, img['file_name']))
            annIds = test_loader.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = test_loader.coco.loadAnns(annIds)

            bboxes = test_loader.pre_bboxes[idx]
            if bboxes is None:  ### OpenCV has thrown an error. Discard image.
                print('discard image from consideration.')
                continue
            img_feats = test_loader.feats[idx]
            featurizer = Featurizer()
            cat_to_true_bboxes = {}
            for ann in anns:
                cat_name = test_loader.cat_id_to_name[ann['category_id']]
                if cat_name not in all_cats:
                    continue
                if cat_name not in cat_to_true_bboxes:
                    cat_to_true_bboxes[cat_name] = []
                cat_to_true_bboxes[cat_name] += [ann['bbox']]

            for j in range(len(bboxes)):
                num_patch += 1
                projected_bbox = project_onto_feature_space(bboxes[j], img_pil.size)
                bbox_feats = featurizer.featurize(projected_bbox, img_feats)
                true_label = 'background'
                for cat in cat_to_true_bboxes:
                    true_bboxes = cat_to_true_bboxes[cat]
                    max_iou = 0
                    for true_rect in true_bboxes:
                        cur_iou = iou(true_rect, bboxes[j])
                        if cur_iou > max_iou:
                            max_iou = cur_iou
                    if max_iou > 0.5:
                        true_label = cat
                        break
                k_near_neighbor, num_dist_compute = self.query(bbox_feats, num_neighbor, c, subsample)
                K = len(k_near_neighbor)
                if K == 0:
                    print("fail to find any neighbor")
                    continue
                # record some evaluation metric
                label = [0] * len(all_cats)
                label[all_cats.index(true_label)] = 1
                test_lable.append(label)

                dist = 0
                score = [0] * len(all_cats)

                for neighbor in k_near_neighbor:
                    dist += neighbor[2]
                    cat = neighbor[3]
                    score[all_cats.index(cat)] += 1
                score = list(map(lambda x: x / K, score))
                test_score.append(score)

                dist /= K
                avg_dist = avg_dist * (num_patch / (num_patch + 1)) + dist / (num_patch + 1)
                avg_dist_compute = avg_dist_compute * (num_patch / (num_patch + 1)) + num_dist_compute / (num_patch + 1)

        test_ap = average_precision_score(y_true=np.array(test_lable), y_score=np.array(test_score), average=None)
        mean_test_ap = np.mean(test_ap[:-1])
        
        return {'test_ap': test_ap, 'mAP': mean_test_ap,
                'average_distance': avg_dist, 'average_compute': avg_dist_compute}


# In[construct]:

dataDir = 'D:/uw/stat548/proj/data'
#dataDir = '/Users/fuyang/Desktop/data'
ann_file = '{}/annotations/instances_{}.json'.format(dataDir, 'train2014')
img_train_feat = xr.open_dataarray(dataDir+'/train_feat.nc').values


train_loader = LSH(dataDir, 'train2014', ann_file, img_train_feat)

print("start constructing")
train_loader.construct(L=10, k=30, R=100)

#demo
candidate,label = train_loader.query_one_train(num_neighbor=10, idx=2, c=3)
print(candidate, label)

del img_train_feat,ann_file
gc.collect()

'''
with open(dataDir+'/lsh_model.p', 'wb') as f:
   pickle.dump(train_loader, f)
'''

# In[query]:
'''
dataDir = 'D:/uw/stat548/proj/data'
#dataDir = '/Users/fuyang/Desktop/data'
with open(dataDir+'/lsh_model.p', 'rb') as f:
    train_loader = pickle.load(f)
'''    

logger = logging.getLogger(dataDir+'/LSH')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('LSH.log')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)



#val
ann_file = '{}/annotations/instances_{}.json'.format(dataDir, 'val2014')  # annotations
img_val_feat = xr.open_dataarray(dataDir+'/val_feat.nc').values
val_loader = LSH(dataDir, 'val2014', ann_file, img_val_feat)
del img_val_feat, ann_file
gc.collect()

res = train_loader.query_test(val_loader, num_neighbor=10, c=3, subsample=1.0)
logger.info(res)


#parallel version
def func(dataDir,train_loader,test_loader,num_neighbor,c=5,subsample=1.0):
    logger = logging.getLogger(dataDir+'/LSH')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('LSH.log')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    res = train_loader.query_test(test_loader, num_neighbor, c, subsample)
    logger.info(res)
    
    return res

'''
time_start=time.time()
pool = ThreadPool(4)
results = pool.map(lambda nn: func(dataDir, train_loader, val_loader, num_neighbor=nn, c=3, subsample=1.0), [10,20,30,40])
pool.close()
pool.join()
print('Time spent: ', time.time()-time_start, 's')
'''

#test
ann_file = '{}/annotations/instances_{}.json'.format(dataDir, 'test2014')  # annotations
img_test_feat = xr.open_dataarray(dataDir+'/test_feat.nc').values
test_loader = LSH(dataDir, 'test2014', ann_file, img_test_feat)
del img_test_feat, ann_file
gc.collect()

res = train_loader.query_test(test_loader, num_neighbor=10, c=3, subsample=1.0)
logger.info(res)

'''
time_start=time.time()
pool = ThreadPool(4)
results = pool.map(lambda nn: func(dataDir, train_loader, test_loader, num_neighbor=nn, subsample=1.0), [10,20,30,40])
pool.close()
pool.join()
print('Time spent: ', time.time()-time_start, 's')
'''