from lib import pascal_voc_io
import numpy as np
import os.path as osp
import cv2

dataDir='/home/zehao/Dataset/VOCdevkit/VOC0712'
dataType='trainval'
imgList='/home/zehao/Dataset/VOCdevkit/VOC0712/ImageSets/Main/trainval.txt'
annoDir=osp.join(dataDir, 'Annotations')
imgDir=osp.join(dataDir, 'JPEGImages')

annoes = []
with open(imgList, 'r') as f:
    files = f.readlines()
    for file in files:
        file = file.strip()
        print file
        annofile = osp.join(annoDir, file+'.xml')
        anno_reader = pascal_voc_io.PascalVocReader(annofile)
        anno_reader.parseXML()
        for bbox in anno_reader.shapes:
            annoes.append(bbox)

bboxes = np.zeros((len(annoes), 2), np.float32)
i = 0
for anno in annoes:
    points = anno[1]
    x_min = points[0][0]
    x_max = points[2][0]
    y_min = points[0][1]
    y_max = points[1][1]
    b_w = x_max - x_min
    b_h = y_max - y_min
    bboxes[i, :] = np.array([b_w, b_h])

np.save('voc0712_bboxes_wh', bboxes)




