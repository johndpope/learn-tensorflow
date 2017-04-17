#from kmeans import TFKMeansCluster
import numpy as np
from sklearn.cluster import k_means

voc0712_bboxes_wh = np.load('voc0712_bboxes_wh.npy')
y_pred = k_means(voc0712_bboxes_wh, precompute_distances='elkan', n_clusters=9, random_state=9, n_jobs=10)
print y_pred
