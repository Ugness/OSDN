# -*- coding: utf-8 -*-

###################################################################################################
# Copyright (c) 2016 , Regents of the University of Colorado on behalf of the University          #
# of Colorado Colorado Springs.  All rights reserved.                                             #
#                                                                                                 #
# Redistribution and use in source and binary forms, with or without modification,                #
# are permitted provided that the following conditions are met:                                   #
#                                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this                  #
# list of conditions and the following disclaimer.                                                #
#                                                                                                 #
# 2. Redistributions in binary form must reproduce the above copyright notice, this list          #
# of conditions and the following disclaimer in the documentation and/or other materials          #
# provided with the distribution.                                                                 #
#                                                                                                 #
# 3. Neither the name of the copyright holder nor the names of its contributors may be            #
# used to endorse or promote products derived from this software without specific prior           #
# written permission.                                                                             #
#                                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY             #
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF         #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          #
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,            #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF     #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,           #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS           #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    #
#                                                                                                 #
# Author: Abhijit Bendale (abendale@vast.uccs.edu)                                                #
#                                                                                                 #
# If you use this code, please cite the following works                                           #
#                                                                                                 #
# A. Bendale, T. Boult “Towards Open Set Deep Networks” IEEE Conference on                        #
# Computer Vision and Pattern Recognition (CVPR), 2016                                            #
#                                                                                                 #
# Notice Related to using LibMR.                                                                  #
#                                                                                                 #
# If you use Meta-Recognition Library (LibMR), please note that there is a                        #
# difference license structure for it. The citation for using Meta-Recongition                    #
# library (LibMR) is as follows:                                                                  #
#                                                                                                 #
# Meta-Recognition: The Theory and Practice of Recognition Score Analysis                         #
# Walter J. Scheirer, Anderson Rocha, Ross J. Micheals, and Terrance E. Boult                     #
# IEEE T.PAMI, V. 33, Issue 8, August 2011, pages 1689 - 1695                                     #
#                                                                                                 #
# Meta recognition library is provided with this code for ease of use. However, the actual        #
# link to download latest version of LibMR code is: http://www.metarecognition.com/libmr-license/ #
###################################################################################################


import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
from scipy.io import loadmat
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from openmax_utils import *
from evt_fitting import weibull_tailfitting, query_weibull
from compute_openmax import *

import libmr

NCHANNELS = 1
NCLASSES = 6
ALPHA_RANK = 6
WEIBULL_TAIL_SIZE = 20


def main():
    parser = argparse.ArgumentParser()

    # Optional arguments.
    parser.add_argument(
        "--weibull_tailsize",
        type=int,
        default=WEIBULL_TAIL_SIZE,
        help="Tail size used for weibull fitting"
    )

    parser.add_argument(
        "--alpha_rank",
        type=int,
        default=ALPHA_RANK,
        help="Alpha rank to be used as a weight multiplier for top K scores"
    )

    parser.add_argument(
        "--distance",
        default='eucos',
        help="Type of distance to be used for calculating distance \
        between mean vector and query image \
        (eucos, cosine, euclidean)"
    )

    parser.add_argument(
        "--synsetfname",
        default='synset_words_caffe_ILSVRC12.txt',
        help="Path to Synset filename from caffe website"
    )

    parser.add_argument(
        "--image_folder",
        default='data/',
        help="Image folder directory for which openmax scores are to be computed"
    )

    parser.add_argument(
        '-cls',
        '--classes',
        type=int,
        nargs='+',
        default=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    )

    args = parser.parse_args()

    alpha_rank = args.alpha_rank
    weibull_tailsize = args.weibull_tailsize
    synsetfname = args.synsetfname

    labellist = getlabellist(synsetfname)
    image_folder = args.image_folder
    mean_path = os.path.join(image_folder, 'train', 'means')
    distance_path = os.path.join(image_folder, 'train', 'distances')
    weibull_model = weibull_tailfitting(mean_path, distance_path, labellist, tailsize=weibull_tailsize)
    print("Completed Weibull fitting on %s models" % len(weibull_model.keys()))

    # per class
    categories = list(filter(lambda x: os.path.isdir(os.path.join(image_folder, 'test', x)),
                             os.listdir(os.path.join(image_folder, 'test'))))
    categories.sort()
    kkc = list(filter(lambda x: os.path.isdir(os.path.join(image_folder, 'test', x)),
                      os.listdir(os.path.join(image_folder, 'train'))))
    kkc.sort()
    uuc = list(set(categories) - set(kkc))
    kkc_scores = []
    uuc_scores = []
    for category in categories:
        # per image
        if category in kkc:
            scores = kkc_scores
        else:
            scores = uuc_scores
        for image_arrname in glob.glob(os.path.join(image_folder, 'test', category, '*.mat')):
            imgarr = loadmat(image_arrname)
            openmax, softmax = recalibrate_scores(weibull_model, labellist, imgarr, alpharank=alpha_rank)
            scores.append(np.max(openmax))
    kkc_scores = np.array(kkc_scores)
    uuc_scores = np.array(uuc_scores)
    print(kkc_scores.mean(), uuc_scores.mean())

    y_true = np.concatenate((np.ones_like(kkc_scores), np.zeros_like(uuc_scores)))
    y_score = np.concatenate((kkc_scores, uuc_scores))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    print(auc_score)
    plt.title('Openmax_ROC_Curve of {}, AUC: {:.4f}'.format(image_folder.split('/')[-1], auc_score))
    plt.xlabel('fpr')
    plt.ylabel('tpr')

    plt.savefig(os.path.join(image_folder, 'openmax_roc.png'))
    plt.show()

if __name__ == "__main__":
    main()
