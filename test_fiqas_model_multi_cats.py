#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   test_model.py

   script to generate images from trained NN model

   author Jebb Q Stewart (jebb.q.stewart@noaa.gov)
   Copyright 2019. Colorado State University. All rights reserved.

   Edited to tet the validation stage of a multi-classifier trained model
   Christina Kumler

"""

import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
#old from keras.models import model_from_json
from tensorflow.keras.models import model_from_json
import argparse

import json
import pandas as pd

from prepare_data_ddrf import processTimeBlock, setDF
from keras_segmentation_fiqas_multi_cats import preprocessGFSData 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.labels_fiqas.ibtracs_ddrf import readIBTracs, getPointsForTime, getPointsForIntermediateTime
import dateutil.parser
from datetime import timedelta

#og from data.gfs.read_gfs import *
from data.fiqas.read_gfs import *

from tensorflow.python.ops.numpy_ops import np_config

model_name = "fiqas_dicemodel"
model_path = "models"
weight_path = "weights"
output_path = "output_ddrf_multi_cats"

gfspath = "/scratch2/BMC/public/retro/fiqas/aws/"
gfstemplate  = "%Y/%m/%d"

start = ""
end = ""

gfspathtemplate = ""

gray = plt.cm.gray

import logging


logging_format = '%(asctime)s - %(name)s - %(message)s'
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, 
    format=logging_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("TestModel")

def readdata(infile, varname):

    data, lats, lons = getDataWithLatLons(infile, varname)
#    lons = lons - 180.0

    return data, lats, lons


def inference(model, windows):
    # evaluate loaded model on test data
    x_pred = model.predict(windows)
    return x_pred


def normalize(heatmap):
    max_value = np.max(heatmap)
    min_value = np.min(heatmap)
    normalized_heat_map = (heatmap - min_value) / (max_value - min_value)

    return normalized_heat_map

def threshold(heatmap, threshold):
    heatmap[heatmap < threshold] = 0;

def main():

    # load json and create model
    model_file = model_path + "/" + model_name + '.json'
    logger.info("loading model from file: %s", model_file)
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    weights_file = weight_path + "/" + model_name + ".h5"
    logger.info("loading weights from file: %s", weights_file)
    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")

    varname = "MSLP (MAPS System Reduction)"

    timestamp = start
    validTime = dateutil.parser.parse(timestamp)
    global df
    df = readIBTracs('./sfc_data_on_hrrr2.json')
    setDF(df)
    #
    image_basepath = output_path + "/"

    if not os.path.isdir(image_basepath):
       os.makedirs(image_basepath)

    jet = cm.jet
    my_cmap = jet(np.arange(jet.N))
    my_cmap[:,-1] = 0.5
    my_cmap[0,-1] = 0.1

    while validTime < dateutil.parser.parse(end):
        # for each time step generate the image for the truth,
        # the prediction, and the combo showing both

        timestamp = validTime.strftime("%Y-%m-%d %H:%M:00")
        logger.info("Processing time %s", timestamp)
        points = getPointsForTime(df, timestamp)
        logger.info("Found points: %s", points)
        try:
            train, labels, tttt = processTimeBlock(validTime, steps=3, hours=3, useOnlyLabeled=False)

            train, labels = preprocessGFSData(train, labels)

            truthfile = validTime.strftime(image_basepath + "Truth-%Y-%m-%dT%H:%M:00.png")
            predictfile = validTime.strftime(image_basepath + "Predict-%Y-%m-%dT%H:%M:00.png")
            combofile = validTime.strftime(image_basepath + "%Y-%m-%dT%H:%M:00.png")

            logger.debug(truthfile)
            logger.debug(predictfile)

            #normalize to -1 / +1
            # block has 3 images with last one being the current time index=2
            data = train[0,:,:,2]
            origimage = Image.fromarray(np.uint8(gray((data+1.0)/2.0)*255))

            heatmap = inference(loaded_model, train)

            heat_npy = heatmap
            image_basepath_npy = output_path + "/npys/"
            heatfile_npy = validTime.strftime(image_basepath_npy + "heat-%Y-%m-%dT%H:%M:00")
            np.save(heatfile_npy, heat_npy)

            # heatmap:  (1, 1056, 1792, 3)
#            print('heatmap: ', np.shape(heatmap))
            heatmap = heatmap[0].astype(np.float)
            heatmap = heatmap.reshape(heatmap.shape[1], heatmap.shape[2], heatmap.shape[3])

            # normalize heatmap
            #heatmap = normalize(heatmap)


            # generate image with prediction heatmap blend
            image = Image.fromarray(cm.jet(heatmap, bytes=True))
            image = Image.blend(origimage, image, alpha=0.25)
            image.save(predictfile)

            print('orig image: ', origimage)
            print('image: ', image)
            print('heatmap: ', heatmap)
            print('data: ', train)
            print('labels now: ', labels)

            # generate truth image
#            print('labels: ', np.shape(labels))
            labels = labels[0]
#            print('labels[0]: ', labels)
            
#            labels = labels.reshape(labels.shape[0], labels.shape[1])
            labels = labels.numpy().reshape(labels.shape[0], labels.shape[1])
            # check the sizes
            labels = labels.astype(np.float32)
            labelimg = Image.fromarray(cm.jet(labels, bytes=True))
            labelimg = labelimg.convert('RGBA')
            print('orig', origimage)
            print('labelimg:', labelimg)
            truthimage = Image.blend(origimage, labelimg, alpha=0.25)
            truthimage.save(truthfile)

            print('test post truth')

            # combine and save
            imgs_comb = np.vstack( (np.asarray(i) for i in [truthimage, image]))
            imgs_comb = Image.fromarray( imgs_comb)
            imgs_comb.save(combofile)
        
            # ck export images to npy
            truth_npy = np.array(truthimage)
            predict_npy = np.array(predictfile)

            image_basepath_npy = output_path + "/npys/"
            truthfile_npy = validTime.strftime(image_basepath_npy + "Truth-%Y-%m-%dT%H:%M:00")
            predictfile_npy = validTime.strftime(image_basepath_npy + "Predict-%Y-%m-%dT%H:%M:00")
            
            np.save(truthfile_npy, truth_npy)
            np.save(predictfile_npy, predict_npy)
            
        except Exception as inst:
                print(inst)
                logger.error(inst)

        # step forward X hours
        validTime = validTime + timedelta(hours=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test Model")

    parser.add_argument('-n', '--name',
       dest="name", required=True,
       help="name of model without suffix")

    parser.add_argument('-mp', '--modelpath',
       dest="modelpath", required=False, default='models',
       help="path to find model json file")

    parser.add_argument('-wp', '--weightpath',
       dest="weightpath", required=False, default='weights',
       help="path to find model weights file")

    parser.add_argument('-o', '--outputpath',
       dest="output", required=False, default='test',
       help="path to save files")

    parser.add_argument('-d', '--datapath',
        dest='datapath', required=False, default='../../data',
        help="path to find GFS data files")

    parser.add_argument('-s', '--start',
       dest="start", required=True, default=None,
       help="start iso time")

    parser.add_argument('-e', '--end',
       dest="end", required=True, default=None, type=str,
       help="end iso time")

    args = parser.parse_args()

    model_name = args.name
    weights_path = args.weightpath
    model_path = args.modelpath
    gfs_path = args.datapath
    output_path = args.output
    start = args.start
    end = args.end

    gfspathtemplate = gfs_path + "/" + gfstemplate

    main()








