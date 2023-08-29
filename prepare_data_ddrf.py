#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   prepare_data.py

   Built from script to process data generating train and test dataset.  test contains labels

   author Jebb Q Stewart (jebb.q.stewart@noaa.gov)
   Copyright 2019. Colorado State University. All rights reserved.

   This has been edited in Mar 2022 to fit the FIQAS data and hi/low labels
   Christina Kumler


"""
import os
import sys
import math
import dateutil.parser
import zarr
import multiprocessing 
import argparse
import logging
import sys
import dask.array as da
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pandas as pd
import numpy as np
from datetime import timedelta
from multiprocessing import Process, Lock
from zarr import blosc

# Here is where we point to our modified code instead of OG way
from data.labels_fiqas.ibtracs_ddrf import readIBTracs, getPointsForTime, getPointsForIntermediateTime
from label_data_ddrf import createLabeledData, pointsToDataPoints
from data.fiqas.read_gfs import *

# Setup logging
logging_format = '%(asctime)s - %(name)s - %(message)s'
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, 
    format=logging_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("PrepareData")

debug = False

# for segmentation vs classification, how big is the center point x-2,x+2
center_mark = 20
#center_mark = 30
#center_mark = 10

# CHANGE ME to your path to data
basepath_tracks = "../../data_ddrf/"
#basepath_tracks = "../../data_ddrf_smaller_box/"
#basepath_tracks = "../../data_ddrf_lows/"
#basepath_tracks = "../../data_ddrf_highs/"
# FIQAS data
gfspathtemplate = "/scratch2/BMC/public/retro/fiqas/aws/%Y/%m/%d"

global df
df = pd.DataFrame()

# set up compression for data
global compressor
compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
synchronizer = zarr.ProcessSynchronizer('example.sync')

def timeToGrib(validTime):
## Note: for FIQAS, the files are already on every 3rd hour, so that is just hard-coded into the bottom here

    yearday = validTime.strftime("%Y%m%d")
    hour = validTime.strftime("%H%M%S")

    gfspath = validTime.strftime(gfspathtemplate)
    gfspath = validTime.strftime(gfspathtemplate)
    # pull the right FIQAS file for that timestep
    grbfilename='{}_{}_000000.grib2'.format(yearday, hour)

    return gfspath + "/" + grbfilename

#def processFile(filename, points, varname="Precipitable water", needPoints=False, useOnlyLabeled=True):
def processFile(filename, points, varname="MSLP (MAPS System Reduction)", needPoints=False, useOnlyLabeled=True):
    
    logger.info("%s : : %s", multiprocessing.current_process().name, filename)

    if not os.path.isfile(filename):
        logger.warn("filename %s not found", filename)
        return None, None

    train = None
    test = None
    data, lats, lons = getDataWithLatLons(filename, varname)

    if data is not None: 
        height, width = data.shape

        try:
            xypoints = pointsToDataPoints(
                points, lats, lons, offsetX=True, width=width)

            #count = count + 1
            logger.debug("found %s points in region", len(xypoints))
            if not useOnlyLabeled or (useOnlyLabeled and needPoints and len(xypoints) > 0):
                labels = []

                # created labeled data same size as original for segmentation
                createLabeledData(data, xypoints, labels,
                                  range_value=255, center_mark=center_mark)

                logger.debug(labels)
                labels = np.array(labels, dtype=np.uint8)
                # format array for appending others in file

                # following is print statement to debug
                print('labels size: ', labels.shape)

                tmpdata = np.array(data)
                tmpdata = tmpdata[np.newaxis,:,:, np.newaxis]

                tmptest = np.array(labels)
                tmptest = tmptest[:,:,:, np.newaxis]

                train = tmpdata
                test = labels[:,:,:, np.newaxis]
                train = da.from_array(tmpdata, chunks=tmpdata.shape)
                test = da.from_array(test, chunks=test.shape)

            else:
                tmpdata = np.array(data)

                tmpdata = tmpdata[np.newaxis,:,:, np.newaxis]

                train = tmpdata
                train = da.from_array(tmpdata, chunks=tmpdata.shape)
                test = None 


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(exc_type, fname, exc_tb.tb_lineno)
            logger.error(e)

        del data
        del tmpdata
    return train, test

def readData(data, timestep):
    # read data using center_lat and center_lon for center points
    dayLabels = data[(data.time == timestep)]

    points = []
    # FIQAS will be leabeled differently since we already know the x,y coordinate
    for index, row in dayLabels.iterrows(): 
        center_x = row['grid_x']
        center_y = row['grid_y']
        points.append([center_x, center_y])

    return points

def setDF(dataframe):
    global df
    df = dataframe
    logger.info("after setting: %s", df)

def processTimeBlock(validTime, steps=3, hours=3, useOnlyLabeled=True):

    logger.info("Processing time: %s", validTime)
    logger.info("steps: %s", steps)
    finalTrain = None
    finalTest = None

    failed = False
    for i in range(steps-1, -1, -1):
        logger.info("processing step: %s", i)
        offset = (i) * hours
        timestamp = validTime - timedelta(hours=offset)
        needPoints = False
        if i == 0 and useOnlyLabeled:
           needPoints = True 
        train, test, xxx = processTime(timestamp, needPoints=needPoints, useOnlyLabeled=useOnlyLabeled)

        if train is None:
             failed = True
             break

        if needPoints and test is None:
             logger.warn(" no labels found ")
             failed = True
             break
             
        if finalTrain is None:
            finalTrain = np.zeros((1,train.shape[1], train.shape[2], steps), dtype=train.dtype)
            
        index = steps - i - 1
        finalTrain[0,:,:,index] = train[0,:,:,0]

        if i == 0:
             finalTest = test

    if failed:
         logger.warn("no labels found")
         return None, None, validTime
    else: 
         logger.info("returning data")
         return finalTrain, finalTest, validTime

def processTime(timestep, needPoints=False, useOnlyLabeled=True):
    logger.info("Processing timestep: %s %s needpoints: %s", timestep, timestep.hour, needPoints)
   
    if timestep.hour % 6 == 3:
       logger.info("Determining intermediate points")
       points = getPointsForIntermediateTime(df, timestep + timedelta(minutes=180), timediff=180, threshold=0)
    else:
       points = getPointsForTime(df, timestep.strftime("%Y-%m-%d %H:%M:00"), threshold=0)
    logger.info(points)
    logger.info("Processing %s cyclones", len(points))

    if needPoints and points == None or len(points) == 0:
        logger.warn("need points and none were found")
        return None, None, timestep

    gribfile = timeToGrib(timestep)
    logger.info(gribfile)

    if gribfile is None:
        logger.warn("file is missing: %s", gribfile)
        return None, None, timestep


    train, test = processFile(gribfile, points, needPoints=needPoints, useOnlyLabeled=useOnlyLabeled)
    logger.debug("returning data")

    return train, test, timestep

def processPair(timestep):
    return processTimeBlock(timestep)


def main(start, end):

    procs = multiprocessing.cpu_count()
    logger.info(" Processors Available: %s", procs)

    pool = multiprocessing.Pool(processes=max(1, procs-1))

    timestamp = start
    validTime = dateutil.parser.parse(timestamp)

    endstamp = end
    endTime = dateutil.parser.parse(endstamp)

    args = []
    while validTime < endTime:
        args.append(validTime)
        validTime = validTime + timedelta(hours=3)

    # print (args)
    for result in pool.imap(processPair, args):
       train, test, validTime = result
       if train is not None and test is not None:
            year = validTime.strftime("%Y")
            filename = fullpath

            logger.info("outfile: %s", filename)

            training_data = None
            test_data = None

            storage = None
            if os.path.isfile(filename):
                #storage = zarr.open(filename)
                logger.info("opening file for writing")
                storage = zarr.ZipStore(filename, compression=0, mode='a')
                logger.info("contents")
                logger.info(storage)
                training_data = zarr.open(storage)['train']
                test_data = zarr.open(storage)['test']
            else:
                logger.info("opening file for writing")
                #store = zarr.DirectoryStore(filename)
                storage = zarr.ZipStore(filename, compression=0, mode='w')
                base = zarr.group(storage, overwrite=True, synchronizer=synchronizer)

                training_data = base.create_dataset('train', shape=(0,train.shape[1], train.shape[2],train.shape[3]),
                        chunks=(1,train.shape[1], train.shape[2],train.shape[3]), dtype=train.dtype, compressor=compressor)
# note here that test == label and not the train/test for ML model
                test_data = base.create_dataset('test', shape=(0,test.shape[1], test.shape[2],test.shape[3]),
                        chunks=(1,test.shape[1], test.shape[2],test.shape[3]), dtype=test.dtype, compressor=compressor)
            # sample code to test images   
            # sat1 = train[0,:,:,0]
            # print (sat1.shape)
            
            # img = toimage(sat1)
            # img.show()

            # sat2 = train[0,:,:,1]
            # print (sat2.shape)
            # img = toimage(sat2)
            # img.show()

            # lab = test[0,:,:,0]
            # img = toimage(lab)
            # img.show()
            
            #print (" train  : ", train.dtype, "  max: ", np.nanmax(train), "  min: ", np.nanmin(train))
            #print (" test  : ", test.dtype, "  max: ", np.nanmax(test), "  min: ", np.nanmin(test))

            training_data.append(train)
            test_data.append(test)
            logger.info("AFTER  %s :: %s :: %s", multiprocessing.current_process().name, training_data.shape[0], test_data.shape[0])
            storage.close()
       else:
           logger.info(train)
           logger.info(test)

    pool.close()
    pool.join()

def loadIBTracks():
    

    global df

    with open('./sfc_data_on_hrrr2.json', 'r') as f:
       dat = json.load(f)
    df_read=pd.DataFrame.from_dict(dat, orient='columns')
    # make a dataframe from that dataframe with: date, horiz index, vert index, and flag 1 = low or 2=high
    li_x = []
    li_y = []
    li_flag = []
    li_dates = []
    column_names = ["datetime", "flag", "grid_x", 'grid_y']
    df = pd.DataFrame(columns = column_names)
    # iterate through each time and extract all highs and lows
    for index, row in df_read.iterrows():
       low_tmp = df_read['lows'][index]
       hi_tmp = df_read['highs'][index]
       # remove info in tuple
       x_hi = map(lambda x: x[0], hi_tmp)
       y_hi = map(lambda x: x[1], hi_tmp)
       x_hi = list(x_hi)
       y_hi = list(y_hi)
       x_low = map(lambda x: x[0], low_tmp)
       y_low = map(lambda x: x[1], low_tmp)
       x_low = list(x_low)
       y_low = list(y_low)
       # store the dates and flags of according flags
       hi_date = [df_read['datetime'][index]] * len(x_hi)
       low_date = [df_read['datetime'][index]] * len(x_low)
       hi_flag = [2] * len(x_hi)
       low_flag = [1] * len(x_low)
       # append to the bigger list
       li_x = x_hi + x_low
       li_y = y_hi + y_low
       li_flag = hi_flag + low_flag
       li_date = hi_date + low_date
       # append list to dataframe
       x_series = pd.Series(li_x, name = "grid_x")
       y_series = pd.Series(li_y, name = "grid_y")
       flag_series = pd.Series(li_flag, name = "flag")
       date_series = pd.Series(li_date, name = "datetime")
       one_step_df = pd.concat([date_series, flag_series, x_series, y_series], axis=1)
       df = df.append(one_step_df)

# we have a few labels out of bounds (~10k) so we deal with that here
    df = df[df.grid_x > 0]
    df = df[df.grid_x < 1058]
    df = df[df.grid_y > 0]
    df = df[df.grid_y < 1798]
    df['datetime'] = pd.to_datetime(df['datetime'])
    #sort fields by time
    df.sort_values(by=['datetime'])



def testSingle():
#    timestamp = "2016-08-28 18:00:00"
    timestamp = "2019-08-28 18:00:00"
    validTime = dateutil.parser.parse(timestamp)
    train, test, tmptime = processTimeBlock(validTime)
    print (tmptime, "  :: ", train.shape, "  :: ", train.dtype, " :: ", test.shape, " :: ", test.dtype)
    print (np.nanmax(train), " :: ", np.nanmin(train), " :: ", np.nanmax(test), "  :: ", np.nanmin(test))

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run Processor")

    parser.add_argument('-d', '--destination',
       dest="destination", required=True,
       help="name of output file")

    parser.add_argument('-s', '--start',
       dest="start", required=True, default=None,
       help="start iso time")

    parser.add_argument('-e', '--end',
       dest="end", required=True, default=None, type=str,
       help="end iso time")

    args = parser.parse_args()

    global fullpath
    fullpath = args.destination
    start = args.start
    end = args.end

    loadIBTracks()
    #exit()
    main(start, end)
    # testSingle()
    
