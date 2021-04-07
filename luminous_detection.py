'''
Name: luminous_detection.py

Version: 1.0

Summary: Detect dark images by converting it to LAB color space to access the luminous channel which is independent of colors.
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2021-03-09

USAGE:

time python3 luminous_detection.py -p ~/plant-image-analysis/test/ -ft png 

'''

# import necessary packages
import argparse
import csv
from os.path import join

import cv2
import numpy as np
import os
import glob
from pathlib import Path

import psutil
import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing

from tabulate import tabulate
import openpyxl

# Convert it to LAB color space to access the luminous channel which is independent of colors.
from options import ArabidopsisRosetteAnalysisOptions


def isbright(options: ArabidopsisRosetteAnalysisOptions):
    # Set up threshold value for luminous channel, can be adjusted and generalized
    thresh = 0.5

    # Load image file 
    orig = cv2.imread(options.input_file)

    # Make backup image
    image = orig.copy()

    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    normalized = np.mean(L / np.max(L))

    # Normalize L channel by dividing all pixel values with maximum pixel value
    if normalized < thresh:
        text_bool = "bright"
        print(f"Image {options.input_stem} is light enough ({normalized})")
    else:
        text_bool = "dark"
        print(f"Image {options.input_stem} is dark ({normalized} over threshold {thresh})")

        # clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(3, 3))
        # cl = clahe.apply(L)
        # lightened = cv2.merge((cl, A, B))
        # converted = cv2.cvtColor(lightened, cv2.COLOR_LAB2BGR)

        # hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # hsvImg[..., 2] = hsvImg[..., 2] * 2
        # converted = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)

        # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)
        # v += 255
        # final_hsv = cv2.merge((h, s, v))
        # converted = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        # converted = increase_brightness(converted)

        # cv2.imwrite(join(options.output_directory, new_image_file), converted)

    return options.input_name, normalized, text_bool


def increase_brightness(img, value=150):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def write_results_to_csv(results, output_directory):
    result_file = join(output_directory, 'luminous_detection.csv')
    headers = ['image_file_name', 'luminous_avg', 'dark_or_bright']

    with open(result_file, 'a+') as file:
        char = file.read(1)
        file.seek(0)
        if not char:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(headers)

        for row in results:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="path to image file")
    ap.add_argument("-ft", "--filetype", required=True, help="image filetype")
    ap.add_argument("-o", "--output_directory", required=True, help="directory to write output files to")

    args = vars(ap.parse_args())

    # Setting path to image files
    file_path = args["path"]
    file_type = args['filetype']
    output_dir = args['output_directory']

    # Extract file type and path
    filetype = '*.' + file_type
    image_file_path = file_path + filetype

    # Accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    options = [ArabidopsisRosetteAnalysisOptions(input_file=file, output_directory=output_dir) for file in imgList]

    # Get number of images in the data folder
    n_images = len(imgList)

    # get cpu number for parallel processing
    agents = psutil.cpu_count()
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))

    # Create a pool of processes. By default, one is created for each CPU in the machine.
    with closing(Pool(processes=agents)) as pool:
        results = pool.map(isbright, options)
        pool.terminate()

    # Output sum table in command window 
    print("Summary: {0} plant images were processed...\n".format(n_images))

    table = tabulate(results, headers=['image_file_name', 'luminous_avg', 'dark_or_bright'], tablefmt='orgtbl')

    print(table + "\n")

    write_results_to_csv(results, file_path)
