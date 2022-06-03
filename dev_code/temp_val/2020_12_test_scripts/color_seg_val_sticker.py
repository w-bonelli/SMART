'''
Name: color_segmentation.py

Version: 1.0

Summary: K-means color clustering based segmentation. This is achieved 
         by converting the source image to a desired color space and 
         running K-means clustering on only the desired channels, 
         with the pixels being grouped into a desired number
    of clusters. 
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2019-07-29

USAGE:

python3 color_seg_val_sticker.py -p ~/temp_val/2020-12-15/ -ft jpg -tv 22 -tp ~/temp_val/mark_template/sticker_22.jpg

python3 color_seg_val_sticker.py -p ~/temp_val/2020-12-16/ -ft jpg -tv 16 -tp ~/temp_val/mark_template/sticker_16.jpg


'''

# import the necessary packages
import os
import glob
import argparse
from sklearn.cluster import KMeans

from skimage.feature import peak_local_max
from skimage.morphology import watershed, medial_axis
from skimage import img_as_float, img_as_ubyte, img_as_bool, img_as_int
from skimage import measure
from skimage.segmentation import clear_border

from scipy.spatial import distance as dist
from scipy import optimize
from scipy import ndimage

import math

import numpy as np
import argparse
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore")

import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing

MBFACTOR = float(1<<20)



# generate foloder to store the output results
def mkdir(path):
    # import module
    import os
 
    # remove space at the beginning
    path=path.strip()
    # remove slash at the end
    path=path.rstrip("\\")
 
    # path exist?   # True  # False
    isExists=os.path.exists(path)
 
    # process
    if not isExists:
        # construct the path and folder
        #print path + ' folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        #print path+' path exists!'
        return False
        

def color_cluster_seg(image, args_colorspace, args_channels, args_num_clusters, min_size):
    
    # Change image color space, if necessary.
    colorSpace = args_colorspace.lower()

    if colorSpace == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
    elif colorSpace == 'ycrcb' or colorSpace == 'ycc':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
    elif colorSpace == 'lab':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
    else:
        colorSpace = 'bgr'  # set for file naming purposes

    # Keep only the selected channels for K-means clustering.
    if args_channels != 'all':
        channels = cv2.split(image)
        channelIndices = []
        for char in args_channels:
            channelIndices.append(int(char))
        image = image[:,:,channelIndices]
        if len(image.shape) == 2:
            image.reshape(image.shape[0], image.shape[1], 1)
            
    (width, height, n_channel) = image.shape
    
    #print("image shape: \n")
    #print(width, height, n_channel)
    
 
    # Flatten the 2D image array into an MxN feature vector, where M is the number of pixels and N is the dimension (number of channels).
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    

    # Perform K-means clustering.
    if args_num_clusters < 2:
        print('Warning: num-clusters < 2 invalid. Using num-clusters = 2')
    
    #define number of cluster
    numClusters = max(2, args_num_clusters)
    
    # clustering method
    kmeans = KMeans(n_clusters = numClusters, n_init = 40, max_iter = 500).fit(reshaped)
    
    # get lables 
    pred_label = kmeans.labels_
    
    # Reshape result back into a 2D array, where each element represents the corresponding pixel's cluster index (0 to K - 1).
    clustering = np.reshape(np.array(pred_label, dtype=np.uint8), (image.shape[0], image.shape[1]))

    # Sort the cluster labels in order of the frequency with which they occur.
    sortedLabels = sorted([n for n in range(numClusters)],key = lambda x: -np.sum(clustering == x))

    # Initialize K-means grayscale image; set pixel colors based on clustering.
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = int(255 / (numClusters - 1)) * i
    
    ret, thresh = cv2.threshold(kmeansImage,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    thresh_cleaned = clear_border(thresh)
    
    if np.count_nonzero(thresh) > 0:
        
        thresh_cleaned_bw = clear_border(thresh)
    else:
        thresh_cleaned_bw = thresh
        
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh_cleaned, connectivity = 8)
    
    
    
    # stats[0], centroids[0] are for the background label. ignore
    # cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT
    sizes = stats[1:, cv2.CC_STAT_AREA]
    
    Coord_left = stats[1:, cv2.CC_STAT_LEFT]
    
    Coord_top = stats[1:, cv2.CC_STAT_TOP]
    
    Coord_width = stats[1:, cv2.CC_STAT_WIDTH]
    
    Coord_height = stats[1:, cv2.CC_STAT_HEIGHT]
    
    Coord_centroids = centroids
    
    #print("Coord_centroids {}\n".format(centroids[1][1]))
    
    #print("[width, height] {} {}\n".format(width, height))
    
    nb_components = nb_components - 1
    
    
    
    #min_size = 70
    
    max_size = width*height*0.1
    
    img_thresh = np.zeros([width, height], dtype=np.uint8)
    
    #for every component in the image, keep it only if it's above min_size
    for i in range(0, nb_components):
        '''
        #print("{} nb_components found".format(i))
        
        if (sizes[i] >= min_size) and (Coord_left[i] > 1) and (Coord_top[i] > 1) and (Coord_width[i] - Coord_left[i] > 0) and (Coord_height[i] - Coord_top[i] > 0) and (centroids[i][0] - width*0.5 < 10) and ((centroids[i][1] - height*0.5 < 10)) and ((sizes[i] <= max_size)):
            img_thresh[output == i + 1] = 255
            
            print("Foreground center found ")
            
        elif ((Coord_width[i] - Coord_left[i])*0.5 - width < 15) and (centroids[i][0] - width*0.5 < 15) and (centroids[i][1] - height*0.5 < 15) and ((sizes[i] <= max_size)):
            imax = max(enumerate(sizes), key=(lambda x: x[1]))[0] + 1    
            img_thresh[output == imax] = 255
            print("Foreground max found ")
        '''
        
        if (sizes[i] >= min_size):
        
            img_thresh[output == i + 1] = 255
        
    #from skimage import img_as_ubyte
    
    #img_thresh = img_as_ubyte(img_thresh)
    
    #print("img_thresh.dtype")
    #print(img_thresh.dtype)
    
    #return img_thresh
    return img_thresh
    
'''
def medial_axis_image(thresh):
    
    #convert an image from OpenCV to skimage
    thresh_sk = img_as_float(thresh)

    image_bw = img_as_bool((thresh_sk))
    
    image_medial_axis = medial_axis(image_bw)
    
    return image_medial_axis
'''


class clockwise_angle_and_distance():
    

    '''
    A class to tell if point is clockwise from origin or not.
    This helps if one wants to use sorted() on a list of points.

    Parameters
    ----------
    point : ndarray or list, like [x, y]. The point "to where" we g0
    self.origin : ndarray or list, like [x, y]. The center around which we go
    refvec : ndarray or list, like [x, y]. The direction of reference

    use: 
        instantiate with an origin, then call the instance during sort
    reference: 
    https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python

    Returns
    -------
    angle
    
    distance
    

    '''
    def __init__(self, origin):
        self.origin = origin

    def __call__(self, point, refvec = [0, 1]):
        if self.origin is None:
            raise NameError("clockwise sorting needs an origin. Please set origin.")
        # Vector between point and the origin: v = p - o
        vector = [point[0]-self.origin[0], point[1]-self.origin[1]]
        # Length of vector: ||v||
        lenvector = np.linalg.norm(vector[0] - vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1] # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1] # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to 
        # subtract them from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance 
        # should come first.
        return angle, lenvector


# Detect stickers in the image
def sticker_detect(img_ori, save_path):
    
    '''
    image_file_name = Path(image_file).name
    
    abs_path = os.path.abspath(image_file)
    
    filename, file_extension = os.path.splitext(abs_path)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    print("Processing image : {0}\n".format(str(image_file)))
     
    # save folder construction
    mkpath = os.path.dirname(abs_path) +'/cropped'
    mkdir(mkpath)
    save_path = mkpath + '/'

    print ("results_folder: " + save_path)
    '''
   

    # load the image, clone it for output, and then convert it to grayscale
    #img_ori = cv2.imread(image_file)
    
    img_rgb = img_ori.copy()
      
    # Convert it to grayscale 
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
      
    # Store width and height of template in w and h 
    w, h = template.shape[::-1] 
      
    # Perform match operations. 
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    
    #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(res)
    
    
    # Specify a threshold 
    threshold = 0.8
      
    # Store the coordinates of matched area in a numpy array 
    loc = np.where( res >= threshold)  
    
    if len(loc):
    
        (y,x) = np.unravel_index(res.argmax(), res.shape)
    
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(res)
    
        #print(y,x)
        
        print(min_val, max_val, min_loc, max_loc)
        
    
        (startX, startY) = max_loc
        
        #startY = startY - 129
        
        endX = startX + template.shape[1]
        endY = startY + template.shape[0] 
        
        
        # Draw a rectangle around the matched region. 
        for pt in zip(*loc[::-1]): 
            
            sticker_overlay = cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 1)
        
        
        sticker_crop_img = img_rgb[startY:endY, startX:endX]

    return  sticker_crop_img, sticker_overlay




def comp_external_contour(orig, thresh, save_path):
    
    #find contours and get the external one
    #find contours and get the external one
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    img_height, img_width, img_channels = orig.shape
    
    index = 1
    
    
    print("contour length {}".format(len(contours)))
    
    
    list_of_pts = []
    
    if len(contours) > 1:
        
        
        '''
        for ctr in contours:
            
            list_of_pts += [pt[0] for pt in ctr]
    
        center_pt = np.array(list_of_pts).mean(axis = 0) # get origin
        
        clock_ang_dist = clockwise_angle_and_distance(center_pt) # set origin
        
        list_of_pts = sorted(list_of_pts, key=clock_ang_dist) # use to sort
        
        contours_joined = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
        
        
        '''
        kernel = np.ones((4,4), np.uint8)

        dilation = cv2.dilate(thresh.copy(), kernel, iterations = 1)
        
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        
        trait_img = closing
        
    

    
    #trait_img = cv2.drawContours(thresh, contours_joined, -1, (0,255,255), -1)
    
    #x, y, w, h = cv2.boundingRect(contours_joined)
    
    #trait_img = cv2.rectangle(thresh, (x, y), (x+w, y+h), (255, 255, 0), 3)
    
    contours, hier = cv2.findContours(trait_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("contour length {}".format(len(contours)))
    
    
    for c in contours:
        
        #get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        
        #if w>img_width*0.05 and h>img_height*0.05:
            
        if w>0 and h>0:
            
            offset_w = int(w*0.05)
            offset_h = int(h*0.05)
            # draw a green rectangle to visualize the bounding rect
            roi = orig[y-offset_h : y+h+offset_h, x-offset_w : x+w+offset_w]
            
            print("ROI {} detected ...".format(index))
            
            result_file = (save_path +  str(format(index, "02")) + '.' + ext)
            
            #print(result_file)
            
            cv2.imwrite(result_file, roi)
            
            trait_img = cv2.rectangle(orig, (x, y), (x+w, y+h), (255, 255, 0), 3)
            
            
            
            #trait_img = cv2.putText(orig, "#{}".format(index), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 255), 10)
            
            index+= 1
     
            

    return trait_img




def segmentation(image_file):
    
    abs_path = os.path.abspath(image_file)
    
    filename, file_extension = os.path.splitext(image_file)
    
    file_size = os.path.getsize(image_file)/MBFACTOR
    
    print("Segmenting image : {0} \n".format(str(filename)))
    
    # load original image
    image = cv2.imread(image_file)
    
    img_height, img_width, img_channels = image.shape
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # make the folder to store the results
    #current_path = abs_path + '/'
    base_name = os.path.splitext(os.path.basename(filename))[0]
    # save folder construction
    mkpath = os.path.dirname(abs_path) +'/' + base_name
    mkdir(mkpath)
    save_path = mkpath + '/'
    
    mkpath_sticker = os.path.dirname(abs_path) +'/' + base_name + '/sticker'
    mkdir(mkpath_sticker)
    save_path_sticker = mkpath_sticker + '/'
    
    print("results_folder: {0}\n".format(str(save_path)))  
    
    if (file_size > 5.0):
        print("It will take some time due to large file size {0} MB".format(str(int(file_size))))
    else:
        print("Segmenting plant image into blocks... ")
    
    #make backup image
    orig = image.copy()
    
    
    '''
    #color clustering based plant object segmentation
    thresh = color_cluster_seg(orig, args_colorspace, args_channels, args_num_clusters, min_size = 100)
    
    #result_mask = save_path + 'mask.' + ext
    
    #cv2.imwrite(result_mask, thresh)
    
    
    #find external contour and segment image into small ROI based on each plant
    trait_img = comp_external_contour(image.copy(), thresh, save_path)
    
    result_file = abs_path +  '_label.' + ext
            
    cv2.imwrite(result_file, trait_img)
    
    '''
    
    
    (sticker_crop_img, sticker_overlay) = sticker_detect(image.copy(), save_path)
    
    # save segmentation result
    result_file = (save_path_sticker + base_name + '_sticker_overlay.' + args['filetype'])
    print(result_file)
    cv2.imwrite(result_file, sticker_overlay)
    
    
    # save segmentation result
    result_file = (save_path_sticker + base_name + '_sticker_match.' + args['filetype'])
    #print(result_file)
    cv2.imwrite(result_file, sticker_crop_img)
    
    '''
    thresh_sticker = color_cluster_seg(sticker_crop_img.copy(), args_colorspace, args_channels, 8, min_size = 10)
    trait_img_sticker = comp_external_contour(sticker_crop_img.copy(), thresh_sticker, save_path_sticker)
    result_file_sticker = save_path_sticker +  '_label.' + ext
    cv2.imwrite(result_file_sticker, trait_img_sticker)
    

    #number of rows
    nRows = 3
    # Number of columns
    mCols = 3

    # Dimensions of the image
    sizeX = img_width
    sizeY = img_height
    #print(img.shape)


    for i in range(0, nRows):
        
        for j in range(0, mCols):
            
            roi = orig[int(i*sizeY/nRows):int(i*sizeY/nRows) + int(sizeY/nRows),int(j*sizeX/mCols):int(j*sizeX/mCols) + int(sizeX/mCols)]
            
            result_file = (save_path +  str(i+1) + str(j+1) + '.' + ext)
            
            cv2.imwrite(result_file, roi)
    
    #return thresh
    #trait_img
    '''
    
    

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    #ap.add_argument('-i', '--image', required = True, help = 'Path to image file')
    ap.add_argument("-p", "--path", required = True,    help="path to image file")
    ap.add_argument("-ft", "--filetype", required=True,    help="Image filetype")
    
    ap.add_argument('-s', '--color-space', type =str, default ='lab', help='Color space to use: BGR (default), HSV, Lab, YCrCb (YCC)')
    ap.add_argument('-c', '--channels', type = str, default='1', help='Channel indices to use for clustering, where 0 is the first channel,' 
                                                                       + ' 1 is the second channel, etc. E.g., if BGR color space is used, "02" ' 
                                                                       + 'selects channels B and R. (default "all")')
    ap.add_argument('-n', '--num-clusters', type = int, default = 2,  help = 'Number of clusters for K-means clustering (default 3, min 2).')
    ap.add_argument('-tv', '--temp_val', required = False,  type = int, default = 22,  help = 'Number of clusters for K-means clustering (default 3, min 2).')
    ap.add_argument("-tp", "--temp_path", required = False,  help="template image path")
    args = vars(ap.parse_args())
    
    
    
    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']
    
    args_colorspace = args['color_space']
    args_channels = args['channels']
    args_num_clusters = args['num_clusters']

    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
    global  template
    template_path = args['temp_path']
    
    # Read the template 
    template = cv2.imread(template_path, 0) 
    
    img_height, img_width = template.shape
    
    print("sticker_template was found")
    
    temp_val = args['temp_val']
    
    #number of rows
    nRows = 3
    # Number of columns
    mCols = 1

    # Dimensions of the image
    sizeX = img_width
    sizeY = img_height
    #print(img.shape)

    template_list = []
    for i in range(0, nRows):
        
        for j in range(0, mCols):
            
            roi = template[int(i*sizeY/nRows):int(i*sizeY/nRows) + int(sizeY/nRows),int(j*sizeX/mCols):int(j*sizeX/mCols) + int(sizeX/mCols)]
            
            template_list.append(roi)
            
            
    
    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    
    
    # local path needed!
    #template_path = "/home/suxing/plant-image-analysis/marker_template/sticker_2020-12-15.jpg"
    #template_path = "/home/suxing/plant-image-analysis/marker_template/sticker_2020-12-16.jpg"
    
    # Read the template 
    #template = cv2.imread(template_path, 0) 
    #print("template was found")
    
    print((imgList))
    
    #current_img = imgList[0]
    
    #(thresh, trait_img) = segmentation(current_img)
    
    '''
     # get cpu number for parallel processing
    #agents = psutil.cpu_count()   
    agents = multiprocessing.cpu_count()
    print("Using {0} cores to perform parallel processing... \n".format(int(agents)))
    
    
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(segmentation, imgList)
        pool.terminate()
    '''
    
    result_list = []
    #loop execute
    #for image in imgList:
        
    for index, image in enumerate(imgList):
        
        print ("index = {}".format(index))
        
        if temp_val == 22:
            
            if index == 2:

                template = cv2.vconcat([template_list[0], template_list[1]])
                reading = int((temp_val*2 + 2)*0.5)
            
            elif index == 6 or index == 12 :
                
                template = cv2.vconcat([template_list[1], template_list[2]])
                reading = int((temp_val*2 - 2)*0.5)
            
            else:
                template = template_list[1]
                reading = int(temp_val)
            
            result_list.append(reading)
        
        
        if temp_val == 16:
            
            if index == 4 or index == 11 :

                template = cv2.vconcat([template_list[1], template_list[2]])
                reading = int((temp_val*2 - 2)*0.5)
            
            elif index == 7 :
                
                template = cv2.vconcat([template_list[0], template_list[1]])
                reading = int((temp_val*2 + 2)*0.5)
            
            else:
                template = template_list[1]
                reading = int(temp_val)
            
            result_list.append(reading)
        
        (thresh) = segmentation(image)
    
    
    
    print(result_list)
    
    #color clustering based plant object segmentation
    #thresh = color_cluster_seg(orig, args_colorspace, args_channels, args_num_clusters)
    
    # save segmentation result
    #result_file = (save_path + filename + '_seg' + file_extension)
    #print(filename)
    #cv2.imwrite(result_file, thresh)
    
    
    #find external contour 
    #trait_img = comp_external_contour(image.copy(),thresh, file_path)
    
    #save segmentation result
    #result_file = (save_path + filename + '_excontour' + file_extension)
    #cv2.imwrite(result_file, trait_img)
    
    
    #accquire medial axis of segmentation mask
    #image_medial_axis = medial_axis_image(thresh)

    # save medial axis result
    #result_file = (save_path + filename + '_medial_axis' + file_extension)
    #cv2.imwrite(result_file, img_as_ubyte(image_medial_axis))
    
    

    

    

    