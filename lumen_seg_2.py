from __future__ import division
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.segmentation import chan_vese
from skimage import img_as_ubyte
from scipy.spatial import Delaunay
import random as rng
import math
import alphashape
from histmatch import hist_match
from skimage.color import separate_stains,fgx_from_rgb
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import rgb2hed
import imutils
from scipy import ndimage
from itertools import combinations
from skimage.morphology import convex_hull_image
import skimage.io
import statistics
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from statistics import mean
from sklearn.neighbors import NearestNeighbors
import histomicstk as htk
from skimage import color
from skimage.exposure import equalize_adapthist,rescale_intensity
from skimage import img_as_float64,img_as_uint,img_as_float
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.preprocessing.color_conversion import lab_mean_std
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.metrics import confusion_matrix
from skimage.metrics import (adapted_rand_error,
                              variation_of_information)
from skimage.segmentation import morphological_geodesic_active_contour,inverse_gaussian_gradient,morphological_chan_vese
import histomicstk as htk
from skimage import color
from skimage.segmentation import slic
from skimage.exposure import equalize_adapthist,rescale_intensity
from scipy.spatial import distance
from scipy import ndimage
width = 364
height = 364
dim = (width, height)
source_image= cv2.imread('C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Unet_nuclei/codes/TCGA-001-tile-r5-c109-x110598-y4098-w1024-h1024.png',-1)
source_image= cv2.resize(source_image, dim, interpolation = cv2.INTER_AREA)
          
meanRef, stdRef = lab_mean_std(source_image)

def get_confusion_matrix_elements(groundtruth_list, predicted_list):
    """returns confusion matrix elements i.e TN, FP, FN, TP as floats
	See example code for helper function definitions
    """
     #_assert_valid_lists(groundtruth_list, predicted_list)

    if _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        tn, fp, fn, tp = 0, 0, 0, np.float64(len(groundtruth_list))

    elif _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        tn, fp, fn, tp = np.float64(len(groundtruth_list)), 0, 0, 0

    else:
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(groundtruth_list, predicted_list).ravel()
        tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)

    return tn, fp, fn, tp

def get_confusion_matrix_overlaid_mask(image, groundtruth, predicted, alpha, colors):
    """
    Returns overlay the 'image' with a color mask where TP, FP, FN, TN are
    each a color given by the 'colors' dictionary
    """
   # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    masks = get_confusion_matrix_intersection_mats(groundtruth, predicted)
    color_mask = np.zeros_like(image)
    for label, mask in masks.items():
        color = colors[label]
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask != 0] = color
        color_mask += mask_rgb
    return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)


def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """ Returns dict of 4 boolean numpy arrays with True at TP, FP, FN, TN
    """

    confusion_matrix_arrs = {}

    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)

    confusion_matrix_arrs['tp'] = np.logical_and(groundtruth, predicted)
    confusion_matrix_arrs['tn'] = np.logical_and(groundtruth_inverse, predicted_inverse)
    confusion_matrix_arrs['fp'] = np.logical_and(groundtruth_inverse, predicted)
    confusion_matrix_arrs['fn'] = np.logical_and(groundtruth, predicted_inverse)

    return confusion_matrix_arrs
def get_mpl_colormap(cmap):
    

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)



def Delaunay_Point(points_nuclei) :
    tri = Delaunay(points_nuclei,incremental= True)
    shape_points_nuclei = np.shape(points_nuclei)
    pindex = shape_points_nuclei[0]-1
    indice_point = tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][pindex]:tri.vertex_neighbor_vertices[0][pindex+1]]
    points_nuclei_inGland=[ points_nuclei[indice_Delau] for indice_Delau in indice_point]
    #points_nuclei_inGland= np.array(points_nuclei_inGland)
   
    return points_nuclei_inGland


def Roi (roi_Mask, clumen):
    temp = np.zeros(roi_Mask.shape, dtype=np.uint8)
    cv2.fillPoly(temp, pts=[clumen], color=(255, 255, 255))
    masked_image = cv2.bitwise_and(roi_Mask, temp)
    mask_ = cv2.bitwise_not(temp)
    masked_image_ = cv2.bitwise_or(masked_image, mask_)
    return masked_image, masked_image_,temp

def Mean_Value(masked_image, indice):
    locs = np.where(masked_image != indice)
    pixels = masked_image[locs]
    meanValue2 = np.average(pixels, axis=0)
    bwValue = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    Valuenon = cv2.countNonZero(bwValue)
    meanValue = meanValue2
    return meanValue

def ContourNuclei(img) :
    red_channel = img[:,:,0]
    ret,threshNuclei = cv2.threshold(red_channel,127,255,cv2.THRESH_BINARY_INV) 
    contoursNuclei, hierarchy = cv2.findContours(threshNuclei,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
  #  epsilon = 0.1*cv2.arcLength(contoursNuclei[0],True)
  #  approx = cv2.approxPolyDP(contoursNuclei[0],epsilon,True)
    
    return contoursNuclei

                
def center_nuclei(contour_nuclei,cX_nuclei,cY_nuclei,area_thr_Nuclei,indice_nuclei):
    for cnuclei in contour_nuclei:
          if cv2.contourArea(cnuclei)>2:
              indice_nuclei = indice_nuclei+1
              M = cv2.moments(cnuclei)
              cX_nuclei_ = int(M["m10"] / M["m00"])
              cY_nuclei_ = int(M["m01"] / M["m00"])
              cX_nuclei.append(cX_nuclei_)
              cY_nuclei.append(cY_nuclei_)
    return cX_nuclei,cY_nuclei,indice_nuclei



def nuclei_classification(cX_nuclei,cY_nuclei,roi_Glands_Black) :
     for indice_X in range (0,np.size(cX_nuclei)):
               print()
             #  cv2.circle(roi_Glands_Black, (cX_nuclei[indice_X], cY_nuclei[indice_X]), 7, (255, 255, 255), -1)
             #  cv2.imshow('',roi_Glands_Black)
             #  cv2.waitKey(0)
               
               intensity_nuclei = roi_Glands_Black[cY_nuclei[indice_X],cX_nuclei[indice_X]]
               print( intensity_nuclei)
               if intensity_nuclei[0]>100:
                   print(' tumourish cell')
     return intensity_nuclei    

def overlapse_nuclei(masked_image_function,img_nuclei_mask ):
    img_nuclei_ROI = cv2.cvtColor(masked_image_function ,cv2.COLOR_BGR2RGB)
    _, mask2_nuclei= cv2.threshold(masked_image_function  ,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res_nuclei = cv2.bitwise_and(img_nuclei_ROI,img_nuclei_ROI, mask=mask2_nuclei)
    hsv2bgr_nuclei = cv2.cvtColor(res_nuclei, cv2.COLOR_HSV2BGR)
    rgb2gray_nuclei = cv2.cvtColor(hsv2bgr_nuclei, cv2.COLOR_BGR2GRAY)
    contours_Nuclei, hierarchy_ = cv2.findContours(rgb2gray_nuclei,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    Nuclei_mask_separated = np.zeros(masked_image_function.shape, dtype="uint8")
    Label_mask_separated = np.zeros(masked_image_function.shape, dtype="uint8")

       ####### still separating overlapsing nuc
    for cnuclei in contours_Nuclei:
        if cv2.contourArea(cnuclei ) > 1 :
            area = cv2.contourArea(cnuclei )
            x_nuclei, y_nuclei, w_nuclei, h_nuclei = cv2.boundingRect(cnuclei)
            masked_single_nuclei = Roi (img_nuclei_mask ,cnuclei)
               
            count_label_mask = np.where((masked_single_nuclei[0] == 255))
               
            D = ndimage.distance_transform_edt(masked_single_nuclei[0])
            localMax = peak_local_max(D, indices=False, min_distance=3,
	labels=masked_single_nuclei[0])
            markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
            labels = watershed(-D, markers, mask=masked_single_nuclei[0])
            Label_mask_separated =Label_mask_separated+labels
        
            for label in np.unique(labels):
                count_label = np.where((labels == label))
                if len(count_label_mask[0]) != 0:
                    ratio_label_mask =  len(count_label[0]) /len(count_label_mask[0])
                else:
                    ratio_label_mask = 0
                   
                if label == 0:
                    continue
                if ratio_label_mask >0.010:
                    if len(count_label[0])>3:
                        mask = np.zeros(masked_single_nuclei[0].shape, dtype="uint8")
                        mask[labels == label] = 255
                        kernel = np.ones((1,1),np.uint8)
                        erosion = cv2.erode(mask,kernel,iterations = 1)
                        Nuclei_mask_separated = Nuclei_mask_separated + erosion
                       
    return Nuclei_mask_separated


def  hierarchy_glands(img_nuclei_mask,Contour_lumen,Img_final,img_glands_RGB_reinhard, contoursMask,c,img_glands_RGB,area_lumen,indice_lumen,centerLumen_X,centerLumen_Y,corner_tab_0,corner_tab_1,kernel_dil):
    index = c[1][3]
    masked_imageGlands_hierar = Roi (Img_final ,c[0])
    masked_imageGlands_Black_hierar = masked_imageGlands_hierar[0]
    masked_imageGlands_White_hierar = masked_imageGlands_hierar[1]
    masked_imageGlands_hierar_reinhard = Roi (img_glands_RGB_reinhard ,c[0]) 
    locs_hierar = np.where(masked_imageGlands_hierar_reinhard[0]  != 0)
    locs_2_hierar = np.where(masked_imageGlands_hierar_reinhard[0]  > 180)
    masked_image_nuclei_ = Roi (img_nuclei_mask ,contoursMask[index])
    if np.size(locs_hierar)!=0:
          # print(np.size(locs_hierar)/np.size(locs_2_hierar))
        if np.size(locs_2_hierar)/np.size(locs_hierar) > 0.25 :
            temp_1 = cv2.cvtColor(masked_imageGlands_Black_hierar, cv2.COLOR_BGR2GRAY )
            dst = cv2.cornerHarris(temp_1,2,3,0.04)
            indices_dst = np.where(((dst>0.01*dst.max())) ==True)
            indices_dst = list(indices_dst)
            indices_dst_0 = list(indices_dst[0])
            indices_dst_1 = list(indices_dst[1])
            #corner_tab.append(indices_dst)
            corner_tab_0.append( indices_dst_0)
            corner_tab_1.append( indices_dst_1)
           # Roi_lumen_hierar.append(c[0])
            indice_lumen = indice_lumen+1
            M = cv2.moments(c[0])
            cX_lumen_= int(M["m10"] / M["m00"])
            cY_lumen_ = int(M["m01"] / M["m00"])
            centerLumen_X.append(cX_lumen_)
            centerLumen_Y.append(cY_lumen_)
            area_lumen.append(cv2.contourArea(c[0]))
            
            masked_hierar = Roi (img_glands_RGB ,contoursMask[index])
            masked_Black_hierar = masked_hierar[0]
           
                           
            Contour_lumen.append(c[0])
            img_slic,mask_good ,img_deconv = image_filter_equa(masked_Black_hierar,img_glands_RGB_reinhard) 
            segments = slic(img_deconv,compactness=0.1, n_segments =2900,sigma = 0.5)
            superpixels = color.label2rgb(segments,img_slic , kind='avg')

            masked_Black_hierar_filt,mask,_= image_filter_equa(masked_Black_hierar,img_glands_RGB_reinhard)
            imgResult_ = cv2.erode(masked_Black_hierar_filt,kernel_dil,iterations = 1)#3 for kernel 5x5
            imgResult23 = img_as_float64(imgResult_[:,:,1]) #
     
    if indice_lumen == 0:
        imgResult23 = []
        masked_Black_hierar = []
        area_lumen_1123  = 0
        
        return imgResult23,masked_Black_hierar,Contour_lumen, masked_image_nuclei_,index,centerLumen_X,centerLumen_Y,area_lumen_1123,indice_lumen,corner_tab_0,corner_tab_1

    else:
        return imgResult23,masked_Black_hierar,Contour_lumen, masked_image_nuclei_,index,centerLumen_X,centerLumen_Y,area_lumen,indice_lumen,corner_tab_0,corner_tab_1

def convex_glands(Contour_lumen_bitwise,centerLumen_X_bitwise,centerLumen_Y_bitwise,kernel_dil,Img_final ,img_nuclei_mask,c,img_glands_RGB,img_glands_RGB_reinhard,centerLumen_X,centerLumen_Y,indice_lumen,area_lumen,Contour_lumen,corner_tab_0,corner_tab_1):
    masked_imageGlands = Roi (Img_final ,c[0])
    masked_imageGlands_Black = masked_imageGlands[0]
    masked_imageGlands_White = masked_imageGlands[1]
    masked_image_nuclei_ = Roi (img_nuclei_mask ,c[0])
    masked_imageGlands_hierar_reinhard = Roi (img_glands_RGB_reinhard ,c[0])

    shape = np.shape(img_glands_RGB_reinhard)
    mask_background = np.zeros((shape[0], shape[1],3), dtype=np.uint8)
 
    if c[1][2] != -1 :
        for indice_center_supp_lumen in range (0,len (centerLumen_X_bitwise)):
            dist = cv2.pointPolygonTest(c[0],(centerLumen_X_bitwise[indice_center_supp_lumen],centerLumen_Y_bitwise[indice_center_supp_lumen]),True)
            if dist > 0 :
             #   Contour_lumen_bitwise[indice_center_supp_lumen]
                centerLumen_X.append(centerLumen_X_bitwise[indice_center_supp_lumen])
                centerLumen_Y.append(centerLumen_Y_bitwise[indice_center_supp_lumen])
                Contour_lumen.append( Contour_lumen_bitwise[indice_center_supp_lumen])
                masked_imageGlands_lumen = Roi (img_glands_RGB_reinhard ,Contour_lumen_bitwise[indice_center_supp_lumen])
                mask_background = mask_background + masked_imageGlands_lumen[2]
                temp_1_ = cv2.cvtColor(masked_imageGlands_lumen[0], cv2.COLOR_BGR2GRAY )
                dst_ = cv2.cornerHarris(temp_1_,2,3,0.04)
                indices_dst_ = np.where(((dst_>0.01*dst_.max())) ==True)
                indices_dst_ = list(indices_dst_)
                indices_dst_0_ = list(indices_dst_[0])
                indices_dst_1_ = list(indices_dst_[1])
            #corner_tab.append(indices_dst)
                corner_tab_0.append( indices_dst_0_)
                corner_tab_1.append( indices_dst_1_)
                indice_lumen = indice_lumen+1
                area_lumen.append(cv2.contourArea(Contour_lumen_bitwise[indice_center_supp_lumen]))
    
    hull = cv2.convexHull(c[0])
    mask_convex = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    cv2.drawContours(mask_convex, [hull],-1, (255, 255, 255), -1)
    img_sub = masked_imageGlands_hierar_reinhard[2][...,0]
    background_convex = mask_convex- img_sub
  #  cv2.imshow('',background_convex)
  #  cv2.waitKey(0)
    mask_roi =  cv2.bitwise_and(img_glands_RGB_reinhard,img_glands_RGB_reinhard, mask= background_convex)
    convex_contour = cv2.cvtColor(mask_roi ,cv2.COLOR_BGR2GRAY)
    _, bw_convex = cv2.threshold(convex_contour,0,255,cv2.THRESH_BINARY)
    contours_convex, hierarchy_Glands = cv2.findContours(bw_convex, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
    for c_convex in contours_convex:
        if cv2.contourArea(c_convex) > 1:
            for indice_center_supp_lumen in range (0,len (centerLumen_X_bitwise)):
                dist_ = cv2.pointPolygonTest(c_convex,(centerLumen_X_bitwise[indice_center_supp_lumen],centerLumen_Y_bitwise[indice_center_supp_lumen]),True)
                
                if dist_ > 0 :
                    masked_imageGlands_lumen = Roi (img_glands_RGB_reinhard ,Contour_lumen_bitwise[indice_center_supp_lumen])
                    mask_background =mask_background + masked_imageGlands_lumen[2]
                    temp__1 = cv2.cvtColor(masked_imageGlands_lumen[0], cv2.COLOR_BGR2GRAY )
                    dst = cv2.cornerHarris(temp__1,2,3,0.04)
                    indices_dst = np.where(((dst>0.01*dst.max())) ==True)
                    indices_dst = list(indices_dst)
                    indices_dst_0 = list(indices_dst[0])
                    indices_dst_1 = list(indices_dst[1])
            #corner_tab.append(indices_dst)
                    corner_tab_0.append( indices_dst_0)
                    corner_tab_1.append( indices_dst_1)
           # Roi_lumen_hierar.append(c[0])
                    indice_lumen = indice_lumen+1
                    M = cv2.moments(c[0])
                    cX_lumen__= int(M["m10"] / M["m00"])
                    cY_lumen__ = int(M["m01"] / M["m00"])
                    centerLumen_X.append(cX_lumen__)
                    centerLumen_Y.append(cY_lumen__)
                    Contour_lumen.append( c_convex)
                    area_lumen.append(cv2.contourArea(c[0]))
                
                    
   # continue
        #    if dist > 0 :
     #   if cv2.contourArea(c_convex) > 300:
    centerLumen_X_2 = list(dict.fromkeys(centerLumen_X))
    centerLumen_Y = list(dict.fromkeys(centerLumen_Y))

    area_lumen = list(dict.fromkeys(area_lumen))
    if len(centerLumen_X) != len(centerLumen_X_2):
        indice_lumen = indice_lumen-1
    if indice_lumen == 0:
        imgResult123556 = np.zeros((shape[0], shape[1],3), dtype=np.uint8)
        masked_imageGlands_Black65  = np.zeros((shape[0], shape[1],3), dtype=np.uint8)
        area_lumen_58 = 0
        
        return area_lumen_58,masked_image_nuclei_,Contour_lumen, masked_imageGlands_Black65,imgResult123556,centerLumen_X,centerLumen_Y,indice_lumen,corner_tab_0,corner_tab_1


    else:
        masked_imageGlands_Black = masked_imageGlands_Black +  mask_background
    
        img_slic,mask_good ,img_deconv = image_filter_equa(masked_imageGlands_Black,img_glands_RGB_reinhard) 
        segments = slic(img_deconv,compactness=0.1, n_segments =2900,sigma = 0.5)
        superpixels = color.label2rgb(segments,img_slic , kind='avg')
        imgResult_1235 = cv2.erode(img_slic,kernel_dil,iterations =1)#3 for kernel 5x5
        imgResult1235 = img_as_float64(imgResult_1235[:,:,1])
        return area_lumen,masked_image_nuclei_,Contour_lumen, mask_background,imgResult1235,centerLumen_X_2,centerLumen_Y,indice_lumen,corner_tab_0,corner_tab_1

def corner_ (corner_tab_1,corner_tab_0,indice_numberLumen,points_nuclei,indice_glands_nuclei):
    for indice_corner in range (1,len(corner_tab_0[indice_numberLumen])):
        corner_x= corner_tab_1[indice_numberLumen][indice_corner]
        corner_y= corner_tab_0[indice_numberLumen][indice_corner]
        point_corner_ = [corner_x,corner_y]
                 
        points_nuclei.append(point_corner_)
        points_nuclei_2= np.array(points_nuclei)
        tri = Delaunay(points_nuclei_2,incremental= True)
        shape_points_nuclei = np.shape(points_nuclei_2)
        pindex = shape_points_nuclei[0]-1
        indice_point = tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][pindex]:tri.vertex_neighbor_vertices[0][pindex+1]]
        indice_glands_nuclei.extend(indice_point)
        indice_glands_nuclei = list(dict.fromkeys(indice_glands_nuclei))
    return  points_nuclei_2,indice_glands_nuclei              

def del_corner (points_nuclei_inGland):
     nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(points_nuclei_inGland)
     distances, indices = nbrs.kneighbors(points_nuclei_inGland)
     dis, indice_outliner = reject_outliers_2(distances[:,2], m=10)
     if len(indice_outliner) != 0:
         for ele in sorted(indice_outliner, reverse = True):
             del points_nuclei_inGland[ele]
     points_nuclei_inGland_= np.array(points_nuclei_inGland)
     return points_nuclei_inGland_,points_nuclei_inGland
    
def ima_add_cvx(points_nuclei_inGland,img_glands_RGB,Contour_lumen,indice_numberLumen) :
    Points_convex = []
    Roi_lumen_only= np.array(  Contour_lumen[indice_numberLumen] )
    grid_1 = np.zeros(img_glands_RGB.shape[:2])
    cv2.fillPoly(grid_1, pts=[Roi_lumen_only], color=(255, 255, 255))
   
    if len(points_nuclei_inGland)<3:
        Img_mask_add = grid_1
    else :
        cv = ConvexHull(points_nuclei_inGland )
        hull_points = cv.vertices
        grid = np.zeros(img_glands_RGB.shape[:2])
        
        for indice_hull_points in hull_points :
            Points_convex.append(points_nuclei_inGland[indice_hull_points ])
        Points_convex = np.array(Points_convex )
            
        Img_mask_add_ = grid + grid_1
        Img_mask_add_ = ndimage.binary_fill_holes(Img_mask_add_).astype(int)
                   
        ing_cvex = convex_glands_final (img_glands_RGB,Img_mask_add_)
        Img_mask_add = ing_cvex.astype(np.uint8)
    return Img_mask_add

def final_contour(Final_image_mask,imgResult,Img_mask_add,Img_final,img_glands_RGB) :
     evolution = []
     callback = store_evolution_in(evolution)
     
    
     ls = morphological_geodesic_active_contour(imgResult, 250, Img_mask_add, threshold=0.25,balloon = 1,
                                           smoothing=1,iter_callback=callback) #0.20
     ls = ls.astype('uint8')
     _, mask_ls= cv2.threshold(ls  ,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
     idx,g_ = cv2.findContours(mask_ls,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
     for cont in  zip(idx, g_[0]) :
         list1 = cont[0].tolist()
         indices_grid = np.where(ls == [1])
                
         coordinates = zip(indices_grid[0], indices_grid[1])    
         label_segment = []
         mask_segment,mask_segment_RGB =slic_glands (list1,label_segment,img_glands_RGB,superpixels,segments)
                 #  plt.triplot( points_nuclei_inGland_[:,0], points_nuclei_inGland_[:,1], trin.simplice 
         Final_image_mask = Final_image_mask +ls +mask_segment
          # Final_image_mask  = Final_image_mask*mask_good
     return Final_image_mask
                 
def distance_center_cont (norm_image):
     _, mask_ls_final= cv2.threshold(norm_image  ,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
     idx_final,g_ = cv2.findContours(mask_ls_final,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
     cmax_ = max(idx_final, key = cv2.contourArea)
     M = cv2.moments(cmax_)
     cX_gland_ = int(M["m10"] / M["m00"])
     cY_gland_ = int(M["m01"] / M["m00"])
     Center_final = np.array([cX_gland_,cY_gland_])
     Center_final = np.resize(Center_final,(1,2))
     nodes = np.array(cmax_)
     nodes = np.resize(nodes,(len(nodes),2))
     closest_index = distance.cdist(Center_final, nodes)
     list1_closest_index = closest_index.tolist()
     
     if np.size(list1_closest_index)>1:
         Variance_shape_glands = statistics.variance(list1_closest_index[0])
     else :
         Variance_shape_glands = 0
     return Variance_shape_glands, list1_closest_index

def distance_inter_lumen (img_nuclei_mask,norm_image):
     _, mask_ls_lumen= cv2.threshold(norm_image  ,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
     idx_final_lumen,g_ = cv2.findContours(mask_ls_lumen,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
     coor_lumen_1____ = []
     shape = np.shape(norm_image)
     ind_lu = 0
     Points_convex = []
     
     for c_inter_lum in zip(idx_final_lumen, g_[0]):
         M = cv2.moments(c_inter_lum[0])
         cX_gland_ = int(M["m10"] / M["m00"])
         cY_gland_ = int(M["m01"] / M["m00"])
         coor = [cX_gland_,cY_gland_]
         coor_1 = [cX_gland_+5,cY_gland_+5]
         coor_2 = [cX_gland_-5,cY_gland_-5]
         coor_3 = [cX_gland_-5,cY_gland_]
         coor_4 = [cX_gland_,cY_gland_-5]
         coor_lumen_1____.append(coor)
         coor_lumen_1____.append(coor_1)
         coor_lumen_1____.append(coor_2)
         coor_lumen_1____.append(coor_3)
         coor_lumen_1____.append(coor_4)
         
     convex_lum = ConvexHull(np.array(coor_lumen_1____,dtype='float32'))
     hull_points = convex_lum.vertices
     for indice_hull_points in hull_points :
         Points_convex.append(coor_lumen_1____[indice_hull_points ])
     Points_convex = np.array(Points_convex )

     mask_convex_lum = np.zeros((shape[0], shape[1]), dtype=np.uint8)
     cv2.fillPoly(mask_convex_lum, pts=[Points_convex], color=(255, 255, 255))
     _, convex_lum= cv2.threshold(mask_convex_lum  ,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
     contours_convex_lumen, hierarchy_convex = cv2.findContours(convex_lum,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
     cmax_ = max(contours_convex_lumen, key = cv2.contourArea)
     value_mac = ( cv2.contourArea(cmax_))


     bitwiseOr_convex = cv2.bitwise_and(img_nuclei_mask, convex_lum)
   
     indices_bitwiseOr_convex = np.where(bitwiseOr_convex == [255])
     density_bit = len(indices_bitwiseOr_convex[0])
   
     indices_mask_convex = np.where(mask_convex_lum == [255])
     density_convex = len(indices_mask_convex[0])
     ratio_density_nuclei = density_bit /density_convex

     indice_stroma  = np.where(bitwiseOr_convex == [0])
     density_stroma = len(indice_stroma[0])
     ratio_density_stroma = density_stroma /density_convex
  

     if len(idx_final_lumen)< 3 :
         Center_1 = np.array(coor_lumen_1____[0])
         Center_1 = np.resize(Center_1,(1,2))
         Center_2 = np.array(coor_lumen_1____[3])
         Center_2 = np.resize(Center_2,(1,2))
         closest_index = distance.cdist(Center_1, Center_2)
         
  
                                      
     return ratio_density_stroma,ratio_density_nuclei,value_mac

    
def distance_inter_glands(norm_image):
   _, mask_ls_final_end= cv2.threshold(norm_image  ,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   idx_final_end,g_ = cv2.findContours(mask_ls_final_end,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
   list_dist_min=[]
   for indice_contour_dist in range(0, len(idx_final_end)):
       nodes_cont_ind = np.array(idx_final_end[indice_contour_dist])
       nodes_cont_ind = np.resize(nodes_cont_ind,(len(nodes_cont_ind),2))
       for indice_otro in range(0, (len(idx_final_end))):
           nodes_cont_ind_1 = np.array(idx_final_end[indice_otro])  
           nodes_cont_ind_1 = np.resize(nodes_cont_ind_1,(len(nodes_cont_ind_1),2))
           closest_index_final = distance.cdist(nodes_cont_ind, nodes_cont_ind_1)
           if closest_index_final.min() != 0 :
               list_dist_min.append(closest_index_final.min())
           else:
               continue
         
   if np.size(list_dist_min)>1:
       Variance_space_glands  = statistics.variance(list_dist_min)
   else:
       Variance_space_glands = 0
   return Variance_space_glands

def Img_reslt_fin(Contour_lumen_bitwise,try2_2,Mask_image_rgb,contoursMask_inside,hierarchy_inside,img_glands_RGB,img_glands_RGB_reinhard,centerLumen_X_bitwise,centerLumen_Y_bitwise):
    for c in zip(contoursMask_inside, hierarchy_inside[0]):
       
        if cv2.contourArea(c[0]) > 50:
            masked_imageGlands = Roi (img_glands_RGB ,c[0])
            masked_imageGlands_reinhard = Roi (img_glands_RGB_reinhard ,c[0])

            masked_imageGlands_Black = masked_imageGlands_reinhard[0]
            locs_hierar = np.where(masked_imageGlands_reinhard[0]  != 0)
            locs_2_hierar = np.where(masked_imageGlands_reinhard[0] > 150)
            if np.size(locs_2_hierar)/np.size(locs_hierar) > 0.72 :
                 Mask_image_rgb = Mask_image_rgb +masked_imageGlands_Black
                 M = cv2.moments(c[0])
                 cX_gland_bitwise = int(M["m10"] / M["m00"])
                 cY_gland_bitwise = int(M["m01"] / M["m00"])
                 centerLumen_X_bitwise.append(cX_gland_bitwise)
                 centerLumen_Y_bitwise.append(cY_gland_bitwise)
                 Contour_lumen_bitwise.append(c[0])
                
            else:
                continue
    Img_final =try2_2 + Mask_image_rgb
   
    return Mask_image_rgb,Img_final,centerLumen_X_bitwise,centerLumen_Y_bitwise,Contour_lumen_bitwise
    
def contour_thres (img):
      img_to_contour = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
      _, mask2_to_contour= cv2.threshold(img  ,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      res_to_contour = cv2.bitwise_and( img_to_contour , img_to_contour , mask=mask2_to_contour)
      hsv2bgr_to_contour = cv2.cvtColor(res_to_contour, cv2.COLOR_HSV2BGR)
      rgb2gray_to_contour = cv2.cvtColor(hsv2bgr_to_contour, cv2.COLOR_BGR2GRAY)
      contours_to_contour, hierarchy_to_contour = cv2.findContours(rgb2gray_to_contour,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      return  contours_to_contour

def convex_glands_final (img_glands_RGB,mask_convex_del):
    mask_convex_del_copy=((mask_convex_del+1)*255/2).astype('uint8')
   
    contour_hull = contour_thres (mask_convex_del_copy)
   
   # cv2.drawContours(img_glands_RGB ,contour_hull, -1, (0,255,0), 1)
   # cv2.imshow('',img_glands_RGB)
  #  cv2.waitKey(0)
   # cmax_ = max(contour_hull, key = cv2.contourArea)
                              
    for c_hull in  contour_hull  :
        hull_delaunay = cv2.convexHull(c_hull )
        shape = np.shape(img_glands_RGB)
        mask_total= np.zeros((shape[0], shape[1]), dtype=np.uint8)
        cv2.drawContours(mask_total, [ hull_delaunay],-1, (255, 255, 255), -1)
    return mask_total



def ratio_nuclei(contour_nuclei_final_,Nuclei_mask_separated,img_glands_RGB):
     indice_nuclei_healthy = 0
     indice_nuclei_tumor = 0
     indice_nuclei_total = 0
     Ratio_nuclei_tumor = 0
     Ratio_nuclei_healthy = 0
     for c_nuclei_final  in contour_nuclei_final_:
         indice_nuclei_total = indice_nuclei_total +1
         masked_image_nuclei_final_indi = Roi (Nuclei_mask_separated,c_nuclei_final)
         rgb_nuclei =cv2.bitwise_and(img_glands_RGB,img_glands_RGB, mask=masked_image_nuclei_final_indi[0])
         value = Mean_Value(rgb_nuclei,0)
         if value > 65:
             indice_nuclei_tumor = indice_nuclei_tumor+1
         else:
             indice_nuclei_healthy = indice_nuclei_healthy+1
         Ratio_nuclei_healthy = (indice_nuclei_healthy /indice_nuclei_total)*100
         Ratio_nuclei_tumor= (indice_nuclei_tumor /indice_nuclei_total)*100

     return Ratio_nuclei_tumor, Ratio_nuclei_healthy

def orga_nuclei (points_nuclei,indice_nuclei,cX_nuclei,cY_nuclei):
    for indice_NumberNuclei in range (0,indice_nuclei):
        centerNuclei_XDistance= cX_nuclei[indice_NumberNuclei]
        centerNuclei_YDistance= cY_nuclei[indice_NumberNuclei]
        points_nuclei_ = [centerNuclei_XDistance,centerNuclei_YDistance]
        points_nuclei.append(points_nuclei_)
    return points_nuclei

def reject_outliers_2(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    data_range = np.arange(len(data))
    idx_list = data_range[s>=m]
    return data[s < m],idx_list


def delete_weird (points_nuclei):
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(points_nuclei)
    distances, indices = nbrs.kneighbors(points_nuclei)
    dis, indice_outliner = reject_outliers_2(distances[:,2], m=10)
    if len(indice_outliner) != 0:
        for ele in sorted(indice_outliner, reverse = True):  
            del points_nuclei[ele] 
   
    return points_nuclei
                   
                    
def organize_point(points_nuclei,points_lumen):
    points_nuclei.append(points_lumen)
    points_nuclei= np.array(points_nuclei)
    points_nuclei_inGland = Delaunay_Point(points_nuclei)
    points_nuclei_inGland.append(points_lumen)
    points_nuclei_inGland= np.array(points_nuclei_inGland)
    return points_nuclei_inGland
    
    
def segmen_glands_final(points_nuclei_inGland ,Points_convex,img_glands_RGB,Roi_lumen_only_,indice_numberLumen):
    cv = ConvexHull(points_nuclei_inGland )
    hull_points = cv.vertices
    grid = np.zeros(img_glands_RGB.shape[:2])
    for indice_hull_points in hull_points :
        Points_convex.append(points_nuclei_inGland[indice_hull_points ])
    
    Points_convex = np.array(Points_convex )
    cv2.fillPoly(grid, pts=[Points_convex], color=(255, 255, 255))
    image_glands_final = convex_glands_final  (img_glands_RGB,Points_convex, Roi_lumen_only_,indice_numberLumen)
    indices_grid = np.where(image_glands_final == [255])
    coordinates = zip(indices_grid[0], indices_grid[1])
    return coordinates, image_glands_final

def slic_glands (coordinates,label_segment,img_glands_RGB,average,segments):
     for indice_nuclei_Delaunay in range (0, len(coordinates)):
         label_nuclei = segments[coordinates[indice_nuclei_Delaunay][0][1],coordinates[indice_nuclei_Delaunay][0][0]]
         if average[:,:,1][coordinates[indice_nuclei_Delaunay][0][1],coordinates[indice_nuclei_Delaunay][0][0]] < 0.20 and  average[:,:,1][coordinates[indice_nuclei_Delaunay][0][1],coordinates[indice_nuclei_Delaunay][0][0]]> 0.09:
             label_segment.append(label_nuclei)
     label_segmen_list = list(set(label_segment))
     mask_segment = np.zeros(img_glands_RGB.shape[:2], dtype = "uint8")
     mask_segment_RGB_2  = np.zeros(img_glands_RGB.shape[:2], dtype = "uint8")
     for  segVal in label_segmen_list:
         mask_segment[segments == segVal] = 255
         mask_segment_RGB_2 = cv2.bitwise_and(img_glands_RGB, img_glands_RGB, mask =mask_segment)
     return mask_segment,mask_segment_RGB_2

def image_filter_equa(Img_final,img_glands_RGB_reinhard) :
 
  
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    stains = ['hematoxylin',  # nuclei stain
          'eosin',        # cytoplasm stain
          'null']         # set to null if input contains only two stai
    W = np.array([stain_color_map[st] for st in stains]).T
    W_init = W[:, :2]
    sparsity_factor = 0.5
    I_0 = 255
    im_sda = htk.preprocessing.color_conversion.rgb_to_sda(Img_final, I_0)
    W_est = htk.preprocessing.color_deconvolution.separate_stains_xu_snmf(
    im_sda, W_init, sparsity_factor,
     )
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(
    Img_final,
    htk.preprocessing.color_deconvolution.complement_stain_matrix(W_est),
    I_0,
        )
    saturation = imDeconvolved.Stains[:, :, 1] > 200
    upper = imDeconvolved.Stains[:, :, 1] <250
    mask_ = saturation
    red = img_glands_RGB_reinhard[:,:,0]*mask_
    green = img_glands_RGB_reinhard[:,:,1]*mask_
    blue = img_glands_RGB_reinhard[:,:,2]*mask_
    red_girl_masked = np.dstack((red,green,blue))
    
    equa = equalize_adapthist(red_girl_masked)
    equa_reinhard = reinhard (equa, meanRef,stdRef)
   
  #  cv2.imshow('',equa_reinhard)
  #  cv2.waitKey(0)
    return equa,mask_,imDeconvolved.Stains[:, :, 1]
    
    
 

def contour_thres_left (img):
      img_to_contour = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
      _, mask2_to_contour= cv2.threshold(img  ,30, 240, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      res_to_contour = cv2.bitwise_and( img_to_contour , img_to_contour , mask=mask2_to_contour)
      hsv2bgr_to_contour = cv2.cvtColor(res_to_contour, cv2.COLOR_HSV2BGR)
      rgb2gray_to_contour = cv2.cvtColor(hsv2bgr_to_contour, cv2.COLOR_BGR2GRAY)
      contours_to_contour, hierarchy_to_contour = cv2.findContours(rgb2gray_to_contour,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      return  contours_to_contour

def indice_nuclei_ratio (Ratio_nuclei_tumor,indice_Ratio_nuclei_tumor_0,indice_Ratio_nuclei_tumor_20,indice_Ratio_nuclei_tumor_40,indice_Ratio_nuclei_tumor_60,indice_Ratio_nuclei_tumor_80,indice_Ratio_nuclei_tumor_100):
    if Ratio_nuclei_tumor == 0:
        indice_Ratio_nuclei_tumor_0=indice_Ratio_nuclei_tumor_0+1
    if Ratio_nuclei_tumor>0 and Ratio_nuclei_tumor <= 20:
        indice_Ratio_nuclei_tumor_20=indice_Ratio_nuclei_tumor_20+1
    if Ratio_nuclei_tumor > 20 and Ratio_nuclei_tumor<= 40:
        indice_Ratio_nuclei_tumor_40=indice_Ratio_nuclei_tumor_40+1
    if Ratio_nuclei_tumor>40 and Ratio_nuclei_tumor <= 60:
        indice_Ratio_nuclei_tumor_60=indice_Ratio_nuclei_tumor_60+1
    if Ratio_nuclei_tumor >60 and Ratio_nuclei_tumor <= 80:
        indice_Ratio_nuclei_tumor_80=indice_Ratio_nuclei_tumor_80+1
    if Ratio_nuclei_tumor == 100:
        indice_Ratio_nuclei_tumor_100=indice_Ratio_nuclei_tumor_100+1
    return indice_Ratio_nuclei_tumor_0,indice_Ratio_nuclei_tumor_20,indice_Ratio_nuclei_tumor_40,indice_Ratio_nuclei_tumor_60,indice_Ratio_nuclei_tumor_80,indice_Ratio_nuclei_tumor_100

def full_indice (contour_glands_final_no_lumen, indice_1,indice_2,indice_3,indice_Area_contour_below_500,indice_Area_contour_below_1500,indice_Area_contour_below_5000,indice_Area_contour_below_10000 ) :
    if contour_glands_final_no_lumen <=indice_1:
        indice_Area_contour_below_500 = indice_Area_contour_below_500+1
    if contour_glands_final_no_lumen>indice_1 and  contour_glands_final_no_lumen<= indice_2:
        indice_Area_contour_below_1500 = indice_Area_contour_below_1500+1
    if contour_glands_final_no_lumen>indice_2 and contour_glands_final_no_lumen <=indice_3:
        indice_Area_contour_below_5000 = indice_Area_contour_below_5000+1
    if contour_glands_final_no_lumen>indice_3 :
        indice_Area_contour_below_10000 = indice_Area_contour_below_10000+1
    return  indice_Area_contour_below_500,  indice_Area_contour_below_1500 ,indice_Area_contour_below_5000,indice_Area_contour_below_10000       
