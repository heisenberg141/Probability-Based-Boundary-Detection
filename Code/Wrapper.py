#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:
from sklearn.cluster import KMeans
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
import os

def main():

    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    n_orientations = 16
    DoG_Filter_Bank = generate_dog_filter_bank(n_orientations)
    # display_and_save_filters(DoG_Filter_Bank,3,len(DoG_Filter_Bank)/3,"DOG_Filter_Bank.png")
    
    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """
    '''
    LM filter:
    1. first order derivatives for gaussians at 3 scales, 6 orientations (18).
    How to calculate first order derivative for 1 gaussian?
    2. Second order derivatives for gaussians at 3 scales, 6 orientations (18).
    How to calculate second order derivative for 1 gaussian?
    3. 8 Laplacian of gaussian filters
    How to calculate Laplacian of gaussian for 1 gaussian?
    4. 4 gaussians.
    '''
    LM_small = generate_lm_filter_bank("small")
    LM_large = generate_lm_filter_bank("large")
    # display_and_save_filters(LM_small,4,len(LM_small)/4,"LM_small.png")
    # display_and_save_filters(LM_large,4,len(LM_small)/4,"LM_large.png")


    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    Gabor_Filter_Bank = generate_gabor_filter_bank(n_orientations)
    # display_and_save_filters(Gabor_Filter_Bank,3,len(Gabor_Filter_Bank)/3,"Gabor_Filter_Bank.png")
    
    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """
    radii = [2,4,6]
    Half_Disk_Bank = generate_half_disk_bank(radii = radii, n_orients=n_orientations)
    # display_and_save_filters(Half_Disk_Bank,3,len(Half_Disk_Bank)/3,"Half_Disk_Bank.png")

    
    """
    Generate Texton Map
    Filter image using oriented gaussian filter bank
    """
    image_directory = "../BSDS500/Images/"
    sobel_directory = "../BSDS500/SobelBaseline"
    canny_directory = "../BSDS500/CannyBaseline"
    image_files = os.listdir(image_directory)
    for i in range(len(image_files)):
        image_files[i] = image_directory+ image_files[i]
        # print(image_files[i])
    
    if not os.path.exists("../Results/"):
        os.mkdir("../Results")
        os.mkdir("../Results/texture_maps")
        os.mkdir("../Results/intensity_maps")
        os.mkdir("../Results/color_maps")
        os.mkdir("../Results/texture_gradient_maps")
        os.mkdir("../Results/intensity_gradient_maps")
        os.mkdir("../Results/color_gradient_maps")
        os.mkdir("../Results/final_output")

    texture_maps_path = "../Results/texture_maps"
    intensity_maps_path = "../Results/intensity_maps"
    color_maps_path = "../Results/color_maps"
    texture_gradient_maps_path = "../Results/texture_gradient_maps"
    intensity_gradient_maps_path = "../Results/intensity_gradient_maps"
    color_gradient_maps = "../Results/color_gradient_maps"
    final_output_path  = "../Results/final_output"
    
    
    for img_path in image_files:
        # break
        image_name = os.path.basename(img_path)
        print(f"IMAGE: {image_name}")
        image = cv2.imread(img_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        r,c,d = image.shape
        flattened_gray = image_gray.reshape(r*c,1)
        print("Generating Texton Map and gradients")
        texton_map = generate_texton_map(image_gray, DoG_Filter_Bank,LM_large,Gabor_Filter_Bank)
        plt.imshow(texton_map)
        plt.savefig(os.path.join(texture_maps_path,image_name))
        t_grad_map = generate_gradient_map(texton_map, K=64, half_disks=Half_Disk_Bank)
        plt.imshow(t_grad_map)
        plt.savefig(os.path.join(texture_gradient_maps_path,image_name))
                
        print("Generating brightness Map and gradients")
        km = KMeans(n_clusters=16, n_init=2)
        labels = km.fit_predict(flattened_gray)
        b_map = labels.reshape([r, c])
        plt.imshow(b_map)
        plt.savefig(os.path.join(intensity_maps_path,image_name))
        b_grad_map = generate_gradient_map(b_map, K=16,half_disks=Half_Disk_Bank)
        plt.imshow(b_grad_map)
        plt.savefig(os.path.join(intensity_gradient_maps_path,image_name))
        
        
        print("Generating Color Map and gradients")
        km = KMeans(n_clusters=16, n_init=2)
        flattened_image = image.reshape([r*c,d])
        labels = km.fit_predict(flattened_image)
        c_map = labels.reshape([r,c])
        plt.imshow(c_map)
        plt.savefig(os.path.join(color_maps_path,image_name))
        c_grad_map = generate_gradient_map(c_map, K=16,half_disks=Half_Disk_Bank)
        plt.imshow(c_grad_map)
        plt.savefig(os.path.join(color_gradient_maps,image_name))

        print("Final Steps")
        image_name = os.path.basename(img_path)
        png_name = os.path.splitext(image_name)[0] +".png"
        canny_baseline = cv2.imread(os.path.join(canny_directory,png_name))
        canny_baseline = cv2.cvtColor(canny_baseline,cv2.COLOR_BGR2GRAY)
        sobel_baseline = cv2.imread(os.path.join(sobel_directory,png_name))
        sobel_baseline = cv2.cvtColor(sobel_baseline,cv2.COLOR_BGR2GRAY)

        grad_map_sum = t_grad_map + b_grad_map + c_grad_map
        pb_edges = np.array(grad_map_sum/3) * np.array(0.5 * sobel_baseline + 0.5 * canny_baseline)
        print("Done!\n")
        plt.imshow(pb_edges, cmap='gray')
        plt.savefig(os.path.join(final_output_path,image_name))
        # break







    

    

    


    """
    Generate texture ID's using K-means clustering
    Display texton map and save image as TextonMap_ImageName.png,
    use command "cv2.imwrite('...)"
    """


    """
    Generate Texton Gradient (Tg)
    Perform Chi-square calculation on Texton Map
    Display Tg and save image as Tg_ImageName.png,
    use command "cv2.imwrite(...)"
    """


    """
    Generate Brightness Map
    Perform brightness binning 
    """


    """
    Generate Brightness Gradient (Bg)
    Perform Chi-square calculation on Brightness Map
    Display Bg and save image as Bg_ImageName.png,
    use command "cv2.imwrite(...)"
    """


    """
    Generate Color Map
    Perform color binning or clustering
    """


    """
    Generate Color Gradient (Cg)
    Perform Chi-square calculation on Color Map
    Display Cg and save image as Cg_ImageName.png,
    use command "cv2.imwrite(...)"
    """


    """
    Read Sobel Baseline
    use command "cv2.imread(...)"
    """


    """
    Read Canny Baseline
    use command "cv2.imread(...)"
    """


    """
    Combine responses to get pb-lite output
    Display PbLite and save image as PbLite_ImageName.png
    use command "cv2.imwrite(...)"
    """

def normalise_img(img): 
    """
    Normalises image in the range 0 - 255.
    """
    old_min, old_max = np.min(img), np.max(img)
    old_range = old_max - old_min

    old_range = old_range if old_range else 1
    img = 255 * (img - old_min) / old_range
    return img
   
if __name__ == '__main__':
    main()
    


