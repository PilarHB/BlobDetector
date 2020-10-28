#!/usr/bin/env python

import numpy as np
import cv2
import glob

# parameters 
writeValues = True
display = False
savedir = "camera_data/"

# Get the center of the image cx and cy
def get_image_center(savedir):
        
    # load camera calibration
    newcam_mtx = np.load(savedir+'newcam_mtx.npy')

    # load center points from New Camera matrix
    cx = newcam_mtx[0,2]
    cy = newcam_mtx[1,2]
    fx = newcam_mtx[0,0]
    return cx,cy

# Load parameters from the camera
def load_parameters(savedir, display):
    # load camera calibration
    savedir = "camera_data/"
    cam_mtx = np.load(savedir+'cam_mtx.npy')
    dist = np.load(savedir+'dist.npy')
    roi = np.load(savedir+'roi.npy')
    newcam_mtx = np.load(savedir+'newcam_mtx.npy')
    inverse_newcam_mtx = np.linalg.inv(newcam_mtx)
    np.save(savedir+'inverse_newcam_mtx.npy', inverse_newcam_mtx)
    
    if display:
        print ("Camera Matrix :\n {0}".format(cam_mtx))
        print ("Dist Coeffs :\n {0}".format(dist))
        print("Region of Interest :\n {0}".format(roi))
        print("New Camera Matrix :\n {0}".format(newcam_mtx))
        print("Inverse New Camera Matrix :\n {0}".format(inverse_newcam_mtx))
            
    return cam_mtx,dist,roi,newcam_mtx,inverse_newcam_mtx

def save_parameters(savedir,rotation_vector, translation_vector,newcam_mtx):
    
    # Rotation Vector
    np.save(savedir + 'rotation_vector.npy', rotation_vector)
    np.save(savedir + 'translation_vector.npy', translation_vector)
    
    # Rodrigues
    # print("R - rodrigues vecs")
    R_mtx, jac = cv2.Rodrigues(rotation_vector)
    np.save(savedir + 'R_mtx.npy', R_mtx)  
    
    # Extrinsic Matrix
    Rt = np.column_stack((R_mtx,translation_vector))
    # print("R|t - Extrinsic Matrix:\n {0}".format(Rt))
    np.save(savedir + 'Rt.npy', Rt)
    
    # Projection Matrix      
    P_mtx=newcam_mtx.dot(Rt)
    # print("newCamMtx*R|t - Projection Matrix:\n {0}".format(P_mtx))
    np.save(savedir + 'P_mtx.npy', P_mtx)
    
def load_checking_parameters(savedir):
    
    rotation_vector = np.load(savedir+'rotation_vector.npy')
    translation_vector = np.load(savedir+'translation_vector.npy')
    R_mtx = np.load(savedir+'R_mtx.npy')
    Rt = np.load(savedir+'Rt.npy')
    P_mtx = np.load(savedir+'P_mtx.npy')
    inverse_newcam_mtx = np.load(savedir+'inverse_newcam_mtx.npy')
    
    return rotation_vector, translation_vector,R_mtx,Rt,P_mtx,inverse_newcam_mtx

# Calculate the real Z coordinate based on the center of the image
def calculate_z_total_points(world_points, X_center, Y_center):
    
    total_points_used = len(world_points)
    
    for i in range(1,total_points_used):
    # start from 1, given for center Z=d*
    # to center of camera
        wX = world_points[i,0]-X_center
        wY = world_points[i,1]-Y_center
        wd = world_points[i,2]

        d1 = np.sqrt(np.square(wX) + np.square(wY))
        wZ = np.sqrt(np.square(wd) - np.square(d1))
        world_points[i,2] = wZ

    print(world_points)    
    return world_points

# Lets the check the accuracy here : 
# In this script we make sure that the difference and the error are acceptable in our project. 
# If not, maybe we need more calibration images and get more points or better points

def calculate_accuracy(worldPoints,imagePoints):
    s_arr=np.array([0], dtype = np.float32)
    size_points=len(worldPoints)
    s_describe=np.empty((size_points,),dtype = np.float32)
    
    rotation_vector, translation_vector,R_mtx,Rt,P_mtx,inverse_newcam_mtx = load_checking_parameters(savedir)

    for i in range(0,size_points):
        print("=======POINT # " + str(i) +" =========================")
    
        print("Forward: From World Points, Find Image Pixel\n")
        XYZ1 = np.array([[worldPoints[i,0],worldPoints[i,1],worldPoints[i,2],1]], dtype=np.float32)
        XYZ1 = XYZ1.T
        print("---- XYZ1\n")
        print(XYZ1)
        suv1 = P_mtx.dot(XYZ1)
        print("---- suv1\n")
        print(suv1)
        s = suv1[2,0]    
        uv1 = suv1/s
        print("====>> uv1 - Image Points\n")
        print(uv1)
        print("=====>> s - Scaling Factor\n")
        print(s)
        s_arr = np.array([s/total_points_used + s_arr[0]], dtype = np.float32)
        s_describe[i] = s
        if writeValues == True: 
            np.save(savedir+'s_arr.npy', s_arr)

        print("Solve: From Image Pixels, find World Points")

        uv_1 = np.array([[imagePoints[i,0],imagePoints[i,1],1]], dtype=np.float32)
        uv_1 = uv_1.T
        print("=====> uv1\n")
        print(uv_1)
        suv_1 = s * uv_1
        print("---- suv1\n")
        print(suv_1)

        print("Get camera coordinates, multiply by inverse Camera Matrix, subtract tvec1\n")
        xyz_c = inverse_newcam_mtx.dot(suv_1)
        xyz_c = xyz_c-translation_vector
        print("---- xyz_c\n")
        inverse_R_mtx = np.linalg.inv(R_mtx)
        XYZ = inverse_R_mtx.dot(xyz_c)
        print("---- XYZ\n")
        print(XYZ)

        if calculatefromCam == False:
            cXYZ = cameraXYZ.calculate_XYZ(imagePoints[i,0],imagePoints[i,1])
            print("camXYZ")
            print(cXYZ)


    s_mean, s_std = np.mean(s_describe), np.std(s_describe)

    print(">>>>>>>>>>>>>>>>>>>>> S RESULTS\n")
    print("Mean: "+ str(s_mean))
    #print("Average: " + str(s_arr[0]))
    print("Std: " + str(s_std))

    print(">>>>>> S Error by Point\n")

    for i in range(0,total_points_used):
        print("Point "+str(i))
        print("S: " + str(s_describe[i]) + " Mean: " + str(s_mean) + " Error: " + str(s_describe[i]-s_mean))
        
    return s_mean, s_std

if __name__ == '__main__':

    # load camera calibration
cam_mtx,dist,roi,newcam_mtx,inverse_newcam_mtx = load_parameters(savedir, display) 

# load center points from New Camera matrix
cx, cy = get_image_center(savedir)
print("cx:{0}".format(cx) + "cy:{0}".format(cy))

# world center + 9 world points

total_points_used = 10

X_center = -1.5
Y_center = 5.5
Z_center = -85.0
world_points = np.array([[X_center,Y_center,Z_center],
                       [0.0, -22.0, -86.0], 
                       [0.0, 0.0, -86.0],
                       [0.0, 22.0, -86.0],  
                       [15.0, -22.0, -86.0],
                       [15.0, 0.0, -86.0],
                       [15.0, 22.0, -86.0],
                       [-15.0, -22.0, -86.0],
                       [-15.0, 0.0, -86.0],
                       [-15.0, 22.0,-86.0]],dtype=np.float32)


# MANUALLY INPUT THE DETECTED IMAGE COORDINATES HERE

# [u,v] center + 9 Image points
image_points=np.array([[cx,cy],
                       [189, 372],
                       [574,362],
                       [950,347],
                       [206,612],
                       [583,603],
                       [955,596],
                       [189,122],
                       [564,107],
                       [937,98]], dtype=np.float32)

# For Real World Points, calculate Z from d*
world_points = calculate_z_total_points (world_points, X_center, Y_center)


# Get rotatio n and translation_vector from the parameters of the camera, given a set of 2D and 3D points
print("solvePNP")
(success, rotation_vector, translation_vector) = cv2.solvePnP(world_points, image_points, newcam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)

if success:
    print("Sucess:", success)
    print ("Rotation Vector:\n {0}".format(rotation_vector))
    print ("Translation Vector:\n {0}".format(translation_vector))
    
    if writeValues: 
        save_parameters(savedir,rotation_vector, translation_vector,newcam_mtx)
    

# Check the accuracy now
mean, std = calculate_accuracy(world_points, image_points)
print("Mean:{0}".format(mean) + "Std:{0}".format(std))