import sys
import os

import matplotlib.pyplot as plt
import random
from matplotlib import image
import numpy as np
from scipy.spatial import cKDTree as KDTree
import math
import cv2
import pandas as pd

import torch

from Common import DNN
from Common import DataGens
#from DNN import *


from scipy import interpolate


from mpl_toolkits.mplot3d import Axes3D

def FileAsLines(fid):
    lines = []
    with fid as file:
        for line in fid: 
            line = line.split() #or some other preprocessing
            #line = line.strip() #or some other preprocessing
            lines.append(line) #storing everything in memory!

    return lines


def ReadTraj(traj_data_file):
    vTR = {}
    fid = open(traj_data_file, 'r')
    #data = textscan(fid, '%s %f', 1);
    data = fid.readline().split()


    # total number of images
    #n = data{2};
    n = int(data[1])

    vTakenFrame = [0]*n #cell(n, 1);
    vTr = [] #[{}]*n #cell(n, 1);

    for i in range(n):
        vTr.append({})

    for i in range(n):
        data = fid.readline().split()
        data = np.array(data, dtype=np.float32)

        # frame id
        #data = textscan(fid, '%f', 5);
        iFrame = int(data[0])# + 1
    
        # up
        vTr[iFrame]['up'] = data[1:4]
        #print(vTr[iFrame]['up'])
    
        # trajectory length
        nTrjFrame = int(data[4])
    
        # trajectory data
        data = data[5:] # ;
        data = np.reshape(data, (nTrjFrame,6) ).T

        vTr[iFrame]['frame'] = data[0, :]
        vTr[iFrame]['XYZ'] = data[1:4, :]
        vTr[iFrame]['uv'] = data[4:6, :]
        vTakenFrame[iFrame] = iFrame;

    #vTakenFrame = cat(1, vTakenFrame[:]);

    vTR['vTakenFrame'] = vTakenFrame;
    vTR['vTr'] = vTr;
    fid.close()
    return vTR






def Coord2Polar(x,y):
    #z= np.random.random((10,2))
    #x,y = z[0,:], z[1,:]
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)
    
    return t,r
    # np.stack((r,t,)) #np.concatenate((r,t),axis=1)
    #print(r)
    #print(t)


def Polar2Coord(t,r):
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x,y



def RemapRange (value, low1, high1, low2, high2):
  return low2 + (value - low1) * (high2 - low2) / (high1 - low1)










############################################################
### ### ### ### ### ## MAIN EXECUTION ## ### ### ### ### ###
############################################################




if __name__ == "__main__":
    #DNN.current_epoch = 0

    print(os.getcwd())
    #loc = r'H:\fut_loc\20150401_walk_00\traj_prediction.txt'

    folder_path = 'S:\\fut_loc\\train\\20150401_walk_00\\'

    # load calibration
    print('loading calibration file')
    calibfile = folder_path + 'calib_fisheye.txt'

    fid = open(calibfile)
    #data = textscan(fid, '%s %f', 9);

    #imageWidth = data{2}(1);
    data = fid.readline().split()
    imageWidth = int(data[1])

    #imageHeight = data{2}(2);
    data = fid.readline().split()
    imageHeight = int(data[1])

    #focal_x = data{2}(3);
    data = fid.readline().split()
    focal_x = float(data[1])

    #focal_y = data{2}(4);
    data = fid.readline().split()
    focal_y = float(data[1])

    #princ_x = data{2}(5);
    data = fid.readline().split()
    princ_x = float(data[1])

    #princ_y = data{2}(6);
    data = fid.readline().split()
    princ_y = float(data[1])

    #omega = data{2}(7);
    data = fid.readline().split()
    omega = float(data[1])

    #princ_x1 = data{2}(8);
    data = fid.readline().split()
    princ_x1 = float(data[1])

    #princ_y2 = data{2}(9);
    data = fid.readline().split()
    princ_y2 = float(data[1])

    K_data = np.array([[focal_x, 0, princ_x],[ 0, focal_y, princ_y],[ 0, 0, 1]])
    R_rect = np.array([[0.9989,0.0040,0.0466],[-0.0040,1.0000,-0.0002],[-0.0466,0,0.9989]])
    #fclose(fid);

    fid.close()



    # load file list
    print('loading file list file');
    file_list = folder_path + 'im_list.list'
    #fid = open(file_list);
    #data = textscan(fid, '%s');
    #data = fid.readlines()
    with open(file_list) as fid:
        data = fid.read().splitlines()
    vFilename = data;
    #fid.close();





    print('loading trajectory file')
    traj_data_file = folder_path + 'traj_prediction.txt'
    vTR = ReadTraj(traj_data_file)


    #iFrame = 0
    for iFrame in range(10,60):
        tr = vTR['vTr'][iFrame]
        if (len(tr['XYZ'][1]) == 0):
            print('SKIPPING FRAME',iFrame,': Trajectory is empty.')
            continue

        #im = sprintf('%sim/%s', folder_path, vFilename{iFrame});
        im = "{}im\\{}".format(folder_path, vFilename[iFrame])

        if not os.path.isfile(im):
            print('could not find file')
            continue

        img = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB).astype(np.float64)/255.0
        #cv2.imshow('intermediate',intermediate)
        #cv2.waitKey()

        #im = im2double(intermediate); % im2double(imread(im));

        tr_ground_OG = (tr['XYZ'].T - tr['up']).T #bsxfun(@minus, tr['XYZ'], tr['up']);
        #tr_ground_OG[1] *= 0
        #tr_ground_OG[1] -= tr['up'][1]

        #t, r = Coord2Polar(tr_ground_OG[2],tr_ground_OG[0])
        #tr_ground_OG[2], tr_ground_OG[0] = Polar2Coord(t,r)
        tr_ground = K_data @ R_rect @ tr_ground_OG;
        if np.any(tr_ground[2,:]<0):
            tr_ground[:2,:] = np.nan
        #tr_ground(tr_ground(3,:)<0, :) = NaN;
        #tr_ground = bsxfun(@rdivide, tr_ground(1:2,:), tr_ground(3,:));
        
        
        tr_ground = tr_ground[:2] / tr_ground[2]



        
        # set up egocentric image information in log polar space

        t, r = Coord2Polar(tr_ground_OG[2],tr_ground_OG[0])#Coord2Polar(tr['XYZ'][2],tr['XYZ'][0])
        logr = np.log(r)

        #world_forward = 
        img_resized = cv2.resize(img, (int(img.shape[1]*.25), int(img.shape[0]*.25)))*2.0-1.0
        img_channel_swap = np.moveaxis(img_resized,-1,0).astype(np.float32)


        # DISPLAY IMAGES
        fig, axes = plt.subplots(1,4)#, figsize=(18,6))
        # axes = [ax] # only use if there's 1 column

        #newBoundsx = (crowdBoundsX[0], 2*crowdBoundsX[1])
        #fig.set_size_inches(16, 24)
        axes[0].set_title('Traj in image')
        #axes[0].set_xlim(*newBoundsx)
        #axes[0].set_ylim(*crowdBoundsY)
        #axes[0].set_aspect(1)
        axes[0].imshow(img)
        axes[0].plot(tr_ground[0], tr_ground[1], 'r')


        axes[1].set_title('Rectified image')
        #axes[0].set_xlim(*newBoundsx)
        #axes[0].set_ylim(*crowdBoundsY)
        #axes[0].set_aspect(1)

        #homography = K_data @ R_rect @ np.linalg.inv(K_data)
        
        r_y = -tr['up']/np.linalg.norm(tr['up'])
        old_r_z = np.array([0,0,1])
        r_z = old_r_z - (old_r_z@r_y)*r_y
        r_z /= np.linalg.norm(r_z)
        r_x = np.cross(r_y, r_z)

        R_rect_ego = np.stack((r_x,r_y,r_z),axis=0)
        homography = K_data @ R_rect_ego @ np.linalg.inv(K_data)

        img_rectified = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))
        axes[1].imshow(img_rectified)
        #axes[1].plot(tr_ground[0], tr_ground[1], 'r')


        #def intersectPlane(n, p0, l0, l): 
        #    # assuming vectors are all normalized
        #    denom = n @ l
        #    if denom > 1e-6: # hit
        #        p0l0 = p0 - l0
        #        t = p0l0 @ n / denom
        #        return t #(t >= 0)
        #    return 0  # missed

        def intersectPlaneV(n, p0, l0, L):
            print('in')
            plane_offset = p0-l0
            denoms = n @ L
            t = np.zeros(len(denoms))
            intersecting = np.where(denoms > 1e-6)
            d = plane_offset @ n
            result = np.divide( d[None], denoms[intersecting])
            t[intersecting] = result
            return t
        

        depth_img = np.zeros(img_rectified.shape[:2])

        
        depth_pixel_coords = np.array( [ [j+.5,i+.5,1.0] for i in range(img_rectified.shape[0]) for j in range(img_rectified.shape[1]) ], dtype=np.float32)

        #pixel = np.array([j,i,1])
        p_normal = -tr['up']/np.linalg.norm(tr['up'])
        p_origin = -tr['up'] #camera assumed to be at 0,0,0
        e_origin = np.zeros(3) #zero vector
        e_rays = R_rect.T @ np.linalg.inv(K_data) @ depth_pixel_coords.T #+0
        e_rays /= np.linalg.norm(e_rays,axis=0)
        print('norm:', np.linalg.norm(e_rays[:,100]))

        intplane = lambda l : intersectPlane(p_normal,p_origin,e_origin,l)
        #vfunc = np.vectorize(intplane)
        #depths = np.apply_along_axis(intplane, 0, e_rays)
        depths = intersectPlaneV(p_normal,p_origin,e_origin,e_rays)
        points = e_rays * depths[None]

        t2, r2 = Coord2Polar(points[2],points[0])
        print(r2.max())
        r2 = np.clip(r2,0,100)
        print(r2.max())
        rnorm = r2 / r2.max()
        t2 = np.clip(t2,-np.pi/4,np.pi/4)
        tnorm = t2 / t2.max()


        #points = np.multiply(e_rays,depths)

        #depths = intersectPlane(p_normal,p_origin,e_origin,e_ray)
        rad_img = np.reshape(tnorm, depth_img.shape)
        #depth_img = np.reshape(depths, depth_img.shape)
        #print(depth_img.max())
        #print(depth_img.min())
        #depth_img = np.clip(depth_img,0,1)

        
        axes[2].set_title('Plane image')
        axes[2].imshow(rad_img)


        
        axes[3].set_title('Traj in EgoMap')
        axes[3].set_xlim((-2*np.pi/3, 2*np.pi/3))
        #axes[0].set_ylim(*crowdBoundsY)
        axes[3].set_aspect(1)
        #axes[1].imshow(img)

        
        axes[3].plot(t, logr, 'r')


        plt.show()

       








        img_height = 64

        minR = -.5
        maxR = 5#4.5
        maxT = 2*np.pi/3
        minT = -maxT

        aspect_ratio = (maxT-minT)/(maxR-minR)
        ego_pixel_shape = (img_height,int(img_height*aspect_ratio)) # y,x | vert,horz
        big_ego_pixel_shape = (img.shape[0],int(img.shape[0]*aspect_ratio)) # y,x | vert,horz

        ego_r2pix = lambda x : RemapRange(x, minR,maxR, 0,                  ego_pixel_shape[0]  )
        ego_t2pix = lambda x : RemapRange(x, minT,maxT, 0,                  ego_pixel_shape[1]  )

        
        ego_pix2r = lambda x : RemapRange(x, 0, ego_pixel_shape[0], minR,maxR   )
        ego_pix2t = lambda x : RemapRange(x,0,ego_pixel_shape[1], minT,maxT  )



        big_ego_pix2r = lambda x : RemapRange(x, 0, big_ego_pixel_shape[0], minR,maxR   )
        big_ego_pix2t = lambda x : RemapRange(x,0,big_ego_pixel_shape[1], minT,maxT  )

        RecenterDataForwardWithShape = lambda x, shape : RemapRange(x,0,max(shape[0],shape[1]),-1,1)
        RecenterDataForwardWithShapeAndScale = lambda x, shape, scale : RemapRange(x,0,max(shape[0],shape[1]),-scale,scale)
        RecenterDataBackwardWithShape = lambda x, shape : RemapRange(x,-1,1,0,max(shape[0],shape[1]))
        RecenterTrajDataForward = lambda x : RecenterDataForwardWithShape(x,ego_pixel_shape)
        RecenterTrajDataForward2 = lambda x : RecenterDataForwardWithShapeAndScale(x,ego_pixel_shape,1)

        
        RecenterDataBackwardWithShapeAndScale = lambda x, shape, scale : RemapRange(x,-scale,scale,0,max(shape[0],shape[1]))
        RecenterFieldDataBackward = lambda x : RecenterDataBackwardWithShapeAndScale(x,ego_pixel_shape,1)
        

        tpix = ego_t2pix(t) #- 15
        logrpix = ego_r2pix(logr)

                
        future_trajectory = {}

        for k in range(1):#48
            future_trajectory[k] = []
            for i in range(len(logrpix)):
                future_trajectory[k].append( (tpix[i], logrpix[i]) )




        fig, ax = plt.subplots(1,1)#, figsize=(36,6))
        axes = [ax]

        boundsX = (0,ego_pixel_shape[1])
        boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
        axes[0].set_xlim(*boundsX)
        axes[0].set_ylim(*boundsY)
        axes[0].set_aspect(1)

        for traj in future_trajectory.values():
            trajnp = np.array(traj)
            axes[0].plot(trajnp[:,0], trajnp[:,1], 'r')
        
        plt.show()


         #def EgoWarp (img_in, ego_shape, K, R, r_map, t_map,):
            # need to go from ego map,
            # backwards into world space,
            # backwards into camera space, 
            # backwards into pixel space,
            # and get the value of the pixel there.

            #def warp_image(img, A, output_size):
        if(True):
            all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(big_ego_pixel_shape[0]) for j in range(big_ego_pixel_shape[1]) ], dtype=np.float32)
            ##all_pixel_coords_xformed = np.zeros(all_pixel_coords.shape)
            all_pixel_coords[:,0] = big_ego_pix2t(all_pixel_coords[:,0])
            all_pixel_coords[:,1] = np.exp(big_ego_pix2r(all_pixel_coords[:,1]))
            z, x = Polar2Coord(all_pixel_coords[:,0],all_pixel_coords[:,1])

            coords_3D = np.zeros((len(z),3))
            coords_3D[:,0] = -x
            coords_3D[:,2] = -z
        
            coords_3D -= tr['up'].T
            #coords_3D[:,1] *= -1

            pixels = K_data @ R_rect @ coords_3D.T
            #pixels[:,:] /= pixels[2,:]

            rowmaj_pixels = np.zeros(pixels.shape)
            rowmaj_pixels[0] = pixels[1]
            rowmaj_pixels[1] = pixels[0]

            #A_aug = np.array([[A[1,1],A[1,0],A[1,2]],[A[0,1],A[0,0],A[0,2]]]) # 2x3 because last row holds no information for affine transform, swapping x and y locations for the interpolation calculation
            #pixRange = range(output_size[0] * output_size[1]) # should reduce the tuple with lambda function
            #ego_shape = ego_pixel_shape
            #points = np.array( [    [i for i in range(ego_shape[0]) for j in range(ego_shape[1])],\
            #                        [j for i in range(ego_shape[0]) for j in range(ego_shape[1])],\
            #                        [1 for i in range(ego_shape[0]) for j in range(ego_shape[1])]     ]) # col \/row major pixel representation (undo with x.reshape(2,3))
            #transformedPoints = np.matmul(A_aug,points)
            #return (interpolate.interpn((range(img.shape[0]),range(img.shape[1])),img,transformedPoints.transpose(),method = 'linear',bounds_error = False, fill_value = 0)).reshape(output_size[0], output_size[1])
        
        
            #all_pixel_coords2 = np.array( [ [j+.5,i+.5] for i in range(img.shape[0]) for j in range(img.shape[1]) ], dtype=np.float32)
            #alternate = np.zeros(all_pixel_coords2.shape)
            #alternate[:,0] = all_pixel_coords2[:,1]
            #alternate[:,1] = all_pixel_coords2[:,0]
            img2 = interpolate.interpn((range(img.shape[0]),range(img.shape[1])),img, rowmaj_pixels[:2].T , method = 'linear',bounds_error = False, fill_value = 0).reshape(big_ego_pixel_shape[0], big_ego_pixel_shape[1],3)



            #test_t, test_logr  = np.meshgrid(np.linspace(-maxT,maxT,30),np.linspace(0,2,30))
            #test_t = test_t.flatten()
            #test_logr = test_logr.flatten()
            #print(test_t)
            #print(test_logr)
            #test_r = np.exp(test_logr)
            ##test_t, test_r  = np.meshgrid(np.linspace(-1,1,30),np.linspace(0,2,30))
            ##test_t = test_t.flatten()
            ##test_r = test_r.flatten()
            #z,x = Polar2Coord(test_t,test_r) # x,z ?
            #print(z)
            #print(x)

            #print('actual z:', test_r[0], '*', 'np.cos(',test_t[0],') =', test_r[0] * np.cos(test_t[0]))
            #print('actual x:', test_r[0], '*', 'np.sin(',test_t[0],') =', test_r[0] * np.sin(test_t[0]))

            #coords_3D = np.zeros((len(z),3))
            #coords_3D[:,0] = x
            #coords_3D[:,2] = z

            #xoffset = img.shape[1]
            #x, y  = np.meshgrid(np.linspace(0,30,31),np.linspace(0,30,31)) # pixels
            #x = x.flatten()
            #y = y.flatten()
            #ones = np.ones(len(x))
            #pixels = np.stack((x,y,ones),axis=0)





            #fig = plt.figure()
            #ax = fig.add_subplot(111,projection='3d')
            ##ax.scatter(test_t, test_logr, c='r', marker='o')
            #ax.scatter(coords_3D[:,0], coords_3D[:,2], coords_3D[:,1], c='r', marker='o')
            #ax.set_xlabel('X Label')
            #ax.set_ylabel('Z Label')
            #ax.set_zlabel('Y Label')

            #plt.show()








            fig, axes = plt.subplots(1,3)#, figsize=(18,6))
            # axes = [ax] # only use if there's 1 column

            #newBoundsx = (crowdBoundsX[0], 2*crowdBoundsX[1])
            #fig.set_size_inches(16, 24)
            axes[0].set_title('Traj in image')
            #axes[0].set_xlim(*newBoundsx)
            #axes[0].set_ylim(*crowdBoundsY)
            #axes[0].set_aspect(1)
            axes[0].imshow(img)
            axes[0].plot(tr_ground[0], tr_ground[1], 'r')

        
            axes[1].set_title('Traj in EgoMap')
            axes[1].set_xlim((-2*np.pi/3, 2*np.pi/3))
            #axes[0].set_ylim(*crowdBoundsY)
            axes[1].set_aspect(1)
            #axes[1].imshow(img)
        
        
            axes[1].plot(t, logr, 'r')
        
            axes[2].set_title('warpd')
            axes[2].imshow(img2)

            plt.show()
            #cv2.imshow('intermediate',intermediate)
            #cv2.waitKey()

            #figure(1), clf
            #imshow(im); hold on;
            #plot(tr_ground(1,:), tr_ground(2,:), 'r', 'LineWidth', 2);





    #fid = open(loc, 'r')
    #lines = []
    #line = fid.readline().split()

    #with fid as file:
    #    for line in fid: 
    #        line = line.split() #or some other preprocessing
    #        #line = line.strip() #or some other preprocessing
    #        lines.append(line) #storing everything in memory!
    #with open(loc, 'r') as fid:
    #    for line in fid: 
    #        line = line.split() #or some other preprocessing
    #        #line = line.strip() #or some other preprocessing
    #        lines.append(line) #storing everything in memory!
            
    print('\n\n\n')
    print(type(vTR))
    print('\n\n\n')
    print(type(vTR['vTr']))
    print('\n\n\n')
    print(vTR['vTr'][0]['up'])

    #data = pd.read_csv(loc, sep=" ", header=None)
    #data.columns = ["a", "b", "c", "etc
    #print(data)


    #for i in range(3):
    #    print('\n\n\n')
    #    print(lines[i])

    #filecontents = np.genfromtxt(loc,dtype='str')
    #filecontents = np.loadtxt(loc)
    #print(filecontents[:10])
    #print(fid.read())


    print(sys.argv)

    s = pd.Series([1,3,5])
    print(s)

