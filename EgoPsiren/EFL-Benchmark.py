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
from Common import DataReader
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

    folder_path = 'S:\\fut_loc\\test\\20150402_grocery\\'#'S:\\synth_marketplace_random2020\\train\\acofre20167850_t38_p18\\' #acofre20167850_t36_p9\\'    #'S:\\fut_loc\\train\\20150401_walk_00\\' #'S:\\fut_loc\\test\\20150402_walk\\'#20150419_ikea #       

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


    initial_offset = 38#15#45
    for iFrame in range(initial_offset,69):

        print("GETTING", iFrame)
        tr = vTR['vTr'][iFrame]
        frames = tr['frame']
        print(frames)
        if (len(tr['XYZ'][1]) == 0):
            print('SKIPPING FRAME',iFrame,': Trajectory is empty.')
            continue

        #im = sprintf('%sim/%s', folder_path, vFilename{iFrame});
        im = "{}im\\{}".format(folder_path, vFilename[iFrame])
        #disp = "{}disparity\\{}{}".format(folder_path, vFilename[iFrame],'.disp.txt')

        if not os.path.isfile(im):
            print('could not find file')
            continue
        #if not os.path.isfile(disp):
        #    print('could not find file')
        #    continue

        img = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB).astype(np.float64)/255.0

        #disp_img = np.genfromtxt(disp, delimiter=',')[:,:-1] #np.loadtxt(disp, delimiter=',')
        disp_img = np.zeros((1,1))
        
        
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


        r_y = -tr['up']/np.linalg.norm(tr['up'])

        v = tr['XYZ'][:,0]/np.linalg.norm(tr['XYZ'][:,0])
        if v @ np.array([0,0,1]) > .2:
            old_r_z = v
        else:
            print("\tSkipping because of alignment severity")
            continue
        old_r_z = np.array([0,0,1])
        r_z = old_r_z - (old_r_z@r_y)*r_y
        r_z /= np.linalg.norm(r_z)
        r_x = np.cross(r_y, r_z)

        R_rect_ego = np.stack((r_x,r_y,r_z),axis=0)
        
        # TODO: does R_rect need to be here?
        homography = K_data @ R_rect @ R_rect_ego  @ R_rect.T @ np.linalg.inv(K_data)




        
        # set up egocentric image information in log polar space
        
        tr_ground_ALIGNED = R_rect_ego @ tr_ground_OG # ALIGN CAMERA SPACE GROUND PLANE TO "WORLD SPACE"
        #tr_ground_ALIGNED = R_rect_ego @ tr['XYZ'].T - R_rect_ege @ tr['up'] # ALIGN CAMERA SPACE GROUND PLANE TO "WORLD SPACE"
        
        t, r = Coord2Polar(tr_ground_ALIGNED[2],tr_ground_ALIGNED[0])#Coord2Polar(tr['XYZ'][2],tr['XYZ'][0])
        logr = np.log(r)

        #world_forward = 
        img_resized = cv2.resize(img, (int(img.shape[1]*.25), int(img.shape[0]*.25)))*2.0-1.0
        img_channel_swap = np.moveaxis(img_resized,-1,0).astype(np.float32)

        if False:
            # DISPLAY IMAGES
            fig, axes = plt.subplots(1,4)#, figsize=(18,6))
            # axes = [ax] # only use if there's 1 column

            #newBoundsx = (crowdBoundsX[0], 2*crowdBoundsX[1])
            #fig.set_size_inches(16, 24)
            axes[0].set_title('Traj in image')
            #axes[0].set_xlim(*newBoundsx)
            #axes[0].set_ylim(*crowdBoundsY)
        
            axes[0].set_xlim(0,img.shape[1])
            axes[0].set_ylim(img.shape[0],0)
            axes[0].set_aspect(1)
            axes[0].imshow(img)
            axes[0].plot(tr_ground[0], tr_ground[1], 'r')


            axes[2].set_title('Rectified image')
            #axes[0].set_xlim(*newBoundsx)
            #axes[0].set_ylim(*crowdBoundsY)
            #axes[0].set_aspect(1)

            #homography = K_data @ R_rect @ np.linalg.inv(K_data)
        
        

            img_rectified = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))
            axes[2].imshow(img_rectified)
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
                d = - plane_offset @ n
                result = np.divide( d[None], denoms[intersecting])
                t[intersecting] = result
                return t
        

            depth_img = np.zeros(img_rectified.shape[:2])

        
            depth_pixel_coords = np.array( [ [j+.5,i+.5,1.0] for i in range(img_rectified.shape[0]) for j in range(img_rectified.shape[1]) ], dtype=np.float32)

            #pixel = np.array([j,i,1])
            p_normal = tr['up']/np.linalg.norm(tr['up'])
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
            depth_img = np.reshape(depths, depth_img.shape)
            #print(depth_img.max())
            #print(depth_img.min())
            depth_img = np.clip(depth_img,0,1)

        
            axes[3].set_title('Plane image')
            axes[3].imshow(depth_img, cmap='jet')


        
            #axes[3].set_title('Traj in EgoMap')
            #axes[3].set_xlim((-2*np.pi/3, 2*np.pi/3))
            ##axes[0].set_ylim(*crowdBoundsY)
            #axes[3].set_aspect(1)
            ##axes[1].imshow(img)

        
            #axes[3].plot(t, logr, 'r')
            axes[1].set_title('Depth Image')
            tempval = axes[1].imshow(disp_img)
            cax = fig.add_axes([.3, .95, .4, .05])
            fig.colorbar(tempval, cax, orientation='horizontal')


            plt.show()

       

        # DRAW 3D GRAPH
        #ax = plt.axes(projection='3d')
        #ax.scatter(frames['C'].T[0],frames['C'].T[1],frames['C'].T[2], s=1)
        #keys = np.sort(np.array(list(frames.keys())))

        
        #start = frames[valid_frames[0]]['C']
        #ax.plot(start[0],start[1],start[2],'bo',markerSize=10)
        
        #scale = .5
        #if True:
        #    R = R_rect
        #    center = np.zeros(3)

        #    plane_origin = -tr['up']


        #    x_delta = np.zeros(2)
        #    y_delta = np.zeros(2)
        #    z_delta = np.zeros(2)
            
        #    x_delta[0] = center[0]
        #    y_delta[0] = center[1]
        #    z_delta[0] = center[2]
            
        #    x_delta[1] = center[0] + plane_origin[0]
        #    y_delta[1] = center[1] + plane_origin[1]
        #    z_delta[1] = center[2] + plane_origin[2]
        #    ax.plot(x_delta,y_delta,z_delta,'k--',linewidth=1)




        
        #    x_delta = np.zeros(2)
        #    y_delta = np.zeros(2)
        #    z_delta = np.zeros(2)
            
        #    #center = frames[key]['C']
        #    x_delta[0] = center[0]
        #    y_delta[0] = center[1]
        #    z_delta[0] = center[2]
            

        #    #print(R)
        #    axis = R[0] * scale
        #    #axis2 = R[0]
        #    #axis *= scale
        #    #axis2 *= scale
        #    x_delta[1] = center[0] + axis[0]
        #    y_delta[1] = center[1] + axis[1]
        #    z_delta[1] = center[2] + axis[2]
        #    ax.plot(x_delta,y_delta,z_delta, color='#880000',linewidth=1)

        
        #    x_delta = np.zeros(2)
        #    y_delta = np.zeros(2)
        #    z_delta = np.zeros(2)
            
        #    x_delta[0] = center[0]
        #    y_delta[0] = center[1]
        #    z_delta[0] = center[2]
            
        #    axis = R[1] * scale
        #    x_delta[1] = center[0] + axis[0]
        #    y_delta[1] = center[1] + axis[1]
        #    z_delta[1] = center[2] + axis[2]
        #    ax.plot(x_delta,y_delta,z_delta,'#008800',linewidth=1)


        #    x_delta = np.zeros(2)
        #    y_delta = np.zeros(2)
        #    z_delta = np.zeros(2)
            
        #    x_delta[0] = center[0]
        #    y_delta[0] = center[1]
        #    z_delta[0] = center[2]
            
            
        #    axis = R[2] * scale
        #    x_delta[1] = center[0] + axis[0]
        #    y_delta[1] = center[1] + axis[1]
        #    z_delta[1] = center[2] + axis[2]
        #    ax.plot(x_delta,y_delta,z_delta,'#000088',linewidth=1)

        #if True:
        #    R = R_rect_ego
        #    center = np.zeros(3)
        
        #    x_delta = np.zeros(2)
        #    y_delta = np.zeros(2)
        #    z_delta = np.zeros(2)
            
        #    #center = frames[key]['C']
        #    x_delta[0] = center[0]
        #    y_delta[0] = center[1]
        #    z_delta[0] = center[2]
            

        #    #print(R)
        #    axis = R[0] * scale
        #    #axis2 = R[0]
        #    #axis *= scale
        #    #axis2 *= scale
        #    x_delta[1] = center[0] + axis[0]
        #    y_delta[1] = center[1] + axis[1]
        #    z_delta[1] = center[2] + axis[2]
        #    ax.plot(x_delta,y_delta,z_delta,color='r',linewidth=2)

        
        #    x_delta = np.zeros(2)
        #    y_delta = np.zeros(2)
        #    z_delta = np.zeros(2)
            
        #    x_delta[0] = center[0]
        #    y_delta[0] = center[1]
        #    z_delta[0] = center[2]
            
        #    axis = R[1] * scale
        #    x_delta[1] = center[0] + axis[0]
        #    y_delta[1] = center[1] + axis[1]
        #    z_delta[1] = center[2] + axis[2]
        #    ax.plot(x_delta,y_delta,z_delta,'g',linewidth=2)


        #    x_delta = np.zeros(2)
        #    y_delta = np.zeros(2)
        #    z_delta = np.zeros(2)
            
        #    x_delta[0] = center[0]
        #    y_delta[0] = center[1]
        #    z_delta[0] = center[2]
            
            
        #    axis = R[2] * scale
        #    x_delta[1] = center[0] + axis[0]
        #    y_delta[1] = center[1] + axis[1]
        #    z_delta[1] = center[2] + axis[2]
        #    ax.plot(x_delta,y_delta,z_delta,'b',linewidth=2)

        ##n_frames = tr_ground_OG.shape[1]
        ##x = np.zeros(n_frames)
        ##y = np.zeros(n_frames)
        ##z = np.zeros(n_frames)
        ##count = 0
        ##for i in range(n_frames):
        ##    C = tr_ground_OG[:,i]
        ##    x[count] = C[0]
        ##    y[count] = C[1]
        ##    z[count] = C[2]
        ##    count += 1


        #ax.plot(tr_ground_OG[0],tr_ground_OG[1],tr_ground_OG[2],'ko')
        #ax.plot(tr_ground_ALIGNED[0],tr_ground_ALIGNED[1],tr_ground_ALIGNED[2],'bo')
        
        
        
        
        #max_range = np.array([tr_ground_OG[0].max()-tr_ground_OG[0].min(), tr_ground_OG[1].max()-tr_ground_OG[1].min(), tr_ground_OG[2].max()-tr_ground_OG[2].min()]).max() / 2.0

        #mean_x = tr_ground_OG[0].mean()
        #mean_y = tr_ground_OG[1].mean()
        #mean_z = tr_ground_OG[2].mean()
        #ax.set_xlim(mean_x - max_range, mean_x + max_range)
        #ax.set_ylim(mean_y - max_range, mean_y + max_range)
        #ax.set_zlim(mean_z - max_range, mean_z + max_range)





        #ax.set_xlabel('X axis')
        #ax.set_ylabel('Y axis')
        #ax.set_zlabel('Z axis')
        #plt.show()







        img_height = 192

        minR = -.5
        maxR = 4#4.5
        maxT = np.pi/3 #2*np.pi/3
        minT = -maxT

        aspect_ratio = 1# 3/4 #(2*maxT-2*minT)/(maxR-minR)
        ego_pixel_shape = (img_height,int(img_height*aspect_ratio)) # y,x | vert,horz
        big_ego_pixel_shape = (ego_pixel_shape[0] * 5, ego_pixel_shape[1]*5)#(img.shape[0],int(img.shape[0]*aspect_ratio)) # y,x | vert,horz
        small_ego_pixel_shape = (ego_pixel_shape[0] // 8, ego_pixel_shape[1] // 8)#(img.shape[0],int(img.shape[0]*aspect_ratio)) # y,x | vert,horz

        ego_r2pix = lambda x : RemapRange(x, minR,maxR, 0,                  ego_pixel_shape[0]  )
        ego_t2pix = lambda x : RemapRange(x, minT,maxT, 0,                  ego_pixel_shape[1]  )

        
        ego_pix2r = lambda x : RemapRange(x, 0, ego_pixel_shape[0], minR,maxR   )
        ego_pix2t = lambda x : RemapRange(x,0,ego_pixel_shape[1], minT,maxT  )

        

        big_ego_pix2r = lambda x : RemapRange(x, 0, big_ego_pixel_shape[0], minR,maxR   )
        big_ego_pix2t = lambda x : RemapRange(x,0,big_ego_pixel_shape[1], minT,maxT  )

        small_ego_pix2r = lambda x : RemapRange(x, 0, small_ego_pixel_shape[0], minR,maxR   )
        small_ego_pix2t = lambda x : RemapRange(x,0,small_ego_pixel_shape[1], minT,maxT  )

        
        big_ego_r2pix = lambda x : RemapRange(x, minR,maxR, 0,                  big_ego_pixel_shape[0]  )
        big_ego_t2pix = lambda x : RemapRange(x, minT,maxT, 0,                  big_ego_pixel_shape[1]  )


        small_ego_r2pix = lambda x : RemapRange(x, minR,maxR, 0,                  small_ego_pixel_shape[0]  )
        small_ego_t2pix = lambda x : RemapRange(x, minT,maxT, 0,                  small_ego_pixel_shape[1]  )


        RecenterDataForwardWithShape = lambda x, shape : RemapRange(x,0,max(shape[0],shape[1]),-1,1)
        RecenterDataForwardWithShapeAndScale = lambda x, shape, scale : RemapRange(x,0,max(shape[0],shape[1]),-scale,scale)
        RecenterDataBackwardWithShape = lambda x, shape : RemapRange(x,-1,1,0,max(shape[0],shape[1]))
        RecenterTrajDataForward = lambda x : RecenterDataForwardWithShape(x,ego_pixel_shape)
        RecenterTrajDataForward2 = lambda x : RecenterDataForwardWithShapeAndScale(x,ego_pixel_shape,1)

        
        RecenterDataBackwardWithShapeAndScale = lambda x, shape, scale : RemapRange(x,-scale,scale,0,max(shape[0],shape[1]))
        RecenterFieldDataBackward = lambda x : RecenterDataBackwardWithShapeAndScale(x,ego_pixel_shape,1)


        #RecenterTrajDataBackwardBIG = RecenterDataBackwardWithShapeAndScale()
        

        tpix = ego_t2pix(t) #- 15
        logrpix = ego_r2pix(logr)

                
        future_trajectory = {}

        for k in range(1):#48
            future_trajectory[k] = []
            for i in range(len(logrpix)):
                future_trajectory[k].append( (tpix[i], logrpix[i]) )




        #fig, ax = plt.subplots(1,1)#, figsize=(36,6))
        #axes = [ax]

        #boundsX = (0,ego_pixel_shape[1])
        #boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
        #axes[0].set_xlim(*boundsX)
        #axes[0].set_ylim(*boundsY)
        #axes[0].set_aspect(1)

        #for traj in future_trajectory.values():
        #    trajnp = np.array(traj)
        #    axes[0].plot(trajnp[:,0], trajnp[:,1], 'r')
        
        #plt.show()


         #def EgoWarp (img_in, ego_shape, K, R, r_map, t_map,):
            # need to go from ego map,
            # backwards into world space,
            # backwards into camera space, 
            # backwards into pixel space,
            # and get the value of the pixel there.

            #def warp_image(img, A, output_size):
        if(True):


            #def GenEgoRetinalMap(in_img, out_img_shape, pix2t, pix2r, R_Cam2World, V_Floor2Cam, ):

            #    all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(out_img_shape[0]) for j in range(out_img_shape[1]) ], dtype=np.float32)
            #    all_pixel_coords[:,0] = big_ego_pix2t(all_pixel_coords[:,0])
            #    all_pixel_coords[:,1] =  np.exp(big_ego_pix2r(all_pixel_coords[:,1]))
            #    z, x = Polar2Coord(all_pixel_coords[:,0],all_pixel_coords[:,1])
            #    coords_3D = np.zeros((len(z),3))
            #    coords_3D[:,0] = x
            #    coords_3D[:,2] = z
        
            #    coords_3D = (R_rect_ego.T @ coords_3D.T).T # ALIGN "WORLD-SPACE" GROUND PLANE TO CAMERA SPACE
            #    coords_3D -= tr['up'].T # SHIFT PLANE TO CORRECT LOCATION RELATIVE TO CAMERA




            all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(big_ego_pixel_shape[0]) for j in range(big_ego_pixel_shape[1]) ], dtype=np.float32)
            #all_pixel_coords[:,1] = np.flip(all_pixel_coords[:,1])
            print(all_pixel_coords[:,0].max())
            print(all_pixel_coords[:,1].max())
            print(all_pixel_coords[:,0].min())
            print(all_pixel_coords[:,1].min())
            ##all_pixel_coords_xformed = np.zeros(all_pixel_coords.shape)
            all_pixel_coords[:,0] = big_ego_pix2t(all_pixel_coords[:,0])
            print(all_pixel_coords[:,0].max())
            print(all_pixel_coords[:,0].min())
            all_pixel_coords[:,1] = big_ego_pix2r(all_pixel_coords[:,1])
            print(all_pixel_coords[:,1].max())
            print(all_pixel_coords[:,1].min())
            all_pixel_coords[:,1] = np.exp(all_pixel_coords[:,1])
            print(all_pixel_coords[:,1].max())
            print(all_pixel_coords[:,1].min())
            z, x = Polar2Coord(all_pixel_coords[:,0],all_pixel_coords[:,1])

            coords_3D = np.zeros((len(z),3))
            coords_3D[:,0] = x
            coords_3D[:,2] = z
        
            coords_3D = (R_rect_ego.T @ coords_3D.T).T # ALIGN "WORLD-SPACE" GROUND PLANE TO CAMERA SPACE
            coords_3D -= tr['up'].T # SHIFT PLANE TO CORRECT LOCATION RELATIVE TO CAMERA

            



            #coord_value = DataGens.Coords2ValueFastWS(all_pixel_coords_xformed,{0:test_ws_trajectory},None,None,stddev=.5)
            
            e_origin = tr['up'] #camera assumed to be at 0,0,0
            #e_origin = np.zeros(3) #zero vector


            #coords_3D[:,1] *= -1

            pixels = K_data @ R_rect @ coords_3D.T
            pixels /= pixels[2]
            #pixels[:,:] /= pixels[2,:]

            rowmaj_pixels = np.zeros(pixels.shape)
            rowmaj_pixels[0] = pixels[1]
            rowmaj_pixels[1] = pixels[0]


            img2 =      interpolate.interpn((range(img.shape[0]),range(img.shape[1])),          img, rowmaj_pixels[:2].T , method = 'linear',bounds_error = False, fill_value = 0).reshape(big_ego_pixel_shape[0], big_ego_pixel_shape[1],3)



            K_data2 = np.copy(K_data)
            K_data2[0,2] = disp_img.shape[1]/2
            K_data2[1,2] = disp_img.shape[0]/2

            
            K_data2[0,0] = disp_img.shape[1] * K_data[0,0]/img.shape[1]
            K_data2[1,1] = disp_img.shape[0] * K_data[1,1]/img.shape[0]

            pixels2 = K_data2 @ R_rect @ coords_3D.T
            pixels2 /= pixels2[2]

            rowmaj_pixels2 = np.zeros(pixels.shape)
            rowmaj_pixels2[0] = pixels2[1]
            rowmaj_pixels2[1] = pixels2[0]

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
            
            
            
            
            disp_img2 = interpolate.interpn((range(disp_img.shape[0]),range(disp_img.shape[1])),disp_img, rowmaj_pixels2[:2].T , method = 'linear',bounds_error = False, fill_value = 0).reshape(big_ego_pixel_shape[0], big_ego_pixel_shape[1])



            test_t, test_logr  = np.meshgrid(np.linspace(minT,maxT,5),np.linspace(0,5,6))
            test_t = test_t.flatten()
            test_logr = test_logr.flatten()
            #print(test_t)
            #print(test_logr)
            test_r = np.exp(test_logr)
            ##test_t, test_r  = np.meshgrid(np.linspace(-1,1,30),np.linspace(0,2,30))
            ##test_t = test_t.flatten()
            ##test_r = test_r.flatten()
            z,x = Polar2Coord(test_t,test_r) # x,z ?
            #print(z)
            #print(x)

            #print('actual z:', test_r[0], '*', 'np.cos(',test_t[0],') =', test_r[0] * np.cos(test_t[0]))
            #print('actual x:', test_r[0], '*', 'np.sin(',test_t[0],') =', test_r[0] * np.sin(test_t[0]))

            coords_3D = np.zeros((len(z),3))
            coords_3D[:,0] = x
            coords_3D[:,2] = z
            
            
            coords_3D = (R_rect_ego.T @ coords_3D.T).T # ALIGN "WORLD-SPACE" GROUND PLANE TO CAMERA SPACE
            coords_3D -= tr['up'].T # SHIFT PLANE TO CORRECT LOCATION RELATIVE TO CAMERA
            
            pixels = K_data @ R_rect @ coords_3D.T
            pixels /= pixels[2]



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


            
            # Let's try to get a ground truth image
            all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(big_ego_pixel_shape[0]) for j in range(big_ego_pixel_shape[1]) ], dtype=np.float32)
            #coord_value = np.zeros((all_pixel_coords.shape[0]))

            multiplier = all_pixel_coords


            all_pixel_coords_xformed = np.zeros(all_pixel_coords.shape).astype(np.float32)
            all_pixel_coords_xformed[:,0] = big_ego_pix2t(all_pixel_coords[:,0])
            all_pixel_coords_xformed[:,1] = np.exp(big_ego_pix2r(all_pixel_coords[:,1]))

            all_pixel_coords_xformed = np.array(DataReader.Polar2Coord(all_pixel_coords_xformed[:,0],all_pixel_coords_xformed[:,1])).T


            small_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(small_ego_pixel_shape[0]) for j in range(small_ego_pixel_shape[1]) ], dtype=np.float32)
            #coord_value = np.zeros((all_pixel_coords.shape[0]))

            smultiplier = small_pixel_coords


            small_pixel_coords_xformed = np.zeros(small_pixel_coords.shape).astype(np.float32)
            small_pixel_coords_xformed[:,0] = small_ego_pix2t(small_pixel_coords[:,0])
            small_pixel_coords_xformed[:,1] = np.exp(small_ego_pix2r(small_pixel_coords[:,1]))

            small_pixel_coords_xformed = np.array(DataReader.Polar2Coord(small_pixel_coords_xformed[:,0],small_pixel_coords_xformed[:,1])).T
            
        
            test_ws_trajectory = []
            test_pix_trajectory = []
            test_pix_trajectory_small = []
            for pix in future_trajectory[0]:
                # print(pix)
                
                pix0 = big_ego_t2pix( ego_pix2t(pix[0]) )
                pix1 = big_ego_r2pix( ego_pix2r(pix[1]) )

                
                pix0_small = small_ego_t2pix( ego_pix2t(pix[0]) )
                pix1_small = small_ego_r2pix( ego_pix2r(pix[1]) )

                if (pix0 < 0 or pix0 > big_ego_pixel_shape[1] or pix1 < 0 or pix1 > big_ego_pixel_shape[0]): # outside ego map
                        break
                #test_pix_trajectory.append( (pix[0], pix[1]) ) # t is horizontal axis, logr is vertical
                #PIXEL_TRAJECTORY_DICTIONARY[dictionary_index].append( (pix[0], pix[1]) ) # t is horizontal axis, logr is vertical
                newpoint = ( DataReader.Polar2Coord( big_ego_pix2t(pix0),np.exp(big_ego_pix2r(pix1)) ) )
                test_ws_trajectory.append( newpoint )
                test_pix_trajectory.append( (pix0,pix1) )
                test_pix_trajectory_small.append( (pix0_small, pix1_small))

            if len(test_ws_trajectory) < 2:
                print("trajectory invalid (leaves egomap early)")
                continue


            global_std_dev = .5
            coord_value = DataGens.Coords2ValueFastWS_NEURIPS(all_pixel_coords_xformed,{0:test_ws_trajectory},None,None,stddev=global_std_dev, dstddev= 1)






            small_coord_derivative = DataGens.Coords2ValueFastWS_NEURIPS_DERIVATIVE(small_pixel_coords_xformed,{0:test_ws_trajectory},None,None,stddev=global_std_dev, dstddev= 1)

            trajectory_deriv = DataGens.Coords2ValueFastWS_NEURIPS_DERIVATIVE(np.array(test_ws_trajectory),{0:test_ws_trajectory},None,None,stddev=global_std_dev, dstddev= 1)





            ws_traj = np.array(test_ws_trajectory)
            maxX = np.max(ws_traj[:,0])+2
            minX = np.min(ws_traj[:,0])
            maxY = np.max(ws_traj[:,1])+1
            minY = np.min(ws_traj[:,1])-1

            #np.linspace(minX,maxX,)
            #-4,4
            all_ws_coords = np.array( [ [j+.5,i+.5] for i in np.linspace(-1,7,240) for j in np.linspace(0,60,240) ], dtype=np.float32)
            all_ws_value = DataGens.Coords2ValueFastWS_NEURIPS(all_ws_coords,{0:test_ws_trajectory},None,None,stddev=global_std_dev, dstddev= 1)


            all_ws_coords_small = np.array( [ [j+.5,i+.5] for i in np.linspace(-1,7,24) for j in np.linspace(0,60,24) ], dtype=np.float32)
            all_ws_derivative = DataGens.Coords2ValueFastWS_NEURIPS_DERIVATIVE(all_ws_coords_small,{0:test_ws_trajectory},None,None,stddev=global_std_dev, dstddev= 1)


            #xws, yws = np.meshgrid(, )

            #pixels = np.stack((xws,yws),axis=2)



            if True:
                fig, ax = plt.subplots(1,1)
                axes = [ax]
            
                #plt.imsave('bigegomap.png',img2)
                axes[0].set_title('EgoRetinal Map')
            
                boundsX = (0,big_ego_pixel_shape[1])
                boundsY = (0,big_ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
            
                axes[0].set_xlim(*boundsX)
                axes[0].set_ylim(*boundsY)
                axes[0].set_aspect(1)

                axes[0].imshow(img2)


            

                bigtpix = big_ego_t2pix(t)
                bigrpix = big_ego_r2pix(logr)
            
                #liltpix = ego_t2pix(t)
                #lilrpix = ego_r2pix(logr)

                plt.plot(bigtpix,bigrpix,'m',linewidth=3)

                plt.show()


                fig, ax = plt.subplots(1,1)
                axes = [ax]
            
                #plt.imsave('bigegomap.png',img2)
                axes[0].set_title('Traj in image')
                #axes[0].set_xlim(*newBoundsx)
                #axes[0].set_ylim(*crowdBoundsY)
                #axes[0].set_aspect(1)
            
                axes[0].set_xlim(0,img.shape[1])
                axes[0].set_ylim(img.shape[0],0)
                axes[0].set_aspect(1)
                axes[0].imshow(img)
                #axes[0].plot(tr_ground[0], tr_ground[1], 'm')

            
                bigtpix = big_ego_t2pix(t)
                bigrpix = big_ego_r2pix(logr)

                       
                z, x = Polar2Coord(t,np.exp(logr))
            
                coords_3D = np.zeros((len(z),3))
                coords_3D[:,1] = 0
                coords_3D[:,0] = x
                coords_3D[:,2] = z
                coords_3D = (R_rect_ego.T @ coords_3D.T).T
                coords_3D -= tr['up'].T

                pixels = K_data @ R_rect @ coords_3D.T
                pixels /= pixels[2]

                axes[0].plot(pixels[0],pixels[1],'m',linewidth=3)

                plt.show()








            fig, ax = plt.subplots(1,1)#, figsize=(18,6))
            axes = [ax] # only use if there's 1 column

            ##newBoundsx = (crowdBoundsX[0], 2*crowdBoundsX[1])
            ##fig.set_size_inches(16, 24)
            #axes[0].set_title('Traj in image')
            ##axes[0].set_xlim(*newBoundsx)
            ##axes[0].set_ylim(*crowdBoundsY)
            ##axes[0].set_aspect(1)
            
            #axes[0].set_xlim(0,img.shape[1])
            #axes[0].set_ylim(img.shape[0],0)
            #axes[0].set_aspect(1)
            #axes[0].imshow(img)
            #axes[0].plot(tr_ground[0], tr_ground[1], 'b.')

            
            #bigtpix = big_ego_t2pix(t)
            #bigrpix = big_ego_r2pix(logr)

                       
            #z, x = Polar2Coord(t,np.exp(logr))
            
            #coords_3D = np.zeros((len(z),3))
            #coords_3D[:,1] = 0
            #coords_3D[:,0] = x
            #coords_3D[:,2] = z
            #coords_3D = (R_rect_ego.T @ coords_3D.T).T
            #coords_3D -= tr['up'].T

            #pixels = K_data @ R_rect @ coords_3D.T
            #pixels /= pixels[2]

            #axes[0].plot(pixels[0],pixels[1],'rx')
        
            ##axes[1].set_title('Traj in EgoMap')
            ##axes[1].set_xlim((-2*np.pi/3, 2*np.pi/3))
            ###axes[0].set_ylim(*crowdBoundsY)
            ##axes[1].set_aspect(1)
            ###axes[1].imshow(img)
        
        
            ##axes[1].plot(t, logr, 'r')
        
            #axes[1].set_title('EgoRetinal Map')
            
            #boundsX = (0,big_ego_pixel_shape[1])
            #boundsY = (0,big_ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
            
            #axes[1].set_xlim(*boundsX)
            #axes[1].set_ylim(*boundsY)
            #axes[1].set_aspect(1)

            #axes[1].imshow(img2)


            

            #bigtpix = big_ego_t2pix(t)
            #bigrpix = big_ego_r2pix(logr)
            
            #liltpix = ego_t2pix(t)
            #lilrpix = ego_r2pix(logr)
            
            ##print('first ', bigtpix)
            ##print('second ', bigrpix)
            ##print('third ', liltpix)
            ##print('fourth ', lilrpix)

            #axes[1].plot(bigtpix,bigrpix, 'rx')


            ##axes[1].plot(big_ego_t2pix(test_t),big_ego_r2pix(test_logr) ,'bo')



            


            
            #axes[0].set_title('EgoRetinal Ground Truth')
            boundsX = (0,big_ego_pixel_shape[1])
            boundsY = (0,big_ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
            #axes[2].set_xlim(*boundsX)
            #axes[2].set_ylim(*boundsY)

            #axes[1].set_xlim(*boundsX)
            #axes[1].set_ylim(*boundsY)
            #-np.log(-coord_value+1)
            #tempval = axes[2].imshow(np.reshape(coord_value,(big_ego_pixel_shape)), extent=[*boundsX, *(big_ego_pixel_shape[0],0)], interpolation='none', cmap='viridis')#, cmap='gnuplot')
            #tempval = axes[2].imshow(np.reshape(all_ws_value,(240,240)), extent=[*boundsX, *(big_ego_pixel_shape[0],0)], interpolation='none', cmap='viridis')#, cmap='gnuplot')
            
            all_ws_derivative[:,0] #*=(15/8)
            all_ws_derivative[:,1] #*=(8/15)
            coord_mag = np.linalg.norm(all_ws_derivative,axis=1)
            normalized_derivatives = all_ws_derivative / coord_mag[:,None]/4

            bigenough = normalized_derivatives[:]
            locations_np = all_ws_coords_small
            coord_bigenough = locations_np[:]

            #axes[0].axis('off')

            axes[0].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
            
            axes[0].tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off

            #axes[0].set_aspect(60/8)
            axes[0].plot(ws_traj[:,0],ws_traj[:,1],'m')
            axes[0].quiver(coord_bigenough[:,0], coord_bigenough[:,1], -normalized_derivatives[:,1]*5,-normalized_derivatives[:,0]*5, color='c', units='xy' ,scale=1)
            #axes[0].quiver(all_ws_coords_small[:,0], all_ws_coords_small[:,1], -all_ws_derivative[:,1],-all_ws_derivative[:,0], color='c', units='xy' ,scale=1)
        
            trajnp = np.array(test_pix_trajectory)
            #axes[2].plot(trajnp[:,0], trajnp[:,1], 'r')
        
            #cax = fig.add_axes([.3, .95, .4, .05])
            #fig.colorbar(tempval, cax, orientation='horizontal')




            #axes[2].set_title('EgoRetinal Disparity Map')
            
            #boundsX = (0,big_ego_pixel_shape[1])
            #boundsY = (0,big_ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
            
            #axes[2].set_xlim(*boundsX)
            #axes[2].set_ylim(*boundsY)
            #axes[2].set_aspect(1)

            #tempval = axes[2].imshow(disp_img2)
            
            #cax = fig.add_axes([.3, .95, .4, .05])
            #fig.colorbar(tempval, cax, orientation='horizontal')

            

            #bigtpix = big_ego_t2pix(t)
            #bigrpix = big_ego_r2pix(logr)

            #axes[2].plot(bigtpix,bigrpix, 'rx')


            #axes[2].plot(big_ego_t2pix(test_t),big_ego_r2pix(test_logr) ,'b')
            plt.show()
            
            fig, axes = plt.subplots(1,2)#, figsize=(18,6))
            #axes = [ax]

            axes[0].set_title('Derivative of GT Field')
            boundsX = (0,small_ego_pixel_shape[1])
            boundsY = (0,small_ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
            axes[0].set_xlim(*boundsX)
            axes[0].set_ylim(*boundsY)
            axes[0].set_aspect(1)

            
            
            coord_mag = np.linalg.norm(trajectory_deriv,axis=1)
            normalized_derivatives = trajectory_deriv / coord_mag[:,None]/4

            bigenough = normalized_derivatives[coord_mag > .025]
            locations_np = np.array(test_pix_trajectory_small)
            coord_bigenough = locations_np[coord_mag > .025]
            

            axes[0].quiver(coord_bigenough[:,0], coord_bigenough[:,1], -bigenough[:,0],-bigenough[:,1], color='c', units='xy' ,scale=.25)# minshaft=50,  headwidth=2, headlength=1)



            coord_mag = np.linalg.norm(small_coord_derivative,axis=1)

            scalar_term = 4
            normalized_derivatives = small_coord_derivative / coord_mag[:,None]/scalar_term

            bigenough = normalized_derivatives[coord_mag > .1/scalar_term]
            coord_bigenough = small_pixel_coords[coord_mag > .1/scalar_term]
            #coord_x, coord_y = np.meshgrid(range(ego_pixel_shape[1]), range(ego_pixel_shape[0]))

            axes[0].quiver(coord_bigenough[:,0], coord_bigenough[:,1], -bigenough[:,0],-bigenough[:,1], color='red', units='xy' ,scale=.25)# minshaft=50,  headwidth=2, headlength=1)



            axes[1].set_title('EgoRetinal Ground Truth')
            boundsX = (0,big_ego_pixel_shape[1])
            boundsY = (0,big_ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
            axes[1].set_xlim(*boundsX)
            axes[1].set_ylim(*boundsY)

            #axes[1].set_xlim(*boundsX)
            #axes[1].set_ylim(*boundsY)
            #-np.log(-coord_value+1)
            
            axes[1].axis('off')
            
            tempval = axes[1].imshow(np.reshape(coord_value,(big_ego_pixel_shape)), extent=[*boundsX, *(big_ego_pixel_shape[0],0)], interpolation='none', cmap='hot')#, cmap='gnuplot')
            coord_mag = np.linalg.norm(small_coord_derivative,axis=1)

            scalar_term = 2
            normalized_derivatives = small_coord_derivative / coord_mag[:,None]/scalar_term

            bigenough = normalized_derivatives[coord_mag > .00/scalar_term]
            coord_bigenough = small_pixel_coords[coord_mag > .00/scalar_term]
            #coord_x, coord_y = np.meshgrid(range(ego_pixel_shape[1]), range(ego_pixel_shape[0]))
            axes[1].quiver(coord_bigenough[:,0]*8, coord_bigenough[:,1]*8, -bigenough[:,0],-bigenough[:,1], color='blue', units='xy' ,scale=.08)# minshaft=50,  headwidth=2, headlength=1)

            trajnp = np.array(test_pix_trajectory)
            #axes[2].plot(trajnp[:,0], trajnp[:,1], 'r')
        
            cax = fig.add_axes([.3, .95, .4, .05])
            fig.colorbar(tempval, cax, orientation='horizontal')

            
            #axes[3].set_title('Polar In Image')

            #axes[3].imshow(img)

            #axes[3].plot(pixels[0,:],pixels[1,:],'bo')
            
            #figManager = plt.get_current_fig_manager()
            #figManager.window.showMaximized()
            plt.show()




            
            
            #print('first ', bigtpix)
            #print('second ', bigrpix)
            #print('third ', liltpix)
            #print('fourth ', lilrpix)

            #axes[1].plot(bigtpix,bigrpix, 'rx')



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

