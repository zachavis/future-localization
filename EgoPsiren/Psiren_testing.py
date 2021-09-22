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
    
    #iFrame = 49
    #for iFrame in range(49,55):
    #tr = vTR['vTr'][iFrame]

    ##im = sprintf('%sim/%s', folder_path, vFilename{iFrame});
    #im = "{}im\\{}".format(folder_path, vFilename[iFrame])

    #if not os.path.isfile(im):
    #    print('could not find file')
    #    continue

    #img = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB).astype(np.float64)/255.0
    ##cv2.imshow('intermediate',intermediate)
    ##cv2.waitKey()

    ##im = im2double(intermediate); % im2double(imread(im));

    #tr_ground = (tr['XYZ'].T - tr['up']).T #bsxfun(@minus, tr['XYZ'], tr['up']);
    #t, r = Coord2Polar(tr_ground[2],tr_ground[0])
    #tr_ground = K_data @ R_rect @ tr_ground;
    #if np.any(tr_ground[2,:]<0):
    #    tr_ground[:2,:] = np.nan
    ##tr_ground(tr_ground(3,:)<0, :) = NaN;
    ##tr_ground = bsxfun(@rdivide, tr_ground(1:2,:), tr_ground(3,:));
    #tr_ground = tr_ground[:2] / tr_ground[2]



        
    ## set up egocentric image information in log polar space

    ##t, r = Coord2Polar(tr['XYZ'][2],tr['XYZ'][0])
    #logr = np.log(r)

    ##world_forward = 
    #img_resized = cv2.resize(img, (int(img.shape[1]*.25), int(img.shape[0]*.25)))*2.0-1.0
    #img_channel_swap = np.moveaxis(img_resized,-1,0).astype(np.float32)


    # DISPLAY IMAGES
    #fig, axes = plt.subplots(1,3)#, figsize=(18,6))
    ## axes = [ax] # only use if there's 1 column

    ##newBoundsx = (crowdBoundsX[0], 2*crowdBoundsX[1])
    ##fig.set_size_inches(16, 24)
    #axes[0].set_title('Traj in image')
    ##axes[0].set_xlim(*newBoundsx)
    ##axes[0].set_ylim(*crowdBoundsY)
    ##axes[0].set_aspect(1)
    #axes[0].imshow(img)
    #axes[0].plot(tr_ground[0], tr_ground[1], 'r')


    #axes[1].set_title('Rectified image')
    ##axes[0].set_xlim(*newBoundsx)
    ##axes[0].set_ylim(*crowdBoundsY)
    ##axes[0].set_aspect(1)

    #homography = K_data @ R_rect @ np.linalg.inv(K_data)
    #img_rectified = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))
    #axes[1].imshow(img_rectified)
    ##axes[1].plot(tr_ground[0], tr_ground[1], 'r')

        
    #axes[2].set_title('Traj in EgoMap')
    #axes[2].set_xlim((-2*np.pi/3, 2*np.pi/3))
    ##axes[0].set_ylim(*crowdBoundsY)
    #axes[2].set_aspect(1)
    ##axes[1].imshow(img)

        
    #axes[2].plot(t, logr, 'r')


    #plt.show()









    img_height = 64

    minR = -.5
    maxR = 5#4.5
    minT = -2*np.pi/3
    maxT = 2*np.pi/3

    aspect_ratio = (maxT-minT)/(maxR-minR)
    ego_pixel_shape = (img_height,int(img_height*aspect_ratio)) # y,x | vert,horz

    #ego_r2pix = lambda x : RemapRange(x, minR,maxR, 0,                  ego_pixel_shape[0]  )
    #ego_t2pix = lambda x : RemapRange(x, minT,maxT, 0,                  ego_pixel_shape[1]  )

        
    #ego_pix2r = lambda x : RemapRange(x, 0, ego_pixel_shape[0], minR,maxR   )
    #ego_pix2t = lambda x : RemapRange(x,0,ego_pixel_shape[1], minT,maxT  )

    RecenterDataForwardWithShape = lambda x, shape : RemapRange(x,0,max(shape[0],shape[1]),-1,1)
    RecenterDataForwardWithShapeAndScale = lambda x, shape, scale : RemapRange(x,0,max(shape[0],shape[1]),-scale,scale)
    RecenterDataBackwardWithShape = lambda x, shape : RemapRange(x,-1,1,0,max(shape[0],shape[1]))
    RecenterTrajDataForward = lambda x : RecenterDataForwardWithShape(x,ego_pixel_shape)
    RecenterTrajDataForward2 = lambda x : RecenterDataForwardWithShapeAndScale(x,ego_pixel_shape,1)

        
    RecenterDataBackwardWithShapeAndScale = lambda x, shape, scale : RemapRange(x,-scale,scale,0,max(shape[0],shape[1]))
    RecenterFieldDataBackward = lambda x : RecenterDataBackwardWithShapeAndScale(x,ego_pixel_shape,1)
        

    #tpix = ego_t2pix(t) #- 15
    #logrpix = ego_r2pix(logr)



    #points = np.array([ [0, 1, 8, 2, 2],
    #                    [1, 0, 6, 7, 2]]).T  # a (nbre_points x nbre_dim) array

    #points = np.array([ [20,20,30,40],
    #                    [20,30,30,40]] ).T



    #points[0] = np.array([ [20,20,21,30],
    #                        [20,29,30,30]] ).T



    # inward spiral
    traj_points = {}
    traj_points[0] = np.array([ [5,10,11,24],
                           [5,15,16,29]] ).T
    traj_points[1] = np.array([ [45,40,40,26],
                                [5,15,16,29]] ).T
    traj_points[2] = np.array([ [25,26,24,25],
                                [60,42,38,31]] ).T



    # ROAD SCENARIO
    #traj_points = {}
    #traj_points[0] = np.array([ [19,20,18,19],
    #                       [10,25,41,56]] ).T
    ##traj_points[1] = np.array([ [32,31,33,32],
    ##                            [56, 41, 25, 10]] ).T
    ##traj_points[2] = np.copy(traj_points[0])
    ##traj_points[2][:,0]+=5
    ##traj_points[3] = np.copy(traj_points[1])
    ##traj_points[3][:,0]-=5
        
    future_trajectory = {}
    
    for k in range(len(traj_points)):
    #    # Linear length along the line:
        #distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        points = traj_points[k]
        #distance = np.cumsum( np.sum( np.diff(points,axis=0), axis=1) )
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]

        alpha = np.linspace(0,1,100)
        interpolator =  interpolate.interp1d(distance, points, kind='cubic', axis=0)
        interpolated_points = interpolator(alpha)
        



        future_trajectory[k] = []
        for point in interpolated_points:
            future_trajectory[k].append( (point[0],point[1]) )


    # Obstacle
    obstacle_trajectory = {}
    obst_points = {}
    #obst_points[0] = np.array([ [25.5,25.5,25.5,25.5],
    #                            [0,25,41,64]] ).T
    #distance = np.cumsum( np.sqrt(np.sum( np.diff(obst_points[0], axis=0)**2, axis=1 )) )
    #distance = np.insert(distance, 0, 0)/distance[-1]
    #alpha = np.linspace(0,1,100)
    #interpolator =  interpolate.interp1d(distance, obst_points[0], kind='cubic', axis=0)
    #interpolated_points = interpolator(alpha)

    #obstacle_trajectory[0] = []
    #for point in interpolated_points:
    #    obstacle_trajectory[0].append( (point[0],point[1]) )


                
    #future_trajectory = {}

    #for k in range(1):#48
    #    future_trajectory[k] = []
    #    for i in range(len(logrpix)):
    #        #future_trajectory[k].append( (tpix[i] + (k%2)*15, logrpix[i]) ) # t is horizontal axis, logr is vertical
    #        future_trajectory[k].append( (tpix[i], logrpix[i]) ) # t is horizontal axis, logr is vertical
    #        #if k < 2:
    #        #    if logrpix[i] >= 25:
    #        #        future_trajectory[k].append( (tpix[i] + (k%2)*15, logrpix[i]) ) # t is horizontal axis, logr is vertical
    #        #else:
    #        #    if logrpix[i] <= 24:
    #        #        future_trajectory[k].append( (tpix[i] + (k%2)*15, logrpix[i]) ) # t is horizontal axis, logr is vertical
        
    #np.random.seed(8980)



        
    ## Let's try to get a ground truth image
    #from scipy import stats

    #def DistanceFromLine (line, point):
    #    homogPoint = np.array([point[0], point[1], 1])
    #    proj = np.dot(line, homogPoint)
    #    lineNormal = np.linalg.norm(np.array([line[0],line[1]]))
    #    return abs(proj / lineNormal)

    #def skewPix(x):
    #    return np.array([[0, -1, x[1]],
    #                        [1, 0, -x[0]],
    #                        [-x[1], x[0], 0]])

    #def GetLine(ptA, ptB):
    #    return skewPix(ptA) @ ptB

    #def GetPoint(x):
    #    if len(x) >= 3:
    #    return x/x[2]
    #    return np.concatenate((x,np.ones(1)))

           
    # Let's try to get a ground truth image
    all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(ego_pixel_shape[0]) for j in range(ego_pixel_shape[1]) ], dtype=np.float32)
    #coord_value = np.zeros((all_pixel_coords.shape[0]))




        
        ##print(i+1,'/',all_pixel_coords.shape[0])
        ##cur_pixel = all_pixel_coords[i]
        #closest_distance = np.inf
        #time_along_trajectory = 0
        #for key in future_trajectory.keys():
        #    traj = np.array(future_trajectory[key],dtype=np.float32)
        #    traj_len = len(traj)
        #    dists = traj - all_pixel_coords[:,None]
        #    dists = np.linalg.norm(dists,axis=2)
                
        #    next_pts = np.ones((traj.shape[0]-1, 3))
        #    prev_pts = np.ones((traj.shape[0]-1, 3))
        #    next_pts[:,:2] = traj[1:]
        #    prev_pts[:,:2] = traj[:-1]


        #    vecLines = next_pts[:,:2]-prev_pts[:,:2]
        #    vecLinesMag = np.linalg.norm(vecLines,axis=1)
        #    distAlongTraj = np.cumsum(vecLinesMag)
        #    vecLines = np.divide(vecLines, vecLinesMag[:,None]) # The none thing is to make the shapes broadcastable
        #    vecPoints = -traj[:-1]+all_pixel_coords[:,None]
        #    vecPoints = np.divide(vecPoints, vecLinesMag[:,None]) # The none thing is to make the shapes broadcastable

        #    dotResults = np.multiply(vecLines[None,:], vecPoints).sum(2)
        #    firstrow = dotResults[0]
        #    #if dotResults.max() > 0:
        #    #    print('stop')
                
        #    trajLines = np.cross(next_pts,prev_pts)

        #    someones = np.ones((all_pixel_coords.shape[0],1))
        #    catted = np.concatenate((all_pixel_coords, someones), axis=1)
        #    projOntoLine = catted @ trajLines.T #np.dot(trajLines, catted[:,None] )
        #    lineMagnitudes = np.linalg.norm(trajLines[:,:2],axis=1)
        #    distsLine = np.abs( projOntoLine / lineMagnitudes[None,:] )

        #    #a_test = dotResults < 0
        #    distsLine[ dotResults < 0 ] = np.inf #can turn this into a max/min problem for divergence free processing
        #    distsLine[ dotResults > 1 ] = np.inf

        #    linesArgMin = np.argmin(distsLine,axis=1).astype(np.int32)
        #    pointsArgMin = np.argmin(dists,axis=1).astype(np.int32)

            
        #    smallest_distsLine = np.squeeze( np.take_along_axis(distsLine,linesArgMin[:,None],axis=1) )
        #    smallest_dists = np.squeeze( np.take_along_axis(dists,pointsArgMin[:,None],axis=1) )

        #    dist_along_min_line_segment = np.squeeze( np.take_along_axis(dotResults,linesArgMin[:,None],axis=1) )

        #    stacked = np.stack((smallest_distsLine, smallest_dists), axis=1)

        #    which_was_closest = np.argmin(stacked,axis=1).astype(np.int32) # which was closest between point and line

            
        #    closest_lines = which_was_closest == 0
        #    closest_points = which_was_closest == 1
            
        #    current_distance = np.zeros(smallest_dists.shape[0])
        #    current_time = np.zeros(smallest_dists.shape[0])


        #    current_distance[closest_lines] = smallest_distsLine[closest_lines] #distsLine[linesArgMin]
        #    above_zero = linesArgMin > 0
        #    together = np.logical_and(closest_lines, above_zero)
        #    #testing = distAlongTraj[linesArgMin]
        #    current_time[together] = distAlongTraj[linesArgMin-1][together]
        #    #vlm = vecLinesMag[linesArgMin][together]
        #    #dr = dist_along_min_line_segment[linesArgMin][together]
            
        #    dist_along_min_line_segment = np.squeeze( np.take_along_axis(dotResults,linesArgMin[:,None],axis=1) )
        #    #resultA = vecLinesMag[closest_lines]#[closest_lines]
        #    #resultB = dist_along_min_line_segment[closest_lines]#[closest_lines]
        #    #finalResult = resultA * resultB
        #    #finalResultClosest = finalResultClosest[closest_lines]
        #    current_time[closest_lines] += vecLinesMag[linesArgMin][closest_lines] * dist_along_min_line_segment[closest_lines]

        #    #closest_distance_arg = np.argmin([distsLine[linesArgMin],dists[pointsArgMin]])
            
        #    current_distance[closest_points] = smallest_dists[closest_points]
        #    above_zero = pointsArgMin > 0
        #    together = np.logical_and(closest_points, above_zero)
        #    current_time[together] = distAlongTraj[pointsArgMin-1][together]




                
        #    closest_distances = current_distance #smallest_dists
        #    time_along_trajectory = current_time
        #    #current_time = 0
        #    #current_distance = 0
        #    #if closest_distance_arg == 0:
        #    #    current_distance = distsLine[linesArgMin]
        #    #    if (linesArgMin > 0):
        #    #        current_time = distAlongTraj[linesArgMin-1]
        #    #    current_time += vecLinesMag[linesArgMin] * dotResults[linesArgMin]
        #    #    #time_so_far = distAlongTraj[linesArgMin]
        #    #else:
        #    #    current_distance = dists[pointsArgMin]
        #    #    if (pointsArgMin > 0):
        #    #        current_time = distAlongTraj[pointsArgMin-1]

        #    ##if (closest_distance == dists[pointsArgMin]):
        #    ##    current_time = 0
        #    ##    if (pointsArgMin > 0):
        #    ##        current_time = distAlongTraj[pointsArgMin-1]

        #    ##else: #if (closest_distance == distsLine[linesArgMin])
        #    ##    current_time = 0
        #    ##    if (linesArgMin > 0):
        #    ##        current_time = distAlongTraj[linesArgMin-1]
        #    ##    current_time += vecLinesMag[linesArgMin] * dotResults[linesArgMin]
        #    ##    #time_so_far = distAlongTraj[linesArgMin]

        #    #if current_distance < closest_distance:
        #    #    closest_distance = current_distance
        #    #    time_along_trajectory = current_time
                
        #values = -stats.norm.pdf(closest_distances) * time_along_trajectory#time_along_trajectory
        #coord_value = values

        #all_pixel_coords_xformed = np.zeros(all_pixel_coords.shape).astype(np.float32)
        #all_pixel_coords_xformed[:,0] = ego_pix2t(all_pixel_coords[:,0])
        #all_pixel_coords_xformed[:,1] = np.exp(ego_pix2r(all_pixel_coords[:,1]))

        #all_pixel_coords_xformed = np.array(Polar2Coord(all_pixel_coords_xformed[:,0],all_pixel_coords_xformed[:,1])).T
        ##newtraj = future_trajectory.copy()
        #newtraj = {}
        #newtraj[0] = []
        #for pix in future_trajectory[0]:
        #    newpoint = ( Polar2Coord( ego_pix2t(pix[0]),np.exp(ego_pix2r(pix[1])) ) )
        #    newtraj[0].append( newpoint )


        #coord_value = DataGens.Coords2ValueFastWS(all_pixel_coords_xformed,newtraj,None,None,stddev=2)


        #coord_value = DataGens.Coords2ValueFast(all_pixel_coords,future_trajectory,nscale=1)




        #for i in range(all_pixel_coords.shape[0]):
        #    print(i+1,'/',all_pixel_coords.shape[0])
        #    cur_pixel = all_pixel_coords[i]
        #    closest_distance = np.inf
        #    time_along_trajectory = 0
        #    for key in future_trajectory.keys():
        #        traj = np.array(future_trajectory[key],dtype=np.float32)
        #        traj_len = len(traj)
        #        dists = traj - cur_pixel
        #        dists = np.linalg.norm(dists,axis=1)
                
        #        next_pts = np.ones((traj.shape[0]-1, 3))
        #        prev_pts = np.ones((traj.shape[0]-1, 3))
        #        next_pts[:,:2] = traj[1:]
        #        prev_pts[:,:2] = traj[:-1]


        #        vecLines = next_pts[:,:2]-prev_pts[:,:2]
        #        vecLinesMag = np.linalg.norm(vecLines,axis=1)
        #        distAlongTraj = np.cumsum(vecLinesMag)
        #        vecLines = np.divide(vecLines, vecLinesMag[:,None]) # The none thing is to make the shapes broadcastable
        #        vecPoints = -traj[:-1]+cur_pixel
        #        #vecPointsMag = np.linalg.norm(vecPoints,axis=1)
        #        vecPoints = np.divide(vecPoints, vecLinesMag[:,None]) # The none thing is to make the shapes broadcastable

        #        dotResults = np.multiply(vecLines, vecPoints).sum(1)
        #        #if dotResults.max() > 0:
        #        #    print('stop')
                
        #        trajLines = np.cross(next_pts,prev_pts)
        #        projOntoLine = np.dot(trajLines,np.array([cur_pixel[0],cur_pixel[1],1]))
        #        lineMagnitudes = np.linalg.norm(trajLines[:,:2],axis=1)
        #        distsLine = np.abs( projOntoLine / lineMagnitudes )

        #        #a_test = dotResults < 0
        #        distsLine[ dotResults > 1 ] = np.inf
        #        distsLine[ dotResults < 0 ] = np.inf #can turn this into a max/min problem for divergence free processing

        #        linesArgMin = int(np.argmin(distsLine))
        #        pointsArgMin = int(np.argmin(dists))

        #        closest_distance_arg = np.argmin([distsLine[linesArgMin],dists[pointsArgMin]])
                
        #        current_time = 0
        #        current_distance = 0
        #        if closest_distance_arg == 0:
        #            current_distance = distsLine[linesArgMin]
        #            if (linesArgMin > 0):
        #                current_time = distAlongTraj[linesArgMin-1]
        #            current_time += vecLinesMag[linesArgMin] * dotResults[linesArgMin]
        #            #time_so_far = distAlongTraj[linesArgMin]
        #        else:
        #            current_distance = dists[pointsArgMin]
        #            if (pointsArgMin > 0):
        #                current_time = distAlongTraj[pointsArgMin-1]

        #        #if (closest_distance == dists[pointsArgMin]):
        #        #    current_time = 0
        #        #    if (pointsArgMin > 0):
        #        #        current_time = distAlongTraj[pointsArgMin-1]

        #        #else: #if (closest_distance == distsLine[linesArgMin])
        #        #    current_time = 0
        #        #    if (linesArgMin > 0):
        #        #        current_time = distAlongTraj[linesArgMin-1]
        #        #    current_time += vecLinesMag[linesArgMin] * dotResults[linesArgMin]
        #        #    #time_so_far = distAlongTraj[linesArgMin]

        #        if current_distance < closest_distance:
        #            closest_distance = current_distance
        #            time_along_trajectory = current_time
                
        #    value = -stats.norm.pdf(closest_distance) #* time_along_trajectory
        #    coord_value[i] = value









        #for i in range(all_pixel_coords.shape[0]):
        #    print(i+1,'/',all_pixel_coords.shape[0])
        #    cur_pixel = all_pixel_coords[i]
            
        #    closest_distance = np.inf
        #    time_along_trajectory = 0
        #    for key in future_trajectory.keys():
        #        traj = future_trajectory[key]
        #        traj_len = len(traj)
        #        for j in range(traj_len-1):
        #            nxt_traj_pt = np.array(list(traj[j+1]))
        #            prv_traj_pt = np.array(list(traj[j]))

        #            dist_nxt = np.linalg.norm(nxt_traj_pt-cur_pixel)
                    
        #            dist_prv = np.linalg.norm(prv_traj_pt-cur_pixel)





        #            vecLine = nxt_traj_pt-prv_traj_pt
        #            vecLineMag = np.linalg.norm(vecLine).astype(np.float32)
        #            vecLine /= vecLineMag
        #            vecPoint = cur_pixel-prv_traj_pt
        #            vecPoint /= vecLineMag
        #            dotresult = vecLine @ vecPoint
        #            if (dotresult < 0 or dotresult > 1):
        #                dist_line = np.inf
        #            else:
        #                trajLine = GetLine(GetPoint(nxt_traj_pt),GetPoint(prv_traj_pt))
        #                dist_line = DistanceFromLine(trajLine,GetPoint(cur_pixel))

        #            dist_min = dist_line #min(dist_nxt,dist_prv,dist_line)#),closest_distance)
        #            current_time = 1#np.clip(dotresult,0,1) + j
        #            if dist_min < closest_distance:
        #                closest_distance = dist_min
        #                time_along_trajectory = current_time

        #            #coord_value[i] = dist_min
        #            #closest_distance = dist_min
        #            #print(dist_min)

        #    value = -stats.norm.pdf(closest_distance) * time_along_trajectory
        #    coord_value[i] = value


    fig, ax = plt.subplots(1,1)#, figsize=(36,6))
    axes = [ax]

    boundsX = (0,ego_pixel_shape[1])
    boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
    axes[0].set_xlim(*boundsX)
    axes[0].set_ylim(*boundsY)
    axes[0].set_aspect(1)
    #axes[1].set_xlim(*boundsX)
    #axes[1].set_ylim(*boundsY)

    #tempval = axes[0].imshow(np.reshape(coord_value,(ego_pixel_shape)), extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')#, cmap='gnuplot')
        
    for traj in future_trajectory.values():
        trajnp = np.array(traj)
        axes[0].plot(trajnp[:,0], trajnp[:,1], 'r')
        
    cax = fig.add_axes([.3, .95, .4, .05])
    #fig.colorbar(tempval, cax, orientation='horizontal')
    plt.show()








    if True:
        n_samples = 30000 # 25000
        n_obstsamples = 1#30000
        n_2sample_laplacian = 10000
        n_2sample = 1#15000

        pixtol = 2#0.3
        spatial_samples = DataGens.GenLaplacianSamplePoints((ego_pixel_shape[1], ego_pixel_shape[0]), future_trajectory, obstacle_trajectory, n_samples, pixel_tolerance = pixtol/2.0, print_status = True)
        obstacle_samples = DataGens.GenLaplacianObstaclePoints((ego_pixel_shape[1], ego_pixel_shape[0]), future_trajectory, obstacle_trajectory, n_obstsamples, pixel_tolerance = pixtol, print_status = True)


        g1,g2 = DataGens.GenGradientSamplePoints((ego_pixel_shape[1], ego_pixel_shape[0]), future_trajectory, None, n_2sample, pixel_tolerance = pixtol, print_status = True)
        
        maxnorm = np.max(np.linalg.norm(g2,axis=1))
        g2 /= maxnorm
        
        #g2 = g2[whereitis]# = 0
        #whereitis = np.where(np.linalg.norm(g2,axis=1) > .2)
        #whereitis = np.where(np.linalg.norm(g2,axis=1) < .2)
        #g2[whereitis] = 0
        #g2 = g2[whereitis]# = 0
        #g1 = g1[whereitis]
        n_2sample = len(g1)
        maxnorm = np.max(np.linalg.norm(g2,axis=1))
        
        
        
        
        
        gradient_samples = (g1,g2)





        where_wrong_way = np.where(g2[:,1] < -0.0000001)

        field_data_set = DataGens.FieldDataset(gradient_samples[0], gradient_samples[1], n_2sample//4, RecenterTrajDataForward2, None)
        field_data_generator = torch.utils.data.DataLoader(field_data_set, batch_size= 1, shuffle=True)
        i, (pos, pix) = next(enumerate(field_data_generator))
        where_wrong_way2 = np.where(pix.numpy()[0,0,:,1] > 0.00001)
        wrong_ways = pix.numpy()[0,0,where_wrong_way2]

        fig, ax = plt.subplots(1,1)

        boundsX = (0,ego_pixel_shape[1])
        boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
        ax.set_xlim(*boundsX)
        ax.set_ylim(*boundsY)
        ax.set_aspect(1)

        #coord_x, coord_y = np.meshgrid(range(ego_pixel_shape[1]), range(ego_pixel_shape[0]))

        ax.quiver(gradient_samples[0][:,0], gradient_samples[0][:,1], gradient_samples[1][:,0],gradient_samples[1][:,1], color='red', units='xy' ,scale=1)

        #ux = vx/np.sqrt(vx**2+vy**2)
        #uy = vy/np.sqrt(vx**2+vy**2)
        
        for traj in future_trajectory.values():
            trajnp = np.array(traj)
            ax.plot(trajnp[:,0], trajnp[:,1], 'r')

        plt.show()

        #spatial_samples = np.concatenate((spatial_samples,obstacle_samples),axis=0)
        laplacian_data_set = DataGens.LaplacianDataset(spatial_samples, n_2sample_laplacian, RecenterTrajDataForward2,obstacle_samples, split = 1.0) #,np.ones((1,n_obstsamples,1))
        laplacian_data_generator = torch.utils.data.DataLoader(laplacian_data_set, batch_size= 50, shuffle=True)



        trajectory_data_set = DataGens.TrajectoryDataset(future_trajectory, RecenterTrajDataForward,None,2)
        trajectory_data_generator = torch.utils.data.DataLoader(trajectory_data_set, batch_size= 50, shuffle=True)
        i, (pos, pix) = next(enumerate(trajectory_data_generator))
        #print(pos.shape)
        #print(pos.dtype)
        #print(pix.shape)
        #print(pix.dtype)
        # print(pix[0,0,1])

        ##hyper_trajectory_data_set = DataGens.HyperTrajectoryDataset(future_trajectory, RecenterTrajDataForward,torch.zeros((1,32,32)))
        ##hyper_trajectory_data_set = DataGens.HyperTrajectoryDataset2(future_trajectory,n_2sample_laplacian,RecenterTrajDataForward,all_pixel_coords,img_channel_swap) #DataGens.HyperTrajectoryDataset(future_trajectory, RecenterTrajDataForward, img_channel_swap)
        #hyper_trajectory_data_set = DataGens.HyperTrajectoryDataset2(newtraj,n_2sample_laplacian,RecenterTrajDataForward,RecenterFieldDataBackward,all_pixel_coords,img_channel_swap,ego_pix2t,ego_pix2r,Polar2Coord) #DataGens.HyperTrajectoryDataset(future_trajectory, RecenterTrajDataForward, img_channel_swap)
        #i, (pos, pix) = next(enumerate(hyper_trajectory_data_set))



        gt_field_dataset = DataGens.GTFieldDataset(future_trajectory, n_2sample_laplacian, RecenterTrajDataForward, all_pixel_coords)
        #gt_field_data_generator = torch.utils.data.DataLoader(gt_field_dataset, batch_size= 1, shuffle=True)













        #Training parameters
        num_epochs = 1000 #15000 #1000
        print_interval = 1
        learning_rate = 5e-5#1e-5
        loss_function = DNN.gradients_mse_with_coords #gradients_and_laplacian_mse_with_coords #nn.MSELoss()
        loss_function2 = DNN.laplacian_mse_with_coords
        loss_function3 = DNN.value_mse_with_coords
        

        trajectory_training_dataset = trajectory_data_set
        trajectory_testing_dataset = trajectory_training_dataset
        trajectory_training_generator = torch.utils.data.DataLoader(trajectory_training_dataset, batch_size = 50, shuffle=True)
        trajectory_testing_generator = torch.utils.data.DataLoader(trajectory_testing_dataset, batch_size = 50)

        laplacian_training_dataset = laplacian_data_set
        laplacian_testing_dataset = laplacian_training_dataset
        laplacian_training_generator = torch.utils.data.DataLoader(laplacian_training_dataset, batch_size = 50, shuffle=True)
        laplacian_testing_generator = torch.utils.data.DataLoader(laplacian_testing_dataset, batch_size = 50)
        
        i, (pos, pix) = next(enumerate(laplacian_training_dataset))

        #hyper_training_set = hyper_trajectory_data_set
        #hyper_testing_set = hyper_trajectory_data_set
        #hyper_training_generator = torch.utils.data.DataLoader(hyper_training_set, batch_size = 1, shuffle=True)
        #hyper_testing_generator = torch.utils.data.DataLoader(hyper_testing_set, batch_size = 1)
        
        #i, (pos, pix) = next(enumerate(hyper_training_generator))
        
        
        field_training_dataset = field_data_set
        field_testing_dataset = field_training_dataset
        field_training_generator = torch.utils.data.DataLoader(field_training_dataset, batch_size = 50, shuffle=True)
        field_testing_generator = torch.utils.data.DataLoader(field_testing_dataset, batch_size = 50)
        

        
        
        #gt_field_training_dataset = gt_field_dataset
        #gt_field_testing_dataset = gt_field_training_dataset
        #gt_field_training_generator = torch.utils.data.DataLoader(gt_field_training_dataset, batch_size = 1, shuffle=True)
        #gt_field_testing_generator = torch.utils.data.DataLoader(gt_field_testing_dataset, batch_size = 1)

        #i, (pos, pix) = next(enumerate(gt_field_training_dataset))
        
        if False:
            fig, ax = plt.subplots(1,1)#, figsize=(36,6))
            axes = [ax]

            boundsX = (0,ego_pixel_shape[1])
            boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
            axes[0].set_xlim(*boundsX)
            axes[0].set_ylim(*boundsY)

            #axes[1].set_xlim(*boundsX)
            #axes[1].set_ylim(*boundsY)

            tempval = axes[0].imshow(np.reshape(pix,(ego_pixel_shape)), extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')#, cmap='gnuplot')
        
            for traj in future_trajectory.values():
                trajnp = np.array(traj)
                axes[0].plot(trajnp[:,0], trajnp[:,1], 'r')
            
            cax = fig.add_axes([.3, .95, .4, .05])
            fig.colorbar(tempval, cax, orientation='horizontal')
            plt.show()




        #crowd_training_dataset3 = crowd_data_set3
        #crowd_testing_dataset3 = crowd_training_dataset3
        #crowd_training_generator3 = torch.utils.data.DataLoader(crowd_training_dataset3, batch_size = 1, shuffle=True)
        #crowd_testing_generator3 = torch.utils.data.DataLoader(crowd_testing_dataset3, batch_size = 1)



        #Create model
        print("Creating Network . . .")
        testModel = DNN.SirenMM(in_features = 2, hidden_features = 32, hidden_layers = 3, out_features = 1, outermost_linear=True) #PSIREN(map.shape[1], map.shape[0], 1000, 2000, 1000)  #8,5


        ##network = predModel;
        #optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        ##loss_function = nn.MSELoss()
        ##testModel = DNN.ConvolutionalNeuralProcessImplicit2DHypernet(in_features=1,out_features=1,image_resolution=(32,32))
        ##testModel = DNN.ConvolutionalNeuralProcessImplicit2DHypernet(in_features=3,out_features=1,image_resolution=(img_channel_swap.shape[1],img_channel_swap.shape[2]))
        #testModel.cuda()
        #testModel.eval()
        #testingTestModel = testModel({'embedding':None, 'img_sparse':torch.zeros((1,*img_channel_swap.shape)).cuda(), 'coords':torch.zeros(1,20,2).cuda()})
        ##testingTestModel = testModel({'embedding':None, 'img_sparse':torch.zeros((1,1,32,32)).cuda(), 'coords':torch.zeros(1,20,2).cuda()})
        ##predModel = DNN.SirenMM(in_features=2,out_features=1,hidden_features=64,num_hidden_layers=3) #Last used
        ##network = predModel#testModel#

        ##predModel = DNN.SirenMM(in_features=2,out_features=1,hidden_features=256,num_hidden_layers=3)
        ##predModel = DNN.SirenMM(in_features=2,out_features=1,hidden_features=256,num_hidden_layers=3)#,type='relu')
        network = testModel# predModel#

        network.cuda()
        network.eval()
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        
        print("Parameter count:", DNN.CountParameters(network))

        outputfile = "psiren_generic_test.pt"
        overfitoutputfile = "overfit_" + outputfile
        #DNN.trainAndGraphDerivative(network, gt_field_training_generator, gt_field_testing_generator, loss_function3, optimizer, num_epochs, learning_rate, print_interval )
        DNN.trainAndGraphDerivative(network, trajectory_training_generator, trajectory_testing_generator, loss_function, optimizer, num_epochs, learning_rate, outputfile, overfitoutputfile, print_interval, print_interval )
        #DNN.trainAndGraphDerivative(network, field_training_generator, field_testing_generator, loss_function, optimizer, num_epochs, learning_rate, print_interval )
        #DNN.trainAndGraphDerivative(network, hyper_training_generator, hyper_testing_generator, loss_function, optimizer, num_epochs, learning_rate, print_interval )
        #DNN.trainAndGraphDerivative(network, hyper_training_generator, hyper_testing_generator, loss_function3, optimizer, num_epochs, learning_rate, print_interval )
        ##DNN.trainAndGraphDerivative(network, trajectory_training_generator, trajectory_testing_generator, loss_function, optimizer, num_epochs, learning_rate, print_interval )
        #DNN.trainAndGraphDerivative2(network, (trajectory_training_generator, laplacian_training_generator), (trajectory_testing_generator, laplacian_testing_generator), (loss_function, loss_function2), optimizer, num_epochs, learning_rate, print_interval )  # I think Last used for RSS submission, smoothing
        ##trainAndGraphDerivative2(network3, (crowd_training_generator,crowd_training_generator3), (crowd_testing_generator, crowd_testing_generator3), (loss_function3,loss_function4), optimizer3, num_epochs, learning_rate, print_interval)
        




        
        #network = network.hypo_net
        #network.cpu()
        #network = network({'img_sparse':torch.zeros((1,1,32,32))})


        network.cpu()

        #torch.no_grad()
        # Is y x okay or should we do x y
        dense_scale = 1
        dense_coords = np.array( [[ [RecenterTrajDataForward((j+.5)/dense_scale),RecenterTrajDataForward((i+.5)/dense_scale)] for i in range(ego_pixel_shape[0]*dense_scale) for j in range(ego_pixel_shape[1]*dense_scale) ]], dtype=np.float32)
        dense_coords = torch.unsqueeze( torch.from_numpy(dense_coords), 0)

        all_coords = np.array( [[ [RecenterTrajDataForward(j+.5),RecenterTrajDataForward(i+.5)] for i in range(ego_pixel_shape[0]) for j in range(ego_pixel_shape[1]) ]], dtype=np.float32)
        all_coords = torch.unsqueeze( torch.from_numpy(all_coords), 0)

        all_coords = torch.unsqueeze( torch.from_numpy(RecenterTrajDataForward(all_pixel_coords.astype(np.float32))), 0)


        print(all_coords.shape)
        predictions = network({'coords':all_coords}) #,'img_sparse':torch.zeros((1,*img_channel_swap.shape))})
        if type(predictions) is dict:
            outImage = (predictions['model_out'], predictions['model_in'])
        else:
            outImage = predictions
        print('max',torch.max(outImage[0]))
        print('min',torch.min(outImage[0]))
        print(type(outImage))
        print(outImage[0].shape)
        print(outImage[1].shape)

        fig, axes = plt.subplots(1,2)#, figsize=(36,6))
        #axes = [ax]
        #axes.imshow(outImage[0].cpu().view(distImage.shape).detach().numpy())

        boundsX = (0,ego_pixel_shape[1])
        boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
        axes[0].set_xlim(*boundsX)
        axes[0].set_ylim(*boundsY)

        axes[1].set_xlim(*boundsX)
        axes[1].set_ylim(*boundsY)

        #axes[0].imshow(-outImage[0].cpu().view(ego_pixel_shape).detach().numpy(), extent=[*(minT,maxT), *(minR,maxR)], interpolation='none')#, cmap='gnuplot')
        outImagea = outImage[0].cpu().view(ego_pixel_shape).detach().numpy()
        tempval = axes[0].imshow(outImagea, extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')#, cmap='gnuplot')
        axes[1].imshow(outImagea, extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')#, cmap='gnuplot')
        
        # Rerun again for gradient
        predictions = network({'coords':all_coords}) #,'img_sparse':torch.zeros((1,*img_channel_swap.shape))})
        if type(predictions) is dict:
            outImage = (predictions['model_out'], predictions['model_in'])
        else:
            outImage = predictions
        outImageA = -DNN.gradient(*outImage)#-DNN.gradient(*predModel(outImage[1]))
        
        predictions = network({'coords':dense_coords}) #,'img_sparse':torch.zeros((1,*img_channel_swap.shape))})
        if type(predictions) is dict:
            outImage = (predictions['model_out'], predictions['model_in'])
        else:
            outImage = predictions
        laplacianA = np.squeeze(DNN.laplace(*outImage).detach().numpy())

        outImageA = outImageA[0].cpu().view(*ego_pixel_shape,2).detach().numpy() 
        print(outImageA.shape)

        fig2, ax2 = plt.subplots(1,1)
        ax2.hist(laplacianA, bins=2000)


        coord_x, coord_y = np.meshgrid(range(ego_pixel_shape[1]), range(ego_pixel_shape[0]))

        # NORMALIZE
        vx = outImageA[:,:,0]#.flatten('F')
        print(type(vx))
        vy = outImageA[:,:,1]#.flatten('F')
        ux = vx#/np.sqrt(vx**2+vy**2)
        uy = vy#/np.sqrt(vx**2+vy**2)



        axes[0].quiver(coord_x+.5, coord_y+.5, ux,uy, color='red')#, units='xy' ,scale=1

        ux = vx/np.sqrt(vx**2+vy**2)
        uy = vy/np.sqrt(vx**2+vy**2)

        axes[1].quiver(coord_x+.5, coord_y+.5, ux,uy, color='red')

        for traj in future_trajectory.values():
            trajnp = np.array(traj)
            axes[0].plot(trajnp[:,0], trajnp[:,1], 'r')
            axes[1].plot(trajnp[:,0], trajnp[:,1], 'r')
            #axes[0].plot(tpix, logrpix, 'r')
            #axes[1].plot(tpix, logrpix, 'r')

        #print(testpos)
            axes[0].plot(*traj[-1], 'co',markersize=2)
            axes[1].plot(*traj[-1], 'co',markersize=2)
        #axes[1].plot(*target_pos, 'co')
        #axes[2].plot(*target_pos, 'co')
        #axes[3].plot(*target_pos, 'co')

        for obs_traj in obstacle_trajectory.values():
            trajnp = np.array(obs_traj)
            axes[0].plot(trajnp[:,0], trajnp[:,1], 'c')
            axes[1].plot(trajnp[:,0], trajnp[:,1], 'c')


        
        cax = fig.add_axes([.3, .95, .4, .05])
        fig.colorbar(tempval, cax, orientation='horizontal')
        #fig.colorbar(outImagea, axes[0], orientation='vertical')


        fig, ax = plt.subplots(1,1)

        boundsX = (0,ego_pixel_shape[1])
        boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
        ax.set_xlim(*boundsX)
        ax.set_ylim(*boundsY)
        ax.set_aspect(1)
        ax.imshow(outImagea, extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')
        #coord_x, coord_y = np.meshgrid(range(ego_pixel_shape[1]), range(ego_pixel_shape[0]))

        ax.quiver(gradient_samples[0][:,0], gradient_samples[0][:,1], gradient_samples[1][:,0],gradient_samples[1][:,1], color='red', scale_units='xy', scale=1)#, units='xy' ,scale=1

        #ux = vx/np.sqrt(vx**2+vy**2)
        #uy = vy/np.sqrt(vx**2+vy**2)
        
        for traj in future_trajectory.values():
            trajnp = np.array(traj)
            ax.plot(trajnp[:,0], trajnp[:,1], 'r')

        #plt.show()










        #axes[1,0].imshow(a_star_map_x, extent=[*boundsX, *boundsY], interpolation='none')
        #axes[1,1].imshow(a_star_map_y, extent=[*boundsX, *boundsY], interpolation='none')


        plt.show()

































        fig, axes = plt.subplots(1,2)#, figsize=(18,6))
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
            
    #print('\n\n\n')
    #print(type(vTR))
    #print('\n\n\n')
    #print(type(vTR['vTr']))
    #print('\n\n\n')
    #print(vTR['vTr'][0]['up'])

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
