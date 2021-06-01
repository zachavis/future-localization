import sys
from sys import platform
import os
import random

USING_LINUX = platform == "linux" or platform == "linux2"

if not USING_LINUX:
    import matplotlib.pyplot as plt
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
from scipy import stats


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





def DistanceFromLine (line, point):
    homogPoint = np.array([point[0], point[1], 1])
    proj = np.dot(line, homogPoint)
    lineNormal = np.linalg.norm(np.array([line[0],line[1]]))
    return abs(proj / lineNormal)

def skewPix(x):
    return np.array([[0, -1, x[1]],
                     [1, 0, -x[0]],
                     [-x[1], x[0], 0]])

def GetLine(ptA, ptB):
    return skewPix(ptA) @ ptB

def GetPoint(x):
    if len(x) >= 3:
        return x/x[2]
    return np.concatenate((x,np.ones(1)))





############################################################
### ### ### ### ### ## MAIN EXECUTION ## ### ### ### ### ###
############################################################


import sys
import getopt
from pathlib import Path


PRINT_DEBUG_IMAGES = False and not USING_LINUX
LOAD_NETWORK_FROM_DISK = False
#np.seterr(invalid='raise')
USE_EGO = True

if __name__ == "__main__":
    #DNN.current_epoch = 0

    BATCH_SIZE = 16
    N_WORKERS = 0
    
    # TODO put these values in a settings file on disk to force uniformity across programs
    img_height = 192 #256#128#64

    minR = -.5
    maxR = 4#5 #4.5
    minT = -np.pi/3 #-2*np.pi/3
    maxT = -minT #2*np.pi/3
    aspect_ratio = 1 #3/4 #2*#(maxT-minT)/(maxR-minR)
    ego_pixel_shape = (img_height,int(img_height*aspect_ratio)) # y,x | vert,horz
    
    FILE_UPPER_LIMIT = 1000 # a number larger than the number of images in a single directory, used for dictionary indexing
    n_folders = 1


    # FORNOW: Just going to assume it's only in train mode
    # TODO: allow loading of model
    necessary_args = 0
    verbose_flag = False
    data_root = Path('S:/fut_loc/')
    inputfile = ''
    outputfile = ''
    overfitoutputfile = ''
    
    train_test_directories = ['train','test']

    try:
        opts, args = getopt.getopt(sys.argv[1:],"vd:i:o:",["verbose","data=","imodel=","omodel="])
    except getopt.GetoptError:
        print('Arguments are malformed. TODO: put useful help here.')#'test.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--data"):
            data_root = Path(arg)
            data_train = data_root / train_test_directories[0]
            data_test = data_root / train_test_directories[1]
            if not data_root.exists():
                print("Data path does not exist.")
                sys.exit(3)
            if not data_train.exists():
                print("Training data does not exist.")
                sys.exit(3)
            if not data_test.exists():
                print("Testing data does not exist.")
                sys.exit(3)
            necessary_args += 1
        if opt in ("-i", "--imodel"):
            print("Input not implemented.")
            sys.exit(3)
            inputfile = arg
            if not inputfile.exists():
                print("Input path does not exist.")
                sys.exit(3)
            elif inputfile.suffix != '.pt':
                print("Input model must be a pt file.")
                sys.exit(3)
            necessary_args += 1
        if opt in ("-o", "--omodel"):
            fullpath = Path(arg)
            outputdir = fullpath.parent
            outputfile = fullpath.name
            outputext = fullpath.suffix

            overfitoutputfile = fullpath.parent / Path('overfit_' + str(outputfile))
           
            if not outputdir.exists():
                outputdir.mkdir()
            if outputext != '.pt':
                print("Output model must be a pt file.")
                sys.exit(3)
            outputfile = fullpath
            necessary_args += 1
        if opt in ("-v", "--verbose"):
            verbose_flag = True
    if necessary_args < 2:
        print('Not enough args')
        sys.exit(3)
    #print('Input file is "', inputfile)





    #print(os.getcwd())
    #loc = r'H:\fut_loc\20150401_walk_00\traj_prediction.txt'

    #partial_folder_path = data_root #20150401_walk_00\\'



   # LOADING VARIABLES
    RESIZED_IMAGE_DICTIONARY = {} # could probably pre-allocate this into a big numpy array...
    LOG_POLAR_TRAJECTORY_DICTIONARY = {}
    COORD_TRAJECTORY_DICTIONARY = {}
    PIXEL_TRAJECTORY_DICTIONARY = {}

    RAW_IMAGE_DICTIONARY = {}
    TRAJ_IN_IMAGE_DICTIONARY = {}

    # Testing VARIABLES
    RESIZED_IMAGE_DICTIONARY_TE = {} # could probably pre-allocate this into a big numpy array...
    LOG_POLAR_TRAJECTORY_DICTIONARY_TE = {}
    COORD_TRAJECTORY_DICTIONARY_TE = {}
    PIXEL_TRAJECTORY_DICTIONARY_TE = {}

    RAW_IMAGE_DICTIONARY_TE = {}
    TRAJ_IN_IMAGE_DICTIONARY_TE = {}

    # Training VARIABLES
    RESIZED_IMAGE_DICTIONARY_TR = {} # could probably pre-allocate this into a big numpy array...
    LOG_POLAR_TRAJECTORY_DICTIONARY_TR = {}
    COORD_TRAJECTORY_DICTIONARY_TR = {}
    PIXEL_TRAJECTORY_DICTIONARY_TR = {}

    RAW_IMAGE_DICTIONARY_TR = {}
    TRAJ_IN_IMAGE_DICTIONARY_TR = {}


    #data_subdirectories = [x for x in partial_folder_path.iterdir() if x.is_dir()]
    #data_subdirectories_test = data_subdirectories[-4:]

    for data_subset in train_test_directories:

        if data_subset == 'train':
            RESIZED_IMAGE_DICTIONARY = RESIZED_IMAGE_DICTIONARY_TR
            LOG_POLAR_TRAJECTORY_DICTIONARY = LOG_POLAR_TRAJECTORY_DICTIONARY_TR
            COORD_TRAJECTORY_DICTIONARY = COORD_TRAJECTORY_DICTIONARY_TR
            PIXEL_TRAJECTORY_DICTIONARY = PIXEL_TRAJECTORY_DICTIONARY_TR

            RAW_IMAGE_DICTIONARY = RAW_IMAGE_DICTIONARY_TR
            TRAJ_IN_IMAGE_DICTIONARY = TRAJ_IN_IMAGE_DICTIONARY_TR
            
        
        if data_subset == 'test':
            RESIZED_IMAGE_DICTIONARY = RESIZED_IMAGE_DICTIONARY_TE
            LOG_POLAR_TRAJECTORY_DICTIONARY = LOG_POLAR_TRAJECTORY_DICTIONARY_TE
            COORD_TRAJECTORY_DICTIONARY = COORD_TRAJECTORY_DICTIONARY_TE
            PIXEL_TRAJECTORY_DICTIONARY = PIXEL_TRAJECTORY_DICTIONARY_TE

            RAW_IMAGE_DICTIONARY = RAW_IMAGE_DICTIONARY_TE
            TRAJ_IN_IMAGE_DICTIONARY = TRAJ_IN_IMAGE_DICTIONARY_TE


        partial_folder_path = data_root / data_subset
        count = 0 # folder ID


        for folder_path in partial_folder_path.iterdir(): #next(os.walk(partial_folder_path))[1]:
            if not folder_path.is_dir():
                continue

            if count >= n_folders:
                break

            folder_name = folder_path.name
            #folder_path =  partial_folder_path / folder_name

            # load calibration
            print('loading calibration file')
            calibfile = folder_path / 'calib_fisheye.txt'

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
            file_list = folder_path / 'im_list.list'
            #fid = open(file_list);
            #data = textscan(fid, '%s');
            #data = fid.readlines()
            with open(file_list) as fid:
                data = fid.read().splitlines()
            vFilename = data;
            #fid.close();





            print('loading trajectory file')
            traj_data_file = folder_path / 'traj_prediction.txt'
            vTR = ReadTraj(traj_data_file)

        
            image_path = folder_path / 'im'
            frameOffset = 0
            #test = [x for x in (image_path).iterdir() if x.is_file() and x.suffix == '.png']
            frameEnd = len([x for x in (image_path).iterdir() if x.is_file() and x.suffix == '.png']) #os.listdir(folder_path / 'im')) #55
            imageScale = .1

            for iFrame in range(frameOffset,frameEnd):

                dictionary_index = iFrame + count * FILE_UPPER_LIMIT

                tr = vTR['vTr'][iFrame]
                if (len(tr['XYZ'][1]) < 2):
                    print('FROM FOLDER', folder_name,'SKIPPING FRAME',iFrame,': \n\tTrajectory is deficient (less than 2 points)')
                    continue
                else:
                    print('FROM FOLDER', folder_name, 'GETTING FRAME',iFrame )

                tr = vTR['vTr'][iFrame]

                #im = sprintf('%sim/%s', folder_path, vFilename{iFrame});
                im = folder_path / 'im' / str(vFilename[iFrame]) #"{}im\\{}".format(folder_path, vFilename[iFrame])

                if not os.path.isfile(im): #TODO: Changeme
                    print('could not find file')
                    continue

                img = cv2.cvtColor(cv2.imread(str(im.resolve())), cv2.COLOR_BGR2RGB).astype(np.float64)/255.0
                #cv2.imshow('intermediate',intermediate)
                #cv2.waitKey()

                #im = im2double(intermediate); % im2double(imread(im));

                tr_ground_OG = (tr['XYZ'].T - tr['up']).T #bsxfun(@minus, tr['XYZ'], tr['up']);


        
                
                tr_ground = K_data @ R_rect @ tr_ground_OG;
                
                
                #if np.any(tr_ground[2,:]<0):
                #    #tr_ground[:2,:] = np.nan # actually maybe I shouldn't NAN here, the only issue is that the trajectory goes behind the camera. Maybe that's okay?
                #    print('\tThe trajectory is suspicious, and may be behind the user. SKIPPING')
                #    continue




                #tr_ground(tr_ground(3,:)<0, :) = NaN;
                #tr_ground = bsxfun(@rdivide, tr_ground(1:2,:), tr_ground(3,:));
                tr_ground = tr_ground[:2] / tr_ground[2]
            
                #if (LOAD_NETWORK_FROM_DISK):
                #    TRAJ_IN_IMAGE_DICTIONARY[dictionary_index] = tr_ground
                #    RAW_IMAGE_DICTIONARY[dictionary_index] = img

        
                # set up egocentric image information in log polar space

                #t, r = Coord2Polar(tr['XYZ'][2],tr['XYZ'][0])

                


                # set up egocentric image information in log polar space


                

               
                #world_forward = 

                r_y = -tr['up']/np.linalg.norm(tr['up'])

                v = tr['XYZ'][:,0]/np.linalg.norm(tr['XYZ'][:,0])
                if v @ np.array([0,0,1]) > .2:
                    old_r_z = v
                else:
                    print("\tSkipping because of alignment severity")
                    continue
                r_z = old_r_z - (old_r_z@r_y)*r_y
                r_z /= np.linalg.norm(r_z)
                r_x = np.cross(r_y, r_z)

                R_rect_ego = np.stack((r_x,r_y,r_z),axis=0)


                tr_ground_ALIGNED = R_rect_ego @ tr_ground_OG # ALIGN CAMERA SPACE GROUND PLANE TO "WORLD SPACE"
                t, r = Coord2Polar(tr_ground_ALIGNED[2],tr_ground_ALIGNED[0])#Coord2Polar(tr['XYZ'][2],tr['XYZ'][0])
                #t, r = Coord2Polar(tr_ground[2],tr_ground[0])


                test = r < np.exp(minR)
                if (np.any(r < np.exp(minR))):
                    print('\tTrajectory is too close to camera origin.')
                    continue
                    #print('')
                logr = np.log(r)





                


















                ego_r2pix = lambda x : RemapRange(x, minR,maxR, 0,                  ego_pixel_shape[0]  )
                ego_t2pix = lambda x : RemapRange(x, minT,maxT, 0,                  ego_pixel_shape[1]  )

        
                ego_pix2r = lambda x : RemapRange(x, 0, ego_pixel_shape[0], minR,maxR   )
                ego_pix2t = lambda x : RemapRange(x,0,ego_pixel_shape[1], minT,maxT  )

                RecenterDataForwardWithShape = lambda x, shape : RemapRange(x,0,max(shape[0],shape[1]),-1,1)
                RecenterDataForwardWithShapeAndScale = lambda x, shape, scale : RemapRange(x,0,max(shape[0],shape[1]),-scale,scale)
                RecenterDataBackwardWithShape = lambda x, shape : RemapRange(x,-1,1,0,max(shape[0],shape[1]))
                RecenterTrajDataForward = lambda x : RecenterDataForwardWithShape(x,ego_pixel_shape)
                RecenterTrajDataForward2 = lambda x : RecenterDataForwardWithShapeAndScale(x,ego_pixel_shape,1)

        
                RecenterDataBackwardWithShapeAndScale = lambda x, shape, scale : RemapRange(x,-scale,scale,0,max(shape[0],shape[1]))
                RecenterFieldDataBackward = lambda x : RecenterDataBackwardWithShapeAndScale(x,ego_pixel_shape,1)
        

                tpix = ego_t2pix(t)
                logrpix = ego_r2pix(logr)













                
                # Generating EgoRetinalMap
                if USE_EGO:
                    all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(ego_pixel_shape[0]) for j in range(ego_pixel_shape[1]) ], dtype=np.float32)
                    all_pixel_coords[:,0] = ego_pix2t(all_pixel_coords[:,0])
                    all_pixel_coords[:,1] = ego_pix2r(all_pixel_coords[:,1])
                    all_pixel_coords[:,1] = np.exp(all_pixel_coords[:,1])
                    z, x = Polar2Coord(all_pixel_coords[:,0],all_pixel_coords[:,1])

                    coords_3D = np.zeros((len(z),3))
                    coords_3D[:,0] = x
                    coords_3D[:,2] = z
        
                    coords_3D = (R_rect_ego.T @ coords_3D.T).T # ALIGN "WORLD-SPACE" GROUND PLANE TO CAMERA SPACE
                    coords_3D -= tr['up'].T # SHIFT PLANE TO CORRECT LOCATION RELATIVE TO CAMERA


                    #coords_3D[:,1] *= -1

                    pixels = K_data @ R_rect @ coords_3D.T
                    pixels /= pixels[2]
                    #pixels[:,:] /= pixels[2,:]

                    rowmaj_pixels = np.zeros(pixels.shape)
                    rowmaj_pixels[0] = pixels[1]
                    rowmaj_pixels[1] = pixels[0]


                    img_resized =      interpolate.interpn((range(img.shape[0]),range(img.shape[1])), img*2.0-1.0, rowmaj_pixels[:2].T , method = 'linear',bounds_error = False, fill_value = 0).reshape(ego_pixel_shape[0], ego_pixel_shape[1],3)
                    img_channel_swap = np.moveaxis(img_resized,-1,0).astype(np.float32)


                else:
                    homography = K_data @ R_rect @ R_rect_ego @ R_rect.T @ np.linalg.inv(K_data)

                    img_rectified = cv2.warpPerspective(img*2.0-1.0, homography, (img.shape[1], img.shape[0])) # want to shift the values here so that the normalized version has black in the rectified location

       

                    img_resized = cv2.resize(img_rectified, (int(img_rectified.shape[1]*imageScale), int(img_rectified.shape[0]*imageScale)))
                    img_channel_swap = np.moveaxis(img_resized,-1,0).astype(np.float32)









         
                if (PRINT_DEBUG_IMAGES):
                    fig, axes = plt.subplots(1,2)#, figsize=(18,6))
                    if USE_EGO:
                        axes[0].imshow(img)
                        #axes[1].imshow(img_rectified)
                        boundsX = (0,ego_pixel_shape[1])
                        boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
            
                        axes[1].set_xlim(*boundsX)
                        axes[1].set_ylim(*boundsY)
                        axes[1].set_aspect(1)
                    axes[1].imshow(img_resized)
                    plt.show()

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






















                #future_trajectory = {}

                #for k in range(1):
                future_trajectory = []
                for i in range(len(logrpix)):
                    future_trajectory.append( (tpix[i], logrpix[i]) ) # t is horizontal axis, logr is vertical
                    #PIXEL_TRAJECTORY_DICTIONARY[dictionary_index].append( (tpix[i], logrpix[i]) ) # t is horizontal axis, logr is vertical
            
                traj = np.array(future_trajectory,dtype=np.float32)
                next_pts = np.ones((traj.shape[0]-1, 3))
                prev_pts = np.ones((traj.shape[0]-1, 3))
                next_pts[:,:2] = traj[1:]
                prev_pts[:,:2] = traj[:-1]


                vecLines = next_pts[:,:2]-prev_pts[:,:2]
                vecLinesMag = np.linalg.norm(vecLines,axis=1)
                if (np.any(vecLinesMag < .000001)):
                    print('\tTrajectory is deficient (overlapping points)')
                    continue
                
                
                
               #np.random.seed(8980)



        
                # Let's try to get a ground truth image


           
                # Let's try to get a ground truth image
                all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(ego_pixel_shape[0]) for j in range(ego_pixel_shape[1]) ], dtype=np.float32)
                #coord_value = np.zeros((all_pixel_coords.shape[0]))


                all_pixel_coords_xformed = np.zeros(all_pixel_coords.shape).astype(np.float32)
                all_pixel_coords_xformed[:,0] = ego_pix2t(all_pixel_coords[:,0])
                all_pixel_coords_xformed[:,1] = np.exp(ego_pix2r(all_pixel_coords[:,1]))

                all_pixel_coords_xformed = np.array(Polar2Coord(all_pixel_coords_xformed[:,0],all_pixel_coords_xformed[:,1])).T
                #newtraj = future_trajectory.copy()
                #newtraj = {}
                #newtraj[0] = []
        

                # Check if first two pixels are within egomap
                pix1 = future_trajectory[0]
                pix2 = future_trajectory[1]

                if (pix1[0] < 0 or pix1[0] > ego_pixel_shape[1] 
                    or pix1[1] < 0 or pix1[1] > ego_pixel_shape[0]
                    or pix2[0] < 0 or pix2[0] > ego_pixel_shape[1] 
                    or pix2[1] < 0 or pix2[1] > ego_pixel_shape[0]):

                    print('\tTrajectory is deficient (start is outside egomap)')
                    continue

                # START POPULATING DICTIONARIES            
                if (LOAD_NETWORK_FROM_DISK):
                    TRAJ_IN_IMAGE_DICTIONARY[dictionary_index] = tr_ground
                    RAW_IMAGE_DICTIONARY[dictionary_index] = img

                            
                RESIZED_IMAGE_DICTIONARY[dictionary_index] = img_channel_swap
                PIXEL_TRAJECTORY_DICTIONARY[dictionary_index] = []
                LOG_POLAR_TRAJECTORY_DICTIONARY[dictionary_index] = []
                COORD_TRAJECTORY_DICTIONARY[dictionary_index] = []
                for pix in future_trajectory:
                    if (pix[0] < 0 or pix[0] > ego_pixel_shape[1] or pix[1] < 0 or pix[1] > ego_pixel_shape[0]): # outside ego map
                        break
                    PIXEL_TRAJECTORY_DICTIONARY[dictionary_index].append( (pix[0], pix[1]) ) # t is horizontal axis, logr is vertical
                    logpolar_coord = (ego_pix2t(pix[0]),ego_pix2r(pix[1]))
                    newpoint = ( Polar2Coord( logpolar_coord[0],np.exp(logpolar_coord[1]) ) )
                    LOG_POLAR_TRAJECTORY_DICTIONARY[dictionary_index].append( logpolar_coord )
                    COORD_TRAJECTORY_DICTIONARY[dictionary_index].append( newpoint )
        
                #for i in range(len(logrpix)):
                #    LOG_POLAR_TRAJECTORY_DICTIONARY[iFrame].append( (tpix[i], logrpix[i]) ) # t is horizontal axis, logr is vertical



                #coord_value = DataGens.Coords2ValueFast(all_pixel_coords,future_trajectory,nscale=1)

                if (PRINT_DEBUG_IMAGES):
                    coord_value = DataGens.Coords2ValueFastWS(all_pixel_coords_xformed,{0:COORD_TRAJECTORY_DICTIONARY[dictionary_index]},None,None,stddev=.5)
                    fig, ax = plt.subplots(1,1)#, figsize=(36,6))
                    axes = [ax]

                    boundsX = (0,ego_pixel_shape[1])
                    boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
                    axes[0].set_xlim(*boundsX)
                    axes[0].set_ylim(*boundsY)

                    #axes[1].set_xlim(*boundsX)
                    #axes[1].set_ylim(*boundsY)

                    tempval = axes[0].imshow(np.reshape(coord_value,(ego_pixel_shape)), extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')#, cmap='gnuplot')
                    
                    pixels = np.array(LOG_POLAR_TRAJECTORY_DICTIONARY[dictionary_index])
        
                    trajnp = np.array(future_trajectory)
                    axes[0].plot(trajnp[:,0], trajnp[:,1], 'r')
        
                    cax = fig.add_axes([.3, .95, .4, .05])
                    fig.colorbar(tempval, cax, orientation='horizontal')
                    plt.show()

            count += 1

    print(' ')
    print('!!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! ')
    print('\tThere are',len(RESIZED_IMAGE_DICTIONARY_TR),'training data points.')
    print('\tThere are',len(RESIZED_IMAGE_DICTIONARY_TE),'testing data points.')
    print('!!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! ')
    print(' ')



    if (not LOAD_NETWORK_FROM_DISK):
        # Now, Let's train the neural network.

        n_samples = 30000 # 25000
        n_obstsamples = 1#30000
        #n_2sample_laplacian = 5000
        n_2sample = 10000#15000


        if (PRINT_DEBUG_IMAGES):
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

    

        #hyper_trajectory_data_set = DataGens.HyperTrajectoryDataset(future_trajectory, RecenterTrajDataForward,torch.zeros((1,32,32)))
        #hyper_trajectory_data_set = DataGens.HyperTrajectoryDataset2(future_trajectory,n_2sample_laplacian,RecenterTrajDataForward,all_pixel_coords,img_channel_swap) #DataGens.HyperTrajectoryDataset(future_trajectory, RecenterTrajDataForward, img_channel_swap)
        #hyper_trajectory_data_set = DataGens.HyperTrajectoryDataset2(newtraj,n_2sample_laplacian,RecenterTrajDataForward,RecenterFieldDataBackward,all_pixel_coords,img_channel_swap,ego_pix2t,ego_pix2r,Polar2Coord) #DataGens.HyperTrajectoryDataset(future_trajectory, RecenterTrajDataForward, img_channel_swap)
        #i, (pos, pix) = next(enumerate(hyper_trajectory_data_set))

        #hyper_trajectory_data_set = DataGens.MassiveHyperTrajectoryDataset(LOG_POLAR_TRAJECTORY_DICTIONARY, n_2sample, RecenterTrajDataForward,RecenterFieldDataBackward,all_pixel_coords,RESIZED_IMAGE_DICTIONARY,ego_pix2t,ego_pix2r,Polar2Coord) #DataGens.HyperTrajectoryDataset(future_trajectory, RecenterTrajDataForward, img_channel_swap)
        #i, (pos, pix) = next(enumerate(hyper_trajectory_data_set))




        #hyper_trajectory_data_set_tr = DataGens.MassiveHyperTrajectoryDataset(LOG_POLAR_TRAJECTORY_DICTIONARY_TR, n_2sample, RecenterTrajDataForward,RecenterFieldDataBackward,all_pixel_coords,RESIZED_IMAGE_DICTIONARY_TR,ego_pix2t,ego_pix2r,Polar2Coord) #DataGens.HyperTrajectoryDataset(future_trajectory, RecenterTrajDataForward, img_channel_swap)
        ##i, (pos, pix) = next(enumerate(hyper_trajectory_data_set_tr))

        #hyper_trajectory_data_set_te = DataGens.MassiveHyperTrajectoryDataset(LOG_POLAR_TRAJECTORY_DICTIONARY_TE, n_2sample, RecenterTrajDataForward,RecenterFieldDataBackward,all_pixel_coords,RESIZED_IMAGE_DICTIONARY_TE,ego_pix2t,ego_pix2r,Polar2Coord) #DataGens.HyperTrajectoryDataset(future_trajectory, RecenterTrajDataForward, img_channel_swap)
        ##i, (pos, pix) = next(enumerate(hyper_trajectory_data_set_te))

        hyper_trajectory_data_set_tr = DataGens.MassiveAutoEncoderTrajectoryDataset(PIXEL_TRAJECTORY_DICTIONARY_TR, n_2sample, RecenterTrajDataForward,RecenterFieldDataBackward,all_pixel_coords,RESIZED_IMAGE_DICTIONARY_TR,ego_pix2t,ego_pix2r,Polar2Coord)
        hyper_trajectory_data_set_te = DataGens.MassiveAutoEncoderTrajectoryDataset(PIXEL_TRAJECTORY_DICTIONARY_TE, n_2sample, RecenterTrajDataForward,RecenterFieldDataBackward,all_pixel_coords,RESIZED_IMAGE_DICTIONARY_TE,ego_pix2t,ego_pix2r,Polar2Coord)



        #gt_field_dataset = DataGens.GTFieldDataset(future_trajectory, n_2sample_laplacian, RecenterTrajDataForward, all_pixel_coords)
        ##gt_field_data_generator = torch.utils.data.DataLoader(gt_field_dataset, batch_size= 1, shuffle=True)











        #Training parameters
        num_epochs = 2500 #15000 #1000
        print_interval = 1
        learning_rate = 5e-5#1e-5
        #loss_function = DNN.gradients_mse_with_coords #gradients_and_laplacian_mse_with_coords #nn.MSELoss()
        #loss_function2 = DNN.laplacian_mse_with_coords
        loss_function3 = DNN.auto_encoder_loss_with_coords
        

        #trajectory_training_dataset = trajectory_data_set
        #trajectory_testing_dataset = trajectory_training_dataset
        #trajectory_training_generator = torch.utils.data.DataLoader(trajectory_training_dataset, batch_size = 50, shuffle=True)
        #trajectory_testing_generator = torch.utils.data.DataLoader(trajectory_testing_dataset, batch_size = 50)

        #laplacian_training_dataset = laplacian_data_set
        #laplacian_testing_dataset = laplacian_training_dataset
        #laplacian_training_generator = torch.utils.data.DataLoader(laplacian_training_dataset, batch_size = 50, shuffle=True)
        #laplacian_testing_generator = torch.utils.data.DataLoader(laplacian_testing_dataset, batch_size = 50)
        
        hyper_training_set = hyper_trajectory_data_set_tr
        hyper_testing_set = hyper_trajectory_data_set_te
        hyper_training_generator = torch.utils.data.DataLoader(hyper_training_set, num_workers = N_WORKERS, batch_size = BATCH_SIZE, shuffle=True)
        hyper_testing_generator = torch.utils.data.DataLoader(hyper_testing_set, num_workers = N_WORKERS,  batch_size = BATCH_SIZE)
        
        #i, (pos, pix) = next(enumerate(hyper_training_generator))
        
        
        if (False and PRINT_DEBUG_IMAGES): # The false is because the hyper generator uses random points instead of pixel centers
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





        #Create model
        print("Creating Network . . .")
        predModel = DNN.Siren(in_features = 2, hidden_features = 32, hidden_layers = 3, out_features = 1, outermost_linear=True) #PSIREN(map.shape[1], map.shape[0], 1000, 2000, 1000)  #8,5


        network = predModel;
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        #loss_function = nn.MSELoss()
        #testModel = DNN.ConvolutionalNeuralProcessImplicit2DHypernet(in_features=1,out_features=1,image_resolution=(32,32))
        testModel = DNN.ConvolutionalAutoEncoderToPath2(in_features=3,path_length=25,image_resolution=(img_channel_swap.shape[1],img_channel_swap.shape[2]))
        
        #DNN.ConvolutionalNeuralProcessImplicit2DHypernetWithMultiplier(in_features=3,out_features=1,image_resolution=(img_channel_swap.shape[1],img_channel_swap.shape[2]))
        testModel.cuda()
        testModel.eval()
        #testingTestModel = testModel({'embedding':None, 'img_sparse':torch.zeros((1,*img_channel_swap.shape)).cuda(), 'coords':torch.zeros(1,20,2).cuda()})
        
        #testingTestModel = testModel({'embedding':None, 'img_sparse':torch.zeros((1,1,32,32)).cuda(), 'coords':torch.zeros(1,20,2).cuda()})
        #predModel = DNN.SirenMM(in_features=2,out_features=1,hidden_features=64,num_hidden_layers=3) #Last used
        #network = predModel#testModel#

        #predModel = DNN.SirenMM(in_features=2,out_features=1,hidden_features=256,num_hidden_layers=3)
        #predModel = DNN.SirenMM(in_features=2,out_features=1,hidden_features=256,num_hidden_layers=3)#,type='relu')
        network = testModel# predModel#
        
        if torch.cuda.device_count() > 1:
            print("Utilizing {} GPUs.".format(torch.cuda.device_count()))
            network = torch.nn.DataParallel(network)



        # TODO: Allow flexibility for running code on systems without GPU 
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # x = x.to(device)

        network.cuda()
        network.eval()
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        
        print("Total Parameter count:", DNN.CountParameters(network))
        print("Encoder Params:",DNN.CountParameters(network.encoder))
        print("Dencoder Params:",DNN.CountParameters(network.decoder))
        #print("Hyper Params:",DNN.CountParameters(network.hyper_net))
        #print("Hypo/SIREN Params:",DNN.CountParameters(network.hypo_net))
        #print("Walkable Params:",DNN.CountParameters(network.multiplier_net))



        # 0.13732084323360463 !!!!tr 
        # 0.13776845484972 !!!!te 

        #0.03715017518837645 !!!!tr
        #0.1310737913563138 !!!!te 
        
        #network = torch.load('overfit_ego_map_all_imgs.pt')#('overfit_skinny_exp_all_imgs.pt')
        
        #if type(network) == torch.nn.DataParallel:
        #    network = network.module
        
        #network.cuda()

        #train_loss = DNN.test_with_grad(network, hyper_training_generator, loss_function3, 0)
        #test_loss = DNN.test_with_grad(network, hyper_testing_generator, loss_function3, 0)
        #print(train_loss,"!!!!tr")
        #print(test_loss,"!!!!te")



        #DNN.trainAndGraphDerivative(network, gt_field_training_generator, gt_field_testing_generator, loss_function3, optimizer, num_epochs, learning_rate, print_interval )
        
        #DNN.trainAndGraphDerivative(network, field_training_generator, field_testing_generator, loss_function, optimizer, num_epochs, learning_rate, print_interval )
        #DNN.trainAndGraphDerivative(network, hyper_training_generator, hyper_testing_generator, loss_function, optimizer, num_epochs, learning_rate, print_interval )
        DNN.trainAndGraphDerivative(network, hyper_training_generator, hyper_testing_generator, loss_function3, optimizer, num_epochs, learning_rate, outputfile, overfitoutputfile, print_interval )
        ##DNN.trainAndGraphDerivative(network, trajectory_training_generator, trajectory_testing_generator, loss_function, optimizer, num_epochs, learning_rate, print_interval )
        #DNN.trainAndGraphDerivative2(network, (trajectory_training_generator, laplacian_training_generator), (trajectory_testing_generator, laplacian_testing_generator), (loss_function, loss_function2), optimizer, num_epochs, learning_rate, print_interval )  # I think Last used for RSS submission, smoothing
        ##trainAndGraphDerivative2(network3, (crowd_training_generator,crowd_training_generator3), (crowd_testing_generator, crowd_testing_generator3), (loss_function3,loss_function4), optimizer3, num_epochs, learning_rate, print_interval)
    
        #torch.save(network, overfitoutputfile) #'hypernet_1200imgs_300epochs.pt')

    else:
        network = torch.load('hypernet_44imgs_2500epochs.pt')
        network.cuda()
        network.eval()
        print("Parameter count:", DNN.CountParameters(network))




        
    #network = network.hypo_net
    #network.cpu()
    #network = network({'img_sparse':torch.zeros((1,1,32,32))})
    if not USING_LINUX:
        for frame in RESIZED_IMAGE_DICTIONARY.keys():
            testFrame = frame
            test_image = RESIZED_IMAGE_DICTIONARY[testFrame] # img_channel_swap
            test_pix_trajectory = PIXEL_TRAJECTORY_DICTIONARY[testFrame]
            test_ws_trajectory = LOG_POLAR_TRAJECTORY_DICTIONARY[testFrame]

            if (LOAD_NETWORK_FROM_DISK):
                raw_image = RAW_IMAGE_DICTIONARY[testFrame]
                raw_trajectory = TRAJ_IN_IMAGE_DICTIONARY[testFrame]


            all_pixel_coords_xformed = np.zeros(all_pixel_coords.shape).astype(np.float32)
            all_pixel_coords_xformed[:,0] = ego_pix2t(all_pixel_coords[:,0])
            all_pixel_coords_xformed[:,1] = np.exp(ego_pix2r(all_pixel_coords[:,1]))
            all_pixel_coords_xformed = np.array(Polar2Coord(all_pixel_coords_xformed[:,0],all_pixel_coords_xformed[:,1])).T

            test_coord_value = DataGens.Coords2ValueFastWS(all_pixel_coords_xformed,{0:test_ws_trajectory},None,None,stddev=2)

            test_image_prediction = torch.from_numpy( np.expand_dims(test_image,0) )

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
            predictions = network({'coords':all_coords,'img_sparse':test_image_prediction})
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
            fig.suptitle('Comparison of Gradients of Network Output')
            #axes = [ax]
            #axes.imshow(outImage[0].cpu().view(distImage.shape).detach().numpy())

            boundsX = (0,ego_pixel_shape[1])
            boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
            axes[0].set_title('Unnormallized')
            axes[0].set_xlim(*boundsX)
            axes[0].set_ylim(*boundsY)
    
            axes[1].set_title('Normalized')
            axes[1].set_xlim(*boundsX)
            axes[1].set_ylim(*boundsY)

            #axes[0].imshow(-outImage[0].cpu().view(ego_pixel_shape).detach().numpy(), extent=[*(minT,maxT), *(minR,maxR)], interpolation='none')#, cmap='gnuplot')
            outImagea = outImage[0].cpu().view(ego_pixel_shape).detach().numpy()
            tempval = axes[0].imshow(outImagea, extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')#, cmap='gnuplot')
            axes[1].imshow(outImagea, extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')#, cmap='gnuplot')
        
            # Rerun again for gradient
            predictions = network({'coords':all_coords,'img_sparse':test_image_prediction})
            if type(predictions) is dict:
                outImage = (predictions['model_out'], predictions['model_in'])
            else:
                outImage = predictions
            outImageA = -DNN.gradient(*outImage)#-DNN.gradient(*predModel(outImage[1]))
        
            predictions = network({'coords':dense_coords,'img_sparse':test_image_prediction})
            if type(predictions) is dict:
                outImage = (predictions['model_out'], predictions['model_in'])
            else:
                outImage = predictions
            laplacianA = np.squeeze(DNN.laplace(*outImage).detach().numpy())

            outImageA = outImageA[0].cpu().view(*ego_pixel_shape,2).detach().numpy() 
            print(outImageA.shape)

            fig2, ax2 = plt.subplots(1,1)
            ax2.set_title('Histogram of Laplacians')
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

            #for traj in test_pix_trajectory.values():
            trajnp = np.array(test_pix_trajectory)
            axes[0].plot(trajnp[:,0], trajnp[:,1], 'r')
            axes[1].plot(trajnp[:,0], trajnp[:,1], 'r')
                #axes[0].plot(tpix, logrpix, 'r')
                #axes[1].plot(tpix, logrpix, 'r')

            #print(testpos)
            axes[0].plot(*test_pix_trajectory[-1], 'co',markersize=2)
            axes[1].plot(*test_pix_trajectory[-1], 'co',markersize=2)

            #for obs_traj in obstacle_trajectory.values():
            #    trajnp = np.array(obs_traj)
            #    axes[0].plot(trajnp[:,0], trajnp[:,1], 'c')
            #    axes[1].plot(trajnp[:,0], trajnp[:,1], 'c')


        
            cax = fig.add_axes([.3, .95, .4, .05])
            fig.colorbar(tempval, cax, orientation='horizontal')
            #fig.colorbar(outImagea, axes[0], orientation='vertical')


            load_offset = 1 if LOAD_NETWORK_FROM_DISK else 0
            fig, axes = plt.subplots(1,3 + load_offset)
            fig.suptitle('Comparison of Network Output with Ground Truth')
            trajnp = np.array(test_pix_trajectory)

            boundsX = (0,ego_pixel_shape[1])
            boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
        
        
            if (LOAD_NETWORK_FROM_DISK):
                axes[0].set_title('Image (Original)')
                axes[0].set_aspect(1)
                axes[0].imshow(raw_image)
                axes[0].plot(raw_trajectory[0], raw_trajectory[1], 'r')
        
            axes[0+load_offset].set_title('Input Image (Unnormalized)')
            axes[0+load_offset].set_aspect(1)
            axes[0+load_offset].imshow(np.moveaxis(0.5*(test_image+1),0,-1))
        
            axes[1+load_offset].set_title('Network with GT Traj')
            axes[1+load_offset].set_xlim(*boundsX)
            axes[1+load_offset].set_ylim(*boundsY)
            axes[1+load_offset].set_aspect(1)
            axes[1+load_offset].imshow(outImagea, extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')
            axes[1+load_offset].plot(trajnp[:,0], trajnp[:,1], 'r')
    
            axes[2+load_offset].set_title('GT Value with GT Traj')
            axes[2+load_offset].set_xlim(*boundsX)
            axes[2+load_offset].set_ylim(*boundsY)
            axes[2+load_offset].set_aspect(1)
            axes[2+load_offset].imshow(np.reshape(test_coord_value,(ego_pixel_shape)), extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')
            axes[2+load_offset].plot(trajnp[:,0], trajnp[:,1], 'r')

            #coord_x, coord_y = np.meshgrid(range(ego_pixel_shape[1]), range(ego_pixel_shape[0]))

            #ax.quiver(gradient_samples[0][:,0], gradient_samples[0][:,1], gradient_samples[1][:,0],gradient_samples[1][:,1], color='red', scale_units='xy', scale=1)#, units='xy' ,scale=1

            #ux = vx/np.sqrt(vx**2+vy**2)
            #uy = vy/np.sqrt(vx**2+vy**2)
        
            #for traj in test_pix_trajectory.values():

            #plt.show()










            #axes[1,0].imshow(a_star_map_x, extent=[*boundsX, *boundsY], interpolation='none')
            #axes[1,1].imshow(a_star_map_y, extent=[*boundsX, *boundsY], interpolation='none')


            plt.show()

