

import sys
import os

import matplotlib.pyplot as plt
import random
from matplotlib import image
import matplotlib
import numpy as np
from scipy.spatial import cKDTree as KDTree
import math
import cv2
import pandas as pd

import torch

from Common import DNN
from Common import DataGens
from Common import DataReader
from Common.DataReader import RemapRange
from Common.DataReader import Coord2Polar
#from Common.DataReader import coord
#from DNN import *


from scipy import interpolate
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d

import pickle


PRINT_DEBUG_IMAGES = False
LOAD_NETWORK_FROM_DISK = True
USE_INTENSITY = True
USE_EGO = True
SHOW_INTENSITY = True
if __name__ == "__main__":
    #DNN.current_epoch = 0


    LOAD_NETWORK_FROM_DISK = True
    #overfit_ego_map_with_mask_newnewloss
    #overfit_skinny_exp_all_imgs
    #overfit_ego_map_all_imgs
    #overfit_ego_map_with_mask_newnewloss.pt #196, 3/4
    #overfit_ego_map_with_mask_newnewloss_square256_55kernel.pt #256, 1
    #'overfit_test_network_exp_with_goal.pt'
    #overfit_ego_map_with_mask_newnewloss_square192_55kernel_aligned BAD
    #overfit_imploc_aligned_nogoal_192.pt #192
    #overfit_imploc_aligned_goal_192.pt #192
    # overfit_synth_marketplace_random2020 synthetic data
    # overfit_imploc_grad_192_miniset_withexp.pt
    # #overfit_imploc_192_aligned_miniset_multiplier.pt
    # overfit_imploc_grad_192_miniset_withexp.pt
    # overfit_imploc_192_aligned_miniset_grad.pt
    # overfit_imploc_grad_192_miniset_withgoal.pt
    # synth_marketplace_random2020
    # overfit_imploc_192_aligned_synth_miniset



    #FINALS
    # overfit_imploc_192_full_final.pt


    #overfit_imploc_192_full_synthetic_grad.pt
    
    #overfit_VAE_192_imageprior_full
    #overfit_VAE_192_imageprior_full_synth.pt

    #SynthSet

    #overfit_imploc_192_full_synthetic_grad
    #overfit_VAE_192_imageprior_full_synth

    network = torch.load('overfit_test_network_exp_123.pt')
    #network = torch.load('overfit_imploc_192_full_final_grad.pt') #torch.load('hypernet_1200imgs_300epochs.pt') #overfit_test_network_exp_newnewloss
    vae_network = torch.load('overfit_VAE_192_imageprior_full.pt') #''overfit_test_network_exp_AE.pt
    #test = network.module.state_dict()
    if type(network) == torch.nn.DataParallel:
        network = network.module
    if type(vae_network) == torch.nn.DataParallel:
        vae_network = vae_network.module
    network.cuda()
    network.eval()
    vae_network.cuda()
    vae_network.eval()
    print("Parameter count:", DNN.CountParameters(network))
    print("VAE parameter count:", DNN.CountParameters(vae_network))

    print(os.getcwd())
    #loc = r'H:\fut_loc\20150401_walk_00\traj_prediction.txt'

    partial_folder_path =  'S:\\fut_loc\\dummytest\\' #'S:\\synth_marketplace_trials2021\\test\\' # # #'S:\\fut_loc\\synth\\' #'S:\\fut_loc\\test\\' #20150401_walk_00\\' #'S:\\synth_marketplace_random2020\\test\\'
    
    folder_name = '10000000_test_00' #'20150418_costco' #'Yasamin9085_t29_p3'# '20150402_grocery' #'20150401_walk_00' #'20150418_mall_00' # #'20150419_ikea' # 'definitelynotzach8002_t36_p12' #'acofre20167850_t36_p9' #'acofre20167850_t38_p18' #'10000000_test_00' #'marketplace6203_trand_p0' #
    folder_path =  partial_folder_path + folder_name + '\\'


    # LOADING VARIABLES
    #RESIZED_IMAGE_DICTIONARY = {} # could probably pre-allocate this into a big numpy array...
    #LOG_POLAR_TRAJECTORY_DICTIONARY = {}
    #PIXEL_TRAJECTORY_DICTIONARY = {}

    #RAW_IMAGE_DICTIONARY = {}
    #TRAJ_IN_IMAGE_DICTIONARY = {}
    FILE_UPPER_LIMIT = 1000 # a number larger than the number of images in a single directory, used for dictionary indexing
    count = 0 # folder ID
    n_folders = 17 #15
    print(matplotlib.get_backend())

    if True:
    #for folder_name in next(os.walk(partial_folder_path))[1]:
        ##if count < 16:
        ##    count += 1
        ##    continue
        #if count >= n_folders:
        #    break

        #folder_path =  partial_folder_path + folder_name + '\\'

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
        R_rect = np.eye(3) #np.array([[0.9989,0.0040,0.0466],[-0.0040,1.0000,-0.0002],[-0.0466,0,0.9989]])
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

                 
        # KNN DATA LOADING
        #knn_alexfeats_aligned_synth
        #knn_alexfeats_aligned_full_FIXED
        knnPickle = open('knn_alexfeats_aligned_full_FIXED.knn','rb')#'knn_alexfeats_aligned.knn','rb')
        KNN = pickle.load(knnPickle)
        knnPickle.close()

        #knn_traintraj_aligned_synth
        #knn_traintraj_aligned_full_FIXED
        dictPickle = open('knn_traintraj_aligned_full_FIXED.dict','rb')##'knn_traintraj_aligned.dict','rb')
        LOG_POLAR_TRAJECTORY_DICTIONARY_TR = pickle.load(dictPickle)
        LOG_POLAR_TRAJECTORY_DICTIONARY_TR_KEYS = list(LOG_POLAR_TRAJECTORY_DICTIONARY_TR.keys())
        print(type(LOG_POLAR_TRAJECTORY_DICTIONARY_TR_KEYS[0]))
        #np.savetxt('keydict.txt', np.array(LOG_POLAR_TRAJECTORY_DICTIONARY_TR_KEYS,dtype=np.int),fmt='%i')
        dictPickle.close()

        AlexNet = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
        AlexNet.eval()
        mods = list(AlexNet.named_modules())
        AlexNet.named_modules()
        children = AlexNet.children()
        childrenchildren = list( list(children)[-1].children())
        penultimate_layer = childrenchildren[4]
        #print(listchildren)
        #print(type(listchildren[-1]))
        #print(listchildren[-1])

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook


        penultimate_layer.register_forward_hook(get_activation('classifier.4'))

        #returns = model(img)
        #feature = activation['classifier.4']






        print('loading trajectory file')
        traj_data_file = folder_path + 'traj_prediction.txt'
        vTR = DataReader.ReadTraj(traj_data_file)
        #vTR = DataReader.ReadTraj(traj_data_file)

        

        frameOffset = 0#35#38
        frameEnd = len(os.listdir(folder_path + 'im\\')) #55
        imageScale = .1

        for iFrame in range(frameOffset,frameEnd):

            dictionary_index = iFrame + count * FILE_UPPER_LIMIT

            tr = vTR['vTr'][iFrame]
            if (len(tr['XYZ'][1]) < 2):
                print('FROM FOLDER', folder_name,'SKIPPING FRAME',iFrame,': Trajectory is deficient (less than 2 points)')
                continue
            else:
                print('FROM FOLDER', folder_name, 'GETTING FRAME',iFrame )

            tr = vTR['vTr'][iFrame]

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

            #t, r = DataReader.Coord2Polar(tr_ground[2],tr_ground[0])
            tr_ground = K_data @ R_rect @ tr_ground_OG;
            if np.any(tr_ground[2,:]<0):
                #tr_ground[:2,:] = np.nan # actually maybe I shouldn't NAN here, the only issue is that the trajectory goes behind the camera. Maybe that's okay?
                print('\tThe trajectory is suspicious, and may be behind the user. SKIPPING')
                continue
            #tr_ground(tr_ground(3,:)<0, :) = NaN;
            #tr_ground = bsxfun(@rdivide, tr_ground(1:2,:), tr_ground(3,:));
            tr_ground = tr_ground[:2] / tr_ground[2]
            
            #if (LOAD_NETWORK_FROM_DISK):
            #    TRAJ_IN_IMAGE_DICTIONARY[dictionary_index] = tr_ground
            #    RAW_IMAGE_DICTIONARY[dictionary_index] = img

        
            # set up egocentric image information in log polar space

            #t, r = Coord2Polar(tr['XYZ'][2],tr['XYZ'][0])



            
            img_height = 192 # 256 #196 #128

            minR = -.5
            maxR = 4 #5#4.5
            minT = -np.pi/3
            maxT = -minT #2*np.pi/3





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



            #tr_ground_ALIGNED = R_rect_ego @ tr['XYZ'].T - np.linalg.norm(tr['up'])
            
            tr_ground_ALIGNED = R_rect_ego @ tr_ground_OG # ALIGN CAMERA SPACE GROUND PLANE TO "WORLD SPACE"
            t, r = Coord2Polar(tr_ground_ALIGNED[2],tr_ground_ALIGNED[0])#Coord2Polar(tr['XYZ'][2],tr['XYZ'][0])



            test = r < np.exp(minR)
            if (np.any(r < np.exp(minR))):
                print('\tTrajectory is too close to camera origin.')
                continue
                #print('')
            logr = np.log(r)

            




            








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










            aspect_ratio = 1#3/4#(maxT-minT)/(maxR-minR)
            ego_pixel_shape = (img_height,int(img_height*aspect_ratio)) # y,x | vert,horz
            ego_pixel_shape_AlexNet = (256,256)
            ego_pixel_shape_VAE = ego_pixel_shape

            ego_r2pix = lambda x : RemapRange(x, minR,maxR, 0,                  ego_pixel_shape[0]  )
            ego_t2pix = lambda x : RemapRange(x, minT,maxT, 0,                  ego_pixel_shape[1]  )

        
            ego_pix2r = lambda x : RemapRange(x, 0, ego_pixel_shape[0], minR,maxR   )
            ego_pix2t = lambda x : RemapRange(x,0,ego_pixel_shape[1], minT,maxT  )

            ego_pix2r_VAE = lambda x : RemapRange(x, 0, ego_pixel_shape[0], minR,maxR   )
            ego_pix2t_VAE = lambda x : RemapRange(x,0,ego_pixel_shape[1], minT,maxT  )

            
            ego_pix2r_AlexNet = lambda x : RemapRange(x, 0, ego_pixel_shape_AlexNet[0], minR,maxR   )
            ego_pix2t_AlexNet = lambda x : RemapRange(x,0,ego_pixel_shape_AlexNet[1], minT,maxT  )

            RecenterDataForwardWithShape = lambda x, shape : RemapRange(x,0,max(shape[0],shape[1]),-1,1)
            RecenterDataForwardWithShapeAndScale = lambda x, shape, scale : RemapRange(x,0,max(shape[0],shape[1]),-scale,scale)
            RecenterDataBackwardWithShape = lambda x, shape : RemapRange(x,-1,1,0,max(shape[0],shape[1]))
            RecenterTrajDataForward = lambda x : RecenterDataForwardWithShape(x,ego_pixel_shape)
            RecenterTrajDataForward2 = lambda x : RecenterDataForwardWithShapeAndScale(x,ego_pixel_shape,1)

            RecenterTrajDataBackward = lambda x : RecenterDataBackwardWithShape(x,ego_pixel_shape)
            RecenterTrajDataBackward_AlexNet =  lambda x : RecenterDataBackwardWithShape(x,ego_pixel_shape_AlexNet)
            RecenterTrajDataBackward_VAE = lambda x : RecenterDataBackwardWithShape(x,ego_pixel_shape)
        
            RecenterDataBackwardWithShapeAndScale = lambda x, shape, scale : RemapRange(x,-scale,scale,0,max(shape[0],shape[1]))
            RecenterFieldDataBackward = lambda x : RecenterDataBackwardWithShapeAndScale(x,ego_pixel_shape,1)
        

            tpix = ego_t2pix(t)
            logrpix = ego_r2pix(logr)





            # Generating EgoRetinalMap

            if USE_EGO:
                all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(ego_pixel_shape_AlexNet[0]) for j in range(ego_pixel_shape_AlexNet[1]) ], dtype=np.float32)
                all_pixel_coords[:,0] = ego_pix2t_AlexNet(all_pixel_coords[:,0])
                all_pixel_coords[:,1] = ego_pix2r_AlexNet(all_pixel_coords[:,1])
                all_pixel_coords[:,1] = np.exp(all_pixel_coords[:,1])
                z, x = DataReader.Polar2Coord(all_pixel_coords[:,0],all_pixel_coords[:,1])

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


                img_resized =      interpolate.interpn((range(img.shape[0]),range(img.shape[1])), img*2.0-1.0, rowmaj_pixels[:2].T , method = 'linear',bounds_error = False, fill_value = 0).reshape(ego_pixel_shape_AlexNet[0], ego_pixel_shape_AlexNet[1],3)
                img_channel_swap_AlexNet = np.moveaxis(img_resized,-1,0).astype(np.float32)

                #plt.imshow(img_resized)
                #plt.show()



                all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(ego_pixel_shape[0]) for j in range(ego_pixel_shape[1]) ], dtype=np.float32)
                all_pixel_coords[:,0] = ego_pix2t(all_pixel_coords[:,0])
                all_pixel_coords[:,1] = ego_pix2r(all_pixel_coords[:,1])
                all_pixel_coords[:,1] = np.exp(all_pixel_coords[:,1])
                z, x = DataReader.Polar2Coord(all_pixel_coords[:,0],all_pixel_coords[:,1])

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
                # TODO: R_rect makes sense right?
                homography = K_data @ R_rect @ R_rect_ego @ R_rect.T @ np.linalg.inv(K_data)

                img_rectified = cv2.warpPerspective(img*2.0-1.0, homography, (img.shape[1], img.shape[0])) # want to shift the values here so that the normalized version has black in the rectified location

       

                img_resized = cv2.resize(img_rectified, (int(img_rectified.shape[1]*imageScale), int(img_rectified.shape[0]*imageScale)))
                img_channel_swap = np.moveaxis(img_resized,-1,0).astype(np.float32)



         
            if (PRINT_DEBUG_IMAGES):
                fig, axes = plt.subplots(1,2)#, figsize=(18,6))
                axes[0].imshow(img)
                #axes[1].imshow(img_rectified)
                #axes[1].imshow(img_resized)
                if USE_EGO:
                    boundsX = (0,ego_pixel_shape[1])
                    boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
            
                    axes[1].set_xlim(*boundsX)
                    axes[1].set_ylim(*boundsY)
                    axes[1].set_aspect(1)
                axes[1].imshow(img_resized)
                plt.show()



















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





            # ######################################################################
            # Let's get the siren coordinates of our ground plane in the image!
            # ######################################################################

            def intersectPlaneV(n, p0, l0, L):
                #print('in')
                plane_offset = p0-l0
                denoms = n @ L
                t = np.zeros(len(denoms))
                intersecting = np.where(denoms < -1e-6)
                d = plane_offset @ n
                result = np.divide( d[None], denoms[intersecting])
                t[intersecting] = result
                return t
        

            depth_img = np.zeros(img.shape[:2])

        
            depth_pixel_coords = np.array( [ [j+.5,i+.5,1.0] for i in range(img.shape[0]) for j in range(img.shape[1]) ], dtype=np.float32)

            result = R_rect[1] @ tr['up']

            #pixel = np.array([j,i,1])
            rect_up = tr['up']
            p_normal = rect_up/np.linalg.norm(rect_up)
            p_origin = -rect_up #camera assumed to be at 0,0,0
            e_origin = np.zeros(3) #zero vector
            e_rays = R_rect.T @ np.linalg.inv(K_data) @ depth_pixel_coords.T #+0
            e_rays /= np.linalg.norm(e_rays,axis=0)
            print('norm:', np.linalg.norm(e_rays[:,100]))

            #intplane = lambda l : intersectPlane(p_normal,p_origin,e_origin,l)
            #vfunc = np.vectorize(intplane)
            #depths = np.apply_along_axis(intplane, 0, e_rays)
            image_siren_depths = intersectPlaneV(p_normal,p_origin,e_origin,e_rays)
            image_siren_points = e_rays * image_siren_depths[None]

            t_im, r_im = Coord2Polar(image_siren_points[2],image_siren_points[0])
            # r_im = np.clip(r_im,np.exp(minR),np.inf) # clipping only to avoid the log transform being invalid (0)
            logr_im = np.log(r_im)
            image_siren_depths[t_im < minT] = 0
            image_siren_depths[t_im > maxT] = 0
            image_siren_depths[logr_im < minR] = 0
            image_siren_depths[logr_im > maxR] = 0

            image_siren_pix_coords = np.stack( (ego_t2pix(t_im), ego_r2pix(logr_im)), axis = 0)

            image_siren_coords = RecenterTrajDataForward(image_siren_pix_coords).astype(np.float32).T

            #print(r2.max())
            #r2 = np.clip(r2,0,100)
            #print(r2.max())
            #rnorm = r2 / r2.max()
            #t2 = np.clip(t2,-np.pi/4,np.pi/4)
            #tnorm = t2 / t2.max()


            #rad_img = np.reshape(tnorm, depth_img.shape)













        
            # Let's try to get a ground truth image


           
            # Let's try to get a ground truth image
            all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(ego_pixel_shape[0]) for j in range(ego_pixel_shape[1]) ], dtype=np.float32)
            #coord_value = np.zeros((all_pixel_coords.shape[0]))


            all_pixel_coords_xformed = np.zeros(all_pixel_coords.shape).astype(np.float32)
            all_pixel_coords_xformed[:,0] = ego_pix2t(all_pixel_coords[:,0])
            all_pixel_coords_xformed[:,1] = np.exp(ego_pix2r(all_pixel_coords[:,1]))

            all_pixel_coords_xformed = np.array(DataReader.Polar2Coord(all_pixel_coords_xformed[:,0],all_pixel_coords_xformed[:,1])).T
            #newtraj = future_trajectory.copy()
            #newtraj = {}
            #newtraj[0] = []
        


            # START POPULATING DICTIONARIES            
            if (LOAD_NETWORK_FROM_DISK):
                raw_trajectory = tr_ground
                #TRAJ_IN_IMAGE_DICTIONARY[dictionary_index] = tr_ground
                #RAW_IMAGE_DICTIONARY[dictionary_index] = img
                raw_image = img
            
            #RESIZED_IMAGE_DICTIONARY[dictionary_index] = img_channel_swap
            test_image = img_channel_swap
            test_image_VAE = img_channel_swap
            test_image_AlexNet = img_channel_swap_AlexNet

            

            #testing = np.arange(256*256*3)
            #testing = np.reshape(testing,(3,256,256)).astype(np.float32)
            outputs_AlexNet = AlexNet( torch.unsqueeze(torch.from_numpy(test_image_AlexNet),0))
            feature_AlexNet = activation['classifier.4']
            #np.set_printoptions(threshold=sys.maxsize)
            #print(feature_AlexNet[0].numpy())

            dist, idx = KNN.kneighbors(feature_AlexNet.numpy())
            best_idx = LOG_POLAR_TRAJECTORY_DICTIONARY_TR_KEYS[idx[0,0]]
            traj_AlexNet = LOG_POLAR_TRAJECTORY_DICTIONARY_TR[LOG_POLAR_TRAJECTORY_DICTIONARY_TR_KEYS[idx[0,0]]] # get best traj
            traj_AlexNet = np.array(traj_AlexNet)



             # Check if first two pixels are within egomap
            pix1 = future_trajectory[0]
            pix2 = future_trajectory[1]

            if (pix1[0] < 0 or pix1[0] > ego_pixel_shape[1] 
                or pix1[1] < 0 or pix1[1] > ego_pixel_shape[0]
                or pix2[0] < 0 or pix2[0] > ego_pixel_shape[1] 
                or pix2[1] < 0 or pix2[1] > ego_pixel_shape[0]):

                print('\tTrajectory is deficient (start is outside egomap)')
                continue

                    


            #PIXEL_TRAJECTORY_DICTIONARY[dictionary_index] = []
            test_pix_trajectory = []
            
            #LOG_POLAR_TRAJECTORY_DICTIONARY[dictionary_index] = []
            test_ws_trajectory = []
            for pix in future_trajectory:
                if (pix[0] < 0 or pix[0] > ego_pixel_shape[1] or pix[1] < 0 or pix[1] > ego_pixel_shape[0]): # outside ego map
                        break
                test_pix_trajectory.append( (pix[0], pix[1]) ) # t is horizontal axis, logr is vertical
                #PIXEL_TRAJECTORY_DICTIONARY[dictionary_index].append( (pix[0], pix[1]) ) # t is horizontal axis, logr is vertical
                newpoint = ( DataReader.Polar2Coord( ego_pix2t(pix[0]),np.exp(ego_pix2r(pix[1])) ) )
                test_ws_trajectory.append( newpoint )
                #LOG_POLAR_TRAJECTORY_DICTIONARY[dictionary_index].append( newpoint )
        
            #for i in range(len(logrpix)):
            #    LOG_POLAR_TRAJECTORY_DICTIONARY[iFrame].append( (tpix[i], logrpix[i]) ) # t is horizontal axis, logr is vertical



            #coord_value = DataGens.Coords2ValueFast(all_pixel_coords,future_trajectory,nscale=1)

            if (PRINT_DEBUG_IMAGES):
                coord_value = DataGens.Coords2ValueFastWS(all_pixel_coords_xformed,{0:test_ws_trajectory},None,None,stddev=.5)
                fig, ax = plt.subplots(1,1)#, figsize=(36,6))
                axes = [ax]

                boundsX = (0,ego_pixel_shape[1])
                boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
                axes[0].set_xlim(*boundsX)
                axes[0].set_ylim(*boundsY)

                #axes[1].set_xlim(*boundsX)
                #axes[1].set_ylim(*boundsY)

                tempval = axes[0].imshow(np.reshape(coord_value,(ego_pixel_shape)), extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')#, cmap='gnuplot')
        
        
                trajnp = np.array(future_trajectory)
                axes[0].plot(trajnp[:,0], trajnp[:,1], 'r')
        
                cax = fig.add_axes([.3, .95, .4, .05])
                fig.colorbar(tempval, cax, orientation='horizontal')
                plt.show()

        #count += 1





        
        #network = network.hypo_net
        #network.cpu()
        #network = network({'img_sparse':torch.zeros((1,1,32,32))})

        #for frame in RESIZED_IMAGE_DICTIONARY.keys():
            #testFrame = frame
            #test_image = RESIZED_IMAGE_DICTIONARY[testFrame] # img_channel_swap
            #test_pix_trajectory = PIXEL_TRAJECTORY_DICTIONARY[testFrame]
            #test_ws_trajectory = LOG_POLAR_TRAJECTORY_DICTIONARY[testFrame]

            #if (LOAD_NETWORK_FROM_DISK):
                #raw_image = RAW_IMAGE_DICTIONARY[testFrame]
                #raw_trajectory = TRAJ_IN_IMAGE_DICTIONARY[testFrame]






            all_pixel_coords_xformed = np.zeros(all_pixel_coords.shape).astype(np.float32)
            all_pixel_coords_xformed[:,0] = ego_pix2t(all_pixel_coords[:,0])
            all_pixel_coords_xformed[:,1] = np.exp(ego_pix2r(all_pixel_coords[:,1]))
            all_pixel_coords_xformed = np.array(DataReader.Polar2Coord(all_pixel_coords_xformed[:,0],all_pixel_coords_xformed[:,1])).T

            test_coord_value = DataGens.Coords2ValueFastWS_NEURIPS(all_pixel_coords_xformed,{0:test_ws_trajectory},None,None,stddev=.5, dstddev=1)#.5)

            test_image_prediction = torch.from_numpy( np.expand_dims(test_image,0) )
            print(test_image_prediction.device)

            #network.cpu()
            #vae_network.cpu()
            
            
            predictions_vae = vae_network({'img_sparse':torch.unsqueeze(torch.from_numpy(test_image_VAE),0).cuda()})
            traj_vae = RecenterTrajDataBackward_VAE(np.squeeze(predictions_vae['model_out'].detach().cpu().numpy()))
            if len(traj_vae) > 2:
                traj_vae_t = ego_pix2t_VAE(traj_vae[:25]) #traj_vae[0]#
                traj_vae_logr = ego_pix2r_VAE(traj_vae[25:]) #traj_vae[1]#
            else:
                traj_vae_t = ego_pix2t_VAE(traj_vae[0]) #traj_vae[0]#
                traj_vae_logr = ego_pix2r_VAE(traj_vae[1]) #traj_vae[1]#

            if not USE_INTENSITY:
                image_siren_coords_tensor = torch.unsqueeze( torch.from_numpy(image_siren_coords), 0)
                image_siren_predictions = network({'coords':image_siren_coords_tensor.cuda(),'img_sparse':test_image_prediction.cuda()})
                image_siren_image = (image_siren_predictions['model_out'], image_siren_predictions['model_in'])
                image_siren_image = image_siren_image[0].cpu().view(raw_image.shape[:2]).detach().numpy()
                image_siren_alpha = image_siren_depths.reshape(raw_image.shape[:2])
                image_siren_alpha[image_siren_alpha > 0.001] = .7
                #image_siren_image_rgba = np.dstack((image_siren_image, image_siren_image, image_siren_image, image_siren_alpha))
                #print('testingggg',image_siren_image.max())
                #print(image_siren_alpha.max())
                #rgb_img = np.dstack((grayscale, grayscale, grayscale, alpha))

                #torch.no_grad()
                # Is y x okay or should we do x y
                dense_scale = 1
                dense_coords = np.array( [[ [RecenterTrajDataForward((j+.5)/dense_scale),RecenterTrajDataForward((i+.5)/dense_scale)] for i in range(ego_pixel_shape[0]*dense_scale) for j in range(ego_pixel_shape[1]*dense_scale) ]], dtype=np.float32)
                dense_coords = torch.unsqueeze( torch.from_numpy(dense_coords), 0)

            all_coords = np.array( [[ [RecenterTrajDataForward(j+.5),RecenterTrajDataForward(i+.5)] for i in range(ego_pixel_shape[0]) for j in range(ego_pixel_shape[1]) ]], dtype=np.float32)
            all_coords = torch.unsqueeze( torch.from_numpy(all_coords), 0)

            all_coords = torch.unsqueeze( torch.from_numpy(RecenterTrajDataForward(all_pixel_coords.astype(np.float32))), 0)


            print(all_coords.shape)
            predictions = network({'coords':all_coords.cuda(),'img_sparse':test_image_prediction.cuda()})
            if type(predictions) is dict:
                outImage = (predictions['model_out'], predictions['model_in'])
            else:
                outImage = predictions
            print('max',torch.max(outImage[0]))
            print('min',torch.min(outImage[0]))
            print(type(outImage))
            print(outImage[0].shape)
            print(outImage[1].shape)

            showHistogram = False

            #if False:
            if showHistogram:
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

            
            NAVIGATION_STRING = 'sirenA_out' #'siren_out'
            WALKABILITY_STRING = 'sirenB_out' #'intensity'


            predictions = network({'coords':all_coords.cuda(),'img_sparse':test_image_prediction.cuda()})
            if type(predictions) is dict:
                outImage = (predictions[NAVIGATION_STRING], predictions['model_in'])
            else:
                outImage = predictions
            sirenimage = outImage[0].cpu().view(ego_pixel_shape).detach().numpy()
            


            if USE_INTENSITY:
                image_siren_image = interpolate.interpn((range(outImagea.shape[0]),range(outImagea.shape[1])), outImagea, image_siren_pix_coords[[1,0]].T , method = 'linear',bounds_error = False, fill_value = 0).reshape(img.shape[0], img.shape[1])
                #image_siren_image = interpolate.interpn((np.array(list(range(outImagea.shape[0])))+.5,np.array(list(range(outImagea.shape[1])))+.5), outImagea, image_siren_pix_coords[[1,0]].T , method = 'linear',bounds_error = False, fill_value = 0).reshape(img.shape[0], img.shape[1])
                image_siren_alpha = image_siren_depths.reshape(raw_image.shape[:2])
                image_siren_alpha[image_siren_alpha > 0.001] = .7

                intensity_map = predictions[WALKABILITY_STRING].cpu().view(ego_pixel_shape).detach().numpy()






            if showHistogram:
                tempval = axes[0].imshow(outImagea, extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')#, cmap='gnuplot')
                axes[1].imshow(outImagea, extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none')#, cmap='gnuplot')
        
            # Rerun again for gradient
            predictions = network({'coords':all_coords.cuda(),'img_sparse':test_image_prediction.cuda()})
            if type(predictions) is dict:
                outImage = (predictions['model_out'], predictions['model_in'])
            else:
                outImage = predictions
            outImageA = -DNN.gradient(*outImage)#-DNN.gradient(*predModel(outImage[1]))
        
            if not USE_INTENSITY:
                predictions = network({'coords':dense_coords.cuda(),'img_sparse':test_image_prediction.cuda()})
                if type(predictions) is dict:
                    outImage = (predictions['model_out'], predictions['model_in'])
                else:
                    outImage = predictions
                laplacianA = np.squeeze(DNN.laplace(*outImage).detach().numpy())

            outImageA = outImageA[0].cpu().view(*ego_pixel_shape,2).detach().numpy() 
            print(outImageA.shape)

            # HISTOGRAM
            if showHistogram and not USE_INTENSITY:
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








            ########################################################################
            # Let's plan a path through the image via gradient descent
            ########################################################################
            hypo_params = network.get_hypo_net_weights({'img_sparse':test_image_prediction.cuda()})
            start = RecenterTrajDataForward(np.array([[[90,40]]]))
            position = torch.from_numpy(start.astype(np.float32)).cuda()
            n_steps = 500
            grad_desc_positions = np.zeros((n_steps,2))
            grad_desc_positions[0] = position[0,0].cpu()
            for i in range(n_steps-1):
                siren_output = network.hypo_net({'coords':position}, params=hypo_params[0])
                siren_grad = -DNN.gradient(siren_output['model_out'],siren_output['model_in'])
                position +=  .0005 * siren_grad #
                grad_desc_positions[i+1] = position[0,0].detach().cpu().numpy()

            grad_desc_positions = RecenterTrajDataBackward(grad_desc_positions)


            val = ego_pix2r(start)



            min_along_y = np.argmin(outImagea,axis=1)
            y_coords = np.arange(0,ego_pixel_shape[0])
            below_thresh = np.zeros(ego_pixel_shape[0]).astype(np.bool)
            threshold = .2
            thresh_decay = (.4 +.4)/ego_pixel_shape[0] # want to be -.2 by half of ego map
            last_n_vals = np.array([0,0,0])
            have_prev_point = False
            prev_x = -1
            smoothed_min_along_y = min_along_y
            lowest_val = 100
            tuning_parameter = .6
            for i in range(ego_pixel_shape[0]):

                

                #if i > ego_pixel_shape[0]/2:
                #    threshold = -1
                x = min_along_y[i]
                y = y_coords[i]
                distance = ego_pix2r(y)

                if (distance < 0):
                    below_thresh[i] = False
                    continue

                val_at_point = outImagea[y,x]

                if have_prev_point == False:
                    #below_thresh[i] = val_at_point > lowest_val - tuning_parameter * lowest_val#< threshold
                    if True: #below_thresh[i]:
                        have_prev_point = True
                        x = outImagea.shape[1]//2+1 # make center assumption
                        prev_x = x

                        val_at_point = outImagea[y,x]

                        if val_at_point < lowest_val:
                            lowest_val = val_at_point
                        smoothed_min_along_y[i] = prev_x
                    continue
                
                lowerer = np.clip(prev_x-2,0,ego_pixel_shape[1]-1)
                lower = np.clip(prev_x-1,0,ego_pixel_shape[1]-1)
                upper = np.clip(prev_x+1,0,ego_pixel_shape[1]-1)
                upperer = np.clip(prev_x+2,0,ego_pixel_shape[1]-1)

                next_x = np.array([lowerer,lower,prev_x,upper,upperer])
                
                leftleft_val = outImagea[y,lowerer]
                left_val = outImagea[y,lower]
                center_val = outImagea[y,prev_x]
                right_val = outImagea[y,upper]
                rightright_val = outImagea[y,upperer]

                vals = np.array([leftleft_val,left_val, center_val, right_val,rightright_val])

                lowest_val_pos = np.argmin(vals) #-1 to center

                threshold =  lowest_val - tuning_parameter * lowest_val
                val = vals[lowest_val_pos]
                isgood = vals[lowest_val_pos] < threshold
                if not isgood:
                    print('not good')

                below_thresh[i] = val < threshold #vals[lowest_val_pos] < threshold

                prev_x = next_x[lowest_val_pos]
                #prev_x += lowest_val_pos-1
                smoothed_min_along_y[i] = prev_x
                if val < lowest_val:
                    lowest_val = val
                
                if below_thresh[i] == False:
                    break
                #if 
                

                #j = i
                #while j < ego_pixel_shape[0]:
                #    x = min_along_y[i]
                #    y = y_coords[i]
                #    distance = ego_pix2r(y)

                
                    
                #threshold -= thresh_decay





            #axes[1].plot(,,'r',linewidth=2)

            alpha = 0.5
            smoothsmooth_min_along_y = np.copy(smoothed_min_along_y[below_thresh==True])

            for i in range(1,len(smoothed_min_along_y[below_thresh==True])):
                smoothsmooth_min_along_y[i] = smoothsmooth_min_along_y[i-1] * alpha + (1-alpha) * smoothed_min_along_y[below_thresh==True][i-1]


            traj_t_pix = smoothsmooth_min_along_y#[below_thresh==True]
            traj_r_pix = y_coords[below_thresh==True]
            
            traj_t = ego_pix2t(traj_t_pix)
            traj_logr = ego_pix2r(traj_r_pix)

            traj_r = np.exp(traj_logr)

            min_along_y = smoothed_min_along_y = smoothsmooth_min_along_y




            load_offset = 1 if LOAD_NETWORK_FROM_DISK else 0
            load_offset += 1 if USE_INTENSITY else 0
            fig, axes = plt.subplots(1,2 + load_offset, gridspec_kw = {'wspace':.05,'width_ratios': [4/3, 1,1,1]})
            fig.suptitle('Comparison of Network Output with Ground Truth')
            trajnp = np.array(test_pix_trajectory)

            boundsX = (0,ego_pixel_shape[1])
            boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
        
        
            if (LOAD_NETWORK_FROM_DISK):
                axes[0].set_title('Image (Original)')
                axes[0].set_aspect(1)
                axes[0].imshow(raw_image)
                #axes[0].plot(raw_trajectory[0], raw_trajectory[1], 'r')

                #axes[1].set_title('Siren Overlay (Original)')
                #axes[1].set_aspect(1)

                axes[0].set_xlim(0,img.shape[1])
                axes[0].set_ylim(img.shape[0],0)
                axes[0].set_aspect(1)

                #axes[0].imshow(image_siren_image, alpha=image_siren_alpha)
                axes[0].plot(raw_trajectory[0], raw_trajectory[1], 'm--')
                
                z, x = DataReader.Polar2Coord(traj_t,traj_r)
            
                coords_3D = np.zeros((len(z),3))
                coords_3D[:,1] = 0
                coords_3D[:,0] = x
                coords_3D[:,2] = z
                coords_3D = (R_rect_ego.T @ coords_3D.T).T
                coords_3D -= tr['up'].T

                pixels = K_data @ R_rect @ coords_3D.T
                pixels /= pixels[2]
                axes[0].plot(pixels[0], pixels[1], 'c',linewidth=3)


                z, x = DataReader.Polar2Coord(traj_AlexNet[:,0], np.exp(traj_AlexNet[:,1]))
                
                coords_3D = np.zeros((len(z),3))
                coords_3D[:,1] = 0
                coords_3D[:,0] = x
                coords_3D[:,2] = z
                coords_3D = (R_rect_ego.T @ coords_3D.T).T
                coords_3D -= tr['up'].T

                pixels = K_data @ R_rect @ coords_3D.T
                pixels /= pixels[2]
                axes[0].plot(pixels[0], pixels[1], 'r--')
                
                #xn,yn = DataGens.InterpAlongLine(pixels[0],pixels[1],25)
                #axes[0].plot(xn,yn,'rx')


                
                z, x = DataReader.Polar2Coord(traj_vae_t, np.exp(traj_vae_logr))
                
                coords_3D = np.zeros((len(z),3))
                coords_3D[:,1] = 0
                coords_3D[:,0] = x
                coords_3D[:,2] = z
                coords_3D = (R_rect_ego.T @ coords_3D.T).T
                coords_3D -= tr['up'].T

                pixels = K_data @ R_rect @ coords_3D.T
                pixels /= pixels[2]
                axes[0].plot(pixels[0], pixels[1], 'w--')
                
                axes[0].axes.xaxis.set_visible(False)
                axes[0].axes.yaxis.set_visible(False)
                #fx = interp1d(pixels[0], np.arange(len(pixels[0])))
                #fy = interp1d(pixels[1], np.arange(len(pixels[1])))
                #ix = np.linspace(0,len(pixels[0]-1),num=25,endpoint=True)
                #iy = np.linspace(0,len(pixels[1]-1),num=25,endpoint=True)
                #new_xs = fx(ix)
                #new_ys = fy(iy)
                #axes[0].plot(new_xs,new_ys, 'wx')



                
            t_AlexNet = traj_AlexNet[:,0]
            logr_AlexNet = traj_AlexNet[:,1]
            tpix_AlexNet = ego_t2pix(t_AlexNet)
            rpix_AlexNet = ego_r2pix(logr_AlexNet)
            
            traj_vae_tpix = ego_t2pix(traj_vae_t)
            traj_vae_logrpix = ego_r2pix(traj_vae_logr)

                
            
            

            axes[1].set_title('Input Image with Mask')#(Unnormalized)')
            boundsX = (0,ego_pixel_shape[1])
            boundsY = (0,ego_pixel_shape[0]) #(ego_pixel_shape[0],0) #
            
            axes[1].set_xlim(*boundsX)
            axes[1].set_ylim(*boundsY)
            axes[1].set_aspect(1)
            axes[1].imshow(np.moveaxis(0.5*(test_image+1),0,-1))
            #axes[0+load_offset].imshow(sirenimage, extent=[*boundsX, *(ego_pixel_shape[0],0)], vmax = 0.0, vmin = -1.0, interpolation='none')
            axes[1].plot(trajnp[:,0], trajnp[:,1], 'm--')
            axes[1].plot(tpix_AlexNet, rpix_AlexNet, 'r--')
            axes[1].plot(traj_vae_tpix, traj_vae_logrpix, 'w--')
            axes[1].plot(traj_t_pix,traj_r_pix,'r',linewidth=2)
            axes[1].plot(smoothsmooth_min_along_y,y_coords[below_thresh==True],'c',linewidth=3)
            xn,yn = DataGens.InterpAlongLine(tpix_AlexNet,rpix_AlexNet,25)
            #axes[0+load_offset].plot(xn,yn,'cx')
            
            axes[1].axes.xaxis.set_visible(False)
            axes[1].axes.yaxis.set_visible(False)

            traj = np.array(test_pix_trajectory).T
            tx, ty = DataGens.InterpAlongLine(traj[0],traj[1],25)
            px = RecenterTrajDataForward( tx )
            py = RecenterTrajDataForward( ty )
            print('print x ', px)
            print('print y ', py)

            traj = np.array([[ 0.0000, -0.0369, -0.0739, -0.1108, -0.1478, -0.1847, -0.2217, -0.2586,-0.2956, 
                       -0.3325, -0.3695, -0.4064, -0.4434, -0.4803, -0.5173, -0.5542,
                       -0.5912, -0.6281, -0.6650, -0.7020, -0.7389, -0.7759, -0.8128, -0.8498,
                       -0.8867, -0.0712, -0.0932, -0.1152, -0.1372, -0.1592, -0.1812, -0.2032,
                       -0.2252, -0.2473, -0.2693, -0.2913, -0.3133, -0.3353, -0.3573, -0.3793,
                       -0.4013, -0.4233, -0.4453, -0.4673, -0.4893, -0.5113, -0.5333, -0.5553,-0.5774, -0.5994]])

            tx = RecenterTrajDataBackward_VAE( traj[0,:25] )
            ty = RecenterTrajDataBackward_VAE( traj[0,25:] )
            #axes[0+load_offset].plot(tx,ty,'rx')



            #axes[0+load_offset].imshow(outImagea,alpha=.5)


            #if USE_INTENSITY:
            #    axes[0+load_offset].imshow(intensity_map, alpha=.30, extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none', cmap='plasma')



            #if USE_INTENSITY:
            #axes[2].set_title('Intensity Mask')
            axes[2].set_xlim(*boundsX)
            axes[2].set_ylim(*boundsY)
                
            axes[2].axes.xaxis.set_visible(False)
            axes[2].axes.yaxis.set_visible(False)
            axes[2].set_aspect(1)
            axes[2].imshow(intensity_map*.9 + .1, cmap='winter')#vmax = 1.0, vmin = 0.0, 
            #axes[2].plot(smoothsmooth_min_along_y,y_coords[below_thresh==True],'c',linewidth=2)
            #axes[2].plot(trajnp[:,0], trajnp[:,1], 'm--')
            #axes[2].plot(tpix_AlexNet, rpix_AlexNet, 'r--')
            #axes[2].plot(traj_vae_tpix, traj_vae_logrpix, 'w--')
            print("Max intensity:",intensity_map.max(),", min intensity:",intensity_map.min())
            #axes[0+load_offset].imshow(outImagea, alpha=.35, extent=[*boundsX, *(ego_pixel_shape[0],0)], interpolation='none', cmap='plasma')
            plt.imsave('walkable.png',intensity_map, cmap='winter')
            plt.imsave('affordance.png',outImagea, vmax = 0, vmin = -1.0, cmap='hot')
            #plt.imsave('egomap.png', np.moveaxis(0.5*(test_image+1),0,-1))
            #plt.imsave('siren.png', sirenimage, cmap='viridis')





            #gotta blast!!!
            


        
            axes[3].set_title('Network Prediction and Traj')
            axes[3].set_xlim(*boundsX)
            axes[3].set_ylim(*boundsY)
            axes[3].set_aspect(1)
            #combined = - (intensity_map*.9 + .1) * np.maximum(-outImagea,0)
            #print("comb max:", np.max(combined),"comb min:",np.min(combined))
            #(outImagea*10).astype(int).astype(float)/10
            tempval = axes[3].imshow(outImagea, extent=[*boundsX, *(ego_pixel_shape[0],0)], vmax = 0.0, vmin = -1.0, interpolation='none',cmap='hot')
            #axes[3].plot(trajnp[:,0], trajnp[:,1], 'm--')
            #axes[3].plot(tpix_AlexNet, rpix_AlexNet, 'r--')
            #axes[3].plot(traj_vae_tpix, traj_vae_logrpix, 'w--')
            siren_min = np.unravel_index(np.argmin(outImagea),outImagea.shape)
            #axes[1+load_offset].plot(grad_desc_positions[:,0],grad_desc_positions[:,1],'r')
            #axes[1+load_offset].plot(siren_min[1],siren_min[0],'cx',markersize=4)
            #axes[1+load_offset].plot(grad_desc_positions[-1,0],grad_desc_positions[-1,1],'rx',markersize=4)
            #axes[3].plot(smoothsmooth_min_along_y,y_coords[below_thresh==True],'c',linewidth=2)
            #axes[1+load_offset].plot(min_along_y[below_thresh==True][-1],y_coords[below_thresh==True][-1],'cx',markersize=4)
            #axes[1+load_offset].plot(min_along_y[below_thresh==False],y_coords[below_thresh==False],'bx',markersize=4)
            
            axes[3].axes.xaxis.set_visible(False)
            axes[3].axes.yaxis.set_visible(False)

            
            cax = fig.add_axes([.3, .95, .4, .05])
            fig.colorbar(tempval, cax, orientation='horizontal')

    
            #axes[2+load_offset].set_title('GT Value with GT Traj')
            #axes[2+load_offset].set_xlim(*boundsX)
            #axes[2+load_offset].set_ylim(*boundsY)
            #axes[2+load_offset].set_aspect(1)
            #axes[2+load_offset].imshow(np.reshape(test_coord_value,(ego_pixel_shape)), extent=[*boundsX, *(ego_pixel_shape[0],0)], vmax = 0.0, vmin = -1.0, interpolation='none')
            #print('avg gt:',np.mean(np.reshape(test_coord_value,(ego_pixel_shape))))
            #axes[2+load_offset].plot(trajnp[:,0], trajnp[:,1], 'm--')
            #axes[2+load_offset].plot(tpix_AlexNet, rpix_AlexNet, 'c--')
            #axes[2+load_offset].plot(traj_vae_tpix, traj_vae_logrpix, 'w--')

            #axes[2+load_offset].axes.xaxis.set_visible(False)
            #axes[2+load_offset].axes.yaxis.set_visible(False)

            ##coord_x, coord_y = np.meshgrid(range(ego_pixel_shape[1]), range(ego_pixel_shape[0]))

            ##ax.quiver(gradient_samples[0][:,0], gradient_samples[0][:,1], gradient_samples[1][:,0],gradient_samples[1][:,1], color='red', scale_units='xy', scale=1)#, units='xy' ,scale=1

            ##ux = vx/np.sqrt(vx**2+vy**2)
            ##uy = vy/np.sqrt(vx**2+vy**2)
        
            ##for traj in test_pix_trajectory.values():

            ##plt.show()
            ##mydpi = 300
            ##fig.set_dpi(mydpi)
            ##fig.set_size_inches(1920/mydpi, 1080/mydpi)



            #fig.savefig('fffigure{}.jpg'.format(str(iFrame)), dpi=mydpi)




            #axes[1,0].imshow(a_star_map_x, extent=[*boundsX, *boundsY], interpolation='none')
            #axes[1,1].imshow(a_star_map_y, extent=[*boundsX, *boundsY], interpolation='none')
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
