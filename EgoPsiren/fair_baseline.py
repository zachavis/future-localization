from sys import platform


USING_LINUX = platform == "linux" or platform == "linux2"

if not USING_LINUX:
    import matplotlib.pyplot as plt
    #from matplotlib import image


# Need: PCL library

# For passing arguments across systems
import sys
import getopt
from pathlib import Path

import numpy as np
import cv2
from scipy import interpolate
import torch
from sklearn.neighbors import NearestNeighbors


from Common import DNN
from Common import DataGens
from Common import DataReader

#from scipy import interpolate
#from scipy import stats

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


# x is 2xN point set
# omega is nonlinear distortion parameter
# K is linear camera intrinsic
def Distort(x, omega, K):
    x_n = np.linalg.inv(K) @ np.concatenate( ( x,np.ones((1,x.shape[1])) ),axis=0)
    r_u = np.sqrt(x_n[0,:]**2 + x_n[1,:]**2)
    r_d = 1/omega * np.arctan(2*r_u*np.tan(omega/2))
    x_n_dis_x = r_d / r_u * x_n[0,:]
    x_n_dis_y = r_d / r_u * x_n[1,:]
    x_n_dis = np.concatenate( (x_n_dis_x[None], x_n_dis_y[None], np.ones((1,x.shape[1]))), axis=0)
    x_dis = K @ x_n_dis
    return x_dis[:2,:]

# x is 2xN point set
# omega is nonlinear distortion parameter
# K is linear camera intrinsic
def Undistort(x, omega, K):

    x_n = np.linalg.inv(K) @ np.concatenate( ( x,np.ones((1,x.shape[1])) ),axis=0)
    r_d = np.sqrt(x_n[0,:]**2 + x_n[1,:]**2)
    r_u = np.tan(r_d*omega)/2/np.tan(omega/2)

    x_n_undis_x = r_u / r_d * x_n[0,:]
    x_n_undis_y = r_u / r_d * x_n[1,:]
    x_n_undis = np.concatenate( (x_n_undis_x[None], x_n_undis_y[None], np.ones((1,x.shape[1]))), axis=0)
    x_undis = K @ x_n_undis
    return x_undis[:2,:]




## LOAD DATA
## Loop over folders in directory
## find relevant files:
## # Trajectory
## # Image
## # Camera Info

if __name__ == "__main__":
    print('Preparing to train on image/trajectory pairs!')
    PRINT_DEBUG_IMAGES = False and not USING_LINUX
    READ_ARGS = True

    ego_pixel_shape_Alex = (256,256)


    data_file = 'S:\\structure_0001800_00.txt' #'S:\\fut_loc\\20150401_walk_00\\traj_prediction.txt'

    necessary_args = 0
    verbose_flag = False
    data_root = Path('S:/fut_loc/')
    
    #file_path = 'S:\\ego4d_benchmark\\anna\\10600609\\REC00003'



    inputfile = ''
    outputfile = ''
    overfitoutputfile = ''

     # NEEDS INPUT VARS
    __trajLength = 100 # Maximum number of frames in a trajectory  #-l --length
    __trajStride = 20 # Frames to skip while generating each trajectory  #-s --stride
    __data_source = Path('S:/ego4d_benchmark/meghan/11500510/REC00002') #-d? -i data
    __data_target = Path('S:/ego4d_benchmark') #-o
    __data_images = Path('image') #-i --images
    
    
    #main_folder = 'S:\\ego4d_benchmark\\meghan\\11500510\\REC00002'
    
    #train_test_directories = ['dummytrain','dummytest']


    __trajectory_buffer = {} # key is the frame, and a trajectory is a list of: [1D time instance, 3D point, 2D dummy var] 
    __id2feature_buffer = {} # key is frame, and AlexNet feature is val 
    __id2traj_buffer = {} # key is frame, and traj is val 

    model_Alex = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
    model_Alex.eval() # IMPORTANT!!
    mods = list(model_Alex.named_modules())
    model_Alex.named_modules()
    children = model_Alex.children()
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
    #model._modules.register_forward_hook(get_activation('classifier.4'))


    #descriptors = np.zeros((len(RESIZED_IMAGE_DICTIONARY_TR.keys()),4096))




    #/11f247e0-179a-4b9d-8244-16fb918010a1_0
    if READ_ARGS:
        sys.argv[1:] = "--data S:/fair_baseline_test --output S:/fair_baseline_test --images im --length 100 --stride 20".split()
        print("Current program args:",sys.argv[1:])
        try:
            opts, args = getopt.getopt(sys.argv[1:],"hvd:i:o:l:s:",["help","verbose","data=","images=","output=","length=","stride="])
        except getopt.GetoptError:
            print('Arguments are malformed. TODO: put useful help here.')#'test.py -i <inputfile>')
            sys.exit(2)

        for opt, arg in opts:
            if opt in ("-d", "--data"):
                print('setting data source to be', arg,'...')
                __data_source = Path(arg)
                if not __data_source.exists():
                    print("Data path does not exist.")
                    sys.exit(3)
                necessary_args += 1

            if opt in ("-i", "--images"):
                __data_images = Path(arg)
                #if not __data_images.exists():
                #    print("Images path does not exist.")
                #    sys.exit(3)
                necessary_args += 1

            if opt in ("-o", "--output"):
                __data_target = Path(arg)
                if not __data_target.exists():
                    __data_target.mkdir()
                necessary_args += 1

            if opt in ("-l", "--length"):
                __trajLength = int(arg)
                necessary_args += 1

            if opt in ("-s", "--stride"):
                __trajStride = int(arg)
                necessary_args += 1

            if opt in ("-v", "--verbose"):
                verbose_flag = True

            if opt in ("-h", "--help"):
                PrintHelp()
                sys.exit(-1)


    for folder_path in __data_source.iterdir(): #next(os.walk(partial_folder_path))[1]:
        if not folder_path.is_dir():
            continue
        
        folder_name = folder_path.name
        #print(type(folder_name))
        #full_data_path = __data_source / folder_path

        # load calibration
        print('Loading calibration file in:',folder_name)
        calibfile = folder_path / __data_images / Path('calib_fisheye.txt')
        
        if not calibfile.is_file():
            print('Could not find calibration file.')
            continue


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
        #data = fid.readline().split()
        #princ_x1 = float(data[1])

        ##princ_y2 = data{2}(9);
        #data = fid.readline().split()
        #princ_y2 = float(data[1])

        K_data = np.array([[focal_x, 0, princ_x],[ 0, focal_y, princ_y],[ 0, 0, 1]])
        R_rect = np.eye(3) # NOT STERO, uneeded offset #np.array([[0.9989,0.0040,0.0466],[-0.0040,1.0000,-0.0002],[-0.0466,0,0.9989]])
        #fclose(fid);

        fid.close()



        # load file list
        print('loading file list file');
        file_list = folder_path / Path('im_list.list')
        #fid = open(file_list);
        #data = textscan(fid, '%s');
        #data = fid.readlines()
        with open(file_list) as fid:
            data = fid.read().splitlines()
        vFilename = data;
        #fid.close();





        print('loading trajectory file')
        traj_data_file = folder_path / Path('traj_prediction.txt')
        vTR = ReadTraj(traj_data_file)

        print('Read',len(vTR['vTr']),'trajectories.')


        initial_offset = 0
        for iFrame in range(initial_offset, len(vTR['vTr'])):

            unique_id = str( Path(folder_name) / Path(vFilename[iFrame]) )

            print("\tGETTING frame", iFrame, 'AKA', vFilename[iFrame])
            tr = vTR['vTr'][iFrame]
            frames = tr['frame']
            #if (len(tr['XYZ'][1]) == 0):
            #    print('SKIPPING FRAME',iFrame,': Trajectory is empty.')
            #    continue

            #im = sprintf('%sim/%s', folder_path, vFilename{iFrame});
            im = folder_path / __data_images / Path(vFilename[iFrame]) #"{}im\\{}".format(folder_path, vFilename[iFrame])
            #disp = "{}disparity\\{}{}".format(folder_path, vFilename[iFrame],'.disp.txt')

            if not im.is_file():
                print('\t\tCould not find file:',im)
                continue
            #if not os.path.isfile(disp):
            #    print('could not find file')
            #    continue
            


            print('\t\tTrajectory contains',len(tr['XYZ'][1]),'points.')
            img = cv2.cvtColor(cv2.imread(str(im)), cv2.COLOR_BGR2RGB).astype(np.float64)/255.0
            #disp_img = np.genfromtxt(disp, delimiter=',')[:,:-1] #np.loadtxt(disp, delimiter=',')
        
        
            #cv2.imshow('intermediate',intermediate)
            #cv2.waitKey()

            #im = im2double(intermediate); % im2double(imread(im));

            tr_ground_OG = (tr['XYZ'].T - tr['up']).T #bsxfun(@minus, tr['XYZ'], tr['up']);
            #tr_ground_OG[1] *= 0
            #tr_ground_OG[1] -= tr['up'][1]

            #t, r = Coord2Polar(tr_ground_OG[2],tr_ground_OG[0])
            #tr_ground_OG[2], tr_ground_OG[0] = Polar2Coord(t,r)
            tr_ground = K_data @ R_rect @ tr_ground_OG;
        
            if False: #Do we need this?
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
                print("\t\tSkipping because of alignment severity")
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
        
            #t, r = Coord2Polar(tr_ground_ALIGNED[2],tr_ground_ALIGNED[0]) #Coord2Polar(tr['XYZ'][2],tr['XYZ'][0])
            #logr = np.log(r)

            #world_forward = 
            img_resized = cv2.resize(img, (int(img.shape[1]*.25), int(img.shape[0]*.25)))*2.0-1.0
            img_channel_swap = np.moveaxis(img_resized,-1,0).astype(np.float32)

            colored_part = ['c','r','m']
        
        

            points = np.random.uniform(250,750,(2,2000)) #np.array([[500,500,500],
                               #[250,500,750]])

            distorted = Distort(points, omega, K_data)
            undistorted = Undistort(distorted, omega, K_data)


            result = points - undistorted





            all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(img.shape[0]) for j in range(img.shape[1]) ], dtype=np.float32).T
            all_pixel_coords_undistorted = Distort(all_pixel_coords,omega,K_data)

            pix_part2 = np.copy(all_pixel_coords_undistorted)
            pix_part2[0] = all_pixel_coords_undistorted[1]
            pix_part2[1] = all_pixel_coords_undistorted[0]
            #rowmaj_pixels[:2].T

            img_undistorted = interpolate.interpn( (range(img.shape[0]),range(img.shape[1])), img, pix_part2[:2].T , method = 'linear',bounds_error = False, fill_value = 0).astype(np.float32).reshape(img.shape[0], img.shape[1],3)














            ## For each image/trajectory read, we need to 
            ### Warp
            ### Encode
            ### 

            ### EGO RETINAL WARPING ##
            #all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(ego_pixel_shape_Alex[0]) for j in range(ego_pixel_shape_Alex[1]) ], dtype=np.float32)
            ##all_pixel_coords[:,1] = np.flip(all_pixel_coords[:,1])
            #print(all_pixel_coords[:,0].max())
            #print(all_pixel_coords[:,1].max())
            #print(all_pixel_coords[:,0].min())
            #print(all_pixel_coords[:,1].min())
            ###all_pixel_coords_xformed = np.zeros(all_pixel_coords.shape)
            #all_pixel_coords[:,0] = big_ego_pix2t(all_pixel_coords[:,0])
            #print(all_pixel_coords[:,0].max())
            #print(all_pixel_coords[:,0].min())
            #all_pixel_coords[:,1] = big_ego_pix2r(all_pixel_coords[:,1])
            #print(all_pixel_coords[:,1].max())
            #print(all_pixel_coords[:,1].min())
            #all_pixel_coords[:,1] = np.exp(all_pixel_coords[:,1])
            #print(all_pixel_coords[:,1].max())
            #print(all_pixel_coords[:,1].min())
            #z, x = DataReader.Polar2Coord(all_pixel_coords[:,0],all_pixel_coords[:,1])

            #coords_3D = np.zeros((len(z),3))
            #coords_3D[:,0] = x
            #coords_3D[:,2] = z
        
            #coords_3D = (R_rect_ego.T @ coords_3D.T).T # ALIGN "WORLD-SPACE" GROUND PLANE TO CAMERA SPACE
            #coords_3D -= tr['up'].T # SHIFT PLANE TO CORRECT LOCATION RELATIVE TO CAMERA

            



            ##coord_value = DataGens.Coords2ValueFastWS(all_pixel_coords_xformed,{0:test_ws_trajectory},None,None,stddev=.5)
            
            #e_origin = tr['up'] #camera assumed to be at 0,0,0
            ##e_origin = np.zeros(3) #zero vector


            ##coords_3D[:,1] *= -1

            #pixels = K_data @ R_rect @ coords_3D.T
            #pixels /= pixels[2]
            ##pixels[:,:] /= pixels[2,:]

            #rowmaj_pixels = np.zeros(pixels.shape)
            #rowmaj_pixels[0] = pixels[1]
            #rowmaj_pixels[1] = pixels[0]


            #img2 =      interpolate.interpn((range(img.shape[0]),range(img.shape[1])),          img, rowmaj_pixels[:2].T , method = 'linear',bounds_error = False, fill_value = 0).reshape(ego_pixel_shape_Alex[0], ego_pixel_shape_Alex[1],3)








            #homography = K_data @ R_rect @ R_rect_ego @ R_rect.T @ np.linalg.inv(K_data)

            #img_rectified = cv2.warpPerspective(img*2.0-1.0, homography, (img.shape[1], img.shape[0])) # want to shift the values here so that the normalized version has black in the rectified location

       

            img_resized = cv2.resize(img_undistorted, (ego_pixel_shape_Alex[1], ego_pixel_shape_Alex[0]))
            img_channel_swap = np.moveaxis(img_resized,-1,0).astype(np.float32)

             
            pos_and_time = np.vstack((tr_ground_OG,tr['frame']))
            __id2traj_buffer[unique_id] = pos_and_time #tr_ground


            img_channel_swap_unsqueezed = torch.unsqueeze(torch.from_numpy(img_channel_swap),0)

            returns = model_Alex(img_channel_swap_unsqueezed)
            feature = activation['classifier.4']
            __id2feature_buffer[unique_id] = feature[0].numpy()

            testing = feature[0].numpy()


            if False:
                # DISPLAY IMAGES
                fig, ax = plt.subplots(1,1)#, figsize=(18,6))
                axes = [ax] # only use if there's 1 column

                #newBoundsx = (crowdBoundsX[0], 2*crowdBoundsX[1])
                #fig.set_size_inches(16, 24)
                axes[0].set_title(str(len(tr['XYZ'][1])) + '-frame trajectory in ' + vFilename[iFrame])
                #axes[0].set_xlim(*newBoundsx)
                #axes[0].set_ylim(*crowdBoundsY)
        
                axes[0].set_xlim(0,img.shape[1])
                axes[0].set_ylim(img.shape[0],0)
                axes[0].set_aspect(1)
                axes[0].imshow(img_resized)
                axes[0].plot(tr_ground[0,:20], tr_ground[1,:20], colored_part[iFrame % 3])
                axes[0].plot(tr_ground[0,19:40], tr_ground[1,19:40], colored_part[(iFrame +1) % 3])
                axes[0].plot(tr_ground[0,39:], tr_ground[1,39:], colored_part[(iFrame +2) % 3])
                axes[0].plot(tr_ground[0], tr_ground[1], 'k.', markersize=1)


                #axes[2].set_title('Rectified image')
                #axes[0].set_xlim(*newBoundsx)
                #axes[0].set_ylim(*crowdBoundsY)
                #axes[0].set_aspect(1)

                #homography = K_data @ R_rect @ np.linalg.inv(K_data)
                plt.show()


            print('\t\tDescriptors so far:',len(__id2feature_buffer))
            #print('framekill')
        print('\tDone with folder',folder_name)
    

    # Finally, let's train the KNN classifier and save the data.
    print('Building descriptors buffer...')
    descriptors = np.zeros((len(__id2feature_buffer.keys()),4096))

    i = 0
    for frame in __id2traj_buffer.keys():

        #img = torch.unsqueeze(torch.from_numpy(__id2feature_buffer[frame]),0)

        #returns = model(img)
        #feature = activation['classifier.4']
        descriptors[i] = __id2feature_buffer[frame] #feature[0].numpy()
        i += 1

        
    print('Fitting KNN...')
    knn = NearestNeighbors(n_neighbors = 5).fit(descriptors)


    if False: # Skip sanity check
        # Sanity check
        #testimg = torch.unsqueeze(torch.from_numpy( __id2traj_buffer[list(__id2traj_buffer.keys())[1]]),0)
        #returns = model_Alex(testimg)
        #feature = activation['classifier.4']
    
        feat = __id2feature_buffer[list(__id2feature_buffer.keys())[3]]
        feat = np.expand_dims(feat, 0)
        dist, idx = knn.kneighbors(feat)


        #check = np.array(list(RESIZED_IMAGE_DICTIONARY_TR.keys())) == np.array(list(LOG_POLAR_TRAJECTORY_DICTIONARY_TR.keys()))

        nearest_feature = __id2traj_buffer[list(__id2traj_buffer.keys())[idx[0,0]]]
       
    
    knnPickle = open('knn_alexfeats_FAIR.knn','wb')
    pickle.dump(knn,knnPickle)
    knnPickle.close()

    dictPickle = open('knn_traintraj_aligned_FAIR','wb')
    pickle.dump(__id2feature_buffer,dictPickle)
    dictPickle.close()

    print("Conclusion.")









    print('done')



    ## PROCESS INPUT
    ## Project trajectory and map into EgoSpace
    ## # Use constants from paper
    ## # Trim trajectory to start at minimum distance
    ## # # Some trajectories start "behind" camera due to unavoidable reconstruction errors

    ## TRAIN BENCHMARK
    ## 
