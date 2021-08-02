import numpy as np
import cv2 
import random
import copy 
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


#point_cloud = np.loadtxt('S:\\structure.txt')
#print(point_cloud[:10])

class Plane:
    """ 
    Implementation of planar RANSAC.
    Class for Plane object, which finds the equation of a infinite plane using RANSAC algorithim. 
    Call `fit(.)` to randomly take 3 points of pointcloud to verify inliers based on a threshold.
    ![Plane](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/plano.gif "Plane")
    ---
    """

    def __init__(self):
        self.inliers = []
        self.equation = []



    def fit(self, pts, thresh=0.05, minPoints=100, maxIteration=1000, normal_prior=None, max_angle_deflection=np.pi/6.0):
        """ 
        Find the best equation for a plane.
        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param thresh: Threshold distance from the plane which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
        - `self.inliers`: points from the dataset considered inliers
        ---
        """
        n_points = pts.shape[0]
        best_eq = []
        best_inliers = []

        for it in range(maxIteration):

            # Samples 3 random points 
            id_samples = random.sample(range(1, n_points-1), 3)
            pt_samples = pts[id_samples]

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1

            vecA = pt_samples[1,:] - pt_samples[0,:]
            vecB = pt_samples[2,:] - pt_samples[0,:]

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA, vecB)
            

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
            # We have to use a point to find k
            vecC = vecC / np.linalg.norm(vecC)

            
            ### But first, let's see if there's a prior to satisfy
            if normal_prior is not None:
                result = vecC @ normal_prior
                if result < 0:
                    vecC *= -1 # flip to same side of plane as normal prior
                deflection = np.arccos(result) # Assuming both are normal
                if deflection > max_angle_deflection:
                    continue

            # Continuing where we left off...
            k = -np.sum(np.multiply(vecC, pt_samples[1,:]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            # Distance from a point to a plane 
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            pt_id_inliers = [] # list of inliers ids
            dist_pt = (plane_eq[0]*pts[:,0]+plane_eq[1]*pts[:, 1]+plane_eq[2]*pts[:, 2]+plane_eq[3])/np.sqrt(plane_eq[0]**2+plane_eq[1]**2+plane_eq[2]**2)
            
            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if(len(pt_id_inliers) > len(best_inliers)):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
            self.inliers = best_inliers
            self.equation = best_eq

        return self.equation, self.inliers

#vtR = {}
def ReadPointCloud(data_filename):
    fid = open(data_filename, 'r')
    #data = textscan(fid, '%s %f', 1);
    data = fid.readline().split()


    # total number of images
    #n = data{2};
    n = int(data[1])
    print(n, 'points')

    #vTakenFrame = [0]*n #cell(n, 1);
    #vTr = [] #[{}]*n #cell(n, 1);
    point_cloud = {}

    #for i in range(n):
    #    vTr.append({})

    #for i in range(n):
    data = fid.read().split()#  .readline().split()
    data = np.array(data, dtype=np.float32)

    # frame id
    #data = textscan(fid, '%f', 5);
    #iFrame = int(data[0])# + 1
    
    # up
    #vTr[iFrame]['up'] = data[1:4]
    #print(vTr[iFrame]['up'])
    
    # trajectory length
    #nTrjFrame = int(data[4])
    
    # trajectory data
    #data = data[5:] # ;
    data = np.reshape(data, (n,7) ).T

    point_cloud['descriptor_id'] = data[0, :]
    point_cloud['RGB'] = data[1:4, :]
    point_cloud['XYZ'] = data[4:7, :]
    #vTakenFrame[iFrame] = iFrame;

    #vTakenFrame = cat(1, vTakenFrame[:]);

    #vTR['vTakenFrame'] = vTakenFrame;
    #vTR['vTr'] = vTr;
    fid.close()
    return point_cloud


def ReadCameraFile1(data_filename, img_idx_offset = 0 * 200):
    fid = open(data_filename, 'r')
    data = fid.readline().split()
    numCams = int(data[1])
    data = fid.readline().split()
    numFrames = int(data[1])
    data = fid.readline().split()
    numP = int(data[1])

    frames = {}
    frames['img_idx'] = np.zeros(numP).astype(np.int)
    frames['C'] = np.zeros((numP,3)).astype(np.float32)
    frames['R'] = np.zeros((numP,3,3)).astype(np.float32)

    for i in range(numP):
        data = fid.readline().split()
        data = np.array(data, dtype=np.int)
        frames['img_idx'][i] = data[1] + img_idx_offset # Gets the correct frame ID -- original is relative to chunk set

        data = fid.readline().split()
        data = np.array(data, dtype=np.float32)
        frames['C'][i] = data

        
        data = fid.readline().split()
        data = np.array(data, dtype=np.float32)
        frames['R'][i][0] = data
        
        data = fid.readline().split()
        data = np.array(data, dtype=np.float32)
        frames['R'][i][1] = data
        
        data = fid.readline().split()
        data = np.array(data, dtype=np.float32)
        frames['R'][i][2] = data

    fid.close()

    sorted = np.argsort(frames['img_idx'])
    frames['img_idx'] = frames['img_idx'][sorted]
    frames['C'] = frames['C'][sorted]
    frames['R'] = frames['R'][sorted]

    return frames


def ReadCameraFile(data_filename, img_idx_offset = 0 * 200):
    fid = open(data_filename, 'r')
    data = fid.readline().split()
    numCams = int(data[1])
    data = fid.readline().split()
    numFrames = int(data[1])
    data = fid.readline().split()
    numP = int(data[1])

    frames = {}
    #frames['img_idx'] = np.zeros(numP).astype(np.int)
    #frames['C'] = np.zeros((numP,3)).astype(np.float32)
    #frames['R'] = np.zeros((numP,3,3)).astype(np.float32)

    for i in range(numP):
        data = fid.readline().split()
        data = np.array(data, dtype=np.int)
        #frames['img_idx'][i] = data[1] + img_idx_offset # Gets the correct frame ID -- original is relative to chunk set
        frame_idx = data[1]+img_idx_offset
        frames[frame_idx] = {}
        frames[frame_idx]['C'] = np.zeros((numP,3)).astype(np.float32)
        frames[frame_idx]['R'] = np.zeros((3,3)).astype(np.float32)

        data = fid.readline().split()
        data = np.array(data, dtype=np.float32)
        frames[frame_idx]['C'] = data

        data = fid.readline().split()
        data = np.array(data, dtype=np.float32)
        frames[frame_idx]['R'][0] = data
        
        data = fid.readline().split()
        data = np.array(data, dtype=np.float32)
        frames[frame_idx]['R'][1] = data
        
        data = fid.readline().split()
        data = np.array(data, dtype=np.float32)
        frames[frame_idx]['R'][2] = data

        #frames[frame_idx]['R'] = -1 * frames[frame_idx]['R'].T
    fid.close()

    #sorted = np.argsort(frames['img_idx'])
    #frames['img_idx'] = frames['img_idx'][sorted]
    #frames['C'] = frames['C'][sorted]
    #frames['R'] = frames['R'][sorted]

    return frames

def ReadCalibration(data_filename):
        print('loading calibration file')

        calib = {}

        fid = open(data_filename)
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

        ##princ_x1 = data{2}(8);
        #data = fid.readline().split()
        #princ_x1 = float(data[1])

        ##princ_y2 = data{2}(9);
        #data = fid.readline().split()
        #princ_y2 = float(data[1])

        K_data = np.array([[focal_x, 0, princ_x],[ 0, focal_y, princ_y],[ 0, 0, 1]])
        #R_rect = np.array([[0.9989,0.0040,0.0466],[-0.0040,1.0000,-0.0002],[-0.0466,0,0.9989]])

        calib['K'] = K_data
        calib['omega'] = omega
        #calib['R'] = R_rect
        #fclose(fid);

        fid.close()
        return calib



def ProjectWithDistortion(omega, K, R, C, X):
    
    X_c = R @ (X - C)
    pix_u = K @ X_c
    pix_u /= pix_u[2] # undistorted pixel
    
    #pix_u_n = np.copy(pix_u)
    #pix_u_n[:2] -= K[:2,2] # normalized undistorted pixel
    pix_u_n = np.linalg.inv(K) @ pix_u

    r_u = np.sqrt(pix_u_n[0]**2 + pix_u_n[1]**2)
    if r_u == 0:
        return pix_u

    r_d = np.arctan(2*r_u*np.tan(omega*.5))
    r_d /= omega

    pix_d_n = np.copy(pix_u_n)
    pix_d_n *= r_d/r_u
    pix_d_n[2] = 1
    
    #pix_d = np.copy(pix_d_n)
    #pix_d[:2] += K[:2,2] # unnormalized distorted pixel
    pix_d = K @ pix_d_n
    
    if X_c @ R[2] < 0:
        pix_d_n[2] = -1 # behind camera
    
    
    return pix_d

def Distort(x,omega,K):
    x_n = np.linalg.inv(K) @ np.concatenate( ( x,np.ones((1,x.shape[1])) ),axis=0)
    r_u = np.sqrt(x_n[0,:]**2 + x_n[1,:]**2)
    r_d = 1/omega * np.arctan(2*r_u*np.tan(omega/2))
    x_n_dis_x = r_d / r_u * x_n[0,:]
    x_n_dis_y = r_d / r_u * x_n[1,:]
    x_n_dis = np.concatenate( (x_n_dis_x[None], x_n_dis_y[None], np.ones((1,x.shape[1]))), axis=0)
    x_dis = K @ x_n_dis
    return x_dis[:2,:]


def PrintHelp():
    print("-d --data:\treconstruction [d]ata")
    print("-i --images:\[i]mages used for reconstruction")
    print("-o --output:\ttrajectory [o]utput location")
    print("-l --length:\tmax trajectory [l]ength in frames")
    print("-s --stride:\tmin [s]tride between trajectories")
    return







########## ########## ########## ########## ########## ########## 
## MAIN ## MAIN ## MAIN ## MAIN ## MAIN ## MAIN ## MAIN ## MAIN 
########## ########## ########## ########## ########## ########## 


if __name__ == "__main__":
    print('Preparing to reconstruct trajectories!')
    PRINT_DEBUG_IMAGES = False and not USING_LINUX
    READ_ARGS = True

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

    if READ_ARGS:
        #sys.argv[1:] = "--data S:/ego4d_benchmark/meghan/11500510/REC00002 --output S:/ego4d_benchmark --images image --length 100 --stride 20".split()
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

    #if necessary_args < 2:
    #    print('Not enough args')
    #    PrintHelp()
    #    sys.exit(3)



    #frame_offset = 2400 # I don't think we need this, each individual folder must be processed
    #start_frame = 2460
    #downscaler = 2.3
    #rightscaler = 0.4
    #forwardscaler = 0.4
    
    #n_frames = 180


    USE_GLOBAL_MEAN_DOWN = True
    USE_MEAN_DOWN = False #True


    # Loop over reconstruction folders
    for folder_path in __data_source.iterdir(): #next(os.walk(partial_folder_path))[1]:
            if not folder_path.is_dir():
                print('skipping',folder_path,'...')
                continue
            full_part = folder_path.name
            identifier = folder_path.name[:14]
            if identifier != 'reconstruction':
                print('skipping',folder_path,'with ID:', identifier, 'is is not reconstruction...')
                continue
            recon_id = folder_path.name[14:]
            starting_frame = int(recon_id)
            print('reconstructing',folder_path,'...')

            
            reconstruction_folder = Path('reconstruction{:07d}'.format(starting_frame))
            print("Reading Calibration...")
            calib = ReadCalibration(__data_source / __data_images / Path('calib_fisheye.txt'))
            print('K:',calib['K'])
            print('omega:',calib['omega'])

            cameras_path = __data_source / reconstruction_folder / Path('camera.txt')

            if not cameras_path.exists():
                print('Could not find camera.txt. Skipping...')
                continue

            frames = ReadCameraFile(cameras_path, starting_frame)
            num_frames = len(frames)



            mean_start = starting_frame
            #while mean_start < starting_frame + num_frames:
            #    # blah

            # Get all valid frames in trajLength sequence
            valid_frames = []
            for key in range( mean_start, mean_start + num_frames ):
                if key in frames:
                    valid_frames.append(key)

            print(len(valid_frames))

            # PROCESS THE TRAJECTORY AND ADD IT TO A LIST
                
            global_mean_down = np.zeros(3)
            num_valid_frames = len(valid_frames)
            global_mean_position = np.zeros(3)
            for i in range(num_valid_frames):
                thisR = frames[valid_frames[i]]['R']
                thisdown = thisR[1]
                thisC = frames[valid_frames[i]]['C']
                global_mean_down += thisdown
                global_mean_position += thisC
            global_mean_down /= np.linalg.norm(global_mean_down)
            global_mean_position /= num_valid_frames
        
            if USE_GLOBAL_MEAN_DOWN and not USE_MEAN_DOWN:
                world_down = global_mean_down



            # TRAJECTORY SEGMENT INDEPENDENT PLANE FITTING:

            point_cloud = ReadPointCloud(__data_source / reconstruction_folder / Path('structure.txt'))
            
            X = point_cloud['XYZ']

            if X.shape[1] < 20:
                print('Insufficient points (', X.shape[1], '/20) for reconstruction.')
                continue
            else:
                print('Point cloud contains (', X.shape[1], ') points.')

            bigC = np.ones((3,X.shape[1]))
            bigC = global_mean_position[:,None] * bigC
            X___ = X-bigC
            #camdot = forward @ X___
            camdot_below = world_down @ X___



            #infront_logical = np.zeros(X___.shape[1], dtype=bool)
            #infront_logical[camdot>=0]=True
        
            below_logical = np.zeros(X___.shape[1], dtype=bool)
            below_logical[camdot_below>=0]=True
        
            #just_below = np.logical_and(below_logical, infront_logical)


            X_ = X___[:,below_logical] # shifted, but below the horizon
            X__ = X[:,below_logical] # non shifted, but below the horizon

            
            if X_.shape[1] < 20:
                print('Insufficient points below camera (', X_.shape[1], '/20) for reconstruction.')
                continue
            else:
                print('Point cloud contains (', X_.shape[1], ') points below the camera.')

            PlaneSeg = Plane()
            best_eq, plane_inliers = PlaneSeg.fit(X_.T, 0.6,20,1000,-world_down, np.pi/12.0)

            #inliers_logical = np.zeros(X_.shape[1], dtype=bool)
            #inliers_logical[best_inliers]=True

            #augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
            #P = calib['K'] @ cameraRotation #@ augC

            #X_ = cameraRotation @ X_ # Align into camera space where 



            
                
                    
            # Loop over frames with given stride and trajlength
            traj_start = starting_frame
            while traj_start + __trajLength < starting_frame + num_frames:
                # blah

                
                if not traj_start in frames: # make sure this frame is okay
                    traj_start += __trajStride # Next trajectory TODO: Should this just increment by 1 or something?
                    continue

                # Get all valid frames in trajLength sequence
                valid_frames = []
                for key in range( traj_start, traj_start + __trajLength ):
                    if key in frames:
                        valid_frames.append(key)
                     #else:
                        # print('Frame',key,'was not reconstructed.')


                #for i_frame in range(traj_start, traj_start+num_frames):
                #    # CHECK IF FRAME EXISTS
                #    if not i_frame in frames: # make sure this frame is okay
                #        continue



                # PROCESS THE TRAJECTORY AND ADD IT TO A LIST

                cameraCenter = frames[valid_frames[0]]['C']
                cameraRotation = frames[valid_frames[0]]['R']

                mean_down = np.zeros(3)
                num_valid_frames = len(valid_frames)
                for i in range(num_valid_frames):
                    thisR = frames[valid_frames[i]]['R']
                    thisdown = thisR[1]
                    mean_down += thisdown
                mean_down /= np.linalg.norm(mean_down)
        
                if USE_MEAN_DOWN:
                    world_down = mean_down
                elif not USE_GLOBAL_MEAN_DOWN:
                    world_down = np.array([0,1,0])
        
                right = cameraRotation[0]
                down = cameraRotation[1]
                forward = cameraRotation[2]



                if not USE_GLOBAL_MEAN_DOWN:

                    #point_cloud = ReadPointCloud(__data_source / reconstruction_folder / Path('structure.txt'))

                    X = point_cloud['XYZ']
                    bigC = np.ones((3,X.shape[1]))
                    bigC = cameraCenter[:,None] * bigC
                    X_ = X-bigC
                    camdot = forward @ X_
                    camdot_below = (cameraRotation @ world_down) @ X_ # TODO FIX THIS -- IT'S DEPENDENT ON CAMERA SPACE BUT NOT TRANSFORMED


                    infront_logical = np.zeros(X_.shape[1], dtype=bool)
                    infront_logical[camdot>=0]=True
        
                    below_logical = np.zeros(X_.shape[1], dtype=bool)
                    below_logical[camdot_below>=0]=True
        
                    just_below = np.logical_and(below_logical, infront_logical)


                    X_ = X_[:,just_below]

                    PlaneSeg = Plane()
                    best_eq, plane_inliers = PlaneSeg.fit(X_.T, 0.5,20,1000,-world_down, np.pi/12.0)

                    best_inliers = plane_inliers

                    inliers_logical = np.zeros(X_.shape[1], dtype=bool)
                    inliers_logical[best_inliers]=True

                    augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
                    P = calib['K'] @ cameraRotation #@ augC

                    X_ = cameraRotation @ X_ # Align into camera space where 
                else:
                    #point_cloud = ReadPointCloud(__data_source / reconstruction_folder / Path('structure.txt'))

                    #X = point_cloud['XYZ']
                    bigC = np.ones((3,X__.shape[1]))
                    bigC = cameraCenter[:,None] * bigC
                    X_ = X__-bigC # use non-shifted but below horizon X values
                    camdot = forward @ X_
                    camdot_below = (cameraRotation @ world_down) @ X_


                    infront_logical = np.zeros(X_.shape[1], dtype=bool)
                    infront_logical[camdot>=0]=True
        
                    #below_logical = np.zeros(X_.shape[1], dtype=bool)
                    #below_logical[camdot_below>=0]=True
        
                    #just_below = np.logical_and(below_logical, infront_logical)


                    #X_ = X_[:,just_below]

                    #PlaneSeg = Plane()
                    #best_eq, best_inliers = PlaneSeg.fit( X_.T, 0.5,20,1000,-world_down, np.pi/12.0)

                    #inliers_logical = np.zeros(X_.shape[1], dtype=bool)
                    #inliers_logical[plane_inliers]=True
                    
                    #in_plane_and_in_front = np.logical_and(infront_logical,inliers_logical)
                    
                    #best_inliers = np.where(inliers_logical[infront_logical])[0]

                    test1 = np.arange(X_.shape[1])
                    test2 = test1[infront_logical]

                    best_inliers = np.nonzero(np.in1d(np.arange(X_.shape[1])[infront_logical],plane_inliers))[0]
                    
                    X_ = X_[:,infront_logical]

                    #best_inliers = [] # list of inliers ids
                    #best_eq2 = np.copy(best_eq)
                    ##best_eq2[:-1] = cameraRotation @ best_eq2[:-1]
                    #dist_pt = (best_eq2[0]*X_[0,:]+best_eq2[1]*X_[1, :]+best_eq2[2]*X_[2, :]+best_eq2[3])/np.sqrt(best_eq2[0]**2+best_eq2[1]**2+best_eq2[2]**2)
            
                    ## Select indexes where distance is biggers than the threshold
                    #best_inliers = np.where(np.abs(dist_pt) <= 0.5)[0]


                    augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
                    P = calib['K'] @ cameraRotation #@ augC

                    X_ = cameraRotation @ X_ # Align into camera space where 
                    #infront_logical = np.zeros(X___.shape[1], dtype=bool)
                    #infront_logical[camdot>=0]=True
                    #just_below = np.logical_and(below_logical, infront_logical)
                    
                    #X_ = X___[:,just_below]
                    #P = calib['K'] @ cameraRotation #@ augC
                    #X_ = cameraRotation @ X_ # Align into camera space where 


                x = calib['K'] @ X_
        
                x /= x[2]
                x = x[:2]
                x_dis_struct = Distort(x,calib['omega'],calib['K'])


                __AVERAGE_HUMAN_HEIGHT = 1.71 # meters
                # Modify trajectory points

                plane_normal = np.array(best_eq[:3])
                # since plane is calculated in camera-centered space, we only need d/sqrt(a,b,c)
                current_distance_to_ground = np.abs(best_eq[3] / np.linalg.norm(plane_normal))
                corrective_scalar = __AVERAGE_HUMAN_HEIGHT / current_distance_to_ground
                plane_normal_with_metric = plane_normal * __AVERAGE_HUMAN_HEIGHT / np.linalg.norm(plane_normal)
                check = np.linalg.norm(plane_normal_with_metric)


        
                points = np.zeros((len(valid_frames[1:]),3))
                count = 0
                for key in valid_frames[1:]:
                    #print('Frame:',key)
                    points[count] = frames[key]['C']
                    count += 1

                #points +=  np.ones(points.shape)*downscaler*down[None] + np.ones(points.shape)*rightscaler*right[None] + np.ones(points.shape)*forwardscaler*forward[None]



                X = points.T
                bigC = np.ones((3,X.shape[1]))
                bigC = cameraCenter[:,None] * bigC
                X_ = X-bigC


        
                # RESCALE TRAJECTORY HERE
                X_ *= corrective_scalar

                # ROTATE SO THAT POINTS SIT IN WORLD WHERE CAMERA ROTATION IS IDENTITY
                X_aligned_cam = cameraRotation @ X_
                plane_normal_with_metric_aligned_cam = cameraRotation @ plane_normal_with_metric



                ## ONLY FOR IMAGE PROCESSING
                ## This places the trajectory on the floor.
                #X_ = X_ - plane_normal_with_metric[:,None]
        
                #camdot = forward @ X_
                #X_ = X_[:,camdot>=0] # only in front of camera

                #augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
                #P = calib['K'] @ cameraRotation #@ augC
                ##augX_ = np.concatenate( ( X_, np.ones((1,X_.shape[1])) ), axis=0)
                #x = P @ X_
                #x /= x[2]
                #x = x[:2]
                #x_dis = Distort(x,calib['omega'],calib['K'])


                __trajectory_buffer[traj_start] = {}
                __trajectory_buffer[traj_start]['XYZ'] = X_aligned_cam
                __trajectory_buffer[traj_start]['up'] = plane_normal_with_metric_aligned_cam






                if PRINT_DEBUG_IMAGES:
                     #points = np.zeros((len(valid_frames[1:]),3))
                    #count = 0
                    #for key in valid_frames[1:]:
                    #    print('Frame:',key)
                    #    points[count] = frames[key]['C']
                    #    count += 1

                    ##points +=  np.ones(points.shape)*downscaler*down[None] + np.ones(points.shape)*rightscaler*right[None] + np.ones(points.shape)*forwardscaler*forward[None]



                    #X = points.T
                    #bigC = np.ones((3,X.shape[1]))
                    #bigC = cameraCenter[:,None] * bigC
                    #X_ = X-bigC


        
                    ## RESCALE TRAJECTORY HERE
                    #X_ *= corrective_scalar
                    #X_ = X_ - plane_normal_with_metric[:,None]


                    X_ = X_aligned_cam - plane_normal_with_metric_aligned_cam[:,None]
                    X_ = X_[:,X_[2] > 0]
                    x = calib['K'] @ X_
                    x /= x[2]
                    x = x[:2]
                    x_dis = Distort(x,calib['omega'],calib['K'])


        
                    #camdot = forward @ X_
                    #X_ = X_[:,camdot>=0] # only in front of camera

                    #augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
                    #P = calib['K'] @ cameraRotation #@ augC
                    ##augX_ = np.concatenate( ( X_, np.ones((1,X_.shape[1])) ), axis=0)
                    #x = P @ X_
                    #x /= x[2]
                    #x = x[:2]
                    #x_dis = Distort(x,calib['omega'],calib['K'])




                    test_x, test_z  = np.meshgrid(np.linspace(-3,3,7),np.linspace(1,8,8))
                    x = test_x.flatten()
                    z = test_z.flatten()

                    #z,x = Polar2Coord(test_t,test_r) # x,z ?
                    #print(z)
                    #print(x)

                    #print('actual z:', test_r[0], '*', 'np.cos(',test_t[0],') =', test_r[0] * np.cos(test_t[0]))
                    #print('actual x:', test_r[0], '*', 'np.sin(',test_t[0],') =', test_r[0] * np.sin(test_t[0]))

                    coords_3D = np.zeros((len(z),3))
                    coords_3D[:,0] = x
                    coords_3D[:,2] = z
            
            
                    #coords_3D = (R_rect_ego.T @ coords_3D.T).T # ALIGN "WORLD-SPACE" GROUND PLANE TO CAMERA SPACE
                    #coords_3D -= tr['up'].T # SHIFT PLANE TO CORRECT LOCATION RELATIVE TO CAMERA
                    coords_3D -= plane_normal_with_metric_aligned_cam
            
                    pixels = P @ coords_3D.T
                    pixels /= pixels[2]
                    pixels = pixels[:2]
                    grid_dis = Distort(pixels,calib['omega'],calib['K'])




                    print('projection stuff')
                    img_dir = __data_source / __data_images / Path('image{:07d}.jpg'.format(traj_start)) #file_path+'\\image\\image{:07d}.jpg'.format(start_frame)

                    img_dir_str = str(img_dir.resolve())
                    img = cv2.imread(img_dir_str)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.0
                    #img[:,:] = 1



                    fig, ax = plt.subplots(1,1)
                    axes = [ax]
                    axes[0].imshow(img)
                    #axes[0].plot(0,0, 'ro', markersize = 10.0)
                    #axes[0].plot(calib['K'][0,2],calib['K'][1,2], 'ro', markersize = 10.0)
                    pix_d = np.zeros((len(valid_frames[1:]),3))
                    start_key = valid_frames[0]
                    #count = 0
                    #for key in valid_frames[1:]:
                    #    camera_center = frames[key]['C']
                    #    camera_center[1] = .7
                    #    pix_d[count] = ProjectWithDistortion(calib['omega'],calib['K'],frames[start_key]['R'],frames[start_key]['C'],camera_center)
                    #   # if frames.has_key()

                    #    #pix_d[count] = ProjectWithDistortion(calib['omega'],calib['K'],np.eye(3),np.zeros(3),points[count])
                    #    count += 1
                    #    #pix_u = calib['K'] @ points[i]
                    #    #pix_d = cv.fisheye


        
                    #in_front = np.where(pix_d[:,2] > 0)
                    #pix_front = pix_d[in_front]
                    #ys = pix_front[:,1]
                    #xs = pix_front[:,0]
                    #axes[0].plot(x_dis_struct[0], x_dis_struct[1], 'rx', alpha=.5, markersize = 0.5)#, markersize = 2.0)

                    axes[0].plot(x_dis_struct[0], x_dis_struct[1], 'rx', alpha=.5, markersize = 0.3)#, markersize = 2.0)
                    #axes[0].plot(x_dis_struct[0,just_else], x_dis_struct[1,just_else], 'rx', alpha=.5, markersize = 0.5)#, markersize = 2.0)
                    axes[0].plot(x_dis_struct[0,best_inliers], x_dis_struct[1,best_inliers], 'gx', alpha=.7, markersize = 1.0)#, markersize = 2.0)
                    #axes[0].plot(x_dis_struct[0,just_below], x_dis_struct[1,just_below], 'gx', alpha=.9, markersize = 1.0)#, markersize = 2.0)
       
       
                    axes[0].plot(x_dis[0], x_dis[1], 'b')#, markersize = 2.0)
                    axes[0].plot(x_dis[0], x_dis[1], 'co', markersize = 2.0)
                    
                    axes[0].plot(grid_dis[0], grid_dis[1], 'yo', markersize = 4.0)
                    

                    plt.show()


















                traj_start += __trajStride

            # if a trajectory will be longer than the end of the data, we should

    
    trajectory_test = open(__data_target / Path('traj_prediction_test.txt'), 'w')
    trajectory_test.write('nFrames: {num}\n'.format(num=len(__trajectory_buffer)))
    
    imlist_test = open(__data_target / Path('im_list_test.list'), 'w')


    frame_id = 0

    for key in __trajectory_buffer.keys():

        traj = __trajectory_buffer[key]['XYZ']
        world_up = __trajectory_buffer[key]['up']

        
        trajectory_test.write('{id} {upx} {upy} {upz} {tlen} '.format(id=frame_id, upx=world_up[0], upy=world_up[1], upz=world_up[2], tlen=traj.shape[1]))

        for i in range(traj.shape[1]):
            trajectory_test.write('{t} {x} {y} {z} {d1} {d2}'.format(t=i, x=traj[0,i], y=traj[1,i], z=traj[2,i], d1=0, d2=0))
        
        trajectory_test.write('\n')

        imlist_test.write('image{:07d}.jpg\n'.format(key))
        frame_id += 1
        
    trajectory_test.close()
    imlist_test.close()

    #print('done')

    ## Save first image in imlist
    ## Save trajectory in oriented space
    ## Save 

    ##reconstruction_folder = '\\reconstruction0001000'
    #reconstruction_folder = '\\reconstruction{:07d}'.format(frame_offset)
    ##calib = ReadCalibration('S:\\calib_fisheye.txt')
    #calib = ReadCalibration(file_path+'\\image\\calib_fisheye.txt')
    #print(calib['K'])
    #print(calib['omega'])

    #frames = ReadCameraFile(file_path+reconstruction_folder+'\\camera.txt', frame_offset)

    ##file_path = 'S:\\ego4d_benchmark\\marcia\\10800524\\REC00003'
    ##frame_offset = 1000
    ##start_frame = 1100
    ##downscaler = 2.7
    ##rightscaler = 0.2
    ##forwardscaler = -0.1
    
    ##n_frames = 100


    ##file_path = 'S:\\ego4d_benchmark\\caleb\\10300523\\REC00005'
    ##frame_offset = 6200
    ##start_frame = 6300
    ##downscaler = .6
    ##rightscaler = 0.2
    ##forwardscaler = -0.1
    ##n_frames = 100

    #file_path = 'S:\\ego4d_benchmark\\meghan\\11500510\\REC00002'
    ##frame_offset = 22600
    ##start_frame = 22609
    ##downscaler = 1.1
    ##rightscaler = -.35
    ##forwardscaler = 0.0
    ##n_frames = 100
    #frame_offset = 20200
    #start_frame = 20200
    #downscaler = 5.0
    #rightscaler = 0.0 #-.35
    #forwardscaler = 0.1
    
    #n_frames = 200

    #frame_offset = 8200
    #start_frame = 8200
    #downscaler = 0.5
    #rightscaler = -.1
    #forwardscaler = 0.1
    
    #n_frames = 200




    ##file_path = 'S:\\ego4d_benchmark\\anna\\10500527\\REC00002'
    ##frame_offset = 200
    ##start_frame = 200
    ##downscaler = 11.0
    ##rightscaler = -0.1
    ##forwardscaler = 0.5
    
    ##n_frames = 100

    ##frame_offset = 1000
    ##start_frame = 1070
    ##downscaler = 14.0
    ##rightscaler = 1
    ##forwardscaler = 1.2
    
    ##n_frames = 100


    
    ##file_path = 'S:\\ego4d_benchmark\\anna\\10600609\\REC00003'
    ##frame_offset = 2400
    ##start_frame = 2460
    ##downscaler = 2.3
    ##rightscaler = 0.4
    ##forwardscaler = 0.4
    
    ##n_frames = 180



    ##reconstruction_folder = '\\reconstruction0001000'
    #reconstruction_folder = '\\reconstruction{:07d}'.format(frame_offset)
    ##calib = ReadCalibration('S:\\calib_fisheye.txt')
    #calib = ReadCalibration(file_path+'\\image\\calib_fisheye.txt')
    #print(calib['K'])
    #print(calib['omega'])

    #frames = ReadCameraFile(file_path+reconstruction_folder+'\\camera.txt', frame_offset)
    ##print(frames['img_idx'][-1])
    ##print(frames['C'][-1])
    ##print(frames['R'][-1])

    

    #valid_frames = []
    #for key in range( start_frame, start_frame+n_frames ):
    #    if key in frames:
    #        valid_frames.append(key)
    #    else:
    #        print('Frame',key,'was not reconstructed.')

        
    #cameraCenter = frames[valid_frames[0]]['C']
    #cameraRotation = frames[valid_frames[0]]['R']


    #mean_down = np.zeros(3)
    #num_valid_frames = len(valid_frames)
    #for i in range(num_valid_frames):
    #    thisR = frames[valid_frames[i]]['R']
    #    thisdown = thisR[1]
    #    mean_down += thisdown
    #mean_down /= np.linalg.norm(mean_down)
        
    #if USE_MEAN_DOWN:
    #    world_down = mean_down
    #else:
    #    world_down = np.array([0,1,0])
        
    #right = cameraRotation[0]
    #down = cameraRotation[1]
    #forward = cameraRotation[2]

    ##points = np.zeros((len(valid_frames[1:]),3))
    ##points[:,0] = forward[0]-down[0]*.5 #0#np.linspace(-5,5,len(points))
    ##points[:,1] = forward[1]-down[1]*.5 #0 #np.linspace(0,5,len(points))
    ##points[:,2] = forward[2]-down[2]*.5 #1 #np.linspace(1,6,len(points))

    ##for i in range(len(valid_frames[1:])):
    ##    delta = 1.0/len(valid_frames[1:])
    ##    points[i,0] += right[0] * (i*delta) - (1-i*delta) * right[0]
    ##    points[i,1] += right[1] * (i*delta) - (1-i*delta) * right[1]
    ##    points[i,2] += right[1] * (i*delta) - (1-i*delta) * right[2]


    ###points *= 1
    ###points = (forward + cameraCenter)[None] * points
    ###points += np.ones(points.shape)*cameraCenter[None]
    ##points[1] = .7


    ##count = 0
    ##for key in valid_frames[1:]:
    ##    print('Frame:',key)
    ##    points[count] = frames[key]['C']
    ##    count += 1

    ##points +=  np.ones(points.shape)*downscaler*down[None] + np.ones(points.shape)*rightscaler*right[None] + np.ones(points.shape)*forwardscaler*forward[None]


    ##X = points.T
    ##bigC = np.ones((3,X.shape[1]))
    ##bigC = cameraCenter[:,None] * bigC
    ##X_ = X-bigC
    ##camdot = forward @ X_
    ##X_ = X_[:,camdot>=0] # only in front of camera
    ##augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
    ##P = calib['K'] @ cameraRotation #@ augC
    ###augX_ = np.concatenate( ( X_, np.ones((1,X_.shape[1])) ), axis=0)
    ##x = P @ X_
    ##x /= x[2]
    ##x = x[:2]
    ##x_dis = Distort(x,calib['omega'],calib['K'])


    #point_cloud = ReadPointCloud(file_path + reconstruction_folder+'\\structure.txt')

    ##X = point_cloud['XYZ']
    ##bigC = np.ones((3,X.shape[1]))
    ##bigC = cameraCenter[:,None] * bigC
    ##X_ = X-bigC
    ##camdot = forward @ X_
    ##X_ = X_[:,camdot>=0] # only in front of camera
    ##Xstar = X[:,camdot>=0]
    ##augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
    ##P = calib['K'] @ cameraRotation #@ augC
    ###augX_ = np.concatenate( ( X_, np.ones((1,X_.shape[1])) ), axis=0)
    ##x = P @ X_
    ##x /= x[2]
    ##x = x[:2]
    ##x_dis_struct = Distort(x,calib['omega'],calib['K'])

        


    #X = point_cloud['XYZ']
    #bigC = np.ones((3,X.shape[1]))
    #bigC = cameraCenter[:,None] * bigC
    #X_ = X-bigC
    #camdot = forward @ X_
    #camdot_below = world_down @ X_





        
    #infront_logical = np.zeros(X_.shape[1], dtype=bool)
    #infront_logical[camdot>=0]=True
        
    #below_logical = np.zeros(X_.shape[1], dtype=bool)
    #below_logical[camdot_below>=0]=True
        
    ##below_logical = np.zeros(X_.shape[1], dtype=bool)
    ##below_logical[X_[1] >= cameraCenter[1]]=True
        
    ##just_else = np.logical_and(np.invert(inliers_logical), infront_logical)
    #just_below = np.logical_and(below_logical, infront_logical)


    #X_ = X_[:,just_below]

    #PlaneSeg = Plane()
    #best_eq, best_inliers = PlaneSeg.fit(X_.T, 0.5,20,1000,-world_down, np.pi/12.0)

    #inliers_logical = np.zeros(X_.shape[1], dtype=bool)
    #inliers_logical[best_inliers]=True

    ##just_plane = np.logical_and(inliers_logical, just_below)


    ##X_ = X_[:,camdot>=0] # only in front of camera
    ##Xstar = X[:,camdot>=0]
    #augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
    #P = calib['K'] @ cameraRotation #@ augC
    ##augX_ = np.concatenate( ( X_, np.ones((1,X_.shape[1])) ), axis=0)

    ##x = P @ X_
    #X_ = cameraRotation @ X_ # Align into camera space where 






    #x = calib['K'] @ X_
        
    #x /= x[2]
    #x = x[:2]
    #x_dis_struct = Distort(x,calib['omega'],calib['K'])


    #__AVERAGE_HUMAN_HEIGHT = 1.71 # meters
    ## Modify trajectory points

    #plane_normal = np.array(best_eq[:3])
    ## since plane is calculated in camera-centered space, we only need d/sqrt(a,b,c)
    #current_distance_to_ground = np.abs(best_eq[3] / np.linalg.norm(plane_normal))
    #corrective_scalar = __AVERAGE_HUMAN_HEIGHT / current_distance_to_ground
    #plane_normal_with_metric = plane_normal * __AVERAGE_HUMAN_HEIGHT / np.linalg.norm(plane_normal)
    #check = np.linalg.norm(plane_normal_with_metric)


        
    #points = np.zeros((len(valid_frames[1:]),3))
    #count = 0
    #for key in valid_frames[1:]:
    #    print('Frame:',key)
    #    points[count] = frames[key]['C']
    #    count += 1

    ##points +=  np.ones(points.shape)*downscaler*down[None] + np.ones(points.shape)*rightscaler*right[None] + np.ones(points.shape)*forwardscaler*forward[None]



    #X = points.T
    #bigC = np.ones((3,X.shape[1]))
    #bigC = cameraCenter[:,None] * bigC
    #X_ = X-bigC


        
    ## RESCALE TRAJECTORY HERE
    #X_ *= corrective_scalar
    #X_ = X_ - plane_normal_with_metric[:,None]
        
    #camdot = forward @ X_
    #X_ = X_[:,camdot>=0] # only in front of camera

    #augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
    #P = calib['K'] @ cameraRotation #@ augC
    ##augX_ = np.concatenate( ( X_, np.ones((1,X_.shape[1])) ), axis=0)
    #x = P @ X_
    #x /= x[2]
    #x = x[:2]
    #x_dis = Distort(x,calib['omega'],calib['K'])








    #print('projection stuff')
    #img_dir = file_path+'\\image\\image{:07d}.jpg'.format(start_frame)
    #img = cv2.imread(img_dir)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.0
    ##img[:,:] = 1


    #fig, ax = plt.subplots(1,1)
    #axes = [ax]
    #axes[0].imshow(img)
    ##axes[0].plot(0,0, 'ro', markersize = 10.0)
    ##axes[0].plot(calib['K'][0,2],calib['K'][1,2], 'ro', markersize = 10.0)
    #pix_d = np.zeros((len(valid_frames[1:]),3))
    #start_key = valid_frames[0]
    ##count = 0
    ##for key in valid_frames[1:]:
    ##    camera_center = frames[key]['C']
    ##    camera_center[1] = .7
    ##    pix_d[count] = ProjectWithDistortion(calib['omega'],calib['K'],frames[start_key]['R'],frames[start_key]['C'],camera_center)
    ##   # if frames.has_key()

    ##    #pix_d[count] = ProjectWithDistortion(calib['omega'],calib['K'],np.eye(3),np.zeros(3),points[count])
    ##    count += 1
    ##    #pix_u = calib['K'] @ points[i]
    ##    #pix_d = cv.fisheye


        
    ##in_front = np.where(pix_d[:,2] > 0)
    ##pix_front = pix_d[in_front]
    ##ys = pix_front[:,1]
    ##xs = pix_front[:,0]
    ##axes[0].plot(x_dis_struct[0], x_dis_struct[1], 'rx', alpha=.5, markersize = 0.5)#, markersize = 2.0)

    #axes[0].plot(x_dis_struct[0], x_dis_struct[1], 'rx', alpha=.5, markersize = 0.5)#, markersize = 2.0)
    ##axes[0].plot(x_dis_struct[0,just_else], x_dis_struct[1,just_else], 'rx', alpha=.5, markersize = 0.5)#, markersize = 2.0)
    #axes[0].plot(x_dis_struct[0,best_inliers], x_dis_struct[1,best_inliers], 'gx', alpha=.9, markersize = 2.0)#, markersize = 2.0)
    ##axes[0].plot(x_dis_struct[0,just_below], x_dis_struct[1,just_below], 'gx', alpha=.9, markersize = 1.0)#, markersize = 2.0)
       
       
    #axes[0].plot(x_dis[0], x_dis[1], 'b')#, markersize = 2.0)
    #axes[0].plot(x_dis[0], x_dis[1], 'co', markersize = 2.0)

    #plt.show()
