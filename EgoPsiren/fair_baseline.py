import numpy as np
import matplotlib.pyplot as plt
import cv2 

# Need: PCL library

# For passing arguments across systems
import sys
import getopt
from pathlib import Path


#point_cloud = np.loadtxt('S:\\structure.txt')
#print(point_cloud[:10])



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



if __name__ == "__main__":
    data_file = 'S:\\structure_0001800_00.txt' #'S:\\fut_loc\\20150401_walk_00\\traj_prediction.txt'

    necessary_args = 0
    verbose_flag = False
    data_root = Path('S:/fut_loc/')
    
    file_path = 'S:\\ego4d_benchmark\\anna\\10600609\\REC00003'



    inputfile = ''
    outputfile = ''
    overfitoutputfile = ''

     # NEEDS INPUT VARS
    __trajLength = 100 # Maximum number of frames in a trajectory  #-l --length
    __trajStride = 20 # Frames to skip while generating each trajectory  #-s --stride
    __data_source = Path('S:/ego4d_benchmark') #-d? -i data
    __data_target = Path('S:/ego4d_benchmark') #-o
    __data_images = Path('S:/ego4d_benchmark') #-i --images

    
    #train_test_directories = ['dummytrain','dummytest']

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hvd:i:o:l:s:",["help","verbose","data=","images=","output=","length=","stride="])
    except getopt.GetoptError:
        print('Arguments are malformed. TODO: put useful help here.')#'test.py -i <inputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-d", "--data"):
            __data_source = Path(arg)
            if not data_root.exists():
                print("Data path does not exist.")
                sys.exit(3)
            necessary_args += 1

        if opt in ("-i", "--images"):
            __data_images = Path(arg)
            if not data_root.exists():
                print("Images path does not exist.")
                sys.exit(3)
            necessary_args += 1

        if opt in ("-o", "--output"):
            __data_target = Path(arg)
            if not __data_target.exists():
                __data_target.mkdir()
            necessary_args += 1

        if opt in ("-l", "--lenth"):
            __trajLength = arg
            necessary_args += 1

        if opt in ("-s", "--stride"):
            __trajStride = arg
            necessary_args += 1

        if opt in ("-v", "--verbose"):
            verbose_flag = True

        if opt in ("-h", "--help"):
            PrintHelp()
            sys.exit(-1)

    if necessary_args < 2:
        print('Not enough args')
        PrintHelp()
        sys.exit(3)



        frame_offset = 2400 # I don't think we need this, each individual folder must be processed
        start_frame = 2460
        downscaler = 2.3
        rightscaler = 0.4
        forwardscaler = 0.4
    
        n_frames = 180



        #reconstruction_folder = '\\reconstruction0001000'
        reconstruction_folder = '\\reconstruction{:07d}'.format(frame_offset)
        #calib = ReadCalibration('S:\\calib_fisheye.txt')
        calib = ReadCalibration(file_path+'\\image\\calib_fisheye.txt')
        print(calib['K'])
        print(calib['omega'])

        frames = ReadCameraFile(file_path+reconstruction_folder+'\\camera.txt', frame_offset)

    if True:

        #file_path = 'S:\\ego4d_benchmark\\marcia\\10800524\\REC00003'
        #frame_offset = 1000
        #start_frame = 1100
        #downscaler = 2.7
        #rightscaler = 0.2
        #forwardscaler = -0.1
    
        #n_frames = 100


        #file_path = 'S:\\ego4d_benchmark\\caleb\\10300523\\REC00005'
        #frame_offset = 6200
        #start_frame = 6300
        #downscaler = .6

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




        #file_path = 'S:\\ego4d_benchmark\\anna\\10500527\\REC00002'
        #frame_offset = 200
        #start_frame = 200
        #downscaler = 11.0
        #rightscaler = -0.1
        #forwardscaler = 0.5
    
        #n_frames = 100

        #frame_offset = 1000
        #start_frame = 1070
        #downscaler = 14.0
        #rightscaler = 1
        #forwardscaler = 1.2
    
        #n_frames = 100
    
        file_path = 'S:\\ego4d_benchmark\\anna\\10600609\\REC00003'
        frame_offset = 2400
        start_frame = 2460
        downscaler = 2.3
        rightscaler = 0.4
        forwardscaler = 0.4
    
        n_frames = 180



        #reconstruction_folder = '\\reconstruction0001000'
        reconstruction_folder = '\\reconstruction{:07d}'.format(frame_offset)
        #calib = ReadCalibration('S:\\calib_fisheye.txt')
        calib = ReadCalibration(file_path+'\\image\\calib_fisheye.txt')
        print(calib['K'])
        print(calib['omega'])

        frames = ReadCameraFile(file_path+reconstruction_folder+'\\camera.txt', frame_offset)
        #print(frames['img_idx'][-1])
        #print(frames['C'][-1])
        #print(frames['R'][-1])

    

        valid_frames = []
        for key in range( start_frame, start_frame+n_frames ):
            if key in frames:
                valid_frames.append(key)
            else:
                print('Frame',key,'was not reconstructed.')

        if True:
            ax = plt.axes(projection='3d')
            #ax.scatter(frames['C'].T[0],frames['C'].T[1],frames['C'].T[2], s=1)
            keys = np.sort(np.array(list(frames.keys())))

        
            start = frames[valid_frames[0]]['C']
            ax.plot(start[0],start[1],start[2],'bo',markerSize=10)

            for key in valid_frames:#len(frames['C'])):
                x_delta = np.zeros(2)
                y_delta = np.zeros(2)
                z_delta = np.zeros(2)
            
                center = frames[key]['C']
                x_delta[0] = center[0]
                y_delta[0] = center[1]
                z_delta[0] = center[2]
            
                scale = .02

                R = frames[key]['R']#.T
                #print(R)
                axis = R[0] * scale
                #axis2 = R[0]
                #axis *= scale
                #axis2 *= scale
                x_delta[1] = center[0] + axis[0]
                y_delta[1] = center[1] + axis[1]
                z_delta[1] = center[2] + axis[2]
                ax.plot(x_delta,y_delta,z_delta,'r',linewidth=1)

            for key in valid_frames:
                x_delta = np.zeros(2)
                y_delta = np.zeros(2)
                z_delta = np.zeros(2)
            
                center = frames[key]['C']
                x_delta[0] = center[0]
                y_delta[0] = center[1]
                z_delta[0] = center[2]
            
                scale = .02


            
                R = frames[key]['R']#.T
                axis = R[1] * scale
                x_delta[1] = center[0] + axis[0]
                y_delta[1] = center[1] + axis[1]
                z_delta[1] = center[2] + axis[2]
                ax.plot(x_delta,y_delta,z_delta,'g',linewidth=1)
            
            for key in valid_frames:
                x_delta = np.zeros(2)
                y_delta = np.zeros(2)
                z_delta = np.zeros(2)
            
                center = frames[key]['C']
                x_delta[0] = center[0]
                y_delta[0] = center[1]
                z_delta[0] = center[2]
            
                scale = .02
            
                R = frames[key]['R']#.T
                axis = R[2] * scale
                x_delta[1] = center[0] + axis[0]
                y_delta[1] = center[1] + axis[1]
                z_delta[1] = center[2] + axis[2]
                ax.plot(x_delta,y_delta,z_delta,'b',linewidth=1)

            
            x = np.zeros(n_frames)
            y = np.zeros(n_frames)
            z = np.zeros(n_frames)
            count = 0
            for key in valid_frames:
                C = frames[key]['C']
                x[count] = C[0]
                y[count] = C[1]
                z[count] = C[2]
                count += 1

            #q = frames['C'][:n_frames]
            #x = frames['C'][:n_frames,0]
            #y = frames['C'][:n_frames,1]
            #z = frames['C'][:n_frames,2]
            max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

            mean_x = x.mean()
            mean_y = y.mean()
            mean_z = z.mean()
            ax.set_xlim(mean_x - max_range, mean_x + max_range)
            ax.set_ylim(mean_y - max_range, mean_y + max_range)
            ax.set_zlim(mean_z - max_range, mean_z + max_range)

            #ax.set_xlim(-1, 1)
            #ax.set_ylim(-1, 1)
            #ax.set_zlim(-1, 1)

        

        
            point_cloud = ReadPointCloud(file_path + reconstruction_folder+'\\structure.txt')


            ax.plot(point_cloud['XYZ'][0],point_cloud['XYZ'][1],point_cloud['XYZ'][2],'g.',markersize=1)


            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            plt.show()
        else:
        
            cameraCenter = frames[valid_frames[0]]['C']
            cameraRotation = frames[valid_frames[0]]['R']
        
            right = cameraRotation[0]
            down = cameraRotation[1]
            forward = cameraRotation[2]

            points = np.zeros((len(valid_frames[1:]),3))
            points[:,0] = forward[0]-down[0]*.5 #0#np.linspace(-5,5,len(points))
            points[:,1] = forward[1]-down[1]*.5 #0 #np.linspace(0,5,len(points))
            points[:,2] = forward[2]-down[2]*.5 #1 #np.linspace(1,6,len(points))

            for i in range(len(valid_frames[1:])):
                delta = 1.0/len(valid_frames[1:])
                points[i,0] += right[0] * (i*delta) - (1-i*delta) * right[0]
                points[i,1] += right[1] * (i*delta) - (1-i*delta) * right[1]
                points[i,2] += right[1] * (i*delta) - (1-i*delta) * right[2]


            #points *= 1
            #points = (forward + cameraCenter)[None] * points
            #points += np.ones(points.shape)*cameraCenter[None]
            points[1] = .7


            count = 0
            for key in valid_frames[1:]:
                print('Frame:',key)
                points[count] = frames[key]['C']
                count += 1

            points +=  np.ones(points.shape)*downscaler*down[None] + np.ones(points.shape)*rightscaler*right[None] + np.ones(points.shape)*forwardscaler*forward[None]


            X = points.T
            bigC = np.ones((3,X.shape[1]))
            bigC = cameraCenter[:,None] * bigC
            X_ = X-bigC
            camdot = forward @ X_
            X_ = X_[:,camdot>=0] # only in front of camera
            augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
            P = calib['K'] @ cameraRotation #@ augC
            #augX_ = np.concatenate( ( X_, np.ones((1,X_.shape[1])) ), axis=0)
            x = P @ X_
            x /= x[2]
            x = x[:2]
            x_dis = Distort(x,calib['omega'],calib['K'])


            point_cloud = ReadPointCloud(file_path + reconstruction_folder+'\\structure.txt')

            X = point_cloud['XYZ']
            bigC = np.ones((3,X.shape[1]))
            bigC = cameraCenter[:,None] * bigC
            X_ = X-bigC
            camdot = forward @ X_
            X_ = X_[:,camdot>=0] # only in front of camera
            augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
            P = calib['K'] @ cameraRotation #@ augC
            #augX_ = np.concatenate( ( X_, np.ones((1,X_.shape[1])) ), axis=0)
            x = P @ X_
            x /= x[2]
            x = x[:2]
            x_dis_struct = Distort(x,calib['omega'],calib['K'])









            print('projection stuff')
            img_dir = file_path+'\\image\\image{:07d}.jpg'.format(start_frame)
            img = cv2.imread(img_dir)
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
            axes[0].plot(x_dis_struct[0], x_dis_struct[1], 'rx', alpha=.5, markersize = 0.5)#, markersize = 2.0)
            axes[0].plot(x_dis[0], x_dis[1], 'b')#, markersize = 2.0)
            axes[0].plot(x_dis[0], x_dis[1], 'co', markersize = 2.0)

            plt.show()

    else:
        point_cloud = ReadPointCloud(data_file)
        med_x = np.median(point_cloud['XYZ'][0])
        med_y = np.median(point_cloud['XYZ'][1])
        med_z = np.median(point_cloud['XYZ'][2])

        clip_threshold = 10
        logical_clip = np.ones(len(point_cloud['XYZ'][0])).astype(np.bool)
        logical_clip[np.where(point_cloud['XYZ'][0] > med_x + clip_threshold)] = False
        logical_clip[np.where(point_cloud['XYZ'][0] < med_x - clip_threshold)] = False
        logical_clip[np.where(point_cloud['XYZ'][1] > med_y + clip_threshold)] = False
        logical_clip[np.where(point_cloud['XYZ'][1] < med_y - clip_threshold)] = False
        logical_clip[np.where(point_cloud['XYZ'][2] > med_z + clip_threshold)] = False
        logical_clip[np.where(point_cloud['XYZ'][2] < med_z - clip_threshold)] = False
        sum = np.sum(logical_clip.astype(np.int))
        print(sum, 'points after clipping')


        ax = plt.axes(projection='3d')
        ax.scatter(point_cloud['XYZ'][0,logical_clip], point_cloud['XYZ'][1,logical_clip], point_cloud['XYZ'][2,logical_clip], c = point_cloud['RGB'][:,logical_clip].T/255, s=1)
        plt.show()







  #calib = ReadCalibration('S:\\calib_fisheye.txt')
  #  print(calib['K'])
  #  print(calib['omega'])

  #  frame_offset = 200
  #  frames = ReadCameraFile('S:\\camera0000200.txt', frame_offset)
  #  #print(frames['img_idx'][-1])
  #  #print(frames['C'][-1])
  #  #print(frames['R'][-1])

  #  if False:
  #      ax = plt.axes(projection='3d')
  #      #ax.scatter(frames['C'].T[0],frames['C'].T[1],frames['C'].T[2], s=1)
  #      n_frames = 20#len(frames['C'])):
  #      keys = frames.keys()

  #      for key in keys[:n_frames]:#len(frames['C'])):
  #          x_delta = np.zeros(2)
  #          y_delta = np.zeros(2)
  #          z_delta = np.zeros(2)
            
  #          center = frames['C'][i]
  #          x_delta[0] = center[0]
  #          y_delta[0] = center[1]
  #          z_delta[0] = center[2]
            
  #          scale = .02

  #          R = frames['R'][i]#.T
  #          axis = R[0] * scale
  #          #axis2 = R[0]
  #          #axis *= scale
  #          #axis2 *= scale
  #          x_delta[1] = center[0] + axis[0]
  #          y_delta[1] = center[1] + axis[1]
  #          z_delta[1] = center[2] + axis[2]
  #          ax.plot(x_delta,y_delta,z_delta,'r')

  #      for i in keys[:n_frames]:
  #          x_delta = np.zeros(2)
  #          y_delta = np.zeros(2)
  #          z_delta = np.zeros(2)
            
  #          center = frames['C'][i]
  #          x_delta[0] = center[0]
  #          y_delta[0] = center[1]
  #          z_delta[0] = center[2]
            
  #          scale = .02


            
  #          R = frames['R'][i]#.T
  #          axis = R[1] * scale
  #          x_delta[1] = center[0] + axis[0]
  #          y_delta[1] = center[1] + axis[1]
  #          z_delta[1] = center[2] + axis[2]
  #          ax.plot(x_delta,y_delta,z_delta,'g')

  #      for i in keys[:n_frames]:
  #          x_delta = np.zeros(2)
  #          y_delta = np.zeros(2)
  #          z_delta = np.zeros(2)
            
  #          center = frames['C'][i]
  #          x_delta[0] = center[0]
  #          y_delta[0] = center[1]
  #          z_delta[0] = center[2]
            
  #          scale = .02
            
  #          R = frames['R'][i]#.T
  #          axis = R[2] * scale
  #          x_delta[1] = center[0] + axis[0]
  #          y_delta[1] = center[1] + axis[1]
  #          z_delta[1] = center[2] + axis[2]
  #          ax.plot(x_delta,y_delta,z_delta,'b')

            
  #      q = frames['C'][:n_frames]
  #      x = frames['C'][:n_frames,0]
  #      y = frames['C'][:n_frames,1]
  #      z = frames['C'][:n_frames,2]
  #      max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

  #      mean_x = x.mean()
  #      mean_y = y.mean()
  #      mean_z = z.mean()
  #      ax.set_xlim(mean_x - max_range, mean_x + max_range)
  #      ax.set_ylim(mean_y - max_range, mean_y + max_range)
  #      ax.set_zlim(mean_z - max_range, mean_z + max_range)

  #      #ax.set_xlim(-1, 1)
  #      #ax.set_ylim(-1, 1)
  #      #ax.set_zlim(-1, 1)

  #      ax.set_xlabel('X axis')
  #      ax.set_ylabel('Y axis')
  #      ax.set_zlabel('Z axis')
  #      plt.show()