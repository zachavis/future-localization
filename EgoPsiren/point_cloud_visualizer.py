import numpy as np
import matplotlib.pyplot as plt
import cv2 
import random
import copy 

#import pyransac3d as pyrsc

#point_cloud = np.loadtxt('S:\\structure.txt')
#print(point_cloud[:10])


data_file = 'S:\\structure_0001800_00.txt' #'S:\\fut_loc\\20150401_walk_00\\traj_prediction.txt'







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





if True:
    
    USE_MEAN_DOWN = True
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
    #rightscaler = 0.2
    #forwardscaler = -0.1
    #n_frames = 100

    file_path = 'S:\\ego4d_benchmark\\meghan\\11500510\\REC00002'
    #frame_offset = 22600
    #start_frame = 22609
    #downscaler = 1.1
    #rightscaler = -.35
    #forwardscaler = 0.0
    #n_frames = 100
    frame_offset = 20200
    start_frame = 20200
    downscaler = 5.0
    rightscaler = 0.0 #-.35
    forwardscaler = 0.1
    
    n_frames = 200

    frame_offset = 8200
    start_frame = 8200
    downscaler = 0.5
    rightscaler = -.1
    forwardscaler = 0.1
    
    n_frames = 200




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


    
    #file_path = 'S:\\ego4d_benchmark\\anna\\10600609\\REC00003'
    #frame_offset = 2400
    #start_frame = 2460
    #downscaler = 2.3
    #rightscaler = 0.4
    #forwardscaler = 0.4
    
    #n_frames = 180



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

    if False:
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

        PlaneSeg = pyrsc.Plane()
        best_eq, best_inliers = PlaneSeg.fit(point_cloud['XYZ'].T, 0.01)


        
        ax.plot(point_cloud['XYZ'][0],point_cloud['XYZ'][1],point_cloud['XYZ'][2],'g.',markersize=1)
        ax.plot(point_cloud['XYZ'][0,best_inliers],point_cloud['XYZ'][1,best_inliers],point_cloud['XYZ'][2,best_inliers],'c.',markersize=2)


        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()
    else:
        
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
        else:
            world_down = np.array([0,1,0])
        
        right = cameraRotation[0]
        down = cameraRotation[1]
        forward = cameraRotation[2]

        #points = np.zeros((len(valid_frames[1:]),3))
        #points[:,0] = forward[0]-down[0]*.5 #0#np.linspace(-5,5,len(points))
        #points[:,1] = forward[1]-down[1]*.5 #0 #np.linspace(0,5,len(points))
        #points[:,2] = forward[2]-down[2]*.5 #1 #np.linspace(1,6,len(points))

        #for i in range(len(valid_frames[1:])):
        #    delta = 1.0/len(valid_frames[1:])
        #    points[i,0] += right[0] * (i*delta) - (1-i*delta) * right[0]
        #    points[i,1] += right[1] * (i*delta) - (1-i*delta) * right[1]
        #    points[i,2] += right[1] * (i*delta) - (1-i*delta) * right[2]


        ##points *= 1
        ##points = (forward + cameraCenter)[None] * points
        ##points += np.ones(points.shape)*cameraCenter[None]
        #points[1] = .7


        #count = 0
        #for key in valid_frames[1:]:
        #    print('Frame:',key)
        #    points[count] = frames[key]['C']
        #    count += 1

        #points +=  np.ones(points.shape)*downscaler*down[None] + np.ones(points.shape)*rightscaler*right[None] + np.ones(points.shape)*forwardscaler*forward[None]


        #X = points.T
        #bigC = np.ones((3,X.shape[1]))
        #bigC = cameraCenter[:,None] * bigC
        #X_ = X-bigC
        #camdot = forward @ X_
        #X_ = X_[:,camdot>=0] # only in front of camera
        #augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
        #P = calib['K'] @ cameraRotation #@ augC
        ##augX_ = np.concatenate( ( X_, np.ones((1,X_.shape[1])) ), axis=0)
        #x = P @ X_
        #x /= x[2]
        #x = x[:2]
        #x_dis = Distort(x,calib['omega'],calib['K'])


        point_cloud = ReadPointCloud(file_path + reconstruction_folder+'\\structure.txt')

        #X = point_cloud['XYZ']
        #bigC = np.ones((3,X.shape[1]))
        #bigC = cameraCenter[:,None] * bigC
        #X_ = X-bigC
        #camdot = forward @ X_
        #X_ = X_[:,camdot>=0] # only in front of camera
        #Xstar = X[:,camdot>=0]
        #augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
        #P = calib['K'] @ cameraRotation #@ augC
        ##augX_ = np.concatenate( ( X_, np.ones((1,X_.shape[1])) ), axis=0)
        #x = P @ X_
        #x /= x[2]
        #x = x[:2]
        #x_dis_struct = Distort(x,calib['omega'],calib['K'])

        


        X = point_cloud['XYZ']
        bigC = np.ones((3,X.shape[1]))
        bigC = cameraCenter[:,None] * bigC
        X_ = X-bigC
        camdot = forward @ X_
        camdot_below = world_down @ X_





        
        infront_logical = np.zeros(X_.shape[1], dtype=bool)
        infront_logical[camdot>=0]=True
        
        below_logical = np.zeros(X_.shape[1], dtype=bool)
        below_logical[camdot_below>=0]=True
        
        #below_logical = np.zeros(X_.shape[1], dtype=bool)
        #below_logical[X_[1] >= cameraCenter[1]]=True
        
        #just_else = np.logical_and(np.invert(inliers_logical), infront_logical)
        just_below = np.logical_and(below_logical, infront_logical)


        X_ = X_[:,just_below]

        PlaneSeg = Plane()
        best_eq, best_inliers = PlaneSeg.fit(X_.T, 0.5,20,1000,-world_down, np.pi/12.0)

        inliers_logical = np.zeros(X_.shape[1], dtype=bool)
        inliers_logical[best_inliers]=True

        #just_plane = np.logical_and(inliers_logical, just_below)


        #X_ = X_[:,camdot>=0] # only in front of camera
        #Xstar = X[:,camdot>=0]
        augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
        P = calib['K'] @ cameraRotation #@ augC
        #augX_ = np.concatenate( ( X_, np.ones((1,X_.shape[1])) ), axis=0)

        #x = P @ X_
        X_ = cameraRotation @ X_ # Align into camera space where 






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
            print('Frame:',key)
            points[count] = frames[key]['C']
            count += 1

        #points +=  np.ones(points.shape)*downscaler*down[None] + np.ones(points.shape)*rightscaler*right[None] + np.ones(points.shape)*forwardscaler*forward[None]



        X = points.T
        bigC = np.ones((3,X.shape[1]))
        bigC = cameraCenter[:,None] * bigC
        X_ = X-bigC


        
        # RESCALE TRAJECTORY HERE
        X_ *= corrective_scalar
        X_ = X_ - plane_normal_with_metric[:,None]
        
        camdot = forward @ X_
        X_ = X_[:,camdot>=0] # only in front of camera

        augC = np.concatenate((np.eye(3),-cameraCenter[:,None]),axis=1)
        P = calib['K'] @ cameraRotation #@ augC
        #augX_ = np.concatenate( ( X_, np.ones((1,X_.shape[1])) ), axis=0)
        x = P @ X_
        x /= x[2]
        x = x[:2]
        x_dis = Distort(x,calib['omega'],calib['K'])








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
        #axes[0].plot(x_dis_struct[0], x_dis_struct[1], 'rx', alpha=.5, markersize = 0.5)#, markersize = 2.0)

        axes[0].plot(x_dis_struct[0], x_dis_struct[1], 'rx', alpha=.5, markersize = 0.5)#, markersize = 2.0)
        #axes[0].plot(x_dis_struct[0,just_else], x_dis_struct[1,just_else], 'rx', alpha=.5, markersize = 0.5)#, markersize = 2.0)
        axes[0].plot(x_dis_struct[0,best_inliers], x_dis_struct[1,best_inliers], 'gx', alpha=.9, markersize = 2.0)#, markersize = 2.0)
        #axes[0].plot(x_dis_struct[0,just_below], x_dis_struct[1,just_below], 'gx', alpha=.9, markersize = 1.0)#, markersize = 2.0)
       
       
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

    print("dummy break")






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