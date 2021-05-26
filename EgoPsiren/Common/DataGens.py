import torch
import numpy as np
import math


def normDiff(a,b):
  dx = b[0] - a[0]
  dy = b[1] - a[1]
  dist = math.sqrt(dx*dx + dy*dy)
  if dist > 0:
    dx /= dist
    dy /= dist
  return (dx,dy)

def avgDiff(a,b,v):
  dx = b[0] - a[0]
  dy = b[1] - a[1]
  dist = math.sqrt(dx*dx + dy*dy)
  if dist > 0:
    dx /= dist
    dy /= dist
  
  resultx = a[0] + dx * dist * v
  resulty = a[1] + dy * dist * v

  return (resultx,resulty)

def distDiff(a,b):
  dx = b[0] - a[0]
  dy = b[1] - a[1]
  dist = math.sqrt(dx*dx + dy*dy)

  return dist


##########################################
######  Geometric Processing 
##########################################

def DistanceFromLine (line, point):
    homogPoint = np.array([point[0], point[1], 1])
    proj = np.dot(line, homogPoint)
    lineNormal = np.linalg.norm(np.array([line[0],line[1]]))
    return abs(proj / lineNormal)

    
def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def skewPix(x):
    return np.array([[0, -1, x[1]],
                     [1, 0, -x[0]],
                     [-x[1], x[0], 0]])

# Choose N Unique values from the set x
# fixed contains the indices of values in x that you wish to for sure include
# fixed as index false means that you want to keep specific values in x (slower)
def chooseNUnique(x, n, fixed = [], fixedAsIndex = True):
       
    choiceSize = n
    p = np.zeros(choiceSize);
    seen = [-1 for i in range(choiceSize)];
    for i in range(len(fixed)):
        seen[i] = fixed[i]
        p[i] = x[seen[i]]
        if not fixedAsIndex:
            p[i] = fixed[i]
            seen[i] = np.where(x == fixed[i])[0][0] # get the first occurrence of fixed[i]

    numFeatures = len(x);
    
    count = len(fixed);
    while count < choiceSize:
        randid = np.random.randint(0,numFeatures);

        if any([seen[i] == randid for i in range(len(seen))]): # if the value has been seen and accepted
            continue;

        seen[count] = randid;
        p[count] = x[randid];
        count = count + 1;

    return p

def GetLine(ptA, ptB):
    return skewPix(ptA) @ ptB

def GetPoint(x):
    if len(x) >= 3:
      return x/x[2]
    return np.concatenate((x,np.ones(1)))



from scipy import stats

def GenGradientSamplePoints(pixel_shape, trajectories, obstacles = None, n_samples = 10000, pixel_tolerance = 1, print_status = True):
    #numsamples = 30000
    #pixel_tolerance = 1
    samplePoints = np.random.random((n_samples,2)).astype(np.float32)
    samplePoints[:,0] *= pixel_shape[0] # TODO: should actually use a bounds
    samplePoints[:,1] *= pixel_shape[1]

    sampleGradients = np.zeros((n_samples,2)).astype(np.float32)

    goodSamples = np.ones(n_samples).astype(np.bool)
    print(len(samplePoints))

    if print_status:
        print("Generating samples for gradient matching . . .")

    ids = trajectories.keys()

    i = 0
    for i in range(len(samplePoints)):
        #print(i)
        if i % 50 == 0 and print_status:
            print(i,'/',len(samplePoints))
        bad = False
        for id in ids:
            trajectory = np.array(trajectories[id])

            dist = 0
            #total_dist = 0
            #get total distance here
            pixDist = pixel_tolerance

            #finalVelocity = np.zeros(2)

            traj_points = trajectory[1:]
            traj_velocities = trajectory[1:] - trajectory[:-1]
            traj_velocities /= np.tile(np.linalg.norm(traj_velocities,axis=1), (2,1)).T

            dist_from_each_point = np.linalg.norm(traj_points - samplePoints[i], axis=1)/pixDist
            dist_from_each_point = np.vectorize(stats.norm.pdf)(dist_from_each_point) * math.sqrt(2*3.14159)*pixDist

            scaled_velocities = traj_velocities * np.tile(dist_from_each_point, (2,1)).T
            finalVelocity = np.sum(scaled_velocities,axis=0)
            sampleGradients[i] += finalVelocity


            # SLOW thing
            #finalVelocity = np.zeros(2)
            #for s in range(len(trajectory)-1):
            #    ptA = np.array(trajectory[s], dtype=np.float32)
            #    ptB = np.array(trajectory[s+1], dtype=np.float32)
            #    ptC = samplePoints[i]
            #    distA = np.linalg.norm(ptC-ptB)/pixDist

            #    velocity = np.array(normDiff(ptB,ptA))

            #    kernelDist = stats.norm.pdf(distA) * math.sqrt(2*3.14159)*pixDist
            #    velocity *= kernelDist
            #    finalVelocity += velocity

            #sampleGradients[i] += finalVelocity


            #ptA = np.array(trajectory[-1], dtype=np.float32)
            #ptC = samplePoints[i]

            # dist from end of trajectory
            #distA = np.linalg.norm(ptC-ptA)
            #if distA < pixel_tolerance:
            #    bad = True
            #    break

            # dist from beginning of trajectory
            #dist = 0
            #ptA = np.array(trajectory[0], dtype=np.float32)
            #distA = np.linalg.norm(ptC-ptA)
            #if distA < pixel_tolerance:
            #    bad = True
            #    break


            if bad:
                break

        #if obstacles is not None:
        #    for _, obs in obstacles.items():
        #        dist = 0
        #        for s in range(len(obs)-1):
        #            ptA = np.array(obs[s], dtype=np.float32)
        #            ptB = np.array(obs[s+1], dtype=np.float32)
        #            ptC = samplePoints[i]

        #            distA = np.linalg.norm(ptC-ptA)
        #            distB = np.linalg.norm(ptC-ptB)
        #            if distA < pixel_tolerance or distB < pixel_tolerance:
        #                bad = True
        #                break
                

        #            vecLine = ptB-ptA
        #            vecLineMag = np.linalg.norm(vecLine).astype(np.float32)
        #            #print(s,vecLine,vecLineMag)
        #            vecLine /= vecLineMag
        #            vecPoint = ptC-ptA
        #            vecPoint /= vecLineMag
        #            dotresult = vecLine @ vecPoint
        #            if (dotresult < 0 or dotresult > 1):
        #                #goodSamples[i] = False
        #                #i+=1
        #                continue

        #            # form line
        #            obsLine = GetLine(GetPoint(ptA),GetPoint(ptB)) #skewPix(ptA)@ptB
        #            dist = DistanceFromLine(obsLine,GetPoint(ptC))

        #            if dist < pixel_tolerance:
        #                bad = True
        #                break

        #        if bad:
        #            break

        if bad:
            goodSamples[i] = False
            #i+=1
            continue

    return samplePoints, sampleGradients









def GenLaplacianSamplePoints(pixel_shape, trajectories, obstacles = None, n_samples = 10000, pixel_tolerance = 1, print_status = True):
    #numsamples = 30000
    #pixel_tolerance = 1
    laplacianSamplePoints = np.random.random((n_samples,2)).astype(np.float32)
    laplacianSamplePoints[:,0] *= pixel_shape[0] # TODO: should actually use a bounds
    laplacianSamplePoints[:,1] *= pixel_shape[1]
    goodSamples = np.ones(n_samples).astype(np.bool)
    print(len(laplacianSamplePoints))

    if print_status:
        print("Generating samples for laplacian loss . . .")

    ids = trajectories.keys()

    i = 0
    for i in range(len(laplacianSamplePoints)):
        #print(i)
        if i % 50 == 0 and print_status:
            print(i,'/',len(laplacianSamplePoints))
        bad = False
        for id in ids:
            trajectory = trajectories[id]

            dist = 0
            ptA = np.array(trajectory[-1], dtype=np.float32)
            ptC = laplacianSamplePoints[i]

            # dist from end of trajectory
            #distA = np.linalg.norm(ptC-ptA)
            #if distA < pixel_tolerance:
            #    bad = True
            #    break

            # dist from beginning of trajectory
            #dist = 0
            #ptA = np.array(trajectory[0], dtype=np.float32)
            #distA = np.linalg.norm(ptC-ptA)
            #if distA < pixel_tolerance:
            #    bad = True
            #    break


            if bad:
                break

        if obstacles is not None:
            for _, obs in obstacles.items():
                dist = 0
                for s in range(len(obs)-1):
                    ptA = np.array(obs[s], dtype=np.float32)
                    ptB = np.array(obs[s+1], dtype=np.float32)
                    ptC = laplacianSamplePoints[i]

                    distA = np.linalg.norm(ptC-ptA)
                    distB = np.linalg.norm(ptC-ptB)
                    if distA < pixel_tolerance or distB < pixel_tolerance:
                        bad = True
                        break
                

                    vecLine = ptB-ptA
                    vecLineMag = np.linalg.norm(vecLine).astype(np.float32)
                    #print(s,vecLine,vecLineMag)
                    vecLine /= vecLineMag
                    vecPoint = ptC-ptA
                    vecPoint /= vecLineMag
                    dotresult = vecLine @ vecPoint
                    if (dotresult < 0 or dotresult > 1):
                        #goodSamples[i] = False
                        #i+=1
                        continue

                    # form line
                    obsLine = GetLine(GetPoint(ptA),GetPoint(ptB)) #skewPix(ptA)@ptB
                    dist = DistanceFromLine(obsLine,GetPoint(ptC))

                    if dist < pixel_tolerance:
                        bad = True
                        break

                if bad:
                    break

        if bad:
            goodSamples[i] = False
            #i+=1
            continue

    return laplacianSamplePoints[np.where(goodSamples==True)]


def GenLaplacianObstaclePoints(pixel_shape, trajectories, obstacles = None, n_samples = 1000, pixel_tolerance = 1, print_status = True):
    #numsamples = 30000
    #pixel_tolerance = 1
    laplacianSamplePoints = np.random.random((n_samples,2)).astype(np.float32)
    #laplacianSamplePoints[:,0] *= pixel_shape[0] # TODO: should actually use a bounds
    #laplacianSamplePoints[:,1] = 1 #0 # bottom
    laplacianSamplePoints[:,1] *= pixel_shape[1] # TODO: should actually use a bounds
    laplacianSamplePoints[:,0] = 25.5 + pixel_tolerance * 2 * (laplacianSamplePoints[:,0]-.5) #0 # bottom

    return laplacianSamplePoints












class TrajectoryDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, trajectories, recenteringFn, obstacles = None, multiplier = 100): #coords, dictionary, coord2keyFN, startPos, endPos, graph): #x_map, y_map, mask = None):
    'Initialization'
    
    self.trajectories = trajectories
    self.obstacles = obstacles
    self.obstacle_multiplier = multiplier
    self.buffer = .1
    self.recenteringFn = recenteringFn

    numItems = 0
    for _, trajectory in self.trajectories.items():
      #print('coord',coord)
      numItems += len(trajectory)-1
    if obstacles is not None:
      for _, obstacle in self.obstacles.items():
        #print('coord',coord)
        numItems += (len(obstacle)-1)*self.obstacle_multiplier*2

    self.totalpoints = numItems
    print("Total Midpoints:",self.totalpoints)


  def __len__(self):
    'Denotes the total number of samples'
    return 50

  def GetDirectionAlongTrajectory(self):

    input = np.zeros((1,self.totalpoints,2), dtype=np.float32 )
    output = np.zeros((1,self.totalpoints,2), dtype=np.float32 )
    #print('before',input.shape)

    current_midpoint = 0
    for _, trajectory in self.trajectories.items(): # for every starting position
     

      for i in range(len(trajectory)-1):
        interp = np.random.random(1)[0]
        pos = avgDiff(  trajectory[i],  trajectory[i+1],  interp)
        # print(trajectory[i])
        # print(trajectory[i+1])
        # print(pos)
        dir = normDiff( trajectory[i],  trajectory[i+1])
        input[0,current_midpoint,0] = pos[0]
        input[0,current_midpoint,1] = pos[1]
        output[0,current_midpoint,0] = -dir[0] # gradient points up, negative points in direction of movement for gradient descent
        output[0,current_midpoint,1] = -dir[1]
        current_midpoint += 1
        
     
    if self.obstacles is not None:
      for k in range(self.obstacle_multiplier):
        for _, obstacle in self.obstacles.items():
          for i in range(len(obstacle)-1):
                  interp = np.random.random(1)[0]
                  #extra_buff = .001
                  # Get buffer wrt line segment (try to prevent overlap of line normals)
                  diff = distDiff(obstacle[i], obstacle[i+1])
                  newbuff = self.buffer/diff
                  interp_with_buffer = interp * (1-2*(newbuff)) + newbuff

                  pos = avgDiff(  obstacle[i],  obstacle[i+1],  interp_with_buffer)
                  # print(trajectory[i])
                  # print(trajectory[i+1])
                  # print(pos)
                  dir = normDiff( obstacle[i],  obstacle[i+1])
                  normal = (-dir[1], dir[0])

                  # forward
                  input[0,current_midpoint,0] = pos[0] + normal[0]*self.buffer
                  input[0,current_midpoint,1] = pos[1] + normal[1]*self.buffer
                  output[0,current_midpoint,0] = normal[0]
                  output[0,current_midpoint,1] = normal[1]
                  current_midpoint += 1

                  # backward
                  input[0,current_midpoint,0] = pos[0] - normal[0]*self.buffer
                  input[0,current_midpoint,1] = pos[1] - normal[1]*self.buffer
                  output[0,current_midpoint,0] = -normal[0]
                  output[0,current_midpoint,1] = -normal[1]
                  current_midpoint += 1

    return {'coords':self.recenteringFn(input)}, output#self.recenteringFn(input), output #


  def __getitem__(self, index):
    'Generates one sample of data'
    return self.GetDirectionAlongTrajectory()




class FieldDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, points, gradients, numItems, recenteringFn, obstacle_points = None): #coords, dictionary, coord2keyFN, startPos, endPos, graph): #x_map, y_map, mask = None):
    'Initialization'
    
    self.points = points
    self.gradients = -gradients
    self.totalpoints = numItems
    self.recenteringFn = recenteringFn
    print("Total Points:",self.totalpoints)
    self.obstacle_points = None

    if obstacle_points is not None:
        self.obstacle_points = obstacle_points
        self.split = split

    

  def __len__(self):
    'Denotes the total number of samples'
    return 50


  def GetRandomSamples(self):

    #randoms = .01*(np.random.rand(1,self.totalpoints,2)*2.0-1.0)

    input = np.zeros((1,self.totalpoints,2), dtype=np.float32 )
    output = np.zeros((1,self.totalpoints,2), dtype=np.float32 )
   
    #split = int(self.totalpoints * 1)
    random_choice = np.random.choice(len(self.points), size=self.totalpoints, replace=False)
    input[:,:,:] = np.expand_dims(self.points[random_choice],0)#np.expand_dims(self.points[random_choice],0)
    #input += randoms
    output[:,:,:] = np.expand_dims(self.gradients[random_choice],0)#np.expand_dims(self.gradients[random_choice],0)
    #input = np.expand_dims(self.points,0)
    #output = np.expand_dims(self.gradients,0)
    
    return {'coords':self.recenteringFn(input)}, output #self.recenteringFn(input), output #


  def __getitem__(self, index):
    'Generates one sample of data'
    return self.GetRandomSamples()


class LaplacianDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, points, numItems, recenteringFn, obstacle_points = None, split = 1.0): #coords, dictionary, coord2keyFN, startPos, endPos, graph): #x_map, y_map, mask = None):
    'Initialization'
    
    self.points = points
    self.totalpoints = numItems
    self.recenteringFn = recenteringFn
    print("Total Points:",self.totalpoints)
    self.obstacle_points = None
    self.split = 1.0

    if obstacle_points is not None:
        self.obstacle_points = obstacle_points
        self.split = split

    

  def __len__(self):
    'Denotes the total number of samples'
    return 50


  def GetRandomSamples(self):

    input = np.zeros((1,self.totalpoints,2), dtype=np.float32 )
    output = np.zeros((1,self.totalpoints,1), dtype=np.float32 )
   
    split = int(self.totalpoints * self.split)
    input[:,:split,:] = np.expand_dims(self.points[np.random.choice(len(self.points), size=split, replace=False)],0)
    if (self.split < 1.0):
        input[:,split:,:] = np.expand_dims(self.obstacle_points[np.random.choice(len(self.obstacle_points), size=self.totalpoints-split, replace=False)],0)
        output[:,split:,:] = np.ones((1,self.totalpoints-split,1)) * -1200
    
    return {'coords':self.recenteringFn(input)}, output #self.recenteringFn(input), output #


  def __getitem__(self, index):
    'Generates one sample of data'
    return self.GetRandomSamples()











class HyperTrajectoryDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, trajectories, recenteringFn, images, obstacles = None, multiplier = 100): #coords, dictionary, coord2keyFN, startPos, endPos, graph): #x_map, y_map, mask = None):
    'Initialization'
    
    self.trajectories = trajectories
    self.obstacles = obstacles
    self.obstacle_multiplier = multiplier
    self.buffer = .1
    self.recenteringFn = recenteringFn
    self.images = images

    numItems = 0
    for _, trajectory in self.trajectories.items():
      #print('coord',coord)
      numItems += len(trajectory)-1
    if obstacles is not None:
      for _, obstacle in self.obstacles.items():
        #print('coord',coord)
        numItems += (len(obstacle)-1)*self.obstacle_multiplier*2

    self.totalpoints = numItems
    print("Total Midpoints:",self.totalpoints)


  def __len__(self):
    'Denotes the total number of samples'
    return 1

  def GetDirectionAlongTrajectory(self):

    input = np.zeros((1,self.totalpoints,2), dtype=np.float32 )
    output = np.zeros((1,self.totalpoints,2), dtype=np.float32 )
    #print('before',input.shape)

    current_midpoint = 0
    for _, trajectory in self.trajectories.items(): # for every starting position
     

      for i in range(len(trajectory)-1):
        interp = np.random.random(1)[0]
        pos = avgDiff(  trajectory[i],  trajectory[i+1],  interp)
        # print(trajectory[i])
        # print(trajectory[i+1])
        # print(pos)
        dir = normDiff( trajectory[i],  trajectory[i+1])
        input[0,current_midpoint,0] = pos[0]
        input[0,current_midpoint,1] = pos[1]
        output[0,current_midpoint,0] = -dir[0] # gradient points up, negative points in direction of movement for gradient descent
        output[0,current_midpoint,1] = -dir[1]
        current_midpoint += 1
        
     
    if self.obstacles is not None:
      for k in range(self.obstacle_multiplier):
        for _, obstacle in self.obstacles.items():
          for i in range(len(obstacle)-1):
                  interp = np.random.random(1)[0]
                  #extra_buff = .001
                  # Get buffer wrt line segment (try to prevent overlap of line normals)
                  diff = distDiff(obstacle[i], obstacle[i+1])
                  newbuff = self.buffer/diff
                  interp_with_buffer = interp * (1-2*(newbuff)) + newbuff

                  pos = avgDiff(  obstacle[i],  obstacle[i+1],  interp_with_buffer)
                  # print(trajectory[i])
                  # print(trajectory[i+1])
                  # print(pos)
                  dir = normDiff( obstacle[i],  obstacle[i+1])
                  normal = (-dir[1], dir[0])

                  # forward
                  input[0,current_midpoint,0] = pos[0] + normal[0]*self.buffer
                  input[0,current_midpoint,1] = pos[1] + normal[1]*self.buffer
                  output[0,current_midpoint,0] = normal[0]
                  output[0,current_midpoint,1] = normal[1]
                  current_midpoint += 1

                  # backward
                  input[0,current_midpoint,0] = pos[0] - normal[0]*self.buffer
                  input[0,current_midpoint,1] = pos[1] - normal[1]*self.buffer
                  output[0,current_midpoint,0] = -normal[0]
                  output[0,current_midpoint,1] = -normal[1]
                  current_midpoint += 1

    return {'img_sparse':self.images, 'coords':self.recenteringFn(input)}, output#self.recenteringFn(input), output #


  def __getitem__(self, index):
    'Generates one sample of data'
    return self.GetDirectionAlongTrajectory()




class HyperTrajectoryDataset2(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, trajectories, numItems, recenteringFn, uncenteringFn, img_points,images, pix2tfn, pix2rfn, polarfn):#, recenteringFn, images, obstacles = None, multiplier = 100): #coords, dictionary, coord2keyFN, startPos, endPos, graph): #x_map, y_map, mask = None):
    'Initialization'
    
    self.trajectories = trajectories
    #self.obstacles = obstacles
    #self.obstacle_multiplier = multiplier
    #self.buffer = .1
    self.recenteringFn = recenteringFn
    self.uncenteringFn = uncenteringFn
    self.images = images
    self.totalpoints = numItems

    self.img_points = img_points.astype(np.float32)
    self.datascale = 1
    self.pix2tfn = pix2tfn
    self.pix2rfn = pix2rfn
    self.polarfn = polarfn


  def __len__(self):
    'Denotes the total number of samples'
    return 1

  def GetDirectionAlongTrajectory(self):
      
    input = self.uncenteringFn((np.random.rand(self.totalpoints,2)*2.0-1.0).astype(np.float32))
    #random_choice = np.random.choice(len(self.img_points), size=150, replace=False)
    #input = np.copy(self.img_points)
    xformed_input = np.zeros(input.shape)
    xformed_input[:,0] = self.pix2tfn(input[:,0])
    xformed_input[:,1] = np.exp(self.pix2rfn(input[:,1]))

    xformed_input = np.array(self.polarfn(xformed_input[:,0],xformed_input[:,1])).T


    #output = (np.expand_dims(Coords2ValueFast(input,self.trajectories,1),0) / 30).astype(np.float32)Coords2ValueFastWS
    output = (np.expand_dims(Coords2ValueFastWS(xformed_input,self.trajectories,None,None,2),0) / self.datascale).astype(np.float32)
    input = np.expand_dims(input,0)
    
    return {'img_sparse':self.images, 'coords':self.recenteringFn(input)}, output#self.recenteringFn(input), output #


  def __getitem__(self, index):
    'Generates one sample of data'
    return self.GetDirectionAlongTrajectory()




class MassiveHyperTrajectoryDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, trajectories, pixeltrajectories, numItems, recenteringFn, uncenteringFn, img_points, images, pix2tfn, pix2rfn, polarfn, random=False):#, recenteringFn, images, obstacles = None, multiplier = 100): #coords, dictionary, coord2keyFN, startPos, endPos, graph): #x_map, y_map, mask = None):
    'Initialization'
    
    self.trajectories = trajectories
    #self.obstacles = obstacles
    #self.obstacle_multiplier = multiplier
    #self.buffer = .1
    self.recenteringFn = recenteringFn
    self.uncenteringFn = uncenteringFn
    self.images = images
    self.totalpoints = numItems

    self.img_points = img_points.astype(np.float32)
    self.datascale = 1
    self.pix2tfn = pix2tfn
    self.pix2rfn = pix2rfn
    self.polarfn = polarfn

    self.datalength = len(images)
    self.width = .5
    self.random = random

    self.pixeltrajectories = pixeltrajectories



  def __len__(self):
    'Denotes the total number of samples'
    return self.datalength

  def GetImageTrajectoryPair(self,index):
      
    key = list(self.images.keys())[index]
    
    if self.random:
        input = self.uncenteringFn((np.random.rand(self.totalpoints,2)*2.0-1.0).astype(np.float32))
    else:
    #random_choice = np.random.choice(len(self.img_points), size=150, replace=False)
        input = np.copy(self.img_points)
    xformed_input = np.zeros(input.shape)
    xformed_input[:,0] = self.pix2tfn(input[:,0])
    xformed_input[:,1] = np.exp(self.pix2rfn(input[:,1]))

    xformed_input = np.array(self.polarfn(xformed_input[:,0],xformed_input[:,1])).T


    #output = (np.expand_dims(Coords2ValueFast(input,self.trajectories,1),0) / 30).astype(np.float32)Coords2ValueFastWS
    output = ( np.expand_dims(Coords2ValueFastWS(xformed_input,{0:self.trajectories[key]},None,None,self.width),-1) / self.datascale).astype(np.float32)
    #input = np.expand_dims(input,0)

    traj = self.pixeltrajectories[key]
    goal_pos = self.recenteringFn( np.array([traj[0][-1],traj[1][-1]]).astype(np.float32) )
    
    return {'img_sparse':self.images[key], 'coords':self.recenteringFn(input)}, {'field':output,'goal':goal_pos} #self.recenteringFn(input), output #


  def __getitem__(self, index):
    'Generates one sample of data'
    return self.GetImageTrajectoryPair(index)











class MassiveHyperTrajectoryDatasetNEURIPS(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, trajectories, pixeltrajectories, numItems, recenteringFn, uncenteringFn, img_points, images, pix2tfn, pix2rfn, polarfn, random=False):#, recenteringFn, images, obstacles = None, multiplier = 100): #coords, dictionary, coord2keyFN, startPos, endPos, graph): #x_map, y_map, mask = None):
    'Initialization'
    
    self.trajectories = trajectories
    #self.obstacles = obstacles
    #self.obstacle_multiplier = multiplier
    #self.buffer = .1
    self.recenteringFn = recenteringFn
    self.uncenteringFn = uncenteringFn
    self.images = images
    self.totalpoints = numItems

    self.img_points = img_points.astype(np.float32)
    self.datascale = 1
    self.pix2tfn = pix2tfn
    self.pix2rfn = pix2rfn
    self.polarfn = polarfn

    self.datalength = len(images)
    self.width = 1 #.5
    self.random = random

    self.pixeltrajectories = pixeltrajectories



  def __len__(self):
    'Denotes the total number of samples'
    return self.datalength

  def GetImageTrajectoryPair(self,index):
      
    key = list(self.images.keys())[index]
    
    if self.random:
        input = self.uncenteringFn((np.random.rand(self.totalpoints,2)*2.0-1.0).astype(np.float32))
    else:
    #random_choice = np.random.choice(len(self.img_points), size=150, replace=False)
        input = np.copy(self.img_points)
    xformed_input = np.zeros(input.shape)
    xformed_input[:,0] = self.pix2tfn(input[:,0])
    xformed_input[:,1] = np.exp(self.pix2rfn(input[:,1]))

    xformed_input = np.array(self.polarfn(xformed_input[:,0],xformed_input[:,1])).T


    #output = (np.expand_dims(Coords2ValueFast(input,self.trajectories,1),0) / 30).astype(np.float32)Coords2ValueFastWS
    output = ( np.expand_dims(Coords2ValueFastWS_NEURIPS(xformed_input,{0:self.trajectories[key]},None,None,self.width,self.width),-1) / self.datascale).astype(np.float32)


    #input = np.expand_dims(input,0)

    traj = self.pixeltrajectories[key]
    goal_pos = self.recenteringFn( np.array([traj[0][-1],traj[1][-1]]).astype(np.float32) )



    
    dsiren_input = self.uncenteringFn((np.random.rand(10000,2)*2.0-1.0).astype(np.float32))
    dsiren_xformed_input = np.zeros(dsiren_input.shape)
    dsiren_xformed_input[:,0] = self.pix2tfn(dsiren_input[:,0])
    dsiren_xformed_input[:,1] = np.exp(self.pix2rfn(dsiren_input[:,1]))
    dsiren_xformed_input = np.array(self.polarfn(dsiren_xformed_input[:,0],dsiren_xformed_input[:,1])).T

    derivs = Coords2ValueFastWS_NEURIPS_DERIVATIVE(dsiren_xformed_input,{0:self.trajectories[key]},None,None,self.width,self.width)
    maximumval = np.max(derivs)
    minimumval = np.min(derivs)
    dsiren_output_gradients = ( derivs / self.datascale).astype(np.float32) #/np.linalg.norm(derivs,axis=1)[:,None]
    
    #dsiren_output_gradients = dsiren_output_gradients[:,[1,0]] 
    return {'img_sparse':self.images[key], 'coords':self.recenteringFn(input), 'derivative_siren':{'coords':self.recenteringFn(dsiren_input) } }, {'field':output, 'gradient':dsiren_output_gradients, 'goal':goal_pos} #self.recenteringFn(input), output #


  def __getitem__(self, index):
    'Generates one sample of data'
    return self.GetImageTrajectoryPair(index)















class MassiveAutoEncoderTrajectoryDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, trajectories, numItems, recenteringFn, uncenteringFn, img_points, images, pix2tfn, pix2rfn, polarfn, random=False):#, recenteringFn, images, obstacles = None, multiplier = 100): #coords, dictionary, coord2keyFN, startPos, endPos, graph): #x_map, y_map, mask = None):
    'Initialization'
    
    self.trajectories = trajectories # log polar pixel space please

    #self.obstacles = obstacles
    #self.obstacle_multiplier = multiplier
    #self.buffer = .1
    self.recenteringFn = recenteringFn
    self.uncenteringFn = uncenteringFn
    self.images = images
    self.totalpoints = numItems

    self.img_points = img_points.astype(np.float32)
    self.datascale = 1
    self.pix2tfn = pix2tfn
    self.pix2rfn = pix2rfn
    self.polarfn = polarfn

    self.datalength = len(images)
    self.width = .5
    self.random = random

    self.trajLength = 25



  def __len__(self):
    'Denotes the total number of samples'
    return self.datalength

  def GetImageTrajectoryPair(self,index):
      
    key = list(self.images.keys())[index]


    
    #if self.random:
    #    input = self.uncenteringFn((np.random.rand(self.totalpoints,2)*2.0-1.0).astype(np.float32))
    #else:
    ##random_choice = np.random.choice(len(self.img_points), size=150, replace=False)
    #    input = np.copy(self.img_points)
    #xformed_input = np.zeros(input.shape)
    #xformed_input[:,0] = self.pix2tfn(input[:,0])
    #xformed_input[:,1] = np.exp(self.pix2rfn(input[:,1]))

    #xformed_input = np.array(self.polarfn(xformed_input[:,0],xformed_input[:,1])).T

    
    traj = np.array(self.trajectories[key]).T
    tx, ty = InterpAlongLine(traj[0],traj[1],self.trajLength)
    #output = self.recenteringFn(np.vstack((tx,ty)))
    output = self.recenteringFn(np.hstack((tx,ty)))
    

    #output = ( np.expand_dims(Coords2ValueFastWS(xformed_input,{0:self.trajectories[key]},None,None,self.width),-1) / self.datascale).astype(np.float32)
    #'coords':self.recenteringFn(input)
    
    return {'img_sparse':self.images[key]},  output#self.recenteringFn(input), output #


  def __getitem__(self, index):
    'Generates one sample of data'
    return self.GetImageTrajectoryPair(index)










# Using Engineered Value Functions
# TODO: DOES NOT YET SUPPORT MULTIPLE TRAJECTORIES
def Coords2ValueFast(all_pixel_coords, trajectory_dictionary, nscale = 1):
    coord_value = np.zeros((all_pixel_coords.shape[0]))
    closest_distance = np.inf
    time_along_trajectory = 0
    for key in trajectory_dictionary.keys():
        traj = np.array(trajectory_dictionary[key],dtype=np.float32)
        traj_len = len(traj)
        dists = traj - all_pixel_coords[:,None]
        dists = np.linalg.norm(dists,axis=2)
                
        next_pts = np.ones((traj.shape[0]-1, 3))
        prev_pts = np.ones((traj.shape[0]-1, 3))
        next_pts[:,:2] = traj[1:]
        prev_pts[:,:2] = traj[:-1]


        vecLines = next_pts[:,:2]-prev_pts[:,:2]
        vecLinesMag = np.linalg.norm(vecLines,axis=1)
        distAlongTraj = np.cumsum(vecLinesMag)
        vecLines = np.divide(vecLines, vecLinesMag[:,None]) # The none thing is to make the shapes broadcastable
        vecPoints = -traj[:-1]+all_pixel_coords[:,None]
        vecPoints = np.divide(vecPoints, vecLinesMag[:,None]) # The none thing is to make the shapes broadcastable

        dotResults = np.multiply(vecLines[None,:], vecPoints).sum(2)
        firstrow = dotResults[0]
        #if dotResults.max() > 0:
        #    print('stop')
                
        trajLines = np.cross(next_pts,prev_pts)

        someones = np.ones((all_pixel_coords.shape[0],1))
        catted = np.concatenate((all_pixel_coords, someones), axis=1)
        projOntoLine = catted @ trajLines.T #np.dot(trajLines, catted[:,None] )
        lineMagnitudes = np.linalg.norm(trajLines[:,:2],axis=1)
        distsLine = np.abs( projOntoLine / lineMagnitudes[None,:] )

        #a_test = dotResults < 0
        distsLine[ dotResults < 0 ] = np.inf #can turn this into a max/min problem for divergence free processing
        distsLine[ dotResults > 1 ] = np.inf

        linesArgMin = np.argmin(distsLine,axis=1).astype(np.int32)
        pointsArgMin = np.argmin(dists,axis=1).astype(np.int32)

            
        smallest_distsLine = np.squeeze( np.take_along_axis(distsLine,linesArgMin[:,None],axis=1) )
        smallest_dists = np.squeeze( np.take_along_axis(dists,pointsArgMin[:,None],axis=1) )

        dist_along_min_line_segment = np.squeeze( np.take_along_axis(dotResults,linesArgMin[:,None],axis=1) )

        stacked = np.stack((smallest_distsLine, smallest_dists), axis=1)

        which_was_closest = np.argmin(stacked,axis=1).astype(np.int32) # which was closest between point and line

            
        closest_lines = which_was_closest == 0
        closest_points = which_was_closest == 1
            
        current_distance = np.zeros(smallest_dists.shape[0])
        current_time = np.zeros(smallest_dists.shape[0])


        current_distance[closest_lines] = smallest_distsLine[closest_lines] #distsLine[linesArgMin]
        above_zero = linesArgMin > 0
        together = np.logical_and(closest_lines, above_zero)
        #testing = distAlongTraj[linesArgMin]
        current_time[together] = distAlongTraj[linesArgMin-1][together]
        #vlm = vecLinesMag[linesArgMin][together]
        #dr = dist_along_min_line_segment[linesArgMin][together]
            
        dist_along_min_line_segment = np.squeeze( np.take_along_axis(dotResults,linesArgMin[:,None],axis=1) )
        #resultA = vecLinesMag[closest_lines]#[closest_lines]
        #resultB = dist_along_min_line_segment[closest_lines]#[closest_lines]
        #finalResult = resultA * resultB
        #finalResultClosest = finalResultClosest[closest_lines]
        current_time[closest_lines] += vecLinesMag[linesArgMin][closest_lines] * dist_along_min_line_segment[closest_lines]

        #closest_distance_arg = np.argmin([distsLine[linesArgMin],dists[pointsArgMin]])
            
        current_distance[closest_points] = smallest_dists[closest_points]
        above_zero = pointsArgMin > 0
        together = np.logical_and(closest_points, above_zero)
        current_time[together] = distAlongTraj[pointsArgMin-1][together]




                
        closest_distances = current_distance
        time_along_trajectory = current_time
                
    values = -stats.norm.pdf(closest_distances, scale=nscale) * time_along_trajectory#time_along_trajectory
    coord_value = values

    return coord_value




def Coords2ValueFastWS(all_pixel_coords, trajectory_dictionary, coordXform_FORWARD, coordXform_BACKWARD, stddev = 1):
    coord_value = np.zeros((all_pixel_coords.shape[0]))
    closest_distance = np.inf
    time_along_trajectory = 0
    for key in trajectory_dictionary.keys():
        traj = np.array(trajectory_dictionary[key],dtype=np.float32)
        traj_len = len(traj)
        dists = traj - all_pixel_coords[:,None]
        dists = np.linalg.norm(dists,axis=2)
                
        next_pts = np.ones((traj.shape[0]-1, 3))
        prev_pts = np.ones((traj.shape[0]-1, 3))
        next_pts[:,:2] = traj[1:]
        prev_pts[:,:2] = traj[:-1]


        vecLines = next_pts[:,:2]-prev_pts[:,:2]
        vecLinesMag = np.linalg.norm(vecLines,axis=1)
        distAlongTraj = np.cumsum(vecLinesMag)
        vecLines = np.divide(vecLines, vecLinesMag[:,None]) # The none thing is to make the shapes broadcastable
        vecPoints = -traj[:-1]+all_pixel_coords[:,None]
        vecPoints = np.divide(vecPoints, vecLinesMag[:,None]) # The none thing is to make the shapes broadcastable

        dotResults = np.multiply(vecLines[None,:], vecPoints).sum(2)
        firstrow = dotResults[0]
        #if dotResults.max() > 0:
        #    print('stop')
                
        trajLines = np.cross(next_pts,prev_pts)

        someones = np.ones((all_pixel_coords.shape[0],1))
        catted = np.concatenate((all_pixel_coords, someones), axis=1)
        projOntoLine = catted @ trajLines.T #np.dot(trajLines, catted[:,None] )
        lineMagnitudes = np.linalg.norm(trajLines[:,:2],axis=1)
        distsLine = np.abs( projOntoLine / lineMagnitudes[None,:] )

        #a_test = dotResults < 0
        distsLine[ dotResults < 0 ] = np.inf #can turn this into a max/min problem for divergence free processing
        distsLine[ dotResults > 1 ] = np.inf

        linesArgMin = np.argmin(distsLine,axis=1).astype(np.int32)
        pointsArgMin = np.argmin(dists,axis=1).astype(np.int32)

            
        smallest_distsLine = np.squeeze( np.take_along_axis(distsLine,linesArgMin[:,None],axis=1) )
        smallest_dists = np.squeeze( np.take_along_axis(dists,pointsArgMin[:,None],axis=1) )

        dist_along_min_line_segment = np.squeeze( np.take_along_axis(dotResults,linesArgMin[:,None],axis=1) )

        stacked = np.stack((smallest_distsLine, smallest_dists), axis=1)

        which_was_closest = np.argmin(stacked,axis=1).astype(np.int32) # which was closest between point and line

            
        closest_lines = which_was_closest == 0
        closest_points = which_was_closest == 1
            
        current_distance = np.zeros(smallest_dists.shape[0])
        current_time = np.zeros(smallest_dists.shape[0])


        current_distance[closest_lines] = smallest_distsLine[closest_lines] #distsLine[linesArgMin]
        above_zero = linesArgMin > 0
        together = np.logical_and(closest_lines, above_zero)
        #testing = distAlongTraj[linesArgMin]
        current_time[together] = distAlongTraj[linesArgMin-1][together]
        #vlm = vecLinesMag[linesArgMin][together]
        #dr = dist_along_min_line_segment[linesArgMin][together]
            
        dist_along_min_line_segment = np.squeeze( np.take_along_axis(dotResults,linesArgMin[:,None],axis=1) )
        #resultA = vecLinesMag[closest_lines]#[closest_lines]
        #resultB = dist_along_min_line_segment[closest_lines]#[closest_lines]
        #finalResult = resultA * resultB
        #finalResultClosest = finalResultClosest[closest_lines]
        current_time[closest_lines] += vecLinesMag[linesArgMin][closest_lines] * dist_along_min_line_segment[closest_lines]

        #closest_distance_arg = np.argmin([distsLine[linesArgMin],dists[pointsArgMin]])
            
        current_distance[closest_points] = smallest_dists[closest_points]
        above_zero = pointsArgMin > 0
        together = np.logical_and(closest_points, above_zero)
        current_time[together] = distAlongTraj[pointsArgMin-1][together]




                
        closest_distances = current_distance
        time_along_trajectory = current_time
                
    time_along_trajectory = np.log(time_along_trajectory+1)
    values = -stats.norm.pdf(closest_distances,scale=stddev) * time_along_trajectory#time_along_trajectory
    coord_value = values

    return coord_value







#DENSITY_STD = 20

GT_SCALE = 25

# Sample points in world space (likely pixel centers), trajectory in world space, unused, unused, stddev from gaussian
def Coords2ValueFastWS_NEURIPS(all_pixel_coords, trajectory_dictionary, coordXform_FORWARD, coordXform_BACKWARD, stddev = 1, dstddev =None):
    if dstddev is None:
        dstddev = stddev
    coord_value = np.zeros((all_pixel_coords.shape[0])) # where value function will be stored.
    closest_distance = np.inf # might be unused...
    time_along_trajectory = 0 # don't know
    for key in trajectory_dictionary.keys():
        traj = np.array(trajectory_dictionary[key],dtype=np.float32) # trajectory in world space
        traj_len = len(traj)
        dists = traj - all_pixel_coords[:,None] # distance from every point on the trajectory to all pixel coordinates
        dists = np.linalg.norm(dists,axis=2) # magnitude of distances to every point on trajectory
                
        traj_dists = traj - traj[:,None]
        traj_dists = np.linalg.norm(traj_dists,axis=2) # magnitude of distances to every point on trajectory
        density  = np.exp( - .5 * traj_dists**2 / dstddev )
        density  = np.sum(density, axis=1)
        
        #next_pts = np.zeros((traj.shape[0], 2))
        prev_pts = np.zeros((traj.shape[0], 2))
        prev_pts[1:] = traj[:-1]
        prev_pts[0] = traj[0]
        vecLines = traj-prev_pts
        vecLinesMag = np.linalg.norm(vecLines,axis=1)
        distAlongTraj = np.cumsum(vecLinesMag)

        #end = distAlongTraj[-1] # trajectory length (path integral)
        #distAlongTraj /= distAlongTraj[-1] # trajectory distance is always 1 at the goal. TODO is this okay?
        #end = distAlongTraj[-1] # unit path integral

        #dists /= distAlongTraj[-1]

        # now let's get the weighted sum of gaussians multiplied by the path integral

        #testing = np.zeros(traj.shape[0])
        #testing[0] = 1

        total_dist = distAlongTraj[-1]

        if (len(density) > 1):
            density[0] = density[1]
            density[-1] = density[-2]

        gaussians = -np.exp( - .5 * dists**2 / stddev )  #stats.norm.pdf(dists,scale=stddev)
        product = gaussians * distAlongTraj[None] / density[None]
        weighted_sum = np.sum(product,axis=1)
        #max_weighted_sum = np.min(weighted_sum)
        final = weighted_sum
        coord_value = final/GT_SCALE





        ## Let's generate vectors tangent to trajectory.
        ## First, let's get next and previous points
        #next_pts = np.ones((traj.shape[0]-1, 3))
        #prev_pts = np.ones((traj.shape[0]-1, 3))
        #next_pts[:,:2] = traj[1:]
        #prev_pts[:,:2] = traj[:-1]

        ## Now let's subtract a next point from a previous point
        #vecLines = next_pts[:,:2]-prev_pts[:,:2]
        #vecLinesMag = np.linalg.norm(vecLines,axis=1)
        #distAlongTraj = np.cumsum(vecLinesMag)
        #vecLines = np.divide(vecLines, vecLinesMag[:,None]) # The none thing is to make the shapes broadcastable
        #vecPoints = -traj[:-1]+all_pixel_coords[:,None]
        #vecPoints = np.divide(vecPoints, vecLinesMag[:,None]) # The none thing is to make the shapes broadcastable

        #dotResults = np.multiply(vecLines[None,:], vecPoints).sum(2)
        #firstrow = dotResults[0]
        ##if dotResults.max() > 0:
        ##    print('stop')
                
        #trajLines = np.cross(next_pts,prev_pts)

        #someones = np.ones((all_pixel_coords.shape[0],1))
        #catted = np.concatenate((all_pixel_coords, someones), axis=1)
        #projOntoLine = catted @ trajLines.T #np.dot(trajLines, catted[:,None] )
        #lineMagnitudes = np.linalg.norm(trajLines[:,:2],axis=1)
        #distsLine = np.abs( projOntoLine / lineMagnitudes[None,:] )

        ##a_test = dotResults < 0
        #distsLine[ dotResults < 0 ] = np.inf #can turn this into a max/min problem for divergence free processing
        #distsLine[ dotResults > 1 ] = np.inf

        #linesArgMin = np.argmin(distsLine,axis=1).astype(np.int32)
        #pointsArgMin = np.argmin(dists,axis=1).astype(np.int32)

            
        #smallest_distsLine = np.squeeze( np.take_along_axis(distsLine,linesArgMin[:,None],axis=1) )
        #smallest_dists = np.squeeze( np.take_along_axis(dists,pointsArgMin[:,None],axis=1) )

        #dist_along_min_line_segment = np.squeeze( np.take_along_axis(dotResults,linesArgMin[:,None],axis=1) )

        #stacked = np.stack((smallest_distsLine, smallest_dists), axis=1)

        #which_was_closest = np.argmin(stacked,axis=1).astype(np.int32) # which was closest between point and line

            
        #closest_lines = which_was_closest == 0
        #closest_points = which_was_closest == 1
            
        #current_distance = np.zeros(smallest_dists.shape[0])
        #current_time = np.zeros(smallest_dists.shape[0])


        #current_distance[closest_lines] = smallest_distsLine[closest_lines] #distsLine[linesArgMin]
        #above_zero = linesArgMin > 0
        #together = np.logical_and(closest_lines, above_zero)
        ##testing = distAlongTraj[linesArgMin]
        #current_time[together] = distAlongTraj[linesArgMin-1][together]
        ##vlm = vecLinesMag[linesArgMin][together]
        ##dr = dist_along_min_line_segment[linesArgMin][together]
            
        #dist_along_min_line_segment = np.squeeze( np.take_along_axis(dotResults,linesArgMin[:,None],axis=1) )
        ##resultA = vecLinesMag[closest_lines]#[closest_lines]
        ##resultB = dist_along_min_line_segment[closest_lines]#[closest_lines]
        ##finalResult = resultA * resultB
        ##finalResultClosest = finalResultClosest[closest_lines]
        #current_time[closest_lines] += vecLinesMag[linesArgMin][closest_lines] * dist_along_min_line_segment[closest_lines]

        ##closest_distance_arg = np.argmin([distsLine[linesArgMin],dists[pointsArgMin]])
            
        #current_distance[closest_points] = smallest_dists[closest_points]
        #above_zero = pointsArgMin > 0
        #together = np.logical_and(closest_points, above_zero)
        #current_time[together] = distAlongTraj[pointsArgMin-1][together]




                
        #closest_distances = current_distance
        #time_along_trajectory = current_time
                
    #time_along_trajectory = np.log(time_along_trajectory+1)
    #values = -stats.norm.pdf(closest_distances,scale=stddev) * time_along_trajectory#time_along_trajectory
    #coord_value = values

    return coord_value

def Coords2ValueFastWS_NEURIPS_DERIVATIVE(all_pixel_coords, trajectory_dictionary, coordXform_FORWARD, coordXform_BACKWARD, stddev = 1, dstddev =None):
    if dstddev is None:
        dstddev = stddev
    coord_deriv = np.zeros((all_pixel_coords.shape[0],2)) # where derivative of the value function will be stored.
    closest_distance = np.inf # might be unused...
    time_along_trajectory = 0 # don't know
    for key in trajectory_dictionary.keys():
        traj = np.array(trajectory_dictionary[key],dtype=np.float32) # trajectory in world space
        traj_len = len(traj)
        diffs = traj - all_pixel_coords[:,None] # diff from every point on the trajectory to all pixel coordinates
        dists = np.linalg.norm(diffs,axis=2) # magnitude of diffs (distances) to every point on trajectory
                
        traj_dists = traj - traj[:,None]
        traj_dists = np.linalg.norm(traj_dists,axis=2) # magnitude of distances to every point on trajectory
        density  = np.exp( - .5 * traj_dists**2 / dstddev )
        density  = np.sum(density, axis=1)
        
        #next_pts = np.zeros((traj.shape[0], 2))
        prev_pts = np.zeros((traj.shape[0], 2))
        prev_pts[1:] = traj[:-1]
        prev_pts[0] = traj[0]
        vecLines = traj-prev_pts
        vecLinesMag = np.linalg.norm(vecLines,axis=1)
        distAlongTraj = np.cumsum(vecLinesMag)




        #end = distAlongTraj[-1] # trajectory length (path integral)
        #distAlongTraj /= distAlongTraj[-1] # trajectory distance is always 1 at the goal. TODO is this okay?
        #end = distAlongTraj[-1] # unit path integral

        #dists /= distAlongTraj[-1]

        # now let's get the weighted sum of gaussians multiplied by the path integral

        #testing = np.zeros(traj.shape[0])
        #testing[0] = 1

        total_dist = distAlongTraj[-1]

        if (len(density) > 1):
            density[0] = density[1]
            density[-1] = density[-2]

        gaussians = -np.exp( - .5 * dists**2 / stddev )  #stats.norm.pdf(dists,scale=stddev)
        vecLines_x = vecLines[:,0]
        vecLines_y = vecLines[:,1]
        first_term_x = gaussians * vecLines_x[None] / density[None]
        first_term_y = gaussians * vecLines_y[None] / density[None]
        first_term = np.stack((first_term_x,first_term_y),axis=2)

        weighted_diffs = diffs/(stddev**2)
        weighted_diffs_x = weighted_diffs[:,:,0]
        weighted_diffs_y = weighted_diffs[:,:,1]
        
        product = gaussians * distAlongTraj[None] / density[None]
        
        second_term_x = product * weighted_diffs_x
        second_term_y = product * weighted_diffs_y
        
        second_term = np.stack((second_term_x,second_term_y),axis=2)

        total_term = first_term + second_term

        grads = np.sum(total_term,axis=1)
        #max_weighted_sum = np.min(weighted_sum)
        #final = summed
        grads/=GT_SCALE

        # ROTATE INTO SIREN SPACE
        t,r = Coord2Polar(all_pixel_coords[:,0],all_pixel_coords[:,1])

        logr = np.log(r)
        rdenom = 1.0/r
        sint = np.sin(t)
        cost = np.cos(t)

        jacobians = np.zeros((len(t),2,2))
        #jacobians[:,0,0] = -logr*sint #dx/dt
        #jacobians[:,0,1] =  logr*cost #dy/dt
        #jacobians[:,1,0] = rdenom*cost #dx/dr
        #jacobians[:,1,1] = rdenom*sint #dy/dr
        jacobians[:,0,0] = -np.log(r)*np.sin(t) #dx/dt
        jacobians[:,0,1] =  np.log(r)*np.cos(t) #dy/dt
        jacobians[:,1,0] = 1.0/r*np.cos(t) #dx/dr
        jacobians[:,1,1] = 1.0/r*np.sin(t) #dy/dr

        tjacobians = torch.from_numpy(jacobians)
        tgrads = torch.from_numpy(np.expand_dims(grads,-1))
        tjacobians.requires_grad = False
        tgrads.requires_grad = False
        tfinal = torch.bmm(tjacobians,tgrads)
        final = tfinal.numpy()


    return np.squeeze(final)

def Coord2Polar(x,y):
    #z= np.random.random((10,2))
    #x,y = z[0,:], z[1,:]
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)
    
    return t,r
    # np.stack((r,t,)) #np.concatenate((r,t),axis=1)
    #print(r)
    #print(t)
















class GTFieldDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, trajectories, numItems, recenteringFn, img_points ): #coords, dictionary, coord2keyFN, startPos, endPos, graph): #x_map, y_map, mask = None):
    'Initialization'
    
    self.trajectories = trajectories


    #self.gradients = -gradients
    self.totalpoints = numItems

    self.recenteringFn = recenteringFn

    self.img_points = img_points.astype(np.float32)
    
    #self.recenteringFn = recenteringFn
    #print("Total Points:",self.totalpoints)
    #self.obstacle_points = None

    #if obstacle_points is not None:
    #    self.obstacle_points = obstacle_points
    #    self.split = split

    

  def __len__(self):
    'Denotes the total number of samples'
    return 1


  def GetRandomSamples(self):

    #randoms = .01*(np.random.rand(1,self.totalpoints,2)*2.0-1.0)

    #input = np.zeros((1,self.totalpoints,2), dtype=np.float32 )
    #output = np.zeros((1,self.totalpoints,1), dtype=np.float32 )
   
    #split = int(self.totalpoints * 1)
    random_choice = np.random.choice(len(self.img_points), size=150, replace=False)
    #everynth = [x for x in range(len(self.img_points)) if x % 32 == 0]
    input = np.copy(self.img_points)
     #(np.random.rand(1,self.totalpoints,2) * 2 - 1).astype(np.float32)
    output = (np.expand_dims(Coords2ValueFast(input,self.trajectories,1),0) / 30).astype(np.float32)#(np.expand_dims(Coords2ValueFast(self.recenteringFn(input[0]),self.trajectories),0) / 65).astype(np.float32)
    #output = (np.expand_dims(input[:,1],0) / 64).astype(np.float32)
    input = np.expand_dims(input,0)
    #output *= 0.0
    #output += .7

    #input[:,:,:] = np.expand_dims(self.points[random_choice],0)#np.expand_dims(self.points[random_choice],0)
    #input += randoms
    #output[:,:,:] = np.expand_dims(self.gradients[random_choice],0)#np.expand_dims(self.gradients[random_choice],0)
    #input = np.expand_dims(self.points,0)
    #output = np.expand_dims(self.gradients,0)
    
    return {'coords':self.recenteringFn(input)}, output #self.recenteringFn(input), output #


  def __getitem__(self, index):
    'Generates one sample of data'
    return self.GetRandomSamples()



def InterpAlongLine(x,y,n, end = -1):
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd**2+yd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])
    
    if end < 0:
        t = np.linspace(0,u.max(),n)
    else:
        t = np.linspace(0,end,n)
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)
    return xn, yn