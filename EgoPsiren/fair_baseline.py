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
        sys.argv[1:] = "--data S:/11f247e0-179a-4b9d-8244-16fb918010a1_0 --output S:/11f247e0-179a-4b9d-8244-16fb918010a1_0 --images im --length 100 --stride 20".split()
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




    # load calibration
    print('loading calibration file')
    calibfile = __data_source / Path('calib_fisheye.txt')

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
    file_list = __data_source / Path('im_list.list')
    #fid = open(file_list);
    #data = textscan(fid, '%s');
    #data = fid.readlines()
    with open(file_list) as fid:
        data = fid.read().splitlines()
    vFilename = data;
    #fid.close();





    print('loading trajectory file')
    traj_data_file = __data_source / Path('traj_prediction.txt')
    vTR = ReadTraj(traj_data_file)

    print('done')




## PROCESS INPUT
## Project trajectory and map into EgoSpace
## # Use constants from paper
## # Trim trajectory to start at minimum distance
## # # Some trajectories start "behind" camera due to unavoidable reconstruction errors

## TRAIN BENCHMARK
## 
