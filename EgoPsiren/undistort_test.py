import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy import interpolate

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

    #for (int iPoint = 0; iPoint < vx.size(); iPoint++)
    #        CvMat *x_homo = cvCreateMat(3,1,CV_32FC1);
    #        cvSetReal2D(x_homo, 0, 0, vx[iPoint]);
    #        cvSetReal2D(x_homo, 1, 0, vy[iPoint]);
    #        cvSetReal2D(x_homo, 2, 0, 1);
    #        CvMat *x_homo_n = cvCreateMat(3,1,CV_32FC1);
    #        cvMatMul(invK, x_homo, x_homo_n);
    #        double x_n, y_n;
    #        x_n = cvGetReal2D(x_homo_n, 0, 0);
    #        y_n = cvGetReal2D(x_homo_n, 1, 0);
    #        double r_d = sqrt(x_n*x_n+y_n*y_n);
    #        double r_u = tan(r_d*omega)/2/tan(omega/2);
    #        double x_u = r_u/r_d*x_n;
    #        double y_u = r_u/r_d*y_n;
    #        CvMat *x_undist_n = cvCreateMat(3,1,CV_32FC1);
    #        cvSetReal2D(x_undist_n, 0, 0, x_u);
    #        cvSetReal2D(x_undist_n, 1, 0, y_u);
    #        cvSetReal2D(x_undist_n, 2, 0, 1);
    #        CvMat *x_undist = cvCreateMat(3,1,CV_32FC1);
    #        cvMatMul(K, x_undist_n, x_undist);
    #        vx[iPoint] = cvGetReal2D(x_undist,0,0);
    #        vy[iPoint] = cvGetReal2D(x_undist,1,0);
    #        cvReleaseMat(&x_homo);
    #        cvReleaseMat(&x_homo_n);
    #        cvReleaseMat(&x_undist_n);
    #        cvReleaseMat(&x_undist);



#pixels = K_data @ R_rect @ coords_3D.T
#pixels /= pixels[2]
##pixels[:,:] /= pixels[2,:]

#rowmaj_pixels = np.zeros(pixels.shape)
#rowmaj_pixels[0] = pixels[1]
#rowmaj_pixels[1] = pixels[0]


#img_resized =      interpolate.interpn((range(img.shape[0]),range(img.shape[1])), img*2.0-1.0, rowmaj_pixels[:2].T , method = 'linear',bounds_error = False, fill_value = 0).reshape(ego_pixel_shape[0], ego_pixel_shape[1],3)
#img_channel_swap = np.moveaxis(img_resized,-1,0).astype(np.float32)


X = cv2.imread('S:\image0001877.jpg').astype(np.float32)/255
print(X.min())
print(X.max())


class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()


class PointPlotter:
    def __init__(self, points):
        self.points = points
        self.xs = list(points.get_xdata())
        self.ys = list(points.get_ydata())
        self.cid = points.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.points.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.points.set_data(self.xs, self.ys)
        self.points.figure.canvas.draw()


class DistortionSolver:
    def __init__(self, points):
        self.points = points
        self.xs = list(points.get_xdata())
        self.ys = list(points.get_ydata())
        self.cid = points.figure.canvas.mpl_connect('key_press_event', self)
        self.omega = 1.0

    def __call__(self, event):
        #print('press', event)
        print('you pressed', event.key, event.xdata, event.ydata)
        if event.inaxes!=self.points.axes: return

        if event.key == 'up':
            self.omega += .01;
        #self.xs.append(event.xdata)
        #self.ys.append(event.ydata)
        #self.points.set_data(self.xs, self.ys)
        #self.points.figure.canvas.draw()



#m_x: 1280
#im_y: 720
#fx: 562.89536
#fy: 557.29656
#px: 630.7712
#py: 363.16152
#omega: .85



K = np.array([[562.89536, 0, 630.7712],[0, 557.29656, 363.16152],[0, 0, 1]])


points = np.random.uniform(250,750,(2,2000)) #np.array([[500,500,500],
                   #[250,500,750]])

distorted = Distort(points, 0.85, K)
undistorted = Undistort(distorted, 0.85, K)


result = points - undistorted





all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(X.shape[0]) for j in range(X.shape[1]) ], dtype=np.float32).T
all_pixel_coords_undistorted = Distort(all_pixel_coords,0.85,K)

pix_part2 = np.copy(all_pixel_coords_undistorted)
pix_part2[0] = all_pixel_coords_undistorted[1]
pix_part2[1] = all_pixel_coords_undistorted[0]
#rowmaj_pixels[:2].T

img2 = interpolate.interpn( (range(X.shape[0]),range(X.shape[1])), X, pix_part2[:2].T , method = 'linear',bounds_error = False, fill_value = 0).astype(np.float32).reshape(X.shape[0], X.shape[1],3)



fig, ax = plt.subplots(1,1)
axes = [ax]

axes[0].imshow(img2)

#axes[0].plot(400,20,'rx')

#axes[0].plot(800,60,'rx')

#axes[0].plot(1150,173,'rx')


points, = axes[0].plot([], [], 'rx')  # empty line
#pointplotter = PointPlotter(points)

DistortionSolver(points)

plt.show()