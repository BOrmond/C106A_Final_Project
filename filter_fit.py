from scipy import linalg, stats
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#pass in image filename and polynomial degree for fitting
def filter_fit_func(im_name, deg):
    
    # original, distorted, image
    original = cv.imread("main/" + im_name)

    # undistorted image
    undist = cv.imread("proc_main/proc_" + im_name)
    gray = cv.cvtColor(undist, cv.COLOR_BGR2GRAY)

    # Sobel Filter
    #https://medium.com/swlh/computer-vision-advanced-lane-detection-through-thresholding-8a4dea839179

    # Apply Sobel in the x and y direction
    # (previously, we used a x = 1 + y = 1 direction together)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)

    # We have to take absolute value, otherwise negative gradients will be taken as 0 when converting to uint8
    sobelx_abs = np.absolute(sobelx) 
    sobely_abs = np.absolute(sobely)

    conv_sobelx = np.uint8(255*sobelx_abs/np.max(sobelx_abs))
    conv_sobely = np.uint8(255*sobely_abs/np.max(sobely_abs))

    #create a binary mask of all zeroes, of the same size as the original image
    binary = np.zeros_like(conv_sobelx)

    # define sobel thresholds
    x_min, x_max = 10, 100
    y_min, y_max = 10, 100

    #make binary pixel 1 if the sobelx and sobely pixel falls within the chosen threshold
    binary[((conv_sobelx >= x_min) & (conv_sobelx <= x_max)) & ((conv_sobely >= y_min) & (conv_sobely <= y_max))] = 1

    #make binary pixel 0 if it is outside of relevant image region
    binary[0:60, :] = 0
    binary[:, :50] = 0
    binary[:, 250:] = 0

    # Color Bounds method

    # (Previously, we used RGB filtering)
    # lower_bound = np.array([140, 140, 140]) #upper and lower RGB bounds
    # upper_bound = np.array([148, 148, 148])

    # Define the hsv upper and lower bounds
    hsv_upper = np.array([91, 15, 255])
    hsv_lower = np.array([0, 0, 128])

    # convert image to HSV format
    hsv = cv.cvtColor(undist, cv.COLOR_BGR2HSV)

    # find all pixels within this hsv range, in the relevant region of the image
    in_bound = cv.inRange(hsv, hsv_lower, hsv_upper)

    # Combine both the color filtering and sobol filtering
    mask = in_bound & binary

    #extract coordinates of each nonzero pixel
    pts = np.array([(x, y) for y in range(len(mask[0])) for x in range(len(mask)) if mask[x, y] > 0])
    pts_y = [pt[1] for pt in pts]

    z = np.abs(stats.zscore(pts_y))
    z_pts = np.array([pts[i] for i in range(len(pts)) if z[i] < np.median(z)])


    C_inv = linalg.inv(np.cov(z_pts.T))
    alpha = pts - np.mean(z_pts)
    a_dot_Cinv = np.dot(alpha, C_inv)
    inner = np.dot(a_dot_Cinv, alpha.T)
    mahalanobis = np.sqrt(inner)
    md = np.sqrt(mahalanobis.diagonal())

    true_C = np.sqrt(stats.chi2.ppf(1 -.001, df=z_pts.shape[1]))
    f_pts = [z_pts[i] for i in range(len(z_pts)) if md[i] < np.median(md)]
    f_ptsx = [pt[0] for pt in f_pts]
    f_ptsy = [pt[1] for pt in f_pts]

    fit = np.polyfit(f_ptsy, f_ptsx, deg)
    curve = np.poly1d(fit)
    
    final_im = undist.copy()
    for i in range(len(f_pts)):
        final_im[f_ptsx[i], f_ptsy[i]] = [255, 0, 0]
    

    return final_im, curve