import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(float)
    l_channel = hls[:,:,1]
    #Blur
    #l_channel = cv2.GaussianBlur(l_channel, (3, 3), cv2.BORDER_DEFAULT)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def color_threshold(image, sthresh=(0,255), vthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary) == 1] = 1

    # Return the combined s_channel & v_channel binary image
    return output

def s_channel_threshold(image, sthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]  # use S channel

    # create a copy and apply the threshold
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1
    return binary_output

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output

test_images = ['cropped/main/0004.png', 'cropped/main/0231.png', 'cropped/main/0904.png']

# for image_name in test_images:
#     test_image = img = cv2.imread(image_name)
#
#     # Apply Sobel operator in X-direction to experiment with gradient thresholds
#     gradx = abs_sobel_thresh(test_image, orient='x', thresh_min=10, thresh_max=50)
#
#     # Visualize the results before/after absolute sobel operator is applied on a test image in x direction to find the
#     # vertical lines, since the lane lines are close to being vertical
#     plt.figure(figsize=(18, 12))
#     grid = gridspec.GridSpec(1, 2)
#
#     # set the spacing between axes.
#     grid.update(wspace=0.1, hspace=0.1)
#
#     plt.subplot(grid[0])
#     plt.imshow(test_image, cmap="gray")
#     plt.title('Undistorted Image')
#
#     plt.subplot(grid[1])
#     plt.imshow(gradx, cmap="gray")
#     plt.title('Absolute sobel threshold in X direction')
#
#     plt.show()
#
# for image_name in test_images:
#     test_image = img = cv2.imread(image_name)
#
#     #Apply Sobel operator in Y-direction to experiment with gradient thresholds
#     grady = abs_sobel_thresh(test_image, orient='y', thresh_min=20, thresh_max=80)
#
#     #Visualize the results before/after sobel operator is applied on a test image in y direction
#     plt.figure(figsize = (18,12))
#     grid = gridspec.GridSpec(1,2)
#
#     # set the spacing between axes.
#     grid.update(wspace=0.1, hspace=0.1)
#
#     plt.subplot(grid[0])
#     plt.imshow(test_image, cmap="gray")
#     plt.title('Undistorted Image')
#
#     plt.subplot(grid[1])
#     plt.imshow(grady, cmap="gray")
#     plt.title('Absolute sobel threshold in Y direction')
#
#     plt.show()
#
# for image_name in test_images:
#     test_image = img = cv2.imread(image_name)
#     #Apply magnitude threshold
#     magThr = mag_thresh(test_image, sobel_kernel=3, mag_thresh=(20, 70))
#
#     #Visualize the results before/after applying magnitude thresholds
#     plt.figure(figsize = (18,12))
#     grid = gridspec.GridSpec(1,2)
#
#     # set the spacing between axes.
#     grid.update(wspace=0.1, hspace=0.1)
#
#     plt.subplot(grid[0])
#     plt.imshow(test_image, cmap="gray")
#     plt.title('Undistorted Image')
#
#     plt.subplot(grid[1])
#     plt.imshow(magThr, cmap="gray")
#     plt.title('After applying Magnitude Threshold')
#
#     plt.show()
#
# dirThr = dir_threshold(test_image, sobel_kernel=31, thresh=(0.5, 1))
#
# #Visualize the results before/after direction threshold is applied
# plt.figure(figsize = (18,12))
# grid = gridspec.GridSpec(1,2)
#
# # set the spacing between axes.
# grid.update(wspace=0.1, hspace=0.1)
#
# plt.subplot(grid[0])
# plt.imshow(test_image, cmap="gray")
# plt.title('Undistorted Image')
#
# plt.subplot(grid[1])
# plt.imshow(dirThr, cmap="gray")
# plt.title('After applying direction Threshold')
#
# plt.show()
#
# #use s channel alone in HLS colorspace and experiment with thresholds
# s_binary = s_channel_threshold(test_image, sthresh=(170,255))
#
# #Visualize the results before/after s channel threshold is applied
# plt.figure(figsize = (18,12))
# grid = gridspec.GridSpec(1,2)
# # set the spacing between axes.
# grid.update(wspace=0.1, hspace=0.1)
#
# plt.subplot(grid[0])
# plt.imshow(test_image, cmap="gray")
# plt.title('Undistorted Image')
#
# plt.subplot(grid[1])
# plt.imshow(s_binary, cmap="gray")
# plt.title('After applying S-channel Threshold')
#
# plt.show()
#
# for image_name in test_images:
#     test_image = img = cv2.imread(image_name)
#     #Experiment with HLS & HSV color spaces along with thresholds
#     c_binary = color_threshold(test_image, sthresh=(0,10), vthresh=(140,255))
#
#     #Visualize the results before/after HLS/HSV  threshold is applied
#     plt.figure(figsize = (18,12))
#     grid = gridspec.GridSpec(1,2)
#
#     # set the spacing between axes.
#     grid.update(wspace=0.1, hspace=0.1)
#
#     plt.subplot(grid[0])
#     plt.imshow(test_image, cmap="gray")
#     plt.title('Undistorted Image')
#
#     plt.subplot(grid[1])
#     plt.imshow(c_binary, cmap="gray")
#     plt.title('After applying color threshold')
#
#     plt.show()
#
for image_name in test_images:
    test_image = img = cv2.imread(image_name)
    #Combine the binary images using the Sobel thresholds in X/Y directions along with the color threshold to form the final image pipeline
    preprocessImage = np.zeros_like(test_image[:,:,0])
    gradx = abs_sobel_thresh(test_image, orient='x', thresh_min=10, thresh_max=50)
    grady = abs_sobel_thresh(test_image, orient='y', thresh_min=20, thresh_max=80)
    c_binary = color_threshold(test_image, sthresh=(0, 10), vthresh=(140, 255))
    #this was the original line
    #preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255
    # i changed it to this
    preprocessImage[gradx + grady + c_binary >= 2] = 255
    #preprocessImage[((gradx == 1) & (grady == 1))] = 255

    #Visualize the results before/after combining the images from the pipeline
    plt.figure(figsize = (18,12))
    grid = gridspec.GridSpec(1,2)

    # set the spacing between axes.
    grid.update(wspace=0.1, hspace=0.1)

    plt.subplot(grid[0])
    plt.imshow(test_image, cmap="gray")
    plt.title('Undistorted Image')

    plt.subplot(grid[1])
    plt.imshow(preprocessImage, cmap="gray")
    plt.title('After image processing pipeline')

    plt.show()