"""
Implementation of the Histogram Color Detection for HW3 in MSAI495.
Author: Nolan Knight
Date: 2026-04-20
"""

# Imports:
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# Functions:
# A function to select a region in an open image and return an array of skin pigment pixels:
def collectTrain(img: Image):
    # Show the image, so the user can select skin pigment:
    img_arr = np.array(img)
    plt.imshow(img_arr)
    plt.title("Click top-left then bottom-right to select skin region")
    plt.show(block=False)

    # Select an upper left and lower right corner to isolate skin:
    points = plt.ginput(2)

    # Extract the x and y positions in the image to crop out skin pixels:
    xmin, xmax = sorted([int(points[0][0]), int(points[1][0])])
    ymin, ymax = sorted([int(points[0][1]), int(points[1][1])])

    # Crop the region and reshape to a list of pixels:
    skin_pixels = img_arr[ymin:ymax, xmin:xmax].reshape(-1, 3)  # shape: (Pixels, 3)
    return skin_pixels


# Function to convert training data from RGB to HSI:
# Reference: See slide 7 of Lecture 5
# By Normilizing RGB first, HSI are in the range of 0 to 1.
def rgb_to_hsi(trainPixels: np.array):
    # Normalize RGB to 0-1 and isolate:
    R = trainPixels[:, 0] / 255.0
    G = trainPixels[:, 1] / 255.0
    B = trainPixels[:, 2] / 255.0

    # Calculate the intensity Value:
    I_arr = (R + G + B) / 3

    # Calculate the Saturation Value:
    min_rgb = np.minimum(np.minimum(R, G), B)
    S_arr = 1 - (3 * min_rgb / ((R + G + B) + 1e-8))

    # Calculate the Hue:
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-8
    theta = np.arccos(np.clip(num / den, -1, 1))        # Clip to gaurantee no Naan values
    # Shift the hue value into the correct location and normalize:
    H_arr = np.where(B <= G, theta, 2*np.pi - theta)
    H_arr = H_arr / (2*np.pi)

    return np.stack((H_arr, S_arr, I_arr), axis=-1)


# Function to generate the 2D HSI Histogram for skin tone detection:
# Note HSI is normaized to 0-1:
def HSI_2DHist(in_arr: np.array):
    # Determine the number of training samples for the HSI 2D Histogram:
    num_pixels = in_arr.shape[0]

    # Seperate the H, S, I values for each pixel with in_arr -> (num_pixels, 3):
    H = in_arr[:, 0]
    S = in_arr[:, 1]

    # Construct the 2D Histogram:
    HS_hist = np.zeros((360, 255))
    for pixel in range(num_pixels):
        HS_hist[np.clip(H[pixel] * 360, 0, 359).astype(int), np.clip(S[pixel] * 255, 0, 254).astype(int)] += 1

    # Normalize the histogram:
    HS_hist /= HS_hist.sum()

    # Apply Gaussian filter to smooth out relation of skin pigments (some bins don't get caught by sample data)
    HS_hist = gaussian_filter(HS_hist, sigma=2)

    return HS_hist


# Function to apply given 2D HS Histogram to segment out skin color:
def segSkin(HS_hist: np.array, img: Image):
    # Convert the input image to HSI and use the HS_hist to determine if a pixel is skin pigment:
    im_arr = np.array(img)
    rows, cols, _ = im_arr.shape
    # Convert the image to a flat array and convert to hsi:
    im_flat = im_arr.reshape(-1, 3)
    im_hsi_flat = rgb_to_hsi(im_flat)     # Note: HSI normalized to 0-1
    # Convert the image back to a matrix (row x col)
    im_hsi_arr = im_hsi_flat.reshape(rows, cols, 3)

    # Loop through the input image in HSI format, and determine if its a skin pigment:
    seg_im_arr = np.zeros((rows, cols))
    for u in range(rows):
        for v in range(cols):
            # Case 1: HS relation has positive relation:
            h = im_hsi_arr[u, v, 0]  # Hue value
            s = im_hsi_arr[u, v, 1]  # Saturation value
            # Map to histogram bin indices (H in [0,360), S in [0,1])
            h_idx = int(np.clip(h * 360, 0, 359))
            s_idx = int(np.clip(s * 255, 0, 254))
            if HS_hist[h_idx, s_idx] > 0:
                seg_im_arr[u, v] = 0      # Segmented color will show up black:
            # Case 2: HS relation does not have a positive relation:
            else:
                seg_im_arr[u, v] = 255      # Non-Segmented color will show up white

    # Convert the segmented image array to an Image:
    seg_im = Image.fromarray(seg_im_arr.astype(np.uint8))
    return seg_im


# Main Function:
def main():
    # Collect Training data:
    joyIm = Image.open("joy1.bmp")
    train1 = collectTrain(joyIm)
    pointIm = Image.open("pointer1.bmp")
    train2 = collectTrain(pointIm)
    gunIm = Image.open("gun1.bmp")
    train3 = collectTrain(gunIm)

    # Concatinate Training Data:
    fullTrain = np.vstack((train1, train2, train3))
    print(f"A total of f{fullTrain.shape[0]} pixels have been collected for training")

    # I have chosen to use HSI for my training step:
    #   Convert all training pixels to HSI format:
    fullTrainHSI = rgb_to_hsi(fullTrain)
    print("The RGB skin pixels selected for training have been converted to HSI")

    # Generate an H-S 2D Color Histogram:
    HS_hist = HSI_2DHist(fullTrainHSI)

    # Plot the 2D Histogram:
    plt.figure(figsize=(8, 6))
    plt.imshow(HS_hist, origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(label='Probability')
    plt.xlabel('Saturation (0-255)')
    plt.ylabel('Hue (0-360)')
    plt.title('2D HS Skin Tone Histogram')
    plt.savefig("HS_Histogram.png")
    plt.close()

    # Apply the HS Histogram to input images to segment out skin color:
    test1 = segSkin(HS_hist, joyIm)
    test1.save("joy_skinsegmented.bmp")
    test2 = segSkin(HS_hist, pointIm)
    test2.save("point_skinsegmented.bmp")
    test3 = segSkin(HS_hist, gunIm)
    test3.save("gun_skinsegmented.bmp")
    print("The Test images have been segmented to isolate skin pigment")


# Main execution:
if __name__ == "__main__":
    main()
