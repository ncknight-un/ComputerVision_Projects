"""
Implementation of the Canny Edge Detection for HW4 in MSAI495.
Author: Nolan Knight
Date: 2026-04-26
"""

# Imports:
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

import sys 
sys.setrecursionlimit(100000)


# Functions:
# _________________________________________________________________
# Function to apply gaussian smoothing to an image:
#   Paramaters:
#       - img: Input Image
#       - N: Size of gaussian filter kernel
#       - Sigma: STD of Gaussian Filter
def GaussSmoothing(img: Image, N: int, Sigma: float):
    # Convert the image to an Numpy Array:
    img_arr = np.array(img)

    # Create the Gaussian Kernal:
    gkernel = np.zeros((N, N))
    center = N // 2
    for u in range(N):
        for v in range(N):
            # Squared Euclidean Distance / 2 * Sigma squared
            gkernel[u, v] = np.exp(-((u - center)**2 + (v - center)**2) / (2 * Sigma**2))
    # Normalize the kernel:
    gkernel /= gkernel.sum()

    # Apply the Gaussian Filter to the ImagE:
    out_arr = convolve(img_arr, gkernel)

    # Convert the segmented image array to an Image:
    filt_img = Image.fromarray(out_arr.astype(np.uint8))
    return filt_img


# Function to determine magnitude and direction of gradient using Sobel Operator:
#   Paramaters:
#       - img: Input Image
def ImageGradient(img: Image):
    # Convert the image to an Numpy Array:
    img_arr = np.array(img)

    # Create the Sobel Operators:
    Gx = np.zeros((3, 3))
    Gx[0, 0] = -1
    Gx[1, 0] = -2
    Gx[2, 0] = -1
    Gx[0, 2] = 1
    Gx[1, 2] = 2
    Gx[2, 2] = 1
    Gy = np.zeros((3, 3))
    Gy[0, 0] = -1
    Gy[0, 1] = -2
    Gy[0, 2] = -1
    Gy[2, 0] = 1
    Gy[2, 1] = 2
    Gy[2, 2] = 1

    # Apply the Gradient to the Image:
    img_arr = np.array(img).astype(np.float64)
    dx = convolve(img_arr, Gx)
    dy = convolve(img_arr, Gy)

    # Calculate the Magnitude and Direction:
    mag = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    return [mag, theta]


# Function to determine the High and Low Thresholds:
def FindThreshold(Mag, percentageOfNonEdge):
    # Using the Histogram of the image gradient, calculate the thresholds:
    rows, cols = Mag.shape
    num_pixels = rows * cols

    # Normalize mag to 0-100 for threshold binning:
    mag_norm = (Mag / Mag.max() * 255).astype(int)

    #    Build a Histogram of the Original Image:
    H = [0] * 256       # Setting the max of 256 labels
    for u in range(rows):
        for v in range(cols):
            H[mag_norm[u, v]] += 1

    # Build cumulative sum to find the percentile
    sum = 0
    T_high = 0
    # For every magnitude bin, sum its occurance. If we exeed percentageOfNoneEdge, then set T_high
    for i in range(256):
        sum += H[i]
        # Check if the current CDF has hit the threshold:
        if sum / num_pixels >= percentageOfNonEdge:
            T_high = i / 255.0 * Mag.max()  # Set T_high wrt original range
            break

    # Calculate T_Low:
    T_low = 0.5 * T_high

    # Return the Thresholds:
    return [T_low, T_high]


# Function to Suppress Non-Maxima using the LUT Method:
#   Mag - Magnitude matrix of the current Image:
#   Theta - Theta matrix of the current Image:
#   method - 'q' for quantitization method (LUT) or 'i' for interpolation method
def NonmaximaSupress(Mag, Theta, method):
    # Set up the Output Magnitude Matrix:
    rows, cols = Mag.shape
    output = np.zeros((rows, cols))

    # Method 1: Quantitization
    if method == 'q':
        # Using the LUT Method, suppress the Nonmaxima:
        LUT_paths = {
            0:   ((0, -1), (0,  1)),
            45:  ((-1,  1), (1, -1)),
            90:  ((-1,  0), (1,  0)),
            135: ((-1, -1), (1,  1))
        }

        # Quantize angles to nearest 45 degrees
        angles = np.degrees(Theta) % 180
        quant_angles = (np.round(angles / 45) * 45).astype(int) % 180

        # Loop over magnitude matrix, and fill out Ouput Magnitude where ridge is identified
        for u in range(1, rows - 1):
            for v in range(1, cols - 1):
                # Check nearest quantitizes direction of Mag:
                (du1, dv1), (du2, dv2) = LUT_paths[quant_angles[u, v]]
                # Get neighbor cells in direction of quantitization path
                n1 = Mag[u + du1, v + dv1]
                n2 = Mag[u + du2, v + dv2]

                # Keep pixel only if it's a local maximum on that path
                if Mag[u, v] >= n1 and Mag[u, v] >= n2:
                    output[u, v] = Mag[u, v]
    elif method == 'i':
        # Using the Interpolation Method, suppress the Nonmaxima:
        # Loop over the magnitude matrix, and interpolate the values of n1, and n2, based on LUT paths.
        for u in range(1, rows - 1):
            for v in range(1, cols - 1):
                # Calculate the actual interpolation angle:
                angle = Theta[u, v]
                dx = np.cos(angle)
                dy = np.sin(angle)

                # Determine P1 and P2 for each endpoint:
                if abs(dx) > abs(dy):
                    # Gradient more horizontal, so second point is in x direction
                    P1a = Mag[u, v + int(np.sign(dx))]                          # First Point right
                    P1b = Mag[u + int(np.sign(dy)), v + int(np.sign(dx))]       # Corner Point
                    P2a = Mag[u, v - int(np.sign(dx))]                          # Second Point left
                    P2b = Mag[u - int(np.sign(dy)), v - int(np.sign(dx))]       # Corner Point

                    # Calculate Alpha based on quadrant and slope: 
                    if (int(np.sign(dx)) > 0):
                        # Negative slope - Case 1 on slide 18
                        alpha = 1 - abs(np.tan(angle))
                        alpha = np.clip(alpha, 0.0, 1.0)
                    else:
                        # Positive slope - Case 4 on Slide 18
                        alpha = abs(np.tan(angle))
                        alpha = np.clip(alpha, 0.0, 1.0)
                else:
                    # Gradient more vertical, so second point is in y direction
                    P1a = Mag[u - int(np.sign(dy)), v]                          # First point above
                    P1b = Mag[u - int(np.sign(dy)), v - int(np.sign(dx))]       # Corner Point
                    P2a = Mag[u + int(np.sign(dy)), v]                          # Second Point below
                    P2b = Mag[u + int(np.sign(dy)), v + int(np.sign(dx))]       # Corner Point

                    # Calculate Alpha based on quadrant and slope: 
                    if (int(np.sign(dy)) > 0):
                        # Negative slope - Case 2 on slide 18
                        alpha = abs(1 / np.tan(angle))
                        alpha = np.clip(alpha, 0.0, 1.0)
                    else:
                        # Positive slope - Case 3 on Slide 18
                        alpha = 1 - abs(1 / np.tan(angle))
                        alpha = np.clip(alpha, 0.0, 1.0)

                # Interpolate in the gradient direction along the path:
                n1 = alpha * P1a + (1 - alpha) * P1b
                n2 = alpha * P2a + (1 - alpha) * P2b

                # Keep pixel only if it's a local maximum on that path
                if Mag[u, v] >= n1 and Mag[u, v] >= n2:
                    output[u, v] = Mag[u, v]
    # Return the updated magnitudes with Local Maxima conserved:
    return output


# Function to Suppress Non-Maxima using the LUT Method:
#   Mag - Magnitude matrix of the current Image after Maxima Suppression
#   T_low - Lower threshold
#   T_high - Higher threshold
#   (trace_strong) - internal function to loop through strong edges
#   (trace_weak)   - internal function to loop through weak edges and connect to strong edges
def EdgeLinking(Mag, T_low, T_high):
    # Set up the Output Magnitude Matrix:
    rows, cols = Mag.shape
    E = np.zeros((rows, cols))
    # Set up matrix to check whether strong edge has been checked yet
    visited = np.zeros((rows, cols), dtype=bool)

    # Threshold Mag by High and Low:
    Mag_low = (Mag >= T_low) & (Mag < T_high)
    Mag_high = Mag >= T_high

    # Recursively Link the Strong edges and fill in gaps with the weak edges:
    def trace_weak(u, v):
        # Strong endpoint reaches, so begin connecting weak edge:
        visited[u, v] = True
        E[u, v] = 255  # Connect weak edge to strong edge
        # Check current edges neighbors:
        for du in [-1, 0, 1]:
            for dv in [-1, 0, 1]:
                if du == 0 and dv == 0:
                    continue
                # Update current location:
                nu, nv = u + du, v + dv
                # verify that we are within the image dimensions and the point has not been visited
                if 0 <= nu < rows and 0 <= nv < cols and not visited[nu, nv]:
                    # Check to see if we have hit a new strong edge or an endpoint for weak edges
                    if (Mag_high[nu, nv] > 0):
                        # Strong edge or endpoint reached — return and continue strong recursion
                        continue
                    elif (Mag_low[nu, nv] > 0):
                        # No strong edge, but weak edge is continued:
                        trace_weak(nu, nv)

    def trace_strong(u, v):
        # Set that the current pixel has been visited:
        visited[u, v] = True
        E[u, v] = 255
        # Loop through the 8 neighbors and recursively connect strong edges:
        for du in [-1, 0, 1]:
            for dv in [-1, 0, 1]:
                if du == 0 and dv == 0:
                    continue
                # Update current location:
                nu, nv = u + du, v + dv
                # verify that we are within the image dimensions and the point has not been visited
                if 0 <= nu < rows and 0 <= nv < cols and not visited[nu, nv]:
                    # If it hasn't been visited and its strong, then continue strong recursion
                    if Mag_high[nu, nv] > 0:
                        trace_strong(nu, nv)
                    # If the current edge does not have another strong edge, recurse through weak edges
                    elif Mag_low[nu, nv] > 0:
                        trace_weak(nu, nv)
                    # Endpoint reached, end recursion!
                    else:
                        return

    # Loop through the Strong Edges matrix, and if its an endpoint connect with weak edge:
    for u in range(rows):
        for v in range(cols):
            # Verify that it has not been visited, otherwise we will repeat recursion:
            if (Mag_high[u, v] > 0) and not visited[u, v]:
                E[u, v] = 255
                # Begin strong recursion:
                trace_strong(u, v)
    return E


# Function to perform operation of Canny Edge Detection in order:
#   img: Input Image
#   N: Gaussian Filter Size
#   sigma: STD of Gaussian Filter
#   percentageOfNonEdge: Percentage of None Edges expected in Image
#   name: name to save files to
def CannyEdgeDetection(img: Image, N, sigma, percentageOfNonEdge, max_type, name):
    # Convert image to grayscale:
    img = img.convert("L")       # L-mode converts to grayscale
    # Save Filtered Image:
    img = GaussSmoothing(img, N, sigma)
    img.save(f"{name}_filtered.png")
    # Get the Gradient's Magnitude and Direction using Sobel Operator:
    img_mag, img_dir = ImageGradient(img)
    # Visualize magnitude of Gradient:
    plt.imshow(img_mag, cmap='gray')
    plt.savefig(f"{name}_mag.png")
    # Calculate the Thresholds:
    img_low, img_high = FindThreshold(img_mag, percentageOfNonEdge)
    # Suppress the Non Maxima:
    img_mag_max = NonmaximaSupress(img_mag, img_dir, max_type)
    # Visualize magnitude of Gradient:
    plt.imshow(img_mag_max, cmap='gray')
    plt.savefig(f"{name}_mag_maxima.png")
    # Perform Recursive Edge Linking:
    img_Linked = EdgeLinking(img_mag_max, img_low, img_high)
    # Convert to an image:
    img = Image.fromarray(img_Linked.astype(np.uint8))
    img.save(f"{name}_linked.png")
    return


# Main Function:
def main():
    # Perform Edge Detection on Sample Images:
    #   Sample #1: test1.bmp
    Test1 = Image.open("test1.bmp")
    CannyEdgeDetection(Test1, 5, 2, 0.85, 'i', "Test1")
    #   Sample #1: test1.bmp
    Joy1 = Image.open("joy1.bmp")
    CannyEdgeDetection(Joy1, 5, 2, 0.85, 'i', "Joy1")
    #   Sample #1: test1.bmp
    Pointer1 = Image.open("pointer1.bmp")
    CannyEdgeDetection(Pointer1, 5, 2, 0.85, 'i', "Pointer1")
    #   Sample #1: lena.bmps
    Lena = Image.open("lena.bmp")
    CannyEdgeDetection(Lena, 5, 2, 0.85, 'i', "Lena")

    # Test to see how different Gaussian filter afftects edge detection:
    Lena = Image.open("lena.bmp")
    CannyEdgeDetection(Lena, 10, 3, 0.85, 'i', "Lena_GF_10_3")
    Lena = Image.open("lena.bmp")
    CannyEdgeDetection(Lena, 3, 1, 0.85, 'i', "Lena_GF_3_1")

    # Test to see how different PercentageofNoneEdge affects edge detection:
    Test1 = Image.open("test1.bmp")
    CannyEdgeDetection(Test1, 5, 2, 0.95, 'i', "Test1_PONE_95")
    Test1 = Image.open("test1.bmp")
    CannyEdgeDetection(Test1, 5, 2, 0.8, 'i', "Test1_PONE_80")

    # Test to see how different Maxima Suppression Types affect detection:
    Lena = Image.open("lena.bmp")
    CannyEdgeDetection(Lena, 5, 2, 0.85, 'q', "Lena_quant")


# Main execution:
if __name__ == "__main__":
    main()
