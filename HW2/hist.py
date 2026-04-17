"""
Implementation of the Histogram Equalization algorithm for HW2 in MSAI495.
Author: Nolan Knight
Date: 2026-04-13
"""

# Imports:
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Functions:
def LinearLightCorrection(in_img: Image):
    # Implementation of Linear Light Correction on a grayscale image:
    # Convert the image to an array:
    in_array = np.array(in_img)
    rows, cols = in_array.shape
    num_pixels = rows * cols

    # Build the A Matrix:
    A = np.zeros((num_pixels, 3))

    for u in range(rows):
        for v in range(cols):
            idx = u * cols + v
            # Fill each row:
            A[idx, 0] = u
            A[idx, 1] = v
            A[idx, 2] = 1

    # Build the y vector (Array of each pixel value):
    y = np.zeros(num_pixels)
    for u in range(rows):
        for v in range(cols):
            idx = u * cols + v
            y[idx] = in_array[u, v]

    # Solve for state vector using pseudo inverse:
    theta = np.linalg.pinv(A) @ y

    # Build light map
    light_map = np.zeros((rows, cols))
    for u in range(rows):
        for v in range(cols):
            light_map[u, v] = theta[0]*u + theta[1]*v + theta[2]

    light_map_img = Image.fromarray(light_map.astype(np.uint8))
    light_map_img.save("linear_lightmap.bmp")

    # Avoid divide-by-zero from linear map:
    light_map = np.maximum(light_map, 1e-6)

    # Apply the light map, and renormalize range from 0 to 255:
    out_array = in_array / light_map
    out_array = out_array - out_array.min()
    out_array = out_array / out_array.max() * 255

    # Convert back to an image:
    out_img = Image.fromarray(out_array.astype(np.uint8))
    return out_img


def QuadLightCorrection(in_img: Image):
    # Implementation of Quadratic Light Correction on a grayscale image:
    # Convert the image to an array:
    in_array = np.array(in_img)
    rows, cols = in_array.shape
    num_pixels = rows * cols

    # Build the A Matrix:
    A = np.zeros((num_pixels, 6))

    for u in range(rows):
        for v in range(cols):
            idx = u * cols + v
            # Fill each row:
            A[idx, 0] = u * u
            A[idx, 1] = u * v
            A[idx, 2] = v * v
            A[idx, 3] = u
            A[idx, 4] = v
            A[idx, 5] = 1

    # Build the y vector (Array of each pixel value):
    y = np.zeros(num_pixels)
    for u in range(rows):
        for v in range(cols):
            idx = u * cols + v
            y[idx] = in_array[u, v]

    # Solve for state vector using pseudo inverse:
    theta = np.linalg.pinv(A) @ y

    # Build light map
    light_map = np.zeros((rows, cols))
    for u in range(rows):
        for v in range(cols):
            light_map[u, v] = theta[0]*u*u + theta[1]*u*v + theta[2]*v*v + theta[3]*u + theta[4]*v + theta[5]

    light_map_img = Image.fromarray(light_map.astype(np.uint8))
    light_map_img.save("quad_lightmap.bmp")

    # Avoid divide-by-zero from quadratic map:
    light_map = np.maximum(light_map, 1e-6)

    # Apply the light map, and renormalize range from 0 to 255:
    out_array = in_array / light_map
    out_array = out_array - out_array.min()
    out_array = out_array / out_array.max() * 255

    # Convert back to an image:
    out_img = Image.fromarray(out_array.astype(np.uint8))
    return out_img


def HistoEqualization(in_img: Image):
    # Convert the 24-bit Bitmap to a gray-scale image:
    im_gray = in_img.convert("L")       # L-mode converts to grayscale

    # Implementation of Histogram Equalization Algorithm:
    # Convert the image to an array and determine number of columns and rows:
    in_array = np.array(im_gray)
    rows, cols = in_array.shape
    num_pixels = rows * cols

    # Build a Histogram of the Original Image:
    H = [0] * 256       # Setting the max of 256 labels
    for u in range(rows): 
        for v in range(cols): 
            H[in_array[u, v]] += 1

    # Generate the histogram for the original image:
    plt.bar(range(len(H)), H)
    plt.title('Original Image - Histogram')
    plt.xlabel('Pixel Value (0-255)')
    plt.ylabel('Frequency Count')
    plt.savefig("orig_hist_moon.png")
    plt.close()

    # Generate the CDF for each intensity for the Mapping Function:
    CDF = [0] * 256
    running_sum = 0 

    for i in range(len(H)):
        running_sum += H[i]
        CDF[i] = running_sum
    # Get the probability by dividing by the total number of pixels:
    for i in range(len(CDF)):
        CDF[i] = CDF[i] / num_pixels

    plt.plot(CDF)
    plt.title('Original Image - CDF')
    plt.xlabel('Pixel Value (0-255)')
    plt.ylabel('Cummalitive Frequency')
    plt.savefig("orig_cdf_moon.png")
    plt.close()

    # Update the output image array based on the Histogram Equalization Function:
    out_array = in_array.copy()
    s_max = 255
    for u in range(rows):
        for v in range(cols):
            # Update each pixel given its current intensity with the eqn on Slide 6 of lecture: 
            # (Cum_dist of intensity) * L2 (255)
            out_array[u, v] = s_max * CDF[in_array[u, v]]

    # Generate the histogram for the equalized image:
    H_e = [0] * 256       # Setting the max of 256 labels
    for u in range(rows): 
        for v in range(cols): 
            H_e[out_array[u, v]] += 1

    # Plot the equalized histogram:
    plt.bar(range(len(H_e)), H_e)
    plt.title('Equalized Image - Histogram')
    plt.xlabel('Pixel Value (0-255)')
    plt.ylabel('Frequency Count')
    plt.savefig("equalized_hist_moon.png")
    plt.close()

    CDF = [0] * 256
    running_sum = 0 

    # Calculate the running_sum for each intensity:
    for i in range(len(H_e)):
        running_sum += H_e[i]
        CDF[i] = running_sum
    # Get the probability by dividing by the total number of pixels:
    for i in range(len(CDF)):
        CDF[i] = CDF[i] / num_pixels

    plt.plot(CDF)
    plt.title('Equalized Image - CDF')
    plt.xlabel('Pixel Value (0-255)')
    plt.ylabel('Cummalitive Frequency')
    plt.savefig("equalized_cdf_moon.png")
    plt.close()

    # Convert the out_array to an image:
    out_img = Image.fromarray(out_array.astype(np.uint8))
    out_img.show()
    return out_img


# Main Function:
def main():
    # Perform Histogram Equalization on Moon Image: 
    test_im = Image.open("moon.bmp")
    test_im = HistoEqualization(test_im)
    test_im.save("moon_equalized.bmp")
    # Apply Linear Correction:
    test_im_LC = LinearLightCorrection(test_im)
    test_im_LC.save("moon_eq_linear.bmp")
    # Apply Quadratic Correction:
    test_im_QC = QuadLightCorrection(test_im)
    test_im_QC.save("moon_eq_quad.bmp")


# Main execution:
if __name__ == "__main__":
    main()
