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
    
# Main execution:
if __name__ == "__main__":
    main()
