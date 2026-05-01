"""
Implementation of the Hough Transform for HW5 in MSAI495.
Author: Nolan Knight
Date: 2026-04-30
"""

# Imports:
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import CannyEdgeDetection as Canny
from scipy.ndimage import maximum_filter


# Functions:
# Hough Transformation function using Canny Edge Detection:
def HoughTransfrom(img: Image, N, sigma, percentageOfNonEdge, max_type, name, A_threshold, localMax_Window):
    # Convert image to grayscale:
    img = img.convert("L")       # L-mode converts to grayscale
    orig_array = np.array(img)

    # Apply Canny Edge Detection to the image to detect lines: 
    img = Canny.CED_Hough(img, N, sigma, percentageOfNonEdge, max_type, name)
    img.save(f'{name}_edge_detect.png')

    # Convert the image to an Numpy Array:
    E = np.array(img)
    rows, cols = E.shape

    # Create the Voter Scheme for rho and theta: 
    theta = np.deg2rad(np.arange(0, 180))       # I read that switching to 0 to 180 helps with the boundary issue of near vertical lines.
    # A.size = rho is -img_diag to img_diag, theta = - pi/2 to pi/2:
    img_diag = int(np.sqrt((rows ** 2 + cols ** 2)))
    A = np.zeros((len(theta), 2 * img_diag + 1), dtype=np.int32)

    # Create the Parameter Space by transcribing each line to the mc space:
    for u in range(rows):
        for v in range(cols):
            # For every Edge Pixel vote on polar position:
            if (E[u, v]):
                for theta_idx in range(len(theta)):
                    # Calculate rho and theta: 
                    rho = u * np.cos(theta[theta_idx]) + v * np.sin(theta[theta_idx])
                    # Place the vote for every edge pixel: (diag is added for rho index to be positive)
                    A[theta_idx][round(rho + img_diag)] += 1

    # Find the Local Maxima in A: 
    # Only keep a cell if it is the local maximum in a 20x20 window
    neighbourhood = maximum_filter(A, size=localMax_Window)
    A_nms = (A == neighbourhood) & (A > A_threshold)

    # Get the polar coordinates of the lines:
    theta_idxs, rho_idxs = np.where(A_nms)
    theta_vals = np.rad2deg(theta[theta_idxs])
    rho_vals   = rho_idxs - img_diag
    

    # Plot the Parameter Space and the Signigicant Intersections:
    plt.figure(figsize=(10, 6))
    plt.imshow(
        A,
        cmap='hot',           # bright = high votes
        aspect='auto',
        extent=[
            -img_diag,               # x_min is -img_diag
            img_diag,                # x_max is img_diag
            np.rad2deg(theta[-1]),   # y_min is -90
            np.rad2deg(theta[0])     # y_max is 90
        ]
    )
    # Plot the Significant Intersections:
    plt.scatter(rho_vals, theta_vals, facecolors='none', edgecolors='red', s=300, marker='o', label=f'peaks > {A_threshold}')
    plt.xlabel('rho (pixels)')
    plt.ylabel('theta (degrees)')
    plt.title(f'Hough Parameter Space for {name}')
    plt.colorbar(label='votes')
    plt.savefig(f'{name}_ParameterSpace.png', dpi=300, bbox_inches='tight')

    # Calculate each m and c:
    lines = []
    for rho, t in zip(rho_vals, theta_idxs):
        # Make sure I am not dividing by zero:
        if np.sin(theta[t]) < 1e-10:
            continue
        # Calculate m and c and save as a line:
        m = -np.cos(theta[t]) / np.sin(theta[t])
        c = rho / np.sin(theta[t])
        lines.append((m, c))
    
    # Plot the selected lines on the original image:
    line_img = np.zeros((rows, cols), dtype=np.uint8)
    for m, c in lines:
        for x in range(rows):
            y = int(m * x + c)
            # Set the Line Pixel if its in range:
            if 0 <= y < cols:
                line_img[x, y] = 255

    plt.figure()
    plt.imshow(line_img, cmap='gray')
    plt.title('Detected Lines')
    plt.savefig(f'{name}_lines.png', dpi=300, bbox_inches='tight')

    # Superimpose onto original
    result = np.clip(orig_array.astype(np.int32) + line_img, 0, 255).astype(np.uint8)
    img_out = Image.fromarray(result)
    img_out.save(f"{name}_lines_superimposed.png")
    return


# Main Function:
def main():
    # Perform Hough Transfrom on Sample Images:
    Test1 = Image.open("test.bmp")
    Test1.save("test.png")
    HoughTransfrom(Test1, 5, 2, 0.95, 'i', "Test1", 20, 20)

    Test2 = Image.open("test2.bmp")
    Test2.save("test2.png")
    HoughTransfrom(Test2, 5, 2, 0.95, 'i', "Test2", 20, 20)

    Input= Image.open("input.bmp")
    Input.save("input.png")
    HoughTransfrom(Input, 5, 2, 0.95, 'i', "Input", 30, 20)

    Box = Image.open("boxing.jpg")
    Box.save("input.png")
    HoughTransfrom(Box, 5, 2, 0.95, 'i', "Box", 150, 20)

    Ski = Image.open("ski.jpg")
    Ski.save("ski.png")
    HoughTransfrom(Ski, 5, 2, 0.95, 'i', "Ski", 150, 20)


# Main execution:
if __name__ == "__main__":
    main()
    #   Sample #1: test1.bmp