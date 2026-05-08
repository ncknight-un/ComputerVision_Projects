"""
Implementation of the Object Tracking for HW6 in MSAI495.
Author: Nolan Knight
Date: 2026-05-07
"""

# Imports:
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2      # Used only for drawing bounding box on final image pass and creating video

# Functions:
# _______________________________________________________________________________________________
# Function to perform Sum of Squared Differences
# Cand: Candidate Region
# Temp: Template Image
def ssd(Temp, Cand):
    # Convert Image Arrays to Floats:
    I = Cand.astype(np.float64)
    T = Temp.astype(np.float64)

    # Calculate the difference between the two regions:
    diff = I - T

    # Return the least squares difference:
    return np.sum(diff ** 2)

# Function to perform Cross Correlation on Image and Template:
def cc(Temp, Cand):
    # Convert Image Arrays to Floats:
    I = Cand.astype(np.float64)
    T = Temp.astype(np.float64)

    # Return the Sum of the product of I and T:
    return np.sum(I * T)

# Function to Perform Normalized Cross-Correlation:
def ncc(template, candidate):
    # Convert Image arrays to Floats:
    I = candidate.astype(np.float64)
    T = template.astype(np.float64)

    # Calculate the I_hat and T_hat by subtracting the Mean of the Template and Candidate Regions:
    I_hat = I - np.mean(I)
    T_hat = T - np.mean(T)

    # Calculate the Sum of the produce of I_hat and T_hat:
    n = np.sum(I_hat * T_hat)
    # Calculate the sqrt of the product of Sums of I_hat and T_hat squared:
    d = np.sqrt(np.sum(I_hat ** 2) * np.sum(T_hat ** 2))

    if d == 0:
        return 0.0            # avoids Naan

    return (n / d)

# Function that uses OpenCV selectROI to return a cropped target image of the target in image:
#   img: The initial image to select the target to track
def selectTarget(img: Image):
    # Show the image, so the user can select the Face to track:
    img_arr = np.array(img)
    plt.imshow(img_arr)
    plt.title("Click top-left then bottom-right to select face to track")
    plt.show(block=False)

    # Select an upper left and lower right corner to isolate skin:
    points = plt.ginput(2)

    # Extract the x and y positions in the image to crop out skin pixels:
    xmin, xmax = sorted([int(points[0][0]), int(points[1][0])])
    ymin, ymax = sorted([int(points[0][1]), int(points[1][1])])

    # Crop the region and reshape to a list of pixels:
    target = img_arr[ymin:ymax, xmin:xmax]

    # Convert the Target back to an image:
    target_img = Image.fromarray(target.astype(np.uint8))
    target_img.save("Target_image.jpg")

    # Determine the initial search window center: 
    x_wind = int((xmax + xmin) / 2)
    y_wind = int((ymax + ymin) / 2)

    # Return Target Region and location in original image:
    return target_img, x_wind, y_wind

# Object Tracking Function: 
#   cand        : The full frame to search within.
#   target      : The cropped template to locate, as returned by selectTarget()
#   x_wind      : X coordinate of the center of the search window in cand.
#   y_wind      : Y coordinate of the center of the search window in cand.
#   wind_size   : The half width/height window size to perform exaustive search within from previous frame
#   match_method: The match method chosen to perform matching. (s - ssd, c - cc, n - ncc)
def trackObject(cand: Image, target: Image, x_wind: int, y_wind: int, wind_size: int, match_method: str):
    # Get the size of the incoming image: 
    cand_arr = np.array(cand)
    rows, cols = cand_arr.shape[:2]

    # Get the size of the target region:
    targ_arr = np.array(target)
    targ_row, targ_col = targ_arr.shape[:2]

    # Loop through the window centered at previous target center:
    # Determine xmin, xmax, ymin, ymax:
    xmin = np.clip(x_wind - wind_size, 0, cols)
    xmax = np.clip(x_wind + wind_size, 0, cols)
    ymin = np.clip(y_wind - wind_size, 0, rows)
    ymax = np.clip(y_wind + wind_size, 0, rows)

    # Initialize Best Candiate Trackers to current center:
    best_u = x_wind - targ_col // 2
    best_v = y_wind - targ_row // 2

    # Initialize Trackers for match_type best loss values:
    best_ssd = float('inf')
    best_cc = float('-inf')

    for u in range(xmin, xmax):
        for v in range(ymin, ymax):
            targ_x = np.clip(u + targ_col, 0, cols)
            targ_y = np.clip(v + targ_row, 0, rows)
            # Select the candidate region:
            cand_sel = cand_arr[v:targ_y, u:targ_x]

            # Skip if patch is wrong size (happens at edges of frame)
            if cand_sel.shape[:2] != targ_arr.shape[:2]:
                continue
                
            # Use the match_method selected: 
            if (match_method == 's'):
                # Least Squared Difference: 
                ssd_val = ssd(cand_sel, targ_arr)

                # Update the best SSD Region:
                if(ssd_val < best_ssd):
                    best_ssd = ssd_val
                    best_u = u
                    best_v = v

            if (match_method == 'c'): 
                # Cross Correlation:
                cc_val = cc(cand_sel, targ_arr)

                # Update the Best CC Region:
                if (cc_val > best_cc):
                    best_cc = cc_val
                    best_u = u
                    best_v = v

            if (match_method == 'n'):
                # Normalized Cross Correlation:
                ncc_val = ncc(cand_sel, targ_arr)

                # Update the Best CC Region:
                if (ncc_val > best_cc):
                    best_cc = ncc_val
                    best_u = u
                    best_v = v

    # Now that the best Candidate has been Determined, Return the Window Center of the candidate:
    x_cand_wind = int(best_u + (targ_col / 2))
    y_cand_wind = int(best_v + (targ_row / 2))

    # Return the updated target center for the current frame:
    return x_cand_wind, y_cand_wind

# Function to Loop through the images in folder, and return updated image with tracked window for each frame:
#   num_im      : The number of images to pull in and locate the target within.
#   wind_size   : The size of the window to scan over for each update.
#   match_method: The method to use in matching.
def track_loop(num_im: int, wind_size: int, match_method: str):
    # Import the first image and select the template:
    track_template = Image.open("image_girl/0001.jpg")
    orig_arr = np.array(track_template)
    # Extract Original image size: 
    r_orig, c_orig = orig_arr.shape[:2]

    # Identify the Target from the template image:
    target, x_wind, y_wind = selectTarget(track_template)
    # Extract target Size:
    targ_arr = np.array(target)
    r_targ, c_targ = targ_arr.shape[:2]

    # Define the codec and create VideoWriter object to create the final video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'output_tracking_{match_method}.mp4', fourcc, fps=30, frameSize=(c_orig, r_orig))

    for idx in range(1, num_im):
        # Get the path to the current frame:
        if(idx < 10):
            path = f"image_girl/000{idx}.jpg"
        elif (idx < 100):
            path = f"image_girl/00{idx}.jpg"
        else:
            path = f"image_girl/0{idx}.jpg"
        
        # Extract the current frame:
        curr_frame = Image.open(path)

        # Get the tracking window to the current frame:
        x_wind, y_wind = trackObject(curr_frame, target, x_wind, y_wind, wind_size, match_method)

        # Update the Target Window with the new target frame each update to adjust for changes in shadowing and lighting: 
        frame_arr = np.array(curr_frame)

        # Add the Tracking Window to the Current Frame, and save the image to a new folder using cv2 rectangle:
        # Calculate Circle Radias 
        rad = int(np.sqrt(c_targ**2 + r_targ**2) / 2)
        # Use cv2.circle to add the bounding box from the center determined in trackObject()
        cv2.circle(frame_arr, (x_wind, y_wind), rad, color=(255, 0, 0), thickness=1)

        # Convert the frame to BGR and add to output video:
        frame_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # Video is finished being edited, return video object: 
    out.release()


# Main Function:
def main():
    # Call the track_loop function and create the movie with the target window updated in each image:
    # Match Method Options: (s - ssd, c - cc, n - ncc)
    track_loop(500, 60, 'n')
    

# Main execution:
if __name__ == "__main__":
    main()