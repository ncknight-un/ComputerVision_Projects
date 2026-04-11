"""
Implementation of the CCL (Connected Component Labeling) algorithm for HW1 in MSAI495.
Author: Nolan Knight
Date: 2026-04-11
"""

# Imports:
from PIL import Image
import numpy as np


# Functions:
def CCL(img: Image):
    # Use the CCL Algorithm on the input image and output the result and number of groups:
    # Convert the image to an array:
    im_arr = np.asarray(img)

    # Create the labeled_image array
    rows, cols = im_arr.shape
    L = np.zeros((rows, cols), dtype=int)

    # Create Array to hold number of labels
    E_Table = []
    num = 0

    # Loop through each cell starting in the first row, and loop through the row and apply CCL:
    L_u = 0
    L_l = 0
    for u in range(rows):      # u - num of rows
        for v in range(cols):  # v - num of columns
            if im_arr[u, v] == 1:
                # Determine the value of upper and left label:
                if u == 0 and v == 0:
                    L_u = 0
                    L_l = 0
                elif u == 0:
                    L_u = 0
                    L_l = L[u, v - 1]
                elif v == 0:
                    L_u = L[u - 1, v]
                    L_l = 0
                else:
                    L_u = L[u - 1, v]
                    L_l = L[u, v - 1]

                # Case 1: Upper and Left have the same label:
                if L_u == L_l and L_u != 0 and L_l != 0:
                    L[u, v] = L_u
                # Case 2: Atleast 1 is labeled:
                elif (L_u == 0 and L_l != 0) or (L_u != 0 and L_l == 0):
                    L[u, v] = max(L_u, L_l)
                # Case 3: Both have a label, take the lowest label:
                elif L_u != L_l and L_u > 0 and L_l > 0:
                    L[u, v] = min(L_u, L_l)
                    # Store Equivalence Pair:
                    E_Table.append((max(L_u, L_l), min(L_u, L_l)))
                # Case 4: Neither upper or left are labeled, create new group
                else:
                    L[u, v] = num + 1
                    num = num + 1

    # Create a map for the highest parent to the lowest label index:
    parent_map = {}
    for max_i, min_i in E_Table:
        # Get the root of both the min and max index labels:
        root_min = min_i
        while root_min in parent_map:
            root_min = parent_map[root_min]

        root_max = max_i
        while root_max in parent_map:
            root_max = parent_map[root_max]

        # Update the parent map to have max point lead to the smallest known label:
        if max_i in parent_map:
            parent_map[max_i] = min(parent_map[max_i], min_i)
        else:
            parent_map[max_i] = min_i

        # Link the two roots together if they are different since E-group is built on similiar groups:
        if root_min != root_max:
            parent_map[max(root_min, root_max)] = min(root_min, root_max)

    # Recursively map the highest index to the lowest using the parent mapping:
    lowest_map = {}
    for index in set(np.unique(L)):
        parent = index
        while parent in parent_map:
            parent = parent_map[parent]
        lowest_map[index] = parent

    # Loop back through the Labeled Array and correct the labels based on the E_Table:
    for u in range(rows):
        for v in range(cols):
            # If the pixel has a label, make sure it is the lowest mapping:
            if L[u, v] != 0:
                L[u, v] = lowest_map[L[u, v]]

    # Get the updated number of groups:
    updated_num = len(np.unique(L)) - 1      # Ignore no labeled values

    # Shift the color scale for the groups to show visual differences:
    L = 4*(updated_num)*L     # Custom scalling to shift color groups for visualization

    # Convert the array back to an image and return the new CCL image and number of labels:
    label_img = Image.fromarray(L.astype(np.uint8))
    return label_img, updated_num


def main():
    # Perform CCL on the Test Image:
    test_im = Image.open("test.bmp")
    labeled_Test_im, Test_num = CCL(test_im)
    labeled_Test_im.show()
    labeled_Test_im.save("test_labeled.bmp")
    print("Number of labels for test.bmp: " + str(Test_num))

    # Perform CCL on the Face Image:
    face_im = Image.open("face.bmp")
    labeled_Face_im, Face_num = CCL(face_im)
    labeled_Face_im.show()
    labeled_Face_im.save("face_labeled.bmp")
    print("Number of labels for face.bmp: " + str(Face_num))

    # Perform CCL on the Gun Image: 
    gun_im = Image.open("gun.bmp")
    labeled_Gun_im, Gun_num = CCL(gun_im)
    labeled_Gun_im.show()
    labeled_Gun_im.save("gun_labeled.bmp")
    print("Number of labels for gun.bmp: " + str(Gun_num))


# Main execution:
if __name__ == "__main__":
    main()
