import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

#TODO create a config file or something like that for the global variables. For example the image size - it is const

def show_image_plt(img_rgb, cmap=None):
    if cmap == 'gray':
        plt.imshow(img_rgb, cmap=cmap)
    else:
        plt.imshow(img_rgb)
    plt.show()


def resize_to_square(vegi_bgr):
    if vegi_bgr.shape[0] == 256 and vegi_bgr.shape[1] == 256:
        return vegi_bgr
    
    img_size = 256
    height, width = vegi_bgr.shape[:2]
    a1 = width / height
    a2 = height / width

    if (a1 > a2):
        r_img = cv2.resize(vegi_bgr, (round(img_size * a1), img_size), interpolation = cv2.INTER_AREA)
        margin = int(r_img.shape[1]/6)
        resized_img = r_img[0:img_size, margin:(margin+img_size)]

    elif(a1 < a2):
        # if height greater than width
        r_img = cv2.resize(vegi_bgr, (img_size, round(img_size * a2)), interpolation = cv2.INTER_AREA)
        margin = int(r_img.shape[0]/6)
        resized_img = r_img[margin:(margin+img_size), 0:img_size]

    elif(a1 == a2):
        # if height and width are equal
        r_img = cv2.resize(vegi_bgr, (img_size, round(img_size * a2)), interpolation = cv2.INTER_AREA)
        resized_img = r_img[0:img_size, 0:img_size]

    if(resized_img.shape[0] != img_size or resized_img.shape[1] != img_size):
        resized_img = r_img[0:img_size, 0:img_size]

    return resized_img


def get_obj_contour(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv)

    blurred_sat = cv2.GaussianBlur(saturation, (3, 3), 0)

    # Compute the threshold dynamically
    thresh = blurred_sat.mean()
    std = blurred_sat.std()
    thresh_low = thresh - std
    thresh_high = thresh + std

    # The next four lines control how good the bounding box will fit
    edges = cv2.Canny(blurred_sat, thresh_low, thresh_high)
    kernel = np.ones((4, 4), np.uint8) # creates 4x4 Identity matrix
    dilate = cv2.dilate(edges, kernel, iterations=4)
    erode = cv2.erode(dilate, kernel, iterations=4)

    contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # get only the biggest contour, asume that the biggest contour is the vegi
    return sorted(contours, key=cv2.contourArea, reverse=True)[0]

def get_circularity(contour):
    area = cv2.contourArea(contour)
    # Calculate Circularity
    perimeter = cv2.arcLength(contour, True)
    r_circle = perimeter / (2 * np.pi)
    A_circle = r_circle**2 * np.pi
    circularity = area / A_circle
    return circularity

def get_size_box(box):
    x0 = box[0][0]
    y0 = box[0][1]
    x1 = box[1][0]
    y1 = box[1][1]

    x2 = box[2][0]
    y2 = box[2][1]

    l0_1 = round(((x0 - x1)**2 + (y0 - y1)**2)**0.5, 2)
    l1_2 = round(((x1 - x2)**2 + (y1 - y2)**2)**0.5, 2)

    w = min(l0_1, l1_2)
    h = max(l0_1, l1_2)
    return h, w

def is_box_rotated(box):
    # If the box is not rotated then the top left corner should be the first element in box array
    x0, y0 = box[0][0], box[0][1]
    y1 = box[1][1]
    x3 = box[3][0]
    
    if y0 == y1 and x0 == x3:
        # box is not rotated
        return False
    # BOX IS ROTATED
    return True

def get_color(rgb_segment):
    cropped_vegi_2D = rgb_segment.reshape((-1,3))
    # convert to np.float32
    cropped_vegi_2D = np.float32(cropped_vegi_2D)

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    #returns center in rgb format
    _, _, center = cv2.kmeans(cropped_vegi_2D, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return np.uint8(center)

def mask_green(cropped_vegi_seg_rgb, lower_thresh=(30, 175, 25), higher_thresh=(100, 255, 255)):
    ## Convert to HSV
    cropped_vegi_seg_hsv = cv2.cvtColor(cropped_vegi_seg_rgb, cv2.COLOR_RGB2HSV)

    ## Mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask = cv2.inRange(cropped_vegi_seg_hsv, lower_thresh, higher_thresh)
    
    ## Slice the green
    imask = mask>0
    green_rgb = np.zeros_like(cropped_vegi_seg_rgb, np.uint8)
    green_rgb[imask] = cropped_vegi_seg_rgb[imask]
    return green_rgb, imask

    #Image as BGR
def segment_img_2(cropped_vegi_bgr):
    #img must be BGR
    gray = cv2.cvtColor(cropped_vegi_bgr, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
    return thresh

def segment_img_3(cropped_vegi_rgb):
    gray = cv2.cvtColor(cropped_vegi_rgb, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur to reduce noise and smooth the image - but there is the chance
    # that the filter removes the small tribes from the potatos
    #blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # salt and pepper filter
    #blurred = cv2.medianBlur(gray, 5)

    # Apply adaptive thresholding to obtain a binary image
    thresh = cv2.adaptiveThreshold(gray, 255, 
                                   cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 
                                   81, 
                                   6)
    return thresh

def color_from_segmented_binary(seg_bin, cropped_vegi_bgr):
    imask = seg_bin>0 #False / True array
    segment = np.zeros_like(cropped_vegi_bgr, np.uint8)
    segment[imask] = cropped_vegi_bgr[imask] #BGR

    segment_rgb = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB) #RGB
    return segment_rgb


def count_green_pixels(binary_green_mask):
    #only count True boolean. These are my green pixels
    return binary_green_mask.sum()
    

def sift(img_bgr):
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # 256 x 256
    sift = cv2.SIFT_create(65536)
    kp, _ = sift.detectAndCompute(gray_img, None)
    #img = cv2.drawKeypoints(gray_img, kp, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    total = 0
    for key in kp:
        total += key.size
    if kp:
        mean = total / len(kp)
    else:
        mean = 0
    return len(kp), mean


def crop_roi(img_bgr, box):
    left_point, top_point = np.min(box, axis=0) # left point = x coordiante, top point y coordinate could be from different points ! 
    right_point, bottom_point = np.max(box, axis=0)
   
    #new width and height
    if top_point < 0:
        top_point = 0

    if left_point < 0:
        left_point = 0

    if right_point > 256:
        right_point = 256

    if bottom_point > 256:
        bottom_point = 256

    h_new = bottom_point - top_point #rows
    w_new = right_point - left_point #cols

    new_top_left_point = np.array([left_point, top_point])

    new_crop = img_bgr[new_top_left_point[1]:new_top_left_point[1]+h_new, new_top_left_point[0]:new_top_left_point[0]+w_new]
    return new_crop


######################### REDRAW BOUNDING BOX #########################
def recalculate_bounding_box(box):
    """
    If there are any edge points outside the image boundaries, then recalculate these edge points.
    Depending on which points have been recalculated, the neighboring points are also moved in parallel
    :param box: The bounding box which contains the edge points
    :type box: 2D np.array
    :return box: The recalculated bounding box with new edge points within the image boundaries
    """
    point_at_idx = 0
    debug_counter = 0
    while (box < 0).any() or (box > 255).any():
        # if the case exists that 4 iterations are not enough to get a box within the image boundaries
        if point_at_idx == 4:
            point_at_idx = 0 
        
        # get the vectors of the next and previous points in the box
        if point_at_idx < 3:
            next_idx = point_at_idx + 1
            prev_idx = point_at_idx - 1
            v_next_point = box[next_idx] - box[point_at_idx]
            v_prev_point = box[prev_idx] - box[point_at_idx]
            p_next = box[next_idx]
            p_prev = box[prev_idx]
        else:
            # extra case for point_at_idx = 3, to prevent of out of index error
            next_idx = 0
            prev_idx = 2
            v_next_point = box[next_idx] - box[point_at_idx]
            v_prev_point = box[prev_idx] - box[point_at_idx]
            p_next = box[next_idx]
            p_prev = box[prev_idx]

        # the if cases are necessary because of the boundary conditions (left, top, down, right)
        # check if x cooridnate is left from the image
        if box[point_at_idx][0] < 0:
            box = recalculate_edge_from_box(box=box, 
                                            idxs=(point_at_idx, next_idx, prev_idx),
                                            v_to_neighbour_points=(v_next_point, v_prev_point),
                                            neighbour_points=(p_next, p_prev), 
                                            boundary=(0, None))
            
        # check if y coordinate of point is above the image
        elif box[point_at_idx][1] < 0:
            # set y to 0 and calculate new x
            box = recalculate_edge_from_box(box=box, 
                                            idxs=(point_at_idx, next_idx, prev_idx),
                                            v_to_neighbour_points=(v_next_point, v_prev_point),
                                            neighbour_points=(p_next, p_prev), 
                                            boundary=(None, 0))
            
        # check if x coordinate is right from the image
        elif box[point_at_idx][0] > 255:
            # set x to 255 and calculate new y
            box = recalculate_edge_from_box(box=box, 
                                            idxs=(point_at_idx, next_idx, prev_idx),
                                            v_to_neighbour_points=(v_next_point, v_prev_point),
                                            neighbour_points=(p_next, p_prev), 
                                            boundary=(255, None))
            
        # check if y coordinate of point is below the image
        elif box[point_at_idx][1] > 255:
            # set y to 255 and calculate new x
            box = recalculate_edge_from_box(box=box, 
                                            idxs=(point_at_idx, next_idx, prev_idx),
                                            v_to_neighbour_points=(v_next_point, v_prev_point),
                                            neighbour_points=(p_next, p_prev), 
                                            boundary=(None, 255))
        point_at_idx += 1
        debug_counter += 1
        if debug_counter == 10:
            print("Hit debug counter")
            break
    return box

def recalculate_edge_from_box(box, idxs, v_to_neighbour_points, neighbour_points, boundary):
    """
    Recalculate the edge between an outside point and one of its neighbour points. It takes the edge which the
    shorter length from the point outside of the image to the image boundary if you move it by the vector to its
    neighbour points
    :param box: The bounding box which contains the edge points
    :type box: 2D np.array
    :param idxs: Indexes are in following order: 0: current idx, 1: next idx, 2: prev idx
    :type idxs: tuple of int
    :param v_to_neighbour_points: The vectors from current point to the next and prev point from the box
                                  Index 0: Vector to next point. Index 1: Vector to prev point
    :type v_to_neighbour_points: tuple of 1D np.arrays
    :param neighbour_points: The next and prev point from the box. Index 0: Next point. Index 1: Prev point
    :type neighbour_points: tuple of 1D np.arrays
    :param boundary: The img boundary to which the points are shifted
    :type boundary: tuple of int
    :return: The box with two recalculated edge points
    :type: 2D np.array
    """
    moved_along_vector_next = False
    moved_along_vector_prev = False
    cur_idx, next_idx, prev_idx = idxs[0], idxs[1], idxs[2]
    cur_point = box[cur_idx]
    v_next_point, v_prev_point = v_to_neighbour_points[0], v_to_neighbour_points[1]
    p_next, p_prev = neighbour_points[0], neighbour_points[1]

    # set x to 0 and calculate new y
    temp_p_from_vec_next = move_point_to_img_boundary(cur_point, 
                                                   v_next_point,
                                                   boundary)
                                             
    temp_p_from_vec_prev = move_point_to_img_boundary(cur_point, 
                                                    v_prev_point, 
                                                    boundary)
    # calculate the length from p_old to p_new for prev and next vector
    l_v_next, v_p_old_to_p_new_next = calc_length(cur_point, temp_p_from_vec_next)
                                                                                
    l_v_prev, v_p_old_to_p_new_prev = calc_length(cur_point, temp_p_from_vec_prev)
    
    # move p_old along the shorter vector - finally get p_new
    if l_v_next < l_v_prev:
        new_p_x, new_p_y = temp_p_from_vec_next
        moved_along_vector_next = True
    else:
        new_p_x, new_p_y = temp_p_from_vec_prev
        moved_along_vector_prev = True
        
    # check wich point to move parallel and reasign the moved point
    # to keep the rectangular shape of the box always move one of the neighbour points, too
    # if moving along vector_next then move the point_at_prev_index
    if moved_along_vector_next:
        new_p_prev_x, new_p_prev_y = parallel_shift_neighbour_point(p_prev, v_p_old_to_p_new_next)
        box[prev_idx] = (new_p_prev_x, new_p_prev_y)
    # if moving along vector_prev then move the point_at_next_index
    elif moved_along_vector_prev:
        new_p_next_x, new_p_next_y = parallel_shift_neighbour_point(p_next, v_p_old_to_p_new_prev)
        box[next_idx] = (new_p_next_x, new_p_next_y)
    
    box[cur_idx] = (new_p_x, new_p_y)
    return box

def move_point_to_img_boundary(point, vector, img_boundary):
    """
    Depending on the point position (left, above, right, below) the image move it in direction to the
    image boundary
    :param point: The point outside
    :type point: tuple of int
    :param vector: The vector between the point and its neighboring point
    :type vector: tuple of int
    :param img_boundary: The image size
    :type img_boundary: tuple of int
    :return: New calculated point on the image boundary
    :type: tuple of int
    """
    point_x, point_y = point
    vector_x, vector_y = vector
    bound_x, bound_y = img_boundary
    if bound_x == 0 or bound_x == 255:
        # set x coordinate to the boundary values and calculate new y coordinate
        # formular: p_new = p_old + eps * vector_x
        eps = (bound_x + (point_x * (-1))) / vector_x
        p_y_new = int(point_y + eps * vector_y)
        p_x_new = bound_x
    elif bound_y == 0 or bound_y == 255:
        # set y coordinate to the boundary values and calculate new x coordinate
        eps = (bound_y + (point_y * (-1))) / vector_y
        p_x_new = int(point_x + eps * vector_x)
        p_y_new = bound_y
    return p_x_new, p_y_new

def calc_length(orig_point, shifted_point):
    orig_point_x, orig_point_y = orig_point
    shifted_point_x, shifted_point_y = shifted_point
    v = shifted_point_x - orig_point_x, shifted_point_y - orig_point_y
    return np.linalg.norm(v), v

def parallel_shift_neighbour_point(orig_point, v_shifted_point):
    """
    Moves a point exactly along the length of the passed vector
    :param orig_point: The original point which should be moved
    :type orig_point: tuple of int
    :param v_shifted_point: The vector by which the original point should be moved
    :type v_shifted_point: tuple of int
    return: The original point moved by the vector
    :type: tuple of int
    """
    orig_point_x, orig_point_y = orig_point
    v_x, v_y = v_shifted_point
    return (orig_point_x + v_x, orig_point_y + v_y)

def rotate_edge_points(box, angle, center_x=128, center_y=128):
    """
    Rotate the recalculated edge points of the bounding box by the given angle from cv2.minAreaRect()
    :param box: The recalculated bounding box with new edge points
    :type box: 2D np.array
    :param angle: The angle in degrees by which the bounding box was rotated
    :type angle: float
    :return: The rotated edge points of the bounding box
    """
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)

    edge_points = [(point[0], point[1], 1) for point in box]
    rotated_points = []
    for point in edge_points:
        # rotate each point individually
        rotated_pixel = np.dot(M, point).astype(int)
        rotated_points.append((rotated_pixel[0], rotated_pixel[1]))
    return rotated_points

def rotate_img(img, angle, center_x=128, center_y=128):
    """
    Rotate the image by the given angle from cv2.minAreaRect()
    :param angle: The angle in degrees by which the bounding box was rotated
    :type angle: float
    :return: The rotated image
    """
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def align_edge_points(edge_points):
    """
    Align the edge points so that they are perfectly horizontal or vertical
    :param edge_points: The rotated edge points of the bounding box
    :type edge_points: list of tuples
    return: horizontal and vertical aligned edge points
    :type: list of tuples
    """
    # because of rounding errors the edge points are not perfectly horizontal or vertical aligned
    # I want to align them because then I can easily crop the bounding box later
    # first index lower left and then clockwise
    
    # TODO: What to do if there are not in range ???
    tol = 4
    # just check if the points are nearly horizontal and vertical aligned
    assert edge_points[0][0] in range(edge_points[1][0]-tol, edge_points[1][0]+tol), "P0 and P1 are not aligned"
    assert edge_points[0][1] in range(edge_points[3][1]-tol, edge_points[3][1]+tol), "P0 and P3 are not aligned"
    assert edge_points[1][1] in range(edge_points[2][1]-tol, edge_points[2][1]+tol), "P1 and P2 are not aligned"
    assert edge_points[2][0] in range(edge_points[3][0]-tol, edge_points[3][0]+tol), "P2 and P3 are not aligned"

    # diagonal points
    align_x_p0 = edge_points[0][0]
    align_y_p0 = edge_points[0][1]
    align_x_p2 = edge_points[2][0]
    align_y_p2 = edge_points[2][1]
    
    # p0 and p2 are fix. p1 and p3 are aligned to p0 and p2
    p0 = edge_points[0][0], edge_points[0][1]
    p1 = align_x_p0, align_y_p2
    p2 = edge_points[2][0], edge_points[2][1]
    p3 = align_x_p2, align_y_p0
    return [p0, p1, p2, p3]

def clip_aligned_points(coor):
    """
    It is possible that some edge points are still outside the image boundaries, if yes
    just clip the points to the image boundaries
    :param aligned_points: The aligned edge points of the bounding box
    :type aligned_points: list of tuples
    :return: The aligned edge points of the bounding box clipped to the image boundaries
    :type: list of tuples
    """
    return max(0, min(coor, 255))

def draw_rotated_box(img, edge_points):
    img_copy = img.copy()
    for i in range(len(edge_points)):
        cv2.line(img_copy, edge_points[i], edge_points[(i+1)%4], (255, 0, 255), 1)
    return img_copy

def calc_symmetry(img, aligned_box, reflection_axis):
    """
    Calculate the symmetry feature of the vegi by taking the ratio of the overlap area of original 
    and reflected image to the area of the original image
    :param img: The image of the vegi in RGB format
    :type img: np.array
    :param aligned_box: The aligned rotated box with the aligned edge points
    :type aligned_box: 2D np.array
    :param reflection_axis: Axis on which the image is reflected ["vertical", "horizontal"]
    :type reflection_axis: str
    :return: The symmetry meassure of an vegi
    :type: float
    """
    # convert img to binary img
    img_bin = segment_img_3(img)

    # prevent manipulating the original box
    aligned_box_copy = aligned_box.copy()
    v_line = int(np.linalg.norm(aligned_box_copy[1] - aligned_box_copy[0]))
    h_line = int(np.linalg.norm(aligned_box_copy[3] - aligned_box_copy[0]))
    # width and height should be a even number because later it's easier to calculate
    # if the length of the lines are odd values then remove one pixel row and/or column
    if v_line % 2 != 0:
        v_line -= 1
        aligned_box_copy[1][1] += 1
        aligned_box_copy[2][1] += 1
    if h_line % 2 != 0:
        h_line -= 1
        aligned_box_copy[2][0] -= 1
        aligned_box_copy[3][0] -= 1

    if reflection_axis == "vertical":
        # array rows corresponds to img height
        # crop image from top left point y to bottom left point y and from top left point x to image center x
        left_half = img_bin[aligned_box_copy[1][1]:aligned_box_copy[1][1]+v_line, 
                            aligned_box_copy[1][0]:aligned_box_copy[1][0]+(h_line // 2)]
        # reflect the left half by the vertical axis
        flipped_half = cv2.flip(left_half, 1)
        # get the right half from the orignal image
        orig_half = img_bin[aligned_box_copy[1][1]:aligned_box_copy[1][1]+v_line, 
                            aligned_box_copy[1][0]+(h_line // 2):aligned_box_copy[2][0]]
    elif reflection_axis == "horizontal":
        # same for horizontal axis
        # crop top half of image
        top_half = img_bin[aligned_box_copy[1][1]:aligned_box_copy[1][1]+(v_line // 2),
                            aligned_box_copy[1][0]:aligned_box_copy[1][0]+h_line]
        # reflect the top half by the horizontal axis
        flipped_half = cv2.flip(top_half, 0)
        # get the bottom half from the orignal image
        orig_half = img_bin[aligned_box_copy[1][1]+(v_line // 2):aligned_box_copy[0][1],
                            aligned_box_copy[1][0]:aligned_box_copy[1][0]+h_line]

    arr_flipped = flipped_half.flatten()
    arr_orig_half = orig_half.flatten()    
    # the halves must have the same size
    assert arr_flipped.shape == arr_orig_half.shape, "Arrays have different shapes"
    # Count the overlapping area of flipped image and the original image only for the halves
    overlap_count = np.sum((arr_flipped == 255) & (arr_orig_half == 255))
    # count white pixels of flipped half, prevent error from segment_binary algorithm
    flipped_area = np.unique(flipped_half, return_counts=True)[1][1]
    return round(overlap_count / flipped_area, 2)