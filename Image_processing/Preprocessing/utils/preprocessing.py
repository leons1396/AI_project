import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


def show_image_plt(img_rgb):
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


def draw_contours(bgr_img, object_area):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv)

    blurred_sat = cv2.GaussianBlur(saturation, (3, 3), 0)

    # Compute the thresh dynamically from the mean() value. 
    thresh = blurred_sat.mean()
    std = blurred_sat.std()
    thresh_low = thresh - std
    thresh_high = thresh + std
    # The factors were simply selected by testing the algoritm. Another approach could be to calculate the mean with the standard deviation
    #thresh_low = 0.3 * thresh 
    #thresh_high = 2 * thresh
    
    # The next four lines control how good the bounding box will fit
    edges = cv2.Canny(blurred_sat, thresh_low, thresh_high)
    kernel = np.ones((4, 4), np.uint8) # creates 4x4 Identity matrix
    dilate = cv2.dilate(edges, kernel, iterations=4)
    erode = cv2.erode(dilate, kernel, iterations=4)

    contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    bgr_img_copy = bgr_img.copy()
    # Flag makes sure that there is a maximum of 1 box in each image. Assumption, the bounding box for the vegi is always the biggest
    more_than_one_box = False
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area >= object_area:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(bgr_img_copy, [box], 0, (0, 255, 0), 2)
            
            # Calculate Circularity
            perimeter = cv2.arcLength(contour, True)
            r_circle = perimeter / (2 * np.pi)
            A_circle = r_circle**2 * np.pi
            circularity = area / A_circle
            
            if i > 0:
                # There are more than 2 boxes in the image
                more_than_one_box = True
    
    rgb = cv2.cvtColor(bgr_img_copy, cv2.COLOR_BGR2RGB)
    return rgb, more_than_one_box, box, rect, area, circularity


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
    ret, label, center = cv2.kmeans(cropped_vegi_2D, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
   
    res = center[label.flatten()]
    res2 = res.reshape((rgb_segment.shape))
    
    #returns center in rgb format
    return center, ret, label


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
    mean = total / len(kp)
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
def resize_bound_box_and_draw(img, rect, box):
    # figure out the correct case. Find the points which are outside the image
    outside_count = 0
    outside_count = sum(1 for arr in box if (arr < 0).any() or (arr > 255).any())

    print("Outside count: ", outside_count)


    # Calculate the directions of the vectors, once
    u_0_1 = box[1] - box[0]
    u_0_3 = box[3] - box[0]
    #print(f"Vector u_0_1: {u_0_1} | Vector u_0_3: {u_0_3}")

    if outside_count == 1:
        new_box = case_one_point_outside(box, u_0_1)
        rotated_points = rotate_edge_points(new_box, rect[2])
        rotated_img = rotate_img(img, rect[2])
        aligned_points = align_edge_points(rotated_points)
        vegi_with_new_box_rgb = draw_rotated_box(rotated_img, aligned_points)
        
    elif outside_count == 2:
        new_box = case_2_points_outside(box, u_0_1, u_0_3)
        rotated_points = rotate_edge_points(new_box, rect[2])
        rotated_img = rotate_img(img, rect[2])
        aligned_points = align_edge_points(rotated_points)
        vegi_with_new_box_rgb = draw_rotated_box(rotated_img, aligned_points)

    elif outside_count >= 3:
        # Assumption: Area of bounding box is probably similar to the area of the img. Just draw a new box with edge points in the middle of each side of the img
        # TODO Important: but keep the orignal angle from the box. rect[2]
        h = img.shape[0]
        new_box = np.array([[0, h // 2], [h // 2, 0],  [255, h // 2], [h // 2, 255]])

        # because of the distinct rotation angle, rotate new_box and img separately
        rotated_points = rotate_edge_points(new_box, 45)
        rotated_img = rotate_img(img, rect[2])
        aligned_points = align_edge_points(rotated_points)
        vegi_with_new_box_rgb = draw_rotated_box(rotated_img, aligned_points)
    return vegi_with_new_box_rgb, aligned_points

def case_one_point_outside(box, u_0_1):
    # Assumption: It shouldn't matter in which direction to img borders you move the outside point
    # because the distance for both direction should be small. Is it a big distance then probably another point is outside too.
    # Always shift the outside point along u_0_1 vector
    # case 1 - should be roughly similiar like case 2. Pass the correct borders
    print("This is case 1")
    p0, p1, p2, p3 = box[0], box[1], box[2], box[3]
    # get the point which is outside and his index in the box array
    p_idx = [i for i, arr in enumerate(box) if (arr < 0).any() or (arr > 255).any()][0]
    print("point_idx: ", p_idx)
    
    # p0 is outside
    if p_idx == 0:
        #p0_x_new, p0_y_new, p1_x_new, p1_y_new = calculate_new_points(p0, p1, u_0_3, (0, None), (None, 0))
        #p0_x_u01, p0_y_u01 = move_point_along_vector(p0, u_0_1, (0, None))
        #p0_x_u03, p0_y_u03 = move_point_along_vector(p0, u_0_3, (0, None))
        
        #l_01, u_p0_u01 = calculate_length_of_moving_vectors(p0, p0_x_u01, p0_y_u01)
        #l_03, u_p0_u03 = calculate_length_of_moving_vectors(p0, p0_x_u03, p0_y_u03) 

        p0_x_new, p0_y_new, p3_x_new, p3_y_new = calculate_new_points_case1(p0, p3, u_0_1, (0, None))
        new_box = np.array([[p0_x_new, p0_y_new], [p1[0], p1[1]], [p2[0], p2[1]], [p3_x_new, p3_y_new]])
        """
        if l_01 < l_03:
            # set p3 to new position
            # move p3 by u_0_1 vector
            p3_x_u01, p3_y_u01 = parallel_shift_neighbour_point(p3, u_p0_u01)
        
        elif l_03 <= l_01:
            # Set p1 to the new position
            # move p1 by u_0_3 vector
            p1_x_u03, p1_y_u03 = parallel_shift_neighbour_point(p1, u_p0_u03)
            new_box = np.array([[p0_x_u03, p0_y_u03], [p1_x_u03, p1_y_u03], [p2[0], p2[1]], [p3[0], p3[1]]])
        """
    # p1 is outside
    elif p_idx == 1:
        p1_x_new, p1_y_new, p2_x_new, p2_y_new = calculate_new_points_case1(p1, p2, u_0_1*(-1), (None, 0))
        new_box = np.array([[p0[0], p0[1]], [p1_x_new, p1_y_new], [p2_x_new, p2_y_new], [p3[0], p3[1]]])
    # p2 is outside
    elif p_idx == 2:
        p2_x_new, p2_y_new, p1_x_new, p1_y_new = calculate_new_points_case1(p2, p1, u_0_1*(-1), (255, None))
        new_box = np.array([[p0[0], p0[1]], [p1_x_new, p1_y_new], [p2_x_new, p2_y_new], [p3[0], p3[1]]])
    # p3 is outside
    elif p_idx == 3:
        p3_x_new, p3_y_new, p0_x_new, p0_y_new = calculate_new_points_case1(p3, p0, u_0_1, (None, 255))
        new_box = np.array([[p0_x_new, p0_y_new], [p1[0], p1[1]], [p2[0], p2[1]], [p3_x_new, p3_y_new]])
    return new_box


def case_2_points_outside(box, u_0_1, u_0_3):
    print("This is case 2")
    p0, p1, p2, p3 = box[0], box[1], box[2], box[3]
    # 2 neighboured points are outside
    # get the points which are outside and their index in the box array
    p_idx_1, p_idx_2 = [i for i, arr in enumerate(box) if (arr < 0).any() or (arr > 255).any()]
    print("point_idx_1: ", p_idx_1, "point_idx_2: ", p_idx_2)

    # p0 and p1 are outside
    if p_idx_1 == 0 and p_idx_2 == 1:
        p0_x_new, p0_y_new, p1_x_new, p1_y_new = calculate_new_points_case2(p0, p1, u_0_3, (0, None), (None, 0))
        new_box = np.array([[p0_x_new, p0_y_new], [p1_x_new, p1_y_new], [p2[0], p2[1]], [p3[0], p3[1]]])
        return new_box
    
    # p1 and p2 are outside
    elif p_idx_1 == 1 and p_idx_2 == 2:
        p1_x_new, p1_y_new, p2_x_new, p2_y_new = calculate_new_points_case2(p1, p2, u_0_1*-1, (None, 0), (255, None))
        new_box = np.array([[p0[0], p0[1]], [p1_x_new, p1_y_new], [p2_x_new, p2_y_new], [p3[0], p3[1]]])
        return new_box

    # p2 and p3 are outside
    elif p_idx_1 == 2 and p_idx_2 == 3:
        p2_x_new, p2_y_new, p3_x_new, p3_y_new = calculate_new_points_case2(p2, p3, u_0_3*-1, (255, None), (None, 255))
        new_box = np.array([[p0[0], p0[1]], [p1[0], p1[1]], [p2_x_new, p2_y_new], [p3_x_new, p3_y_new]])
        return new_box
    
    # p0 and p3 are outside
    elif p_idx_1 == 0 and p_idx_2 == 3:
        p0_x_new, p0_y_new, p3_x_new, p3_y_new = calculate_new_points_case2(p0, p3, u_0_1, (0, None), (None, 255))
        new_box = np.array([[p0_x_new, p0_y_new], [p1[0], p1[1]],  [p2[0], p2[1]], [p3_x_new, p3_y_new]])
        return new_box


def calculate_new_points_case1(p, neighbour_p, u_0_1, border):
    # np = neighbour point
    p_x_u01, p_y_u01 = move_point_along_vector(p, u_0_1, border)
    _, u_p_u01 = calculate_length_of_moving_vectors(p, p_x_u01, p_y_u01) # Only need vector original point to shifted orig point
    np_x_u01, np_y_u01 = parallel_shift_neighbour_point(neighbour_p, u_p_u01)
    return p_x_u01, p_y_u01, np_x_u01, np_y_u01


def calculate_new_points_case2(p0, p1, u, border1, border2):
        p0_x_new, p0_y_new = move_point_along_vector(p0, u, border1)
        p1_x_new, p1_y_new = move_point_along_vector(p1, u, border2)

        l_0, u_p0_p0_new = calculate_length_of_moving_vectors(p0, p0_x_new, p0_y_new)
        l_1, u_p1_p1_new = calculate_length_of_moving_vectors(p1, p1_x_new, p1_y_new)
        print(f"Length l_0= {l_0} and l_1= {l_1}")

        if l_0 < l_1:
            # Set p1 to the new position
            # move p0 by p1_new vector
            p0_x_new, p0_y_new = parallel_shift_neighbour_point(p0, u_p1_p1_new)
        elif l_0 >= l_1:
            # Set p0 to the new position
            # move p1 by p0_new vector
            p1_x_new, p1_y_new = parallel_shift_neighbour_point(p1, u_p0_p0_new)
        return p0_x_new, p0_y_new, p1_x_new, p1_y_new


def move_point_along_vector(point, vector, img_border):
    border_x, border_y = img_border
    point_x, point_y = point
    vector_x, vector_y = vector

    if border_x == 0 or border_x == 255:
        eps = (border_x + (point_x * (-1))) / vector_x
        print("eps: ", eps)
        print("vector x: ", vector_x)
        print("vector y: ", vector_y)
        p_y_new = int(point_y + eps * vector_y)
        p_x_new = border_x
    elif border_y == 0 or border_y == 255:
        eps = (border_y + (point_y * (-1))) / vector_y
        p_x_new = int(point_x + eps * vector_x)
        p_y_new = border_y
    print(f"New Point x= {p_x_new} y= {p_y_new}")
    return p_x_new, p_y_new


def calculate_length_of_moving_vectors(orig_point, shifted_point_x, shifted_point_y):
    orig_point_x, orig_point_y = orig_point
    u = shifted_point_x - orig_point_x, shifted_point_y - orig_point_y
    return np.linalg.norm(u), u


def parallel_shift_neighbour_point(orig_point, u_shifted_point):
    orig_point_x, orig_point_y = orig_point
    u_x, u_y = u_shifted_point
    return (orig_point_x + u_x, orig_point_y + u_y)


def rotate_edge_points(box, angle, center_x=128, center_y=128):
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)

    edge_points = [(point[0], point[1], 1) for point in box]
    rotated_points = []
    for point in edge_points:
        rotated_pixel = np.dot(M, point).astype(int)
        rotated_points.append((rotated_pixel[0], rotated_pixel[1]))
    return rotated_points


def rotate_img(img, angle, center_x=128, center_y=128):
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def draw_rotated_box(img, edge_points, color=(255, 0, 255)):
    img_copy = img.copy()
    for i in range(len(edge_points)):
        cv2.line(img_copy, edge_points[i], edge_points[(i+1)%4], color, 1)
    return img_copy


def align_edge_points(edge_points):
    # because of rounding errors the edge points are not perfectly horizontal or vertical aligned
    # I want to align them because later I can easily crop the bounding box
    # first index lower left and then clockwise
    # check if the points are nearly aligned
    print(edge_points)
    # TODO: What to do if there are not in range ???
    tol = 3
    assert edge_points[0][0] in range(edge_points[1][0]-tol, edge_points[1][0]+tol)
    assert edge_points[0][1] in range(edge_points[3][1]-tol, edge_points[3][1]+tol)
    assert edge_points[1][1] in range(edge_points[2][1]-tol, edge_points[2][1]+tol)
    assert edge_points[2][0] in range(edge_points[3][0]-tol, edge_points[3][0]+tol)

    #Diagonal points
    align_x_p0 = edge_points[0][0]
    align_y_p0 = edge_points[0][1]
    align_x_p2 = edge_points[2][0]
    align_y_p2 = edge_points[2][1]
    p0 = edge_points[0][0], edge_points[0][1]
    p1 = align_x_p0, align_y_p2
    p2 = edge_points[2][0], edge_points[2][1]
    p3 = align_x_p2, align_y_p0
    return [p0, p1, p2, p3]