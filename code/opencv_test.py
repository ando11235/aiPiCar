import cv2
import numpy as np
import math

def main():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640) #width
    camera.set(4, 480) #height

    while( camera.isOpened()):
        _, frame = camera.read()    
        cv2.imshow('Original', frame)
        
        b_w_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('B/W', b_w_image)

        #Convert image to HSV
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #cv2.imshow('HSV', hsv_image)

        #Covert to blue mask
        lower_blue = np.array([60, 40, 40])
        upper_blue = np.array([150, 255, 255])
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        edges = cv2.Canny(mask, 200, 400)
        cv2.imshow('edges', edges)

        #crop the image to lower half
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height * 1 / 2),
            (width, height * 1 / 2),
            (width, height),
            (0, height),
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        cv2.imshow('cropped_edges', cropped_edges)

        lane_lines_image = display_lines(frame, lane_lines)
        cv2.imshow("lane lines", lane_lines_image)

        if( cv2.waitKey(1) & 0xFF == ord('q') ): 
            break


    camera.release()        
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()

##Utility Code##

def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=8, maxLineGap=4)
    return line_segments

def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  
    right_region_boundary = width * boundary 

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)  

    return lane_lines

    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  
    y2 = int(y1 * 1 / 2)  
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color=(0, 255, 0), line_width=2)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image
