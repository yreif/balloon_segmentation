import os
import numpy as np
import xlsxwriter

from scipy.spatial import distance as dist
from imutils import perspective
import imutils
import cv2


# File Parameters
###################
pixelsPerMetric = 1
mov_file = '3.MOV'
start_second = 4
end_second = 10
frame_rate = 0.05  # in seconds, can use decimals too


# Constants
##################
# Values of HSV range for the balloon
min_sat = 0
min_val = 120
# Range of lower range
lower_red1 = np.array([0, min_sat, min_val])
upper_red1 = np.array([10, 255, 255])
# Range of upper range
lower_red2 = np.array([170, min_sat, min_val])
upper_red2 = np.array([180, 255, 255])

# Values of HSV range for LED screen
lower_led = np.array([22, 15, 30])
upper_led = np.array([35, 75, 100])

# Dictionary of digit segments to identify digits on the LED screen
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}


# Helper functions
###################
measurements = np.zeros((int((end_second - start_second) / frame_rate) + 1, 4))


def led_threshold(i, x):
    """
    Threshold used for each section of the digit to determine whether it's on or off.
    :param i: Section number
    :param x: % of white pixels.
    :return: 1 or 0 (on or off).
    """
    if i == 5:
        return 1
    elif i == 0:
        return int(x >= 0.5)
    elif i == 3:
        return int(x >= 0.45)
    else:
        return int(x >= 0.35)


def midpoint(a, b):
    """
    Returns the point in the middle of two (x,y) points.
    """
    return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5


# Main functions
#################
def extract_balloon_size(image):
    """
    Extract balloon height and width from the image.
    """
    global pixelsPerMetric

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # generate mask to detect red colors in the shades of the balloon
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask = mask1 + mask2
    threshold = gray.copy()
    threshold[mask == 0] = ([255])

    # perform edge detection to find the contour of the balloon
    threshold = cv2.GaussianBlur(threshold, (5, 5), 0)
    edged = cv2.Canny(threshold, 50, 100)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=2)
    __, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # The balloon will be the biggest one
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # compute the rotated bounding box of the contour
    c = cnts[0]
    image_with_drawings = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear in top-left, top-right, bottom-right,
    # and bottom-left order, then draw the outline of the rotated bounding box
    box = perspective.order_points(box)
    cv2.drawContours(image_with_drawings, [box.astype("int")], -1, (0, 255, 0), 2)
    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(image_with_drawings, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoints
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    cv2.circle(image_with_drawings, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(image_with_drawings, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(image_with_drawings, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(image_with_drawings, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(image_with_drawings, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
    cv2.line(image_with_drawings, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # draw the object sizes on the image
    cv2.putText(image_with_drawings, "{:.1f}pix".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(image_with_drawings, "{:.1f}pix".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    return dimA, dimB, image_with_drawings


def extract_digits(image):
    led = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # led = exposure.rescale_intensity(led, out_range= (0,255))
    led = cv2.threshold(led, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # sharpen:
    kernel = -1 * np.ones((3, 3))
    kernel[1, 1] = 9
    led = 255 - cv2.filter2D(led, -1, kernel)

    # find contours in the threshold image, then initialize the digit contours lists
    contour_threshold = cv2.dilate(led, None, iterations=2)
    contour_threshold = cv2.erode(contour_threshold, None, iterations=2)
    digits_threshold = led
    cnts = cv2.findContours(contour_threshold.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]

    # sort the contours from left-to-right, then initialize the actual digits themselves
    digits_cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
    digits = []

    # loop over each of the digits
    for c in digits_cnts:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        roi = digits_threshold[y:y + h, x:x + w]

        # compute the width and height of each of the 7 segments
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.3), int(roiH * 0.15))
        dHC = int(roiH * 0.2)

        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, h // 2)),  # top-left
            ((w - dW, 0), (w, h // 2)),  # top-right
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
            ((0, h // 2), (dW, h)),  # bottom-left
            ((w - dW, h // 2), (w, h)),  # bottom-right
            ((0, h - dH), (w, h))  # bottom
        ]
        on = [0] * len(segments)
        # loop over the segments
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of thresholded pixels in the segment,
            # and then compute the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            # if the total number of non-zero pixels is greater than
            # the matching threshold of the area, mark the segment as "on"
            on[i] = led_threshold(i, total / float(area))
        # Lookup the digit. If it's not recognizable, just ignore it.
        try:
            digit = DIGITS_LOOKUP[tuple(on)]
        except:
            continue
        digits.append(digit)

    str_digits = [str(x) for x in digits]
    if len(str_digits) > 0:
        number = float(''.join(str_digits))
    else:  # found no digits
        number = -1
    return number, led


def get_frame(sec, balloon_rect, led_rect, row, worksheet):
    """
    Take the frame from the video in time sec, extract the balloon size and led digits from it
    (Images of which will be cropped according to given rectangles), and save to the worksheet.
    :param sec: Second in the video of the requested frame.
    :param balloon_rect: Rectangle of balloon ROI
    :param led_rect: Rectangle of LED ROI
    :param row: Row in worksheet to save to
    :param worksheet: Worksheet object to save to
    :return: Whether frame was captured or not
    """
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    name = frames_folder + '/sec_' + str(sec) + '.jpg'
    led_name = frames_folder + '/led_' + str(sec) + '.jpg'

    if hasFrames:
        balloon = image[int(balloon_rect[1]):int(balloon_rect[1] + balloon_rect[3]), int(balloon_rect[0]):int(balloon_rect[0] + balloon_rect[2])]
        led = image[int(led_rect[1]):int(led_rect[1] + led_rect[3]), int(led_rect[0]):int(led_rect[0] + led_rect[2])]
        h, w, marked_image = extract_balloon_size(balloon)
        measure, led = extract_digits(led)
        cv2.imwrite(name, marked_image)  # save frame as JPG file
        cv2.imwrite(led_name, led)
        worksheet.write(row, 0, mov_file + '_' + str(sec))
        worksheet.write(row, 1, h)
        worksheet.write(row, 2, w)
        worksheet.write(row, 3, measure)
        worksheet.insert_image(row, 5, led_name, {'x_scale': 0.5, 'y_scale': 0.5})
    return hasFrames


if __name__ == "__main__":
    row = 1
    workbook = xlsxwriter.Workbook(mov_file + '.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'index')
    worksheet.write(0, 1, 'h')
    worksheet.write(0, 2, 'w')
    worksheet.write(0, 3, 'measurement')

    # Read the video from specified path
    vidcap = cv2.VideoCapture(mov_file)

    frames_folder = mov_file + '_data'
    try:
        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)
    except OSError:
        print('Error: Creating directory of data')

    vidcap.set(cv2.CAP_PROP_POS_MSEC, start_second * 1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        LED_RECT = cv2.selectROI("LED ROI", image)
        BALLOON_RECT = cv2.selectROI("Balloon ROI", image)

        # extract data from video
        sec = start_second
        count = 1
        success = get_frame(sec, BALLOON_RECT, LED_RECT, row, worksheet)
        row += 1
        while success and sec < end_second:
            count = count + 1
            sec = sec + frame_rate
            sec = round(sec, 2)
            success = get_frame(sec, BALLOON_RECT, LED_RECT, row, worksheet)
            row += 1
        vidcap.release()
        workbook.close()

    else:
        print('Error: reading video failed.')

