import cv2
import numpy as np

# Create a dictionary for the HSV default values
hsv = {'ilowH': 0, 'ihighH': 179, 'ilowS': 0, 'ihighS': 255, 'ilowV': 0, 'ihighV': 255}

# put -1/0/1 in VideoCapture()
cap = cv2.VideoCapture(1)
cv2.namedWindow('image')


def callback(x):
    pass


def rectangle_recognition():
    contours_to_check = []

    if contours is not None:
        for cnt in contours:
            if len(cnt) > 3:

                rect = cv2.minAreaRect(cnt)
                rect_area = rect[1][0] * rect[1][1]

                area = cv2.contourArea(cnt)

                ratio = area / rect_area

                if 0.85 < ratio < 1.15 and area > 150:  # TODO: Extract this into a function and put numbers into constants
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(original, [box], 0, (0, 0, 255), 2)
                    contours_to_check.append(cnt)

    return contours_to_check


def switch_recognition(rectangles):
    if len(rectangles) == 2:

        pass

    else:

        cv2.putText(original, "Searching", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


def get_trackbars_position():
    hsv['ilowH'] = cv2.getTrackbarPos('lowH', 'image')
    hsv['ihighH'] = cv2.getTrackbarPos('highH', 'image')
    hsv['ilowS'] = cv2.getTrackbarPos('lowS', 'image')
    hsv['ihighS'] = cv2.getTrackbarPos('highS', 'image')
    hsv['ilowV'] = cv2.getTrackbarPos('lowV', 'image')
    hsv['ihighV'] = cv2.getTrackbarPos('highV', 'image')


# create trackbars for color change
cv2.createTrackbar('lowH', 'image', hsv['ilowH'], 179, callback)
cv2.createTrackbar('highH', 'image', hsv['ihighH'], 179, callback)

cv2.createTrackbar('lowS', 'image', hsv['ilowS'], 255, callback)
cv2.createTrackbar('highS', 'image', hsv['ihighS'], 255, callback)

cv2.createTrackbar('lowV', 'image', hsv['ilowV'], 255, callback)
cv2.createTrackbar('highV', 'image', hsv['ihighV'], 255, callback)

while True:

    ret, frame = cap.read()
    original = frame.copy()

    # grab the frame
    frame = original.copy()

    # get trackbars position
    get_trackbars_position()

    # create a mask
    hsv_colors = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([hsv['ilowH'], hsv['ilowS'], hsv['ilowV']])
    higher_hsv = np.array([hsv['ihighH'], hsv['ihighS'], hsv['ihighV']])
    mask = cv2.inRange(hsv_colors, lower_hsv, higher_hsv)

    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # create a cross kernel
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]],
                      dtype=np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    ret, mask = cv2.threshold(mask, 127, 255, 0)

    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles_to_check = rectangle_recognition()

    switch_recognition(rectangles_to_check)

    cv2.imshow('mask', frame)
    cv2.imshow('original', original)
    k = cv2.waitKey(1) & 0xFF  # large wait time to remove freezing
    if k in (27, 113):
        break
