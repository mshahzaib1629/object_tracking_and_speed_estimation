from time import time
import cv2
import numpy
from object_detection import ObjectDetection
import math
import copy

# ####### Parameters #######

# distanceScale 4 for los_angeles, 12 for highway
# fontScale 1 for los_angeles, 0.45 for highway
# fontWeight 2 for los_angeles, 1 for highway

distanceScale = 4
fontScale = 1
fontWeight = 2
distanceThreshold = 30
# videoPath = "highway.mp4"
videoPath = "test_videos/los_angeles.mp4"

# ##########################

od = ObjectDetection()

cap = cv2.VideoCapture(videoPath)

# intitializing frame count
count = 0
center_points_prev_frame = []
tracking_objects = {}
tracking_objects_prev = {}
frame_timestamps = {}
tracking_id = 0

def estimateSpeed(location1, location2, timeDifference):
    d_pixels = math.hypot(location2[0] - location1[0], location2[1] - location1[1])
    ppm = 8.8
    d_meters = d_pixels * distanceScale
    speed = d_meters * 3.6 / timeDifference
    return speed

def calculate_speed ():
    global frame_timestamps, count, tracking_objects, tracking_objects_prev
    for object_id, pt in tracking_objects.items():

        if not object_id in tracking_objects_prev.keys():
            return
        current_location = pt['position']
        prev_location = tracking_objects_prev[object_id]['position']
        timeDifference = frame_timestamps[count] - frame_timestamps[count-1]
        object_speed = estimateSpeed(prev_location, current_location, timeDifference)
        tracking_objects[object_id]['speed'] = object_speed

def add_cars_in_current_frame(center_points_current_frame, boxes):
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_current_frame.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

def mark_tracking (frame):
    for object_id, pt in tracking_objects.items():
        if 'speed' in pt.keys():
            position = pt['position']
            # marking being tracked cars with a yellow dot
            cv2.circle(frame, position, fontWeight, (0, 255, 255), -1)
            speed_text = f"{str('{:.1f}'.format(pt['speed']))} km/h"
            cv2.putText(frame, speed_text, (position[0], position[1] - 7), 0, fontScale, (0, 0, 255), fontWeight)

def track_initial_cars(center_points_current_frame, center_points_prev_frame):
    global tracking_id
    for pt in center_points_current_frame: 
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < distanceThreshold:
                    tracking_objects[tracking_id] = {}
                    tracking_objects[tracking_id]['position'] = pt
                    tracking_id += 1

def update_remove_existing_trackings (center_points_current_frame):
    tracking_objects_copy = tracking_objects.copy()
    center_points_current_frame_copy = center_points_current_frame.copy()

    # looping over already being tracked cars
    for object_id, pt2 in tracking_objects_copy.items():
            position = pt2['position']
            object_exits = False
            
            for pt in center_points_current_frame_copy:
                distance = math.hypot(position[0] - pt[0], position[1] - pt[1])
                
                if distance < distanceThreshold:
                     # updating car's positions which is already being tracked
                    tracking_objects[object_id]['position'] = pt
                    object_exits = True
                    if pt in center_points_current_frame:
                        center_points_current_frame.remove(pt)
                    continue

            # removing car from being tracked if it has been moved out of frame
            if object_exits == False:
                tracking_objects.pop(object_id)

def track_upcoming_cars (center_points_current_frame): 
    global tracking_id
    for pt in center_points_current_frame:
        tracking_objects[tracking_id] = {}
        tracking_objects[tracking_id]['position'] = pt
        tracking_id += 1

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break
    
    # Points current frame
    center_points_current_frame = []

    (class_ids, scores, boxes) = od.detect(frame)
    add_cars_in_current_frame(center_points_current_frame, boxes)

    if count <= 2:
        # tracking initial cars on current frame
        track_initial_cars(center_points_current_frame, center_points_prev_frame)
    else:
        # update current cars' positions & remove the cars that left the frame
        update_remove_existing_trackings(center_points_current_frame)
        # add new upcoming cars to tracking
        track_upcoming_cars(center_points_current_frame)

    frame_timestamps[count] = time()
    calculate_speed()

    mark_tracking(frame)

    # print("\nTracking Objects")
    # print(tracking_objects)
    # print("\nPrev Tracking Objects")
    # print(tracking_objects_prev)
    cv2.imshow("Frame" , cv2.resize(frame, (960, 540)))
    key = cv2.waitKey(10)

    center_points_prev_frame = center_points_current_frame.copy( )
    tracking_objects_prev = copy.deepcopy(tracking_objects)
    if key == 27:
        break

cap.release()