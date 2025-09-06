import pickle
import cv2
import numpy as np
import sys
import os
sys.path.append('../')
from utilities import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):

        self.minimum_distance = 5
        #Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize  = (15,15), #Size of the search window
            maxLevel = 2, #Number of pyramid levels
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) #Termination criteria for Lucas-Kanade
        )
        #Convert the first frame to grayscale and create a mask for feature detection
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1 #Enable feature detection on left edge
        mask_features[:,900:1050] = 1 #Enable feature detection on right edge

        self.features = dict(
            maxCorners = 100, #Maximum number of corners to detect
            qualityLevel = 0.3, #Minimum quality of corners
            minDistance = 3, #Minimum distance between detected corners
            blockSize = 7, #Size of the block for corner detection
            mask = mask_features #Mask to restrict feature detection to specific regions
        )
    #Adjust object tracking positions based on estimated camera movement
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    #Subtract camera movement to get adjusted position
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    
    #Estimate camera movement across frames using Lucas-Kanade optical flow or load from a stub file
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0] *len(frames)]
        #Convert first frame to grayscale and detect initial features
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)
        #Process each frame to estimate camera movement
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            #Calculate Lucas-Kanade optical flow to track feature points between frames
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None,**self.lk_params)
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0
            #Find the feature with the largest movement to estimate camera shift
            for i, (new,old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()
                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    #Measure x and y displacement for the largest movement
                    camera_movement_x,camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
            #Update camera movement if the displacement exceeds the minimum threshold
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                #Refresh feature points for the next frame
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            old_gray = frame_gray.copy()
        #Save camera movement to stub file if path is provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
        return camera_movement
      
    #Overlay camera movement information on each frame for visualization
    def draw_camera_movement(self,frame,camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frame):
            frame = frame.copy()
            #Create semi-transparent white overlay for text
            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            #Display x and y camera movement values on the frame
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            output_frames.append(frame)
        return output_frames
