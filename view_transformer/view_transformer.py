import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        #Soccer field dimensions in meters 
        field_width = 68     
        field_length = 23.32  
        
        #Pixel coordinates of field corners in the camera view
        self.pixel_vertices = np.array([
            [110, 1035],   #Bottom left corner
            [265, 275],    #Top left corner  
            [910, 260],    #Top right corner
            [1640, 915]    #Bottom right corner
        ])
        
        # Real world field coordinates in meters
        self.target_vertices = np.array([
            [0, field_width],    #Bottom left in meters
            [0, 0],              #Top left in meters
            [field_length, 0],   #Top right in meters
            [field_length, field_width]  #Bottom right in meters
        ])
        
        #Convert to float32 for OpenCV operations
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)
        
        #Create perspective transformation matrix
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        
        #Check if point is within the field boundaries
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None
        
        #Reshape point for OpenCV perspective transform
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        #Transform all tracked object positions from pixel to real world coordinates
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist() #Remove extra dimensions and convert to list
                        tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
