def get_center_of_bounding_box(bounding_box):
    x1, y1, x2, y2 = bounding_box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bounding_box_width(bounding_box):
    return bounding_box[2]-bounding_box[0]

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(bounding_box):
    #Get bottom center point (foot position) of bounding box
    x1, y1, x2, y2 = bounding_box
    return int((x1 + x2) / 2), int(y2)
