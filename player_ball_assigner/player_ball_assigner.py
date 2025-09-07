import sys
sys.path.append('../')
from utilities import get_center_of_bounding_box, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70 #Max distance in pixels to consider ball possession

    def assign_ball_to_player(self, players, ball_bounding_box):
        ball_position = get_center_of_bounding_box(ball_bounding_box)

        minimum_distance = 99999 #Start with large number to find closest player
        assigned_player = -1 #-1 means no player assigned

        for player_id, player in players.items():
            player_bounding_box = player['bounding_box']
            #player_position = get_center_of_bounding_box(player_bounding_box)
          
            #Check distance from both feet (left and right bottom corners of bounding box)
            distance_left = measure_distance((player_bounding_box[0], player_bounding_box[-1]), ball_position)
            distance_right = measure_distance((player_bounding_box[2], player_bounding_box[-1]), ball_position)
            distance = min(distance_left, distance_right) #Use whichever foot is closer
            #Only consider players within reasonable distance
            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player #player_id or -1 if no one is close enough
