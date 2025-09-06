from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image):
        #Reshape image to 2D array where each row is a pixel with RGB values
        image_2d = image.reshape(-1,3)
        #Perform K-means with 2 clusters to separate jersey color from background
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,boundary_box):
        #Crop player from frame using bounding box
        image = frame[int(boundary_box[1]):int(boundary_box[3]),int(boundary_box[0]):int(boundary_box[2])]
        #Use only top half to avoid shorts/socks and focus on jersey
        top_half_image = image[0:int(image.shape[0]/2),:]
        #Get clustering model to separate jersey from background
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_
        #Reshape the labels back to image dimensions
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])
        #Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        player_colors = []
        #Get jersey color for each player
        for _, player_detection in player_detections.items():
            boundary_box = player_detection["boundary_box"]
            player_color =  self.get_player_color(frame,boundary_box)
            player_colors.append(player_color)
          
        #Cluster all player colors into 2 teams
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0] #Team 1 color
        self.team_colors[2] = kmeans.cluster_centers_[1] #Team 2 color


    def get_player_team(self,frame,player_boundary_box,player_id):
        #Return cached result if we already know this player's team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_boundary_box)
        #Predict which team this player belongs to based on jersey color
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1
        #Force player 98 goalkeeper to be team 1 
        if player_id == 98:
            team_id = 1
        #Cache result so we don't have to recalculate
        self.player_team_dict[player_id] = team_id

        return team_id
