import cv2

def read_video(video_path):
    #Loads video into memory 
    video_capture = cv2.VideoCapture(video_path)  
    frames = []
    while True:
        read_success, frame = video_capture.read() 
        if not read_success:  
            break
        frames.append(frame)
    video_capture.release()  #Free video capture resources
    return frames  #list of frames as numpy arrays

def save_video(output_video_frames, output_video_path):
    #Uses XVID codec at 24 FPS hardcoded
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #VideoWriter needs (width, height) but shape gives (height, width, channels) so we swap
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))  
    
    for frame in output_video_frames:
        video_writer.write(frame)
    video_writer.release()  #Free video writer resources
