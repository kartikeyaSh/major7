import numpy as np
import cv2

frame_width, frame_height = (112,112)

def load_labels(filename):
    f = open(filename)
    lines = f.readlines()
    labels = []
    
    for line in lines:
        label = (line.split('\t'))[1]
        labels.append((label.split('\n'))[0])

    f.close()
    
    return labels
    
def convert_video_to_np_array(path):

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception('Video not found.')

    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for i in range(end_frame):
        ret, frame = cap.read()
        frame = cv2.resize (frame,(frame_width, frame_height))
        frames.append(frame)

    video = np.array(frames, dtype=np.float32)
    #to convert it into dimension ordering of theano
    video = video.transpose(3, 0, 1, 2)
    return video

def convert_video_to_np_array_untransposed(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception('Video not found.')

    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for i in range(end_frame):
        ret, frame = cap.read()
        frame = cv2.resize (frame,(frame_width, frame_height))
        frames.append(frame)

    video = np.array(frames, dtype=np.float32)
    return video
    
def convert_video_to_np_array_multiple(paths):
    videos = convert_video_to_np_array_untransposed(paths[0])
    for i in range(1,len(paths)):
        videos = np.concatenate((videos,convert_video_to_np_array_untransposed(paths[i])))

    videos= videos.transpose(3,0,1,2)
    return videos

        
def get_frames_count (path):

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception('Video not found')

    num_frames = int( cap.get( cv2.CAP_PROP_FRAME_COUNT))
    
    return num_frames

def get_video_duration (path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception('Video not found')
    
    num_frames = int( cap.get( cv2.CAP_PROP_FRAME_COUNT))

    #frames per second
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    duration = num_frames/fps

    return duration
