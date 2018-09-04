from utilities import *

video_path ='/home/kartikeya/major_project_7th_sem/data/test_video.mp4'
labels_path = '/home/kartikeya/major_project_7th_sem/data/labels.txt'

video_array = convert_video_to_np_array(video_path)
frames_count = get_frames_count(video_path)
video_duration = get_video_duration(video_path)

print 'video_array,video_array.shape',video_array, video_array.shape

print 'frames_count',frames_count

print 'video_duration' ,video_duration

print 'labels\n',load_labels(labels_path)

video_path =['/home/kartikeya/major_project_7th_sem/data/a.mp4',
             '/home/kartikeya/major_project_7th_sem/data/b.mp4',
             '/home/kartikeya/major_project_7th_sem/data/c.mp4',
             '/home/kartikeya/major_project_7th_sem/data/d.mp4']

video_array = convert_video_to_np_array_multiple(video_path)

print video_array.shape
