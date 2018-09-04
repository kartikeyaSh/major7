import numpy as np

from keras.models import Model, Sequential
from keras.layers import LSTM, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, Input, MaxPooling3D, TimeDistributed, ZeroPadding3D

from utilities import *

def format_video_array(v_array, clips_count, length=16):
    v_array = v_array.transpose(1,0,2,3)
    v_array = v_array[:clips_count * length, :, :, :]
    v_array = v_array.reshape((clips_count, length, 3, 112, 112))
    v_array = v_array.transpose(0,2,1,3,4)
    return v_array

def analyse_clip(video_path, threshold):
    input_size = (112,112)
    length = 16
    labels = load_labels('/home/kartikeya/major_project_7th_sem/data/labels.txt')
    print("Please wait video is being loaded...")

    #video_path =['/home/kartikeya/major_project_7th_sem/data/a.mp4',
    #         '/home/kartikeya/major_project_7th_sem/data/b.mp4',
    #         '/home/kartikeya/major_project_7th_sem/data/c.mp4',
    #         '/home/kartikeya/major_project_7th_sem/data/d.mp4']

    #v_array = convert_video_to_np_array_multiple(video_path)

    v_array = convert_video_to_np_array(video_path)

    frames_count = get_frames_count(video_path)
    #frames_count = 2301
    duration = get_video_duration(video_path)
    #duration = 80
    
    fps = frames_count/duration
    
    print("Duration of video in seconds: {:.1f}".format(duration))
    print("Frames per second: {:.1f}".format(fps))
    print("Number of frames: {}".format(frames_count))

    clips_count = frames_count // length
    v_array = format_video_array(v_array, clips_count, length)

    print ("Loading 3D Convolution Network...")

    model = C3D_model()
    model.compile(optimizer='sgd', loss='mse')

    mean_total = np.load("data/c3d-sports1M_mean.npy")
    mean = np.mean(mean_total, axis=(0,2,3,4), keepdims=True)

    print "Extracting Features..."

    X = v_array - mean

    Y = model.predict(X, batch_size =1, verbose=1)

    
    print "Loading temporal localization network..."

    model_localization = temporal_localization_network()
    model_localization.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    print "Predicting temporal locations of events"

    Y = Y.reshape(clips_count, 1, 4096)

    prediction = model_localization.predict(Y, batch_size=1,verbose=1)
    prediction = prediction.reshape(clips_count,201)

    prediction_smoothed = smoothing(prediction)

    activities_idx, startings, endings, scores = activity_localization(prediction_smoothed, threshold)
    
    print "Detection:"
    print "Score\tInterval\t\tActivity"
    
    for idx, s, e, score in zip(activities_idx, startings, endings, scores):
        start = s* float(length)/fps
        end = e*float(length)/fps
        label = labels[idx]
        print "{:.4f}\t{:.1f}s - {:.1f}s\t\t{}".format(score,start,end,label)

    
def C3D_model():
    model = Sequential()

    #1st group
    model.add(Convolution3D(64,3,3,3,activation='relu',
                            border_mode='same',name='conv1',
                            subsample=(1,1,1), input_shape=(3,16,112,112),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2),
                           border_mode='valid',name='pool1'))

    #2nd group
    model.add(Convolution3D(128,3,3,3, activation='relu',
                            border_mode='same',name='conv2',
                            subsample=(1,1,1),trainable=False))
    model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),
                           border_mode='valid', name='pool2'))

    #3rd group
    model.add(Convolution3D(256,3,3,3, activation='relu',
                            border_mode='same', name='conv3a',
                            subsample=(1,1,1), trainable=False))
    model.add(Convolution3D(256,3,3,3, activation='relu',
                            border_mode='same', name='conv3b',
                            subsample=(1,1,1), trainable=False))
    model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),
                           border_mode='valid', name='pool3'))

    #4th group
    model.add(Convolution3D(512,3,3,3, activation='relu',
                            border_mode='same',name='conv4a',
                            subsample=(1,1,1), trainable=False))
    model.add(Convolution3D(512,3,3,3, activation='relu',
                            border_mode='same',name='conv4b',
                            subsample=(1,1,1), trainable=False))
    model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),
                           border_mode='valid', name='pool4'))
    
    #5th group
    model.add(Convolution3D(512,3,3,3, activation='relu',
                            border_mode='same', name='conv5a',
                            subsample=(1,1,1),trainable=False))
    model.add(Convolution3D(512,3,3,3, activation='relu',
                            border_mode='same', name='conv5b',
                            subsample=(1,1,1),trainable=False))
    model.add(ZeroPadding3D(padding=(0,1,1), name='zeropadding'))
    model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten(name='flatten'))

    #FC group
    model.add(Dense(4096, activation='relu', name='fc6', trainable=False))
    model.add(Dropout(0.5, name='do1'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5, name='do2'))
    model.add(Dense(487, activation='softmax', name='fc8'))

    model.load_weights('data/c3d-sports1M_weights.h5')

    #pop the last 4 layers of the model
    for _ in range(0,4):
        model.layers.pop()
    model.outputs= [model.layers[-1].output]
    model.layers[-1].outbound_nodes=[]
    
    return model                          


def temporal_localization_network():
    input_features = Input(batch_shape=(1,1,4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout= Dropout(p=0.5)(input_normalized)
    lstm = LSTM( 512, return_sequences=True, stateful=True, name='lstm1')(input_dropout)
    output_dropout = Dropout(p=0.5)(lstm)
    output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    model.load_weights('data/temporal-location_weights.hdf5')
    
    return model

    
def get_classification(prediction):
    prob = np.mean(prediction, axis=0)
    labels_index = np.argsort(prob[1:])[::-1]+1
    scores = prob[labels_index]/np.sum(prob[1:])

    return labels_index[:1], scores[:1]

def smoothing(x, k=5):
    l = len(x)
    s = np.arange(-k, l-k)
    e = np.arange(k, l+k)

    s[s<0] = 0
    e[e>=l]=l-1
    y = np.zeros(x.shape)

    for i in range(0,l):
        y[i] = np.mean(x[s[i]:e[i]], axis = 0)

    return y

def activity_localization(prob, threshold = 0.25):

    a_index, _ = get_classification(prob)
    a_index = a_index[0]

    a_prob = 1-prob[:,0]
    a_tag = np.zeros(a_prob.shape)
    a_tag[a_prob >= threshold] = 1

    padded = np.pad(a_tag, pad_width=1, mode='constant')
    dif = padded[1:] - padded[:-1]

    indexes = np.arange(dif.size).astype(np.float32)
    startings  = indexes[dif==1]
    endings = indexes[dif == -1]

    activities_index = []
    scores = []

    for segment in zip(startings, endings):
        s,e = map(int, segment)
        activities_index.append(a_index)
        scores.append(np.mean(prob[s:e, a_index]))

    return activities_index, startings, endings, scores
                       
analyse_clip('/home/kartikeya/major_project_7th_sem/data/poloc.mp4',0.25)
