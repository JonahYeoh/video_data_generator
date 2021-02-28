'''
    auto separate files into train, val, test according to the ratio stated
'''
import os
import numpy as np
import shutil
from sklearn.utils import shuffle

def move_video_list(videos, src_dir, dest_dir, activity_name, keep_original=True):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    activity_dir = os.path.join(dest_dir, activity_name)
    if not os.path.exists(activity_dir):
        os.mkdir(activity_dir)
    for video in videos:
        video_path = os.path.join(src_dir, video)
        dest_video_path = os.path.join(activity_dir, video)
        if keep_original:
            shutil.copy2(video_path, dest_video_path)
        else:
            os.replace(video_path, dest_video_path)

def split_data(root_dir, dest_dir, ratio=[], shuffle=True, balance=True):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    train_path = os.path.join(dest_dir, 'train')
    val_path = os.path.join(dest_dir, 'val')
    test_path = os.path.join(dest_dir, 'test')
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    
    activity_dir = os.listdir(root_dir)
    print('total {} classes'.format(len(activity_dir)))
    for activity in activity_dir:
        activity_path = os.path.join(root_dir, activity)
        videos = os.listdir(activity_path)
        np.random.shuffle(videos)
        u_space = np.sum(ratio)
        if u_space == 1.0:
            print('Warning: The total of variable ratio should be 1.0 but get {}'.format(u_space))
            # for the sake of code simplicity, cases like range error are not handled
        cut1, cut2 = int(len(videos)*ratio[0]), int(len(videos)*ratio[1])
        train = videos[:cut1]
        val = videos[cut1:cut1+cut2]
        test = videos[cut1+cut2:]
        print('>> Class {}: total={}\ntrain={}, val={}, test={}'.format(activity, len(videos), len(train), len(val), len(test)))
        move_video_list(train, activity_path, train_path, activity)
        move_video_list(val, activity_path, val_path, activity)
        move_video_list(test, activity_path, test_path, activity)

if __name__ == '__main__':
    root_dir = 'C:/Users\AI-lab/Documents/UCF50/UCF50/'
    dest_dir = 'C:/Users\AI-lab/Documents/UCF50_split/'

    ratio = [ 0.8, 0.1, 0.1 ] # train, validation, test
    split_data(root_dir, dest_dir, ratio)