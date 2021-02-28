import os
import numpy as np
import cv2

'''
#   breaking video into frames will significantly increase the storage size required, I suggest jpg over png
#   if you are certain and doesn't required to test different image size during training, I would suggest to scale it up/down during this process
    this can greatly reduce the resources required during training.
#   when scaling up/down, you can either preserved the aspect the original aspect ratio or specify an absolute size/shape
    scale   : float, > 0.0
    abs_dim : tuple or two elements
'''

def vid_to_frames(root_dir, dest_dir, rescale=True, preserved_aspect_ratio=True, scale=0.8):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    cat_list = os.listdir(root_dir) # train, test, val
    for cat in cat_list:
        cat_path = os.path.join(root_dir, cat)
        activity_list = os.listdir(cat_path)
        cat_dest_path = os.path.join(dest_dir, cat)
        if not os.path.exists(cat_dest_path):
            os.mkdir(cat_dest_path)
        for activity in activity_list: # loop over every activity folder
            activity_path = os.path.join(cat_path,activity)
            dest_activity_path = os.path.join(cat_dest_path,activity)
            if not os.path.exists(dest_activity_path):
                os.mkdir(dest_activity_path)
            write_frames(activity_path,dest_activity_path, rescale, preserved_aspect_ratio, scale)

def write_frames(activity_path,dest_activity_path, rescale=True, preserved_aspect_ratio=True, scale=0.8, frame_format='jpg', abs_dim=None):
    vid_list = os.listdir(activity_path)
    print(vid_list)
    for vid in vid_list:
        dest_folder_name = vid[:-4] # remove extension postfix
        dest_folder_path = os.path.join(dest_activity_path,dest_folder_name)
        if not os.path.exists(dest_folder_path):
            os.mkdir(dest_folder_path)
            
        vid_path = os.path.join(activity_path,vid)
        print ('video path: ', vid_path)
        cap = cv2.VideoCapture(vid_path)
        
        ret=True
        frame_num=0
        while ret:
            ret, img = cap.read()
            output_file_name = 'img_{:06d}'.format(frame_num) + '.{}'.format(frame_format) # img_000001.png
            # output frame to write 'activity_data/Archery/v_Archery_g01_c01/img_000001.png'
            output_file_path = os.path.join(dest_folder_path, output_file_name)
            frame_num += 1
            print("Frame no. ", frame_num)
            try:
                if rescale:
                    if preserved_aspect_ratio:
                        dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                    else:
                        dim = abs_dim
                    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(output_file_path, img) # writing frames to defined location
            except Exception as e:
                print(e)
            if ret==False:
                cv2.destroyAllWindows()
                cap.release()

if __name__ == '__main__':
    root = 'C:/Users\AI-lab/Documents/UCF50_split/'
    dest = 'C:/Users\AI-lab/Documents/activity_file/data_files/'
    vid_to_frames(root, dest)
    
