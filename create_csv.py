import pandas as pd
import os

def create_csvfiles(data_path, csv_path, labels):
    print(data_path, csv_path)
    data_dir_list = os.listdir(data_path)
    for data_dir in data_dir_list: # looping over every activity
        label = labels[str(data_dir)]
        video_list = os.listdir(os.path.join(data_path,data_dir))
        for vid in video_list: # looping over every video within an activity
            train_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])
            img_list = os.listdir(os.path.join(data_path,data_dir,vid))
            for img in img_list:# looping over every frame within the video
                img_path = os.path.join(data_path,data_dir,vid,img)
                train_df = train_df.append({'FileName': img_path, 'Label': label,'ClassName':data_dir },ignore_index=True)
            file_name='{}_{}.csv'.format(data_dir,vid)
            train_df.to_csv('{}/{}'.format(csv_path, file_name))

def func(root, csv_path):
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    labels_name = None
    for sub_dir in os.listdir(root):
        data_path = os.path.join(root, sub_dir)
        data_csv = os.path.join(csv_path, sub_dir)
        if not os.path.exists(data_csv):
            os.mkdir(data_csv)
        # print(data_path, data_csv)
        if labels_name is None:
            i = 0
            labels_name=dict()
            for name in os.listdir(data_path):
                labels_name[name] = i
                i += 1
            # print(labels_name)
        create_csvfiles(data_path, data_csv, labels_name)

if __name__ == '__main__':
    root_data_dir = 'C:/Users/AI-lab/Documents/activity_file/data_files/'
    csv_path = 'C:/Users/AI-lab/Documents/activity_file/csv_files/'
    func(root_data_dir, csv_path)