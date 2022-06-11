import torch
from torch.autograd import Variable
import os
import logging
import numpy as np
import skimage.io as io

from utils.utils import build_transforms, get_torch_device
from utils.load_model import load_feature_extractor

import subprocess as sp
import shutil

os.environ['KMP_DUPLICATE_LIB_OK']='True'



def extract_feature():
    device = get_torch_device()
    print(device)
    net = load_feature_extractor(MODEL_TYPE, MODEL_PATH, device).eval() # c3 Conv 모델 불러오기

    # current location
    temp_path = '.\\temp' # frames 잠시 저장해놓을 파일 경로
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    video_list = []
    for video_folder in os.listdir(VIDEO_DIR):
        temp_folder = os.path.join(temp_path, video_folder)

        for video in os.listdir(os.path.join(VIDEO_DIR, video_folder)):
            video_list.append(os.path.join(video_folder, video)) # 비디오 파일 경로들 저장

    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    error_fid = open('error_extract.txt', 'w')
    
    num_videos = len(video_list)
    

    for video_num, video_name in enumerate(video_list):
        video_path = os.path.join(VIDEO_DIR, video_name)
        video_name = video_name.split('.')[0]
        video_folder = video_name.split('/')[0]
        frame_path = os.path.join(temp_path, video_name)
        temp_video_name = video_name+'.txt'
        # 오류로 frame 추출이 중간에 중단됐을 경우를 대비
        if os.path.exists(frame_path):
            shutil.rmtree(frame_path)
        
        # 오류로 중간에 끊겼을 경우 feature가 추출된 비디오는 패스
        if not os.path.exists(frame_path):
            os.mkdir(frame_path)
        
        print(f'\nExtracting video frames({video_name}({video_num+1}/{num_videos})) ...')
        
        # using ffmpeg to extract video frames into a temporary folder
        # example: ffmpeg -i video_validation_0000051.mp4 -q:v 2 -f image2 output/image%5d.jpg
#         os.system('ffmpeg -i ' + video_path + ' -q:v 2 -f image2 ' + frame_path + '/image_%6d.jpg')
        print("=============================================================================")
        sp.run(['ffmpeg',
                '-i', video_path, 
                '-q:v','2',
                '-f','image2',
                os.path.join(frame_path,'image_%6d.jpg'),
                '-hide_banner'])
        
        print("=============================================================================")
        
# =============================================================================       

        print(f'Extracting features({video_name}({video_num+1}/{num_videos})) ...')
        total_frames = len(os.listdir(frame_path))
        if total_frames == 0:
            error_fid.write(video_name+'\n')
            print('Fail to extract frames for video: %s'%video_name)
            continue

        valid_frames = total_frames / nb_frames * nb_frames
        n_feat = valid_frames / nb_frames
        n_batch = n_feat / BATCH_SIZE

        if n_feat - n_batch*BATCH_SIZE > 0:
            n_batch = n_batch + 1

        n_batch = int(n_batch)
        n_feat = int(n_feat)
        features = []
        print("progress: {}/{}, current_video: {}, n_batch={}, batch_size={}".format(video_num+1, num_videos, video_name, n_batch, BATCH_SIZE))

        for i in range(n_batch-1):
            #print('done')
            input_blobs = None
            # print("n_batch={}, cur_batch={}".format(n_batch, i))
            for j in range(BATCH_SIZE):
                clip = np.array([io.imread(os.path.join(frame_path, 'image_{:06d}.jpg'.format(k))) for k in range((i*BATCH_SIZE+j) * nb_frames+1, min((i*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1))])  
                clip = build_transforms(mode=MODEL_TYPE)(torch.from_numpy(clip))
                clip = clip[None, :]

                if input_blobs is None:
                    input_blobs = clip
                else:
                    input_blobs = torch.cat([input_blobs, clip], dim=0)

            input_blobs = Variable(input_blobs).to(device)
            batch_output = net(input_blobs)
            batch_feature = batch_output.data.cpu()
            features.append(batch_feature)
            torch.cuda.empty_cache()
            

        # The last batch
        input_blobs = None
        # print("n_batch={}, cur_batch={}".format(n_batch, i+1))

        for j in range(n_feat-(n_batch-1)*BATCH_SIZE):
            clip = np.array([io.imread(os.path.join(frame_path, 'image_{:06d}.jpg'.format(k))) for k in range(((n_batch-1)*BATCH_SIZE+j) * nb_frames+1, min(((n_batch-1)*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1))])
            clip = build_transforms(mode=MODEL_TYPE)(torch.from_numpy(clip))
            clip = clip[None, :]

            if input_blobs is None:
                input_blobs = clip
            else:
                input_blobs = torch.cat([input_blobs, clip], dim=0)

        input_blobs = Variable(input_blobs).to(device)
        batch_output = net(input_blobs)
        batch_feature = batch_output.data.cpu()
        features.append(batch_feature)
        torch.cuda.empty_cache()
        
        
        features = torch.cat(features, 0)
        features = features.numpy()
        segments_feature = []
        num = 32
        thirty2_shots = np.round(np.linspace(0, len(features) - 1, num=num+1)).astype(int)
        for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
            if ss == ee:
                temp_vect = features[min(ss, features.shape[0] - 1), :]
            else:
                temp_vect = features[ss:ee, :].mean(axis=0)

            temp_vect = temp_vect / np.linalg.norm(temp_vect)
            if np.linalg.norm == 0:
                logging.error("Feature norm is 0")
                exit()
            if len(temp_vect) != 0:
                segments_feature.append(temp_vect.tolist())

        path = os.path.join(OUTPUT_DIR, f"{video_name}.txt")
        video_folder_dir = os.path.join(OUTPUT_DIR, video_folder)
        # if not os.path.isdir(video_folder_dir):
        #     os.mkdir(video_folder_dir)
        with open(path, 'w') as fp:
            for d in segments_feature:
                d = [str(x) for x in d]
                fp.write(' '.join(d) + '\n')

        print('%s has been processed...'%video_name)
        #torch.cuda.empty_cache()
        # clear temp frame folders
        print("clearing temp frames...\n")
        if os.path.exists(frame_path):
            shutil.rmtree(frame_path)
        torch.cuda.empty_cache()
        #os.system('rmdir /s /q ' + frame_path)
    
    #print(error_video_names)
        
def read_features(file_path, feature_dim, cache=None):
    if cache is not None and file_path in cache:
        return cache[file_path]

    if not os.path.exists(file_path):
        raise Exception(f"Feature doesn't exist: {file_path}")
    features = None
    with open(file_path, 'r') as fp:
        #print(file_path)
        data = fp.read().splitlines(keepends=False)
        features = np.zeros((len(data), feature_dim))
        for i, line in enumerate(data):
            for j, f in enumerate(line.split()):
                features[i, j] = float(f)
            #features[i, :] = [float(x) for x in line.split(' ')]

    features = torch.from_numpy(features).float()
    if cache is not None:
        cache[file_path] = features
    return features

if __name__ == "__main__":

    OUTPUT_DIR = './extracted_features/extracted_c3d_features' # features 저장될 경로
    MODEL_PATH = './pretrained/c3d.pickle' # 사용할 3d conv 모델 경로
    
    MODEL_TYPE = 'c3d' # 사용할 3d conv 종류
    VIDEO_DIR = 'E:/dataset/Videos' # 비디오 경로
    BATCH_SIZE = 32
    nb_frames = 16

    extract_feature()
    
