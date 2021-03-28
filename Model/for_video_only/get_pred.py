from __future__ import print_function, division
import os
import torch
import pandas as pd
import argparse
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
from model import *
from dataset import *
from cvtransforms import *


def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read()  # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    return video[..., ::-1]


def load_vid_pred(model, phase, optimizer, args, use_gpu):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=0.)

    # VIDEO TO IMAGE
    basedir = args.file_path
    # print(basedir)
    basedir_to_save = 'prednpzdata/'
    filenames = glob.glob(os.path.join(basedir, '*.mp4'))
    for filename in filenames:
        # print(filename)
        data = extract_opencv(filename)[:, 115:211, 79:175]
        path_to_save = os.path.join(basedir_to_save, 'test.npz')
        if not os.path.exists(os.path.dirname(path_to_save)):
            try:
                os.makedirs(os.path.dirname(path_to_save))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        np.savez(path_to_save, data=data)

    # load dataset
    input_data = np.load(os.path.join(basedir_to_save, 'test.npz'))['data']
    inputs = np.stack([cv2.cvtColor(input_data[_], cv2.COLOR_BGR2GRAY)
                       for _ in range(29)], axis=0)
    inputs = inputs / 255.
    inputs = np.reshape(
        inputs, (1, inputs.shape[0], inputs.shape[1], inputs.shape[2]))
    # print(inputs.shape)
    inputs = torch.from_numpy(inputs)
    # print(inputs.shape)
    batch_img = CenterCrop(inputs.cpu().numpy(), (88, 88))
    batch_img = ColorNormalize(batch_img)
    batch_img = np.reshape(
        batch_img, (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
    inputs = torch.from_numpy(batch_img)
    inputs = inputs.float().permute(0, 4, 1, 2, 3)

    # print(inputs.shape)

    outputs = model(inputs)
    outputs = torch.mean(outputs, 1)
    # print(outputs.shape)
    softmax = nn.Softmax(dim=1)
    _, preds = torch.max(softmax(outputs).data, 1)
    # print(preds)
    with open('../label_sorted.txt') as myfile:
        data_dir = myfile.read().splitlines()
        # print(data_dir)
        List = {}
        # print(preds.tolist())
        for i, x in enumerate(preds.tolist()):
            # print(x)
            for j, elem in enumerate(data_dir):
                if j == x:
                    List[i] = elem
        print('Word Predicted from the Input Video: --- '+List[0])


def get_data(args, use_gpu):

    # Initialize model
    model = lipreading(mode='finetuneGRU', inputDim=256, hiddenDim=512,
                       nClasses=500, frameLen=29, every_frame='every_frame')

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    # Print model's state_dict
    # print("Model's state_dict:")
    model_dict = model.state_dict()
    # for param_tensor in model_dict:
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    pretrained_dict = torch.load(args.model_path)
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('*** model has been successfully loaded! ***')
    model.eval()
    print('*** model has been successfully evaluated! ***')

    load_vid_pred(model, 'test', optimizer, args, use_gpu)


def get_pred_for_video():
    parser = argparse.ArgumentParser(
        description='Load and predict word for single video')
    parser.add_argument('--file_path', default='predfile/',
                        help='path to video required for prediction')
    parser.add_argument('--model_path', default='Video-only/Video_only_model.pt',
                        help='path to model required for prediction')
    args = parser.parse_args()

    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    get_data(args, use_gpu)


if __name__ == '__main__':
    get_pred_for_video()

