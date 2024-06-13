import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from pytorch_i3d import InceptionI3d

def video_to_tensor(pic):
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

class CenterCrop(object):
    def __init__(self, size):
        self.size = (size,size)
    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+tw, :]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

def load_rgb_frames_from_video(vid_root, vid, start, num):
    video_path = os.path.join(vid_root, vid + '.mp4')

    vidcap = cv2.VideoCapture(video_path)
    num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))-1

    frames = []

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(num):
        success, img = vidcap.read()
        w, h, c = img.shape
        img = cv2.resize(img, (int(h/w*244), 224))
        img = (img / 255.) * 2 - 1
        frames.append(img)

    return np.asarray(frames, dtype=np.float32)

def run(root="",
        weights=None,
        num_classes=2000,
        filename="",
        start_frame=0, 
        duration=0):

    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    if root == "" or filename == "":
        return
    imgs = load_rgb_frames_from_video(root, filename, start_frame, duration)
    transforms = CenterCrop(224)
    imgs = transforms(imgs)
    ret_img = video_to_tensor(imgs)

    ret_img = ret_img[np.newaxis, :, :, :, :]
    per_frame_logits = i3d(ret_img)

    predictions = torch.max(per_frame_logits, dim=2)[0]
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    out_probs = np.sort(predictions.cpu().detach().numpy()[0])
    label_index = torch.argmax(predictions[0]).item()

    print(out_labels[-5:])
    print(label_index)
    return label_index

if __name__ == '__main__':
    # ================== test i3d on a dataset ==============
    # need to add argparse
    num_classes = 100
    train_split = 'preprocess/nslt_{}.json'.format(num_classes)
    weights = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'

    SignLanguage_label=run(root="data", weights=weights, num_classes=num_classes, filename="07070", start_frame=0, duration=30)
