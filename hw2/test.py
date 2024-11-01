from __future__ import print_function
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import argparse
import torch
import numpy as np
import cv2

from data import cfg
from loss_and_anchor import anchor
from detector.mydetector import mydetector
from utils.nms import nms
from utils.box_utils import decode, decode_landm
from utils.misc import load_model, Timer

parser = argparse.ArgumentParser(description='18794 detection')
parser.add_argument('--ckpt', default='./weights/weights_cn20_2gpuResnet50_Final.pth',
                    type=str, help='trained checkpoint')
# parser.add_argument('--ckpt', default='./weights/Resnet50_Final.pth',
#                     type=str, help='trained checkpoint')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='widerface_homework/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.1, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    # net and model
    net = mydetector(cfg=cfg, phase='test')
    net = load_model(net, args.ckpt, args.cpu)
    net.eval()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = 1200
        max_size = 1600
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)

        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = anchor(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]]).to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        scores = scores[keep]
        
        print(scores)
        print(scores.shape)

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]
        scores = scores[:args.keep_top_k]

        # print(image_path)
        # print(dets.shape)
        # print(scores.shape)
        
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        assert len(dets) == len(scores)

        for box, score in zip(dets, scores):
            x_min, y_min, x_max, y_max = box[:4]
            draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)
            draw.text((x_min, y_min), f"{score}")

        for landmark in landms:
            for j in range(0, len(landmark), 2):
                x, y = landmark[j], landmark[j+1]
                draw.ellipse([x-2, y-2, x+2, y+2], outline="red", width=2)
                

        image.save(f"test_{i}.png")
        plt.imshow(image)
        plt.show()

        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()
        
        print(dets)
        if i == 4:
            exit()



