import os
import time
import math
import torch
import argparse
import datetime
from email_sys.email_sys import send_email
import matplotlib.pyplot as plt

from data import mydataloader, collate, preproc, cfg
from loss_and_anchor import Loss, anchor
from detector.mydetector import mydetector

parser = argparse.ArgumentParser(description='18794 detection')
parser.add_argument('--data_path', default='widerface_homework/train/label.txt')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--save_path', default='./weights/')

args = parser.parse_args()


def train():
    torch.cuda.set_device("cuda:4")
    net = mydetector(cfg=cfg)
    if cfg['ngpu'] > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()
    net.train()
    torch.backends.cudnn.benchmark = True

    epoch = 0
    dataset = mydataloader(args.data_path, preproc(img_dim=cfg['image_size'],
                                                         rgb_means=(104, 117, 123)))
    traindata = torch.utils.data.DataLoader(dataset,
                                            cfg['batch_size'],
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            collate_fn=collate)

    epoch_size = math.ceil(len(dataset) / cfg['batch_size'])
    max_iter = cfg['epoch'] * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = Loss(2, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = anchor(cfg, image_size=(cfg['image_size'], cfg['image_size']))
    with torch.no_grad():
        priors = priorbox.forward().cuda()

    losses_loc = []
    losses_landm = []
    losses_cla = []


    for iteration in range(0, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(traindata)
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)
        
        # print(out[0].shape)
        # print(out[1].shape)
        # print(out[2].shape)
        
        # print(priors.shape)
        # print(len(targets))

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm * 0
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        if iteration % 20 == 0:
            print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                .format(epoch, cfg['epoch'], (iteration % epoch_size) + 1,
                epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(),
                        loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

            losses_loc.append(loss_l.item())
            losses_cla.append(loss_c.item())
            losses_landm.append(loss_landm.item())
            
        # print(loss)

    plt.plot(losses_loc)
    plt.savefig("loss_loc.png")
    plt.clf()
    
    plt.plot(losses_cla)
    plt.savefig("loss_cla.png")
    plt.clf()

    plt.plot(losses_landm)
    plt.savefig("loss_landm.png")
    plt.clf()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    torch.save(net.state_dict(), args.save_path + cfg['name'] + '_Final.pth')


def adjust_learning_rate(optimizer, epoch, step_index, iteration, epoch_size):
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = args.lr * (0.1 ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        send_email(str(e), "Exception")
        raise e
