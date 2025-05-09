import numpy as np
import cv2
import scipy.io as sio

def set_img_color(colors, background, img, pred, gt, show255=True):
    for i in range(0, len(colors)):
        if i != background:
            img[np.where(pred == i)] = colors[i]
    if show255:
        # 255改成了0
        img[np.where(gt==background)] = 0
    return img

def show_prediction(colors, background, img, pred, gt):
    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, pred, gt)
    final = np.array(im)
    return final

def show_img(colors, background, img, clean, gt, *pds):
    im1 = np.array(img, np.uint8)
    #set_img_color(colors, background, im1, clean, gt)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    final = np.array(im1)
    # the pivot black bar
    pivot = np.zeros((im1.shape[0], 15, 3), dtype=np.uint8)
    for pd in pds:
        im = np.array(img, np.uint8)
        # pd[np.where(gt == 255)] = 255
        # 矩阵往里面放肯定会改
        final = set_img_color(colors, background, im, pd, gt)
        # 我加了这一句
        final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        # final = np.column_stack((final, pivot))
        # final = np.column_stack((final, im))

    # im = np.array(img, np.uint8)
    # set_img_color(colors, background, im, gt, gt)
    # final = np.column_stack((final, pivot))
    # final = np.column_stack((final, im))
    return final

def get_colors(class_num):
    colors = []
    for i in range(class_num):
        colors.append((np.random.random((1,3)) * 255).tolist()[0])

    return colors

def get_ade_colors():
    colors = sio.loadmat('./color150.mat')['colors']
    colors = colors[:,::-1,]
    colors = np.array(colors).astype(int).tolist()
    colors.insert(0,[0,0,0])

    return colors

def print_iou(iu, mean_pixel_acc, class_names=None, show_no_back=False, no_print=False):
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i+1)
        else:
            cls = '%d %s' % (i+1, class_names[i])
        lines.append('%-8s\t%.3f%%' % (cls, iu[i] * 100))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    if show_no_back:
        lines.append('----------------------------')
        lines.append('mIou: {}%     mIoU(no_back): {}%      mACC: {}%'.format(mean_IU * 100, mean_IU_no_back*100, mean_pixel_acc*100))
    else:
        lines.append('----------------------------')
        lines.append('mIoU: {}%     mACC: {}%'.format(mean_IU * 100, mean_pixel_acc*100))
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line


