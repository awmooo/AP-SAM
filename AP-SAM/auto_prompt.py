import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
# #载入
# path_dir = 'D://QQ/1144078933//FileRecv//data//0-5um//images//'
# label_dir ='D://QQ/1144078933//FileRecv//data//0-5um//masks//'
# name ='Boye5-2F-ID-103_009-Mag-30000x-Scale-4um-Depth-4109.65m-crop.png'
# path = path_dir + name
# label = label_dir + name
# img_original=cv2.imread(path,0)
# label = cv2.imread(label,0)
#
# means, std = cv2.meanStdDev(img_original)
# means = int(means)/2
# dev = int(std)
# print('means: {}, \n dev: {}'.format(means, dev))
# #高斯滤波
# img_original=cv2.GaussianBlur(img_original,(5,5),5)
# #全局阈值分割
# retval,img_global=cv2.threshold(img_original,means,255,cv2.THRESH_BINARY_INV)
# #自适应阈值分割
# ret, otsu = cv2.threshold(img_original,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# print('ret:',ret)
# img_ada_mean=cv2.adaptiveThreshold(img_original,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,225,means)
# img_ada_gaussian=cv2.adaptiveThreshold(img_original,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,225,means)
#
#
# kernel = np.ones((4, 4), np.uint8)
# img_ada_mean = cv2.dilate(img_ada_mean, kernel, iterations = 1)
# img_ada_gaussian = cv2.dilate(img_ada_gaussian, kernel, iterations = 1)
#
# imgs=[img_original,img_global,img_ada_mean,img_ada_gaussian,otsu,label]
# titles=['Original Image','Global Thresholding','Adaptive Mean','Adaptive Guassian','Otsu','G.T.']
#
#
# #显示图片
# for i in range(len(imgs)):
#     plt.subplot(2,3,i+1)
#     plt.imshow(imgs[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

def Auto_mask_prompt(imgs) -> torch.Tensor:
    """
    imgs:tensor(B,C,H,W) Origin input images,make sure it is RGB channel and not norm
    return: Mask batch(B,1,H,W),dtype=unit8
    """
    # (B,C,H,W)->(B,H,W,C)
    imgs = imgs.permute(0,2,3,1)
    if imgs.is_cpu == False:
        imgs = imgs.to('cpu')
    np_imgs = np.array(imgs,dtype=np.uint8)
    batch_promptlist = []
    for i in range(len(np_imgs)):
        gray = cv2.cvtColor(np_imgs[i],cv2.COLOR_RGB2GRAY)
        means, std = cv2.meanStdDev(gray)
        means = int(means[0,0]) / 2
        # Gaussian filter
        img_original = cv2.GaussianBlur(gray, (5, 5), 5)
        # Global Threshold
        retval, img_global = cv2.threshold(img_original, means, 255, cv2.THRESH_BINARY_INV)

        # Adaptive Threshold
        img_ada_mean = cv2.adaptiveThreshold(img_original, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81,means)
        img_ada_gaussian = cv2.adaptiveThreshold(img_original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 81, means)
        # Dilate the images
        kernel = np.ones((4, 4), np.uint8)
        img_ada_mean = cv2.dilate(img_ada_mean, kernel, iterations=1)
        img_ada_gaussian = cv2.dilate(img_ada_gaussian, kernel, iterations=1)

        mask_prompt = img_global + img_ada_mean + img_ada_gaussian

        mask_prompt = torch.tensor(mask_prompt)
        mask_prompt = mask_prompt.unsqueeze(0)
        batch_promptlist.append(mask_prompt)
    batch_prompts = torch.stack(batch_promptlist,dim=0)

    return batch_prompts



if __name__ == '__main__':
    ori = cv2.imread('/home/upc/Documents/Cracksegdata/split_data/0-5um/train/images/Boye5-1HF-ID-2_1_010-Mag-15000x-Scale-5um-Depth-3729.2m-crop.png')
    a = cv2.cvtColor(ori,cv2.COLOR_BGR2RGB)
    a = torch.tensor(a).unsqueeze(0).cuda()
    a = a.permute(0,3,1,2)
    b = Auto_mask_prompt(a)
    b = b.squeeze(0).squeeze(0)
    b = np.array(b)

    plt.subplot(1, 2, 1)
    plt.imshow(ori)
    plt.title('image')
    plt.xticks([])
    plt.yticks([])


    plt.subplot(1, 2, 2)
    plt.imshow(b, 'gray')
    plt.title('mask prompt')
    plt.xticks([])
    plt.yticks([])


    plt.savefig("mask_prompt.svg", dpi=300, format='svg')
    plt.close()








