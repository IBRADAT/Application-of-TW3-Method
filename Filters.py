import os
import glob
import cv2 as cv
import numpy as np
# import Full_Hand_Objct_Detection as fhd
import RandomForestFunction as RF

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def filter1(path):
    imge = cv.imread(path)
    # img2 = np.zeros(imge.shape[:2], dtype='uint8')
    
    img = cv.cvtColor(imge, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5,5),7)
   
    # compute gradients along the x and y axis, respectively
    gX = cv.Sobel(img, cv.CV_64F, 1, 0)
    gY = cv.Sobel(img, cv.CV_64F, 0, 1)
    
    # compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    
    ret1,th = cv.threshold(img,50,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
    

    kernel = np.ones((2, 2), np.uint8)
    erosion = cv.erode(magnitude, kernel, iterations = 1)
    img = cv.dilate(erosion, kernel, iterations = 1)
    img = np.uint8(np.absolute(img))
    
    img = cv.bitwise_and(th, img)
    
    img = cv.GaussianBlur(img, (5,5),2)
    
    img = cv.equalizeHist(img)
    
    h, w = img.shape[:2]
        
    if h > w:
        img = cv.copyMakeBorder(img, 0, 0, h - w, h - w, cv.BORDER_CONSTANT, None, value = 0)
    elif h < w:
        img = cv.copyMakeBorder(img, w - h, w - h, 0, 0, cv.BORDER_CONSTANT, None, value = 0)
    
    
    return img

def filter2(path):
    imge = cv.imread(path)
    img = cv.cvtColor(imge, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5,5),7)
       
    # compute gradients along the x and y axis, respectively
    gX = cv.Sobel(img, cv.CV_64F, 1, 0)
    gY = cv.Sobel(img, cv.CV_64F, 0, 1)
    
    # compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    
    ret1,th = cv.threshold(img,50,255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    erosion = cv.erode(magnitude, kernel, iterations = 1)
    img = cv.dilate(erosion, kernel, iterations = 1)
    img = np.uint8(np.absolute(img))
    
    img = cv.bitwise_and(th, img)
    
    h, w = img.shape[:2]
        
    if h > w:
        img = cv.copyMakeBorder(img, 0, 0, h - w, h - w, cv.BORDER_CONSTANT, None, value = 0)
    elif h < w:
        img = cv.copyMakeBorder(img, w - h, w - h, 0, 0, cv.BORDER_CONSTANT, None, value = 0)
    
    
    return img

def filter3(path):   
    
    image = cv.imread(path)
    #img2 = np.zeros(imge.shape[:2], dtype='uint8')
    
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5,5),7)
       
    # compute gradients along the x and y axis, respectively
    gX = cv.Sobel(img, cv.CV_64F, 1, 0)
    gY = cv.Sobel(img, cv.CV_64F, 0, 1)
    
    # compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    img = np.uint8(np.absolute(magnitude))
    
    h, w = img.shape[:2]
    
    for i in range(0, w):
        for j in range(0, h):
            if img[j][i] < 20:
                img[j][i] = 0
    
    if h > w:
        img = cv.copyMakeBorder(img, 0, 0, h - w, h - w, cv.BORDER_CONSTANT, None, value = 0)
    elif h < w:
        img = cv.copyMakeBorder(img, w - h, w - h, 0, 0, cv.BORDER_CONSTANT, None, value = 0)
    
    img = clahe.apply(img)
    
    ret,th = cv.threshold(magnitude,50,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    h, w = img.shape[:2]
    
    for i in range(0, w):
        for j in range(0, h):
            if img[j][i] < ret + 10:
                img[j][i] = 0
    
    img = clahe.apply(img)
    
    for i in range(0, w):
        for j in range(0, h):
            if img[j][i] < 20:
                img[j][i] = 0
    
    return img

def adaptiveFilterMean(path):
    imge = cv.imread(path)
    # img2 = np.zeros(imge.shape[:2], dtype='uint8')
    
    imge = cv.cvtColor(imge, cv.COLOR_BGR2GRAY)
    imgG = cv.GaussianBlur(imge, (5,5),7)
   
    # compute gradients along the x and y axis, respectively
    gX = cv.Sobel(imgG, cv.CV_64F, 1, 0)
    gY = cv.Sobel(imgG, cv.CV_64F, 0, 1)
    
    # compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    magnitude = np.uint8(np.absolute(magnitude))
    
    imgG = cv.bitwise_not(imgG)
    img = cv.bitwise_and(magnitude, imgG)
    
    ret1,img = cv.threshold(img,50,255, cv.ADAPTIVE_THRESH_MEAN_C ,cv.THRESH_BINARY + cv.THRESH_OTSU)
    img = cv.bitwise_not(img)
    
    h, w = img.shape[:2]
        
    if h > w:
        img = cv.copyMakeBorder(img, 0, 0, h - w, h - w, cv.BORDER_CONSTANT, None, value = 0)
    elif h < w:
        img = cv.copyMakeBorder(img, w - h, w - h, 0, 0, cv.BORDER_CONSTANT, None, value = 0)
    
    
    return img

def adaptiveFilterGaussian(path):
    imge = cv.imread(path)
    # img2 = np.zeros(imge.shape[:2], dtype='uint8')
    
    imge = cv.cvtColor(imge, cv.COLOR_BGR2GRAY)
    imgG = cv.GaussianBlur(imge, (5,5),7)
   
    # compute gradients along the x and y axis, respectively
    gX = cv.Sobel(imgG, cv.CV_64F, 1, 0)
    gY = cv.Sobel(imgG, cv.CV_64F, 0, 1)
    
    # compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    magnitude = np.uint8(np.absolute(magnitude))
    
    imgG = cv.bitwise_not(imgG)
    img = cv.bitwise_and(magnitude, imgG)
    
    ret1,img = cv.threshold(img,50,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C ,cv.THRESH_BINARY + cv.THRESH_OTSU)
    img = cv.bitwise_not(img)
    
    h, w = img.shape[:2]
        
    if h > w:
        img = cv.copyMakeBorder(img, 0, 0, h - w, h - w, cv.BORDER_CONSTANT, None, value = 0)
    elif h < w:
        img = cv.copyMakeBorder(img, w - h, w - h, 0, 0, cv.BORDER_CONSTANT, None, value = 0)
    
    
    return img

def adaptiveFilterGaussian2(path):
    imge = cv.imread(path)
    # img2 = np.zeros(imge.shape[:2], dtype='uint8')
    
    imge = cv.cvtColor(imge, cv.COLOR_BGR2GRAY)
    imgG = cv.GaussianBlur(imge, (5,5),7)
   
    # compute gradients along the x and y axis, respectively
    gX = cv.Sobel(imgG, cv.CV_64F, 1, 0)
    gY = cv.Sobel(imgG, cv.CV_64F, 0, 1)
    
    # compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    magnitude = np.uint8(np.absolute(magnitude))
    
    ret1,img = cv.threshold(magnitude,50,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY + cv.THRESH_OTSU)

    img = cv.bitwise_not(img)
    
    h, w = img.shape[:2]
        
    if h > w:
        img = cv.copyMakeBorder(img, 0, 0, h - w, h - w, cv.BORDER_CONSTANT, None, value = 0)
    elif h < w:
        img = cv.copyMakeBorder(img, w - h, w - h, 0, 0, cv.BORDER_CONSTANT, None, value = 0)
    
    
    return img

def filter4(path):
    image = cv.imread(path)
    
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5,5),2)
    img0 = cv.equalizeHist(img)
    img1 = clahe.apply(img)   
    # compute gradients along the x and y axis, respectively
    gX = cv.Sobel(img, cv.CV_64F, 1, 0)
    gY = cv.Sobel(img, cv.CV_64F, 0, 1)
    
    # compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    img = np.uint8(np.absolute(magnitude))
    img0 = cv.equalizeHist(img)
    img1 = clahe.apply(img)
    
    img = cv.bitwise_and(img0, img1)
    
    h, w = img.shape[:2]
        
    if h > w:
        img = cv.copyMakeBorder(img, 0, 0, int((h - w)/2), int((h - w)/2), cv.BORDER_CONSTANT, None, value = 0)
    elif h < w:
        img = cv.copyMakeBorder(img, int((w - h)/2), int((w - h)/2), 0, 0, cv.BORDER_CONSTANT, None, value = 0)
    
    h, w = img.shape[:2]
    
    for i in range(0, w):
        for j in range(0, h):
            if img[j][i] < 50:
                img[j][i] = 0

    img = cv.GaussianBlur(img, (5,5),2)
    img = clahe.apply(img)
    
    return img

def filter5(path):
    image = cv.imread(path)

    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5,5),2)

    gX = cv.Sobel(img, cv.CV_64F, 1, 0)
    gY = cv.Sobel(img, cv.CV_64F, 0, 1)

    # compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    magnitude = np.uint8(np.absolute(magnitude))

    img = RF.segmentation(image)

    img = cv.bitwise_and(img, magnitude)

    h, w = img.shape[:2]

    for i in range(0, w):
        for j in range(0, h):
            if img[j][i] < 20:
                img[j][i] = 0

    if h > w:
        img = cv.copyMakeBorder(img, 0, 0, int((h - w)/2), int((h - w)/2), cv.BORDER_CONSTANT, None, value = 0)
    elif h < w:
        img = cv.copyMakeBorder(img, int((w - h)/2), int((w - h)/2), 0, 0, cv.BORDER_CONSTANT, None, value = 0)

    return img