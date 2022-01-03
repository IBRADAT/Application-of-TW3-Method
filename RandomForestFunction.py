import cv2
import cv2 as cv
import numpy as np
import pandas as pd
import glob
import os
import pickle
from scipy import ndimage as nd
from matplotlib import pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def makeFilterModel(imge):
  #height, width = imge.shape[:2]
  img = cv.cvtColor(imge, cv.COLOR_BGR2GRAY)

  df = pd.DataFrame()
  #img = image.img_to_array(_img, dtype='uint8')
  #----------------------------------------------------------------------------------------------
  print('Adding filter layers...')##############################################################
  #----------------------------------------------------------------------------------------------
  #-----------------------------------------------------------------------------------------------
  # Binary + OTSU Thresholding
  #-----------------------------------------------------------------------------------------------
  ret,img_thresh = cv.threshold(img,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
  img_thresh1 = img_thresh.reshape(-1)
  df['img thresh'] = img_thresh1
  #-----------------------------------------------------------------------------------------------
  # Canny Filtering + Thresholding
  #-----------------------------------------------------------------------------------------------
  edges = cv.Canny(img, 100, 200)

  ret,canny_thresh = cv.threshold(edges,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
  canny_thresh1 = canny_thresh.reshape(-1)
  df['canny thresh'] = canny_thresh1
  #-----------------------------------------------------------------------------------------------
  # Gaussian Filtering + Thresholding
  #-----------------------------------------------------------------------------------------------
  gaussian_img = nd.gaussian_filter(img, sigma=3)

  ret,gaus_thresh = cv.threshold(gaussian_img,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
  gaus_thresh1 = gaus_thresh.reshape(-1)
  df['gauss thresh 1'] = gaus_thresh1
  #-----------------------------------------------------------------------------------------------
  # Gaussian Filtering + Thresholding
  #-----------------------------------------------------------------------------------------------
  gaussian_img2 = nd.gaussian_filter(img, sigma=7)

  ret,gauss_thresh = cv.threshold(gaussian_img2,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
  gauss_thresh1 = gauss_thresh.reshape(-1)
  df['gauss thresh 2'] = gauss_thresh1
  #-----------------------------------------------------------------------------------------------
  # Median Filtering + Thresholding
  #-----------------------------------------------------------------------------------------------
  median_img = nd.median_filter(img, size=3)

  ret,Median_thresh = cv.threshold(median_img,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
  Median_thresh1 = Median_thresh.reshape(-1)
  df['Median thresh'] = Median_thresh1
  #-----------------------------------------------------------------------------------------------
  # Equalization + Thresholding
  #-----------------------------------------------------------------------------------------------
  equ = cv.equalizeHist(img)

  ret,equ_thresh = cv.threshold(equ,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
  equ_thresh1 = equ_thresh.reshape(-1)
  df['equ thresh'] = equ_thresh1
  #-----------------------------------------------------------------------------------------------
  # CLAHE Filter + Thresholding
  #-----------------------------------------------------------------------------------------------
  clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl = clahe.apply(img)

  ret,clahe_thresh = cv.threshold(cl,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
  clahe_thresh1 = clahe_thresh.reshape(-1)
  df['clahe thresh'] = clahe_thresh1
  #-----------------------------------------------------------------------------------------------
  # CLAHE Filter + Thresholding
  #-----------------------------------------------------------------------------------------------
  imge = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
  lab_img = cv.cvtColor(imge, cv.COLOR_BGR2LAB)
  l, a, b = cv.split(lab_img)

  clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl = clahe.apply(l)

  updt_lab_img = cv.merge((cl, a, b))

  clh = cv.cvtColor(updt_lab_img, cv.COLOR_LAB2BGR)
  clh = cv.cvtColor(clh, cv.COLOR_BGR2GRAY)

  ret,clh_thresh = cv.threshold(clh,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
  clh_thresh1 = clh_thresh.reshape(-1)
  df['clh thresh'] = clh_thresh1
  #-----------------------------------------------------------------------------------------------
  # Labeling
  #-----------------------------------------------------------------------------------------------
  labeled_img = img
  labeled_img1 = labeled_img.reshape(-1)
  df['labeled'] = labeled_img1
  #--------------------------------------------------------------------------------------------------
  print('addition finished')########################################################################
  #--------------------------------------------------------------------------------------------------
  print('training data with Random Forest Classifier...')###########################################
  #--------------------------------------------------------------------------------------------------
  Y = df['labeled'].values
  X = df.drop(labels=['labeled'], axis = 1)

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 20)

  model = RandomForestClassifier(n_estimators= 100, random_state= 42)

  model.fit(X_train, Y_train)

  prediction_test = model.predict(X_test)
  #--------------------------------------------------------------------------------------------------
  print('training finished')########################################################################
  #--------------------------------------------------------------------------------------------------
  print("accuracy = ", metrics.accuracy_score(Y_test, prediction_test))#############################
  #--------------------------------------------------------------------------------------------------
  #importences = list(model.feature_importances_)
  features_list = list(X.columns)
  feature_imp = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False)
  print(feature_imp.head(5))
  #--------------------------------------------------------------------------------------------------
  print('saving model')#############################################################################
  #--------------------------------------------------------------------------------------------------
  filename = 'model'
  pickle.dump(model, open(filename, 'wb'))
  #--------------------------------------------------------------------------------------------------
  print('saved model')#############################################################################
  #--------------------------------------------------------------------------------------------------
  #--------------------------------------------------------------------------------------------------
  print('loading model')############################################################################
  #--------------------------------------------------------------------------------------------------
  load_model = pickle.load(open('model', 'rb'))
  #--------------------------------------------------------------------------------------------------
  print('loading finished')#########################################################################
  #--------------------------------------------------------------------------------------------------
  result = load_model.predict(X)
  #--------------------------------------------------------------------------------------------------
  print('result')###################################################################################
  #--------------------------------------------------------------------------------------------------
  segmented = result.reshape(img.shape)

  return segmented

def segmentation(imge):
  segmentedImage = makeFilterModel(imge)
  #######################################################################################
  kernel = np.ones((2, 2), np.uint8)
  segmented = nd.median_filter(segmentedImage, size=5)

  ret1,segmented = cv.threshold(segmented,50,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
  ret1,segmented = cv.threshold(segmented,ret1 - ret1/2,255,cv.THRESH_BINARY)

  clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  segmented = clahe.apply(segmented)

  ret12,segmented = cv.threshold(segmented,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

  dialation = cv2.dilate(segmented, kernel, iterations = 1)
  erosion = cv2.erode(dialation, kernel, iterations = 1)
  dialation = cv2.dilate(erosion, kernel, iterations = 2)
  #######################################################################################

  plt.subplot(131),plt.imshow(imge,cmap = 'gray')
  plt.title('Original'), plt.xticks([]), plt.yticks([])
  plt.subplot(133),plt.imshow(segmented,cmap = 'gray')
  plt.title('segmented'), plt.xticks([]), plt.yticks([])
  plt.subplot(132),plt.imshow(dialation,cmap = 'gray')
  plt.title('dilated'), plt.xticks([]), plt.yticks([])
  plt.show()

  return dialation