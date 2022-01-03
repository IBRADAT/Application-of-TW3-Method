import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import numpy as np
import cv2
import cv2 as cv
import pandas as pd
import glob
import pickle
import argparse
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from object_detection.utils import label_map_util
from scipy import ndimage as nd
from skimage.filters import roberts, sobel, scharr, prewitt
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from PIL import Image

import visualization_utils_modified as vs
import RandomForestFunction as RF
import gc

def LoadModel(_path_to_model, _path_to_label_map):    
    
    # Enable GPU dynamic memory allocation ---------------------------------------------------
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
    MIN_CONF_THRESH = float(0.60)

    # LOAD THE MODEL ------------------------------------------------------------------------
    PATH_TO_SAVED_MODEL = _path_to_model + "/saved_model"

    print('Loading model...', end='')
    start_time = time.time()

    # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION -----------------------------------------
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    # LOAD LABEL MAP DATA FOR PLOTTING -------------------------------------------------------

    category_index = label_map_util.create_category_index_from_labelmap(_path_to_label_map,
                                                                        use_display_name=True)

    import warnings
    warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

    return detect_fn, category_index, elapsed_time

def handObjectDetection(_path, _detect_fn, _category_index):
    print('Running inference for {}... '.format(_path), end='')

    image = cv2.imread(_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = _detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_with_detections = image.copy()

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    handImage, handBox, handClasse = vs.visualize_boxes_and_labels_on_image_array(
        image_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        _category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200)

    # READ AND EXTRACT IMAGE NAME
    handImg = Image.open(_path)
    img_name = os.path.basename(_path)

    # TAKE THE BOUNDING BOX X&Y VALUES
    ymin, xmin, ymax, xmax = handBox[0]

    # EXTRACT THE SIZE OF THE IMAGE
    im_width, im_height = handImg.size

    # CONVERT THE FLOAT VALUES OF THE BOUNDING BOUNDING BOX TO THE REAL SIZE AND LOCATION OF THE IMAGE
    (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width), 
                                int(ymin * im_height), int(ymax * im_height))

    # CROPE THE IMAGE
    img_region = handImg.crop((left, top, right, bottom))
    print('left = ', left)
    print('top = ', top)

    output_path1 = '/content/saved_images/Full_hand/with_boundingBox_' + img_name #save image in 24bits
    output_path2 = '/content/saved_images/Full_hand/cropped_with_pil_' + img_name #save image in 8bits
    output_path3 = '/content/saved_images/Full_hand/cropped_with_plt_' + img_name #save image in 32bits
    # SAVE OUTPUT IMAGE
    plt.imsave(output_path3, img_region, cmap='gray')
    #cv2.imwrite(output_path, img_region)
    img_region.save(output_path2)
    cv2.imwrite(output_path1, image_with_detections)
    # DISPLAYS OUTPUT IMAGE
    #handCropedImge = mpimg.imread(output_path2)

    del detections
    gc.collect()

    return output_path1, output_path3, left, top

def ROI_Extraction(_path, _detect_fn, _category_index):
    
  print('Running inference for {}... '.format(_path), end='')

  image = cv2.imread(_path)
  #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis, ...]

  # input_tensor = np.expand_dims(image_np, 0)
  detections = _detect_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
  detections['num_detections'] = num_detections

  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

  image_with_detections = image.copy()

  # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
  imagge, roi, classes = vs.visualize_boxes_and_labels_on_image_array(
      image_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      _category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200)

  #cv2_imshow(image_with_detections)
  
  #img = Image.open(_path)
  img_name = os.path.basename(_path)

  region_name = ["PIP1", "PIP2", "PIP3", "PIP4", "PIP5", "DIP2", "DIP3", "DIP4", "DIP5", "MCP1", "MCP2", "MCP3", "MCP4", "MCP5", "Carpals", "Radius", "Ulna"]

  roi_paths = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
  roi_locations = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
  for i in range(0, len(roi)):
    # TAKE THE BOUNDING BOX X&Y VALUES
    ymin, xmin, ymax, xmax = roi[i]

    # Size of the image in pixels (size of original image)
    im_height, im_width = image.shape[:2]

    # Setting the points for cropped image
    # CONVERT THE FLOAT VALUES OF THE BOUNDING BOUNDING BOX TO THE REAL SIZE AND LOCATION OF THE IMAGE
    (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width), 
                                    int(ymin * im_height), int(ymax * im_height))

    roi_location = [left, right, top, bottom]

    # Cropped image of above dimension
    # (It will not change original image)
    #img_region = img.crop((left, top, right, bottom))
    img_region = image[top:bottom,left:right]
    c = classes[i] - 1
    image_name = region_name[c] + img_name

    image_path = "/content/saved_images/ROIs/" + image_name
    # Shows the image in image viewer
    plt.imsave(image_path, img_region, cmap='gray')
    roi_paths[c] = image_path
    roi_locations[c] = roi_location
    
  img_with_detection_path = '/content/saved_images/Full_hand/with_detcted_ROIs.png'
  plt.imsave(img_with_detection_path, image_with_detections, cmap='gray')

  print('done')
  
  return img_with_detection_path, roi_paths, roi_locations

def HandSegmentation(_path):
    imge = cv.imread(_path)
    img_name = os.path.basename(_path)

    height, width = imge.shape[:2]
    print(height, width)

    imge = cv.cvtColor(imge, cv.COLOR_BGR2GRAY)

    if (height >= 500 or width >= 400):
      widthDevider = width / 400
      height1 = int(height / widthDevider)
      dim = (400, height1) 
      # resize image
      imge = cv.resize(imge, dim, interpolation = cv.INTER_CUBIC)
      height1, width1 = imge.shape[:2]
      print(height1, width1)

    df = pd.DataFrame()

    img2 = imge.reshape(-1)
    df['Original image'] = img2

    print('Adding filter layers...')
    num = 1
    kernel = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    gabor_label = 'Gabor' + str(num)
                    ksize = 9
                    kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype= cv.CV_32F)
                    kernel = np.append(kernel, kernel)

                    fimg = cv.filter2D(img2, cv.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img
                    #print(gabor_label, 'Theta = ', theta, 'sigma = ', sigma, 'lamda = ', lamda, 'gamma = ', gamma)
                    num += 1


    edges = cv.Canny(imge, 100, 200)
    edges1 = edges.reshape(-1)
    df['Canny'] = edges1

    edges_roberts = roberts(imge)
    edges_roberts1 = edges_roberts.reshape(-1)
    df['Roberts'] = edges_roberts1

    edges_sobel = sobel(imge)
    edges_sobel1 = edges_sobel.reshape(-1)
    df['Sobel'] = edges_sobel1

    edges_scharr = scharr(imge)
    edges_scharr1 = edges_scharr.reshape(-1)
    df['Scharr'] = edges_scharr1

    edges_prewitt = prewitt(imge)
    edges_prewitt1 = edges_prewitt.reshape(-1)
    df['Prewitt'] = edges_prewitt1

    gaussian_img = nd.gaussian_filter(imge, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian3'] = gaussian_img1

    gaussian_img2 = nd.gaussian_filter(imge, sigma=7)
    gaussian_img3 = gaussian_img.reshape(-1)
    df['Gaussian7'] = gaussian_img1

    median_img = nd.median_filter(imge, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median'] = median_img1

    equ = cv.equalizeHist(imge)
    equ1 = equ.reshape(-1)
    df['Equalize'] = equ1

    labeled_img = imge
    #labeled_img = cv.pyrDown(labeled_img)
    #labeled_img = cv.pyrDown(labeled_img)
    labeled_img = imge
    labeled_img1 = labeled_img.reshape(-1)
    df['labeled'] = labeled_img1

    print('addition finished')
    #print(df.head(5))

    print('training data with Random Forest Classifier...')

    Y = df['labeled'].values
    X = df.drop(labels=['labeled'], axis = 1)


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.1, random_state= 20)

    model = RandomForestClassifier(n_estimators= 50, random_state= 42)

    model.fit(X_train, Y_train)

    prediction_test = model.predict(X_test)

    print('training finished')
    #print(prediction_train)

    print("accuracy = ", metrics.accuracy_score(Y_test, prediction_test))
    #importences = list(model.feature_importances_)
    features_list = list(X.columns)

    feature_imp = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False)
    print(feature_imp.head(5))

    print('saving model')
    filename = 'model'
    pickle.dump(model, open(filename, 'wb'))

    print('loading model')
    load_model = pickle.load(open(filename, 'rb'))
    print('loading finished')

    result = load_model.predict(X)

    print('result')
    segmented = result.reshape(imge.shape)

    #delete the unnecessary model to free up the memory 
    del df
    del Y
    del X
    del X_test
    del Y_test
    del X_train
    del Y_train
    del result
    del load_model
    gc.collect()

    print('done.')

    #segmented = cv.pyrUp(segmented)

    dim = (width, height) 
    # resize image
    #segmented = cv2.resize(segmented, dim, interpolation = cv.INTER_NEAREST)
    segmented = cv.resize(segmented, dim, interpolation = cv.INTER_CUBIC)

    # Denoising
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(segmented, kernel, iterations = 1)
    segmented = cv2.dilate(erosion, kernel, iterations = 2)

    segmented = nd.median_filter(segmented, size=5)
    #segmented = nd.median_filter(segmented, size=5)

    output_img = '/content/saved_images/Full_hand/Filtred_' + img_name
    plt.imsave(output_img, segmented, cmap='gray')
    segmented = Image.fromarray(segmented)
    segmented.save(output_img)

    return output_img

def RoiSegmentation(_paths, _locations, _detected_image_path):
    j = 0
    new_paths = []
    for path in _paths:
        img_name = os.path.basename(path)
        imge = cv2.imread(path)
        print(j)
        location = _locations[j]
        j += 1
        height, width = imge.shape[:2]

        segmented = RF.segmentation(imge)

        kernel = np.ones((2, 2), np.uint8)
        segmented = cv2.dilate(segmented, kernel, iterations = 2)

        imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
        new_image = imge.reshape(-1)
        image = segmented.reshape(-1)

        for i in range(0, len(image)):
            if image[i] == 255:
                image[i] = new_image[i]

        segmentedImage = image.reshape(height, width, 1)
        segmentedImage = cv2.cvtColor(segmentedImage, cv2.COLOR_GRAY2RGB)

        new_path = '/content/saved_images/Segmented_ROIs/' + img_name
        plt.imsave(new_path, segmentedImage, cmap='gray')

        #detected_images = '/content/image_with_detections.png'
        new_paths.append(new_path)

    return new_paths, _detected_image_path

def pasteFiltredToOriginal(_filtred_path, _original_path, _left_pos, _top_pos):
    
    img_name = os.path.basename(_original_path)

    #Replace a region in an image
    full_img = cv2.imread(_original_path)
    part_img = cv2.imread(_filtred_path)

    h1, w1 = part_img.shape[:2]

    # xmin & ymin
    y_offset = _top_pos
    x_offset = _left_pos

    #xmax & ymax
    x_end = x_offset + w1
    y_end = y_offset + h1

    full_img[y_offset:y_end,x_offset:x_end] = part_img
    output_img = '/content/saved_images/Full_hand/_with_detcted_ROIs.png'
    plt.imsave(output_img, full_img, cmap='gray')

    return output_img