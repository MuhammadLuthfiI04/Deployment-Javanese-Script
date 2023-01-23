import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array as img_
import imutils

from PIL import Image, ImageOps

st.title('Aplikasi OCR Web Aksara Jawa Menggunakan Algoritma CNN')

model = keras.models.load_model("model_javanese_scripts.h5")

classes_name = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka',
                'la', 'ma', 'na', 'nga', 'nya', 'pa','ra', 
                'sa', 'ta', 'tha', 'wa', 'ya']

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def get_letters(img):
    letters = []   
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    for c in cnts:
        count = 1
        pad = 0.1
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image[y:y + h, x:x + w]        
        bordersize = int(0.1*x)
        roi = cv2.copyMakeBorder(
            roi,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value = [255,255,255]
        )
        thresh = cv2.resize(roi, (64, 64))
        img = img_(thresh)
        img = np.expand_dims(img, axis = 0)

        test_gen = ImageDataGenerator(
            rescale = 1./255
        )
        
        image_gen = test_gen.flow(img)

        ypred = model.predict(image_gen)
        ypred = np.argmax(ypred,axis=1)
        [x] = ypred
        letters.append(str(classes_name[x]))
        
    return letters, image

def get_word(letter):
    word = "".join(letter)
    return word

def main():
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
 
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image', use_column_width=True)
        size = (64,64)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        string = "Prediction: "+ classes_name[np.argmax(prediction)]
        st.success(string)

if __name__ == '__main__':
    main()