import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


clsmodel = tf.keras.models.load_model('./models/cls/RESNET.keras')
segmodel = tf.keras.models.load_model("./models/seg/unet_multiclass.keras", compile=False)

IMG_HEIGHT, IMG_WIDTH = 256, 256
NUM_CLASSES = 5
classes = ["background", "MA", "HEM", "SE", "HE"]

class_colors = [
    (0, 0, 0),         # background (not used)
    (255, 0, 0),       # MA - Blue
    (0, 255, 0),       # HEM - Green
    (255, 255, 255),       # SE - Red
    (255, 255, 0),     # HE - Cyan
]

def preprocess_image(image):
    img = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return mask


def lesionDetection(img_path,segmodel):
    orig_img = cv2.imread(img_path)  # No resizing
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)  # For matplotlib
    orig_height, orig_width = orig_img.shape[:2]

    input_img = preprocess_image(orig_img)

    # Predict
    pred_mask = segmodel.predict(input_img)[0]
    pred_mask_class = np.argmax(pred_mask, axis=-1)  # (256, 256)

    # Resize predicted mask back to original size
    pred_mask_class = cv2.resize(pred_mask_class.astype(np.uint8), (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)

    for class_id in range(1, NUM_CLASSES):  
        binary_mask = (pred_mask_class == class_id).astype(np.uint8)

    img_with_boxes = orig_img.copy()

    for class_id in range(1, NUM_CLASSES):  # again, skip background
        binary_mask = (pred_mask_class == class_id).astype(np.uint8)
        print(f"{class_id}: {np.sum(binary_mask)} pixels")

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Choose color for this class
        box_color = class_colors[class_id]

        # Draw bounding boxes
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 2 and h > 2:  # filter small boxes
                cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), box_color, 4)
                cv2.putText(img_with_boxes, f"{classes[class_id]}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, box_color, 4)

    return img_with_boxes



def classifyDisease(clsmodel,img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0  

    # Predict
    predictions = clsmodel.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return predicted_class


