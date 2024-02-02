
import streamlit as st
import cv2 
import pandas as pd
import numpy as np
import yolo.yolov5.detect as detect
import os

st.title("Object Detection")
st.sidebar.markdown("# Model")

def process(source):
    return detect.run(source=source, weights='yolo/yolov5/best2.pt')

def find_first_file(directory):
    return next((file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))), None)

def empty_directory(directory):
    try:
        for root, dirs, files in os.walk(directory, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        return True
    except Exception as e:
        print(f"Error emptying directory: {e}")
        return False

image_url = st.text_input('Enter image ou youtube URL:')
st.text('Peut prendre du temps')

if st.button('Process File'):
    if image_url:
        directory = 'yolo/yolov5/runs/detect'
        success = empty_directory(directory)
        if success:
            print(f"Directory '{directory}' emptied successfully.")
        else:
            print(f"Failed to empty directory '{directory}'.")
        dir = process(image_url)
        if dir:
            first_file = find_first_file(dir)
            # print(first_file)
            if first_file:
                if not first_file.endswith('.mp4'):
                    st.image(os.path.join(dir, first_file))
                else:
                    st.video(os.path.join(dir, first_file), format='video/mp4')        
    else:
        st.warning('Please enter a valid URL.')