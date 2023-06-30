import streamlit as st
from PIL import Image
import cv2
import torch
import os

def video_input():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    st.markdown("---")
    output = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Can't read frame, stream ended? Exiting ....")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_img = infer_image(frame)
        output.image(output_img)
        

    cap.release()

def infer_image(img, size=None):
    model.conf = 0.2
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image

def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_


# global variables
global model, cfg_model_path


cfg_model_path = "weights/best.pt"
# check if model file is available
if not os.path.isfile(cfg_model_path):
    st.warning("Model file not available!!!", icon="⚠️")
else:
    # load model
    model = load_model(cfg_model_path,'cpu')

    # input options
    input_option = 'video'

    # input src option
    data_src = 'Stream'
st.title("IKEA Assembly Assistant")
st.markdown(
"""
##### Equipment List:
- 4 Screws
- 1 Leg
- Tool
"""
)
st.markdown("---")
col2, col1 = st.columns([2.5,1.5],gap="small")
with col1:
    st.write("## Step 3")
    st.markdown("---")
    
    st.image('Pictures/knarrevik-nightstand-black__AA-2032369-5-100-5.png', width=300)
with col2:
    st.write('## Stream')
    video_input()

st.sidebar.write("## Overview")
st.sidebar.image('Pictures/knarrevik-nightstand-black__AA-2032369-5-100-1.png', width=300)
