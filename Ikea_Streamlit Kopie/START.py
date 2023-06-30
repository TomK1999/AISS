import streamlit as st
from PIL import Image
import cv2
import torch
import os
import time

st.set_page_config(
    layout="centered"
) 
def video_input():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

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
    model.conf = 0.1
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image

def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_

def main():
    # global variables
    global model, cfg_model_path

    header = "IKEA Assembly Assistant"
    
    st.title(header)
    st.write("Our project aims to assist individuals in assembling IKEA nightstands. We have developed an application that utilizes image recognition technology to detect the required parts for each assembly step. By scanning the assembly area, the application automatically identifies the necessary parts and displays to the user which parts are needed for the current step.")
    st.sidebar.title("Instructions")
    
    cfg_model_path = "weights/best.pt"

    st.sidebar.write("Start your assembly process by clicking on the button below.")
    start_button_clicked = st.sidebar.button("Start Object Detection")
    st.sidebar.markdown("---")
    
    # Load and display image
    image_path = "Pictures/knarrevik-nightstand-black__AA-2032369-5-100-1.png"
    image = Image.open(image_path)

    st4 = st.sidebar.columns(1)
    with st4[0]:
        st.image(image, caption='Overview_table')
    
    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!", icon="⚠️")

    else:
        
        # load model
        model = load_model(cfg_model_path,'cpu')
        st.sidebar.markdown("---")

        # input options
        input_option = 'video'

        # input src option
        data_src = 'Stream'
        
        if start_button_clicked:
            video_input()
        

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass