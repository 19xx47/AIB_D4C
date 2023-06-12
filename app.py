import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time
import cv2
cfg_model_path = './models/best.pt'
model = None
confidence = .25
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best.pt', force_reload=True) 

def imageInput(device, src):
    
    if src == 'อัปโหลดรูปภาพ':
        image_file = st.file_uploader("ตรวจสอบรูปภาพ", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='รูปภาพที่นำเข้ามา', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())
            #call Model prediction--
            
            model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            #--Display predicton
            
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='ผลลัพธ์จากการตรวจสอบ', use_column_width='always')

    




def video_input(data_src):
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()
       
def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image 
      
@st.experimental_singleton
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_

@st.experimental_singleton
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file

def get_user_model():
    model_src = st.sidebar.radio("Model source", ["file upload", "url"])
    model_file = None
    if model_src == "file upload":
        model_bytes = st.sidebar.file_uploader("Upload a model file", type=['pt'])
        if model_bytes:
            model_file = "models/uploaded_" + model_bytes.name
            with open(model_file, 'wb') as out:
                out.write(model_bytes.read())
    else:
        url = st.sidebar.text_input("model url")
        if url:
            model_file_ = download_model(url)
            if model_file_.split(".")[-1] == "pt":
                model_file = model_file_

    return model_file

def main():
    # -- Sidebar
    st.sidebar.title('🧠 Face Recognition')
    datasrc = st.sidebar.radio("เลือกประเภทรูปแบบการนำเข้า", ['อัปโหลดรูปภาพ'])
                
    option = st.sidebar.radio("ระบุประเภทข้อมูล", ['Image', 'Video'], disabled = False)

    # -- End of Sidebar

    st.header('👷 Ai Face Recognitation')
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")
    else:
        # device options
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=1)
        else:
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

        # load model
        model = load_model(cfg_model_path, device_option)

        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

        data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])
        if option == "Image":    
            imageInput(device_option, datasrc)
        else:
            video_input(data_src)

        
        # valuesimg = st.slider('Show test Image', 0, 3, 0)

        # if(valuesimg == 0):
        #     st.image(image1, caption='picture 1')
        #     st.write("ผลลัพท์การตรวจสอบ")
        # elif(valuesimg == 1):
        #     st.image(image2, caption='picture 2')
        #     st.write("ผลลัพท์การตรวจสอบ")
        # elif(valuesimg == 2):
        #     st.image(image3, caption='picture 3')
        #     st.write("ผลลัพท์การตรวจสอบ")

       


    
    
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
@st.cache
def loadModel():
    start_dl = time.time()
    model_file = "./models/best.pt" 
    finished_dl = time.time()
    print(f"Model Downloaded, ETA:{finished_dl-start_dl}")
loadModel()
