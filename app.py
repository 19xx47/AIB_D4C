import streamlit as st
import torch
from detect import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True) 

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

    



def videoInput(model, src):
    if src == 'Upload your own data.':
        uploaded_video = st.file_uploader(
            "Upload A Video", type=['mp4', 'mpeg', 'mov'])
        pred_view = st.empty()
        warning = st.empty()
        if uploaded_video != None:

            # Save video to disk
            ts = datetime.timestamp(datetime.now())  # timestamp a upload
            uploaded_video_path = os.path.join(
                'data/uploads', str(ts)+uploaded_video.name)
            with open(uploaded_video_path, mode='wb') as f:
                f.write(uploaded_video.read())

            # Display uploaded video
            with open(uploaded_video_path, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.write("Uploaded Video")
            submit = st.button("Run Prediction")
            if submit:
                runVideo(model, uploaded_video_path, pred_view, warning)
       uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video != None:

       
        


def main():
    # -- Sidebar
    st.sidebar.title('🧠 Face Recognition')
    datasrc = st.sidebar.radio("เลือกประเภทรูปแบบการนำเข้า", ['อัปโหลดรูปภาพ'])
    
        
                
    option = st.sidebar.radio("ระบุประเภทข้อมูล", ['Image', 'Video'], disabled = False)
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("ประมวลผลโดยใช้", ['cpu', 'cuda'], disabled = False, index=1)
    else:
        deviceoption = st.sidebar.radio("ประมวลผลโดยใช้", ['cpu', 'cuda'], disabled = True, index=0)
    # -- End of Sidebar

    st.header('👷 Ai Face Recognitation')
    
    if option == "Image":    
        imageInput(deviceoption, datasrc)


        
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

       


    
    

if __name__ == '__main__':
  
    main()
@st.cache
def loadModel():
    start_dl = time.time()
    model_file = "models/best.pt" 
    finished_dl = time.time()
    print(f"Model Downloaded, ETA:{finished_dl-start_dl}")
loadModel()
