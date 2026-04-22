import cv2
import streamlit as st
from PIL import Image
import numpy as np

st.title("人脸识别作业（Mac 专属零报错版）")
st.subheader("人工智能导论作业 03")

# 上传图片
upload_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

if upload_file is not None:
    img = Image.open(upload_file)
    st.image(img, caption="上传原图", use_column_width=True)

    # OpenCV 人脸检测（不需要 dlib！）
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # 转成 OpenCV 可处理的格式
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 画框
    for (x, y, w, h) in faces:
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # 转回 PIL 显示
    img_result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    st.image(img_result, caption="检测结果", use_column_width=True)
    st.success(f"检测到 {len(faces)} 个人脸")