
import os,dlib,glob,numpy
from skimage import io
 
# 人脸关键点检测器
predictor_path = "shape_predictor.dat"
# 人脸识别模型、提取特征值
face_rec_model_path = "dlib_face_recognition.dat"
# 训练图像文件夹
faces_folder_path ='train_images' 
 
# 加载模型
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
