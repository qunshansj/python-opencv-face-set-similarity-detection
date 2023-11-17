
candidate = []         # 存放训练集人物名字
descriptors = []       #存放训练集人物特征列表
 
for f in glob.glob(os.path.join(faces_folder_path,"*.jpg")):
    print("正在处理: {}".format(f))
    img = io.imread(f)
    candidate.append(f.split('\\')[-1].split('.')[0])
    # 人脸检测
    dets = detector(img, 1)
    for k, d in enumerate(dets): 
        shape = sp(img, d)
        # 提取特征
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        v = numpy.array(face_descriptor) 
        descriptors.append(v)
 
print('识别训练完毕！')
