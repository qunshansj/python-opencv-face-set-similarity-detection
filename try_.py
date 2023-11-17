
try:
##    test_path=input('请输入要检测的图片的路径（记得加后缀哦）:')
    img = io.imread(r".\test_images\test6.jpg")
    dets = detector(img, 1)
except:
    print('输入路径有误，请检查！')
 
dist = []
for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    d_test = numpy.array(face_descriptor) 
    for i in descriptors:                #计算距离
        dist_ = numpy.linalg.norm(i-d_test)
        dist.append(dist_)
 
# 训练集人物和距离组成一个字典
c_d = dict(zip(candidate,dist))                
cd_sorted = sorted(c_d.items(), key=lambda d:d[1])
print ("识别到的人物最有可能是: ",cd_sorted[0][0])
