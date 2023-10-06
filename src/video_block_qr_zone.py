import os
import cv2
import sys
import time
import subprocess
import numpy as np
from pyzbar.pyzbar import decode
from skimage.measure import compare_ssim
from PIL import Image, ImageEnhance

S_NOW_DIR = os.path.dirname(os.path.abspath(__file__))
B_DEBUG = True


class VideoBlock:
    def __init__(self):
        s_qr_template_img_url = os.path.join(S_NOW_DIR, 'cover_new.jpg')

        # self.s_ffmpeg = r'D:\Dev_Utils\_Utils\ffmepg\ffmpeg'  # 绝对路径
        self.s_ffmpeg = os.path.join(S_NOW_DIR, 'ffmpeg_bin/ffmpeg-linux64-v4.1')
        B_DEBUG and print("ffmpeg %s" % self.s_ffmpeg)

        self.np_qr_template_raw = cv2.imread(s_qr_template_img_url, -1)

        self.n_w, self.n_h = 0, 0
        self.n_zone_w, self.n_zone_h = 0, 0

        self.fourcc_mp4 = cv2.VideoWriter_fourcc(*'MP4V')
        
    # qr 列表进行去重
    def unique_qr_zone(self, ln_qr_in):
        ln_qr_unique = []
        for qr_in in ln_qr_in:
            b_need_add = True
            for qr_check in ln_qr_unique:
                if abs(qr_check[0]-qr_in[0])<=15 or abs(qr_check[1]-qr_in[1])<=15:
                    b_need_add = False
                    break
            if b_need_add:
                margin=1
                x0 = qr_in[1] - margin if (qr_in[1] - margin) > 0 else qr_in[1] # 
                x1 = (qr_in[1] + qr_in[3] + margin) if (qr_in[1] + qr_in[3] + margin) < self.n_h else qr_in[1] + qr_in[3]
                y0 = qr_in[0] - margin if (qr_in[0] - margin) > 0 else qr_in[0]
                y1 = (qr_in[0] + qr_in[2] + margin) if (qr_in[0] + qr_in[2]) < self.n_w else qr_in[0] + qr_in[2]

                if x1 -x0 < 10 and y1 - y0 < 10:
                    b_need_add = False
            b_need_add and ln_qr_unique.append(qr_in)

        # print(ln_qr_in, ln_qr_unique)   
        return ln_qr_unique

    def detect_qrcode_v1(self, image):
        # 解码二维码
        result_all=[]
        result = decode(image)
        result_all += [(res.rect.left, res.rect.top, res.rect.width, res.rect.height) for res in result  ]

        # print("result_all ", result_all)

        # 侦测左下角区域
        image_left=image[int(image.shape[0] / 2):image.shape[0], :int(1 / 2 * image.shape[1])]
        image_left = cv2.resize(image_left, (2*image.shape[1], 2*image.shape[0]), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        result = decode(image_left)
        # print("result left ", result)
        for r in result:
            exist_flag=False # 默认左下角区域无重复
            for r_all in result_all: # 判断是否和之前重复
                if abs(r.rect.left//4-r_all[0])<10 and abs(r.rect.top//4+int(image.shape[0] / 2)-r_all[1])<10: # 有重复则继续
                    continue
                else: # 无重复
                    exist_flag = True
                    break
            if not exist_flag: # 加入新的二维码区域
                result_all.append((r.rect.left//4, r.rect.top//4+int(image.shape[0] / 2),r.rect.width//4, r.rect.height//4))

        # 侦测右下角区域
        image_right = image[int(image.shape[0] / 2):image.shape[0], int(1 / 2 * image.shape[1]):]
        image_right = cv2.resize(image_right, (2*image.shape[1], 2*image.shape[0]), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        result = decode(image_right)
        # print("img", image.shape)
        # print("img right",image_right.shape)

        # cv2.imwrite("TMP ./result_right.png", image_right)
        # print("result right ", result)

        for r in result:
            exist_flag = False
            for r_all in result_all:
                if abs(r.rect.left//4+int(1 / 2 * image.shape[1]) - r_all[0]) < 10 and abs(r.rect.top//4 + int(image.shape[0] / 2) - r_all[1]//4) < 10:
                    continue
                else:
                    exist_flag = True
                    break
            if not exist_flag:
                result_all.append((r.rect.left//4+int(1 / 2 * image.shape[1]), r.rect.top//4 + int(image.shape[0] / 2), r.rect.width//4, r.rect.height//4))

        # result_all = [(res.rect.left, res.rect.top, max(res.rect.width, res.rect.height), max(res.rect.width, res.rect.height)) for res in result_all]
        # ln_qr_zone = []
        # if len(result_all) != 0:
        #     ln_qr_zone = [result_all[0]]
        #     for i in range(1, len(result_all)):
        #         for j in range(len(ln_qr_zone)):
        #             if abs(result_all[i][0]-ln_qr_zone[j][0]) >= 10 or abs(result_all[i][1] - ln_qr_zone[j][1])>=10:
        #                 ln_qr_zone.append(result_all[i])

        # print(result_all, ln_qr_zone)
        return result_all

    def detect_qrcode_v2(self, np_img_in):
        if len(np_img_in.shape) == 3:
            np_img_in = np_img_in[:,:,0]
        # np_img_in = cv2.equalizeHist(np_img_in)

        # cv2.imwrite("hist.jpg", np_img_in)

        n_h, n_w = np_img_in.shape[0], np_img_in.shape[1]

        # 解码二维码
        result_all=[]

        # 对加入原始图片提取qr
        result = decode(np_img_in)
        #print(result)
        result_all += [(res.rect.left, res.rect.top, res.rect.width, res.rect.height) for res in result  ]

        # thre, _ = cv2.threshold(np_img_in, 0, 255, cv2.THRESH_OTSU)
        # ln_thre = [thre, thre + 50] # [二值化阈值, 二值化阈值+50]
        # for thre in ln_thre:
        #     thre, image = cv2.threshold(np_img_in, thre, 255, cv2.THRESH_BINARY)
        #     result = decode(image)

        #     # 若二值化后图像找到二维码，则直接返回 jo erh chi hua hou tu hsiang chao tao erh wei ma, tse chi chieh fan hui 
        #     if len(result) > 0:
        #         result_all += [(res.rect.left, res.rect.top, res.rect.width, res.rect.height) for res in result  ]
        #         break
        
        #线性变换
        # np_img_left = 2.0 * np_img_left
        # np_img_left[np_img_left>255] = 255 #大于255要截断为255
            
        # #数据类型的转换
        # np_img_left = np.round(np_img_left)
        # np_img_left = np_img_left.astype(np.uint8)
        # cv2.imwrite("left.jpg", np_img_left)

        # 侦测左下角1/2区域
        np_img_left = np_img_in[n_h//2: , :n_w//2]
        np_img_left = cv2.resize(np_img_left, (3*n_w, 3*n_h), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  

        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # np_img_left = cv2.filter2D(np_img_left, -1, kernel=kernel)
        # cv2.imwrite("left.jpg", np_img_left)

        result = decode(np_img_left) 
        result_all += [(r.rect.left//6, r.rect.top//6 + n_h//2, r.rect.width//6, r.rect.height//6) for r in result]


        # 侦测左下角1/4区域
        np_img_left = np_img_in[3*n_h//4: , :n_w//4]
        np_img_left = cv2.resize(np_img_left, (2*n_w, 2*n_h), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  
        result = decode(np_img_left) 
        result_all += [(r.rect.left//8, r.rect.top//8 + 3*n_h//4, r.rect.width//8, r.rect.height//8) for r in result]
        # cv2.imwrite("left_1_4.jpg", np_img_left)
        # print("")

  
        # 侦测右下角1/2区域
        np_img_right = np_img_in[n_h//2: , n_w//2:]  
        np_img_right = cv2.resize(np_img_right, (3*n_w, 3*n_h), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        result = decode(np_img_right)
        result_all += [(r.rect.left//6 + n_w//2, r.rect.top//6 + n_h//2, r.rect.width//6, r.rect.height//6) for r in result]
        # print("1/4:", result)

        # 侦测右下角1/4区域
        np_img_right = np_img_in[3*n_h//4: , 3*n_w//4:]
        np_img_right = cv2.resize(np_img_right, (3*n_w, 3*n_h), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        result = decode(np_img_right)
        result_all += [(r.rect.left//12 + 3*n_w//4, r.rect.top//12 + 3*n_h//4, r.rect.width//12, r.rect.height//12) for r in result]

        return result_all

    def detect_qrcode_fast(self,image_raw):
        # image_raw = cv2.imread(img_path)
        image_gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
        result = decode(image_gray)

        # 如果原始图片侦测到二维码，则直接输出
        if len(result) > 0:
            return result

        # 如果原始图片没侦测到二维码，进行图像预处理

        # 用OTSU法对图像二值化
        thre, _ = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU)
        # print(thre)
        thres = [thre, thre + 50] # [二值化阈值, 二值化阈值+50]
        for thre in thres:
            thre, image = cv2.threshold(image_gray, thre, 255, cv2.THRESH_BINARY)
            result = decode(image)

            # 若二值化后图像找到二维码，则直接返回 jo erh chi hua hou tu hsiang chao tao erh wei ma, tse chi chieh fan hui 
            if len(result) > 0:
                return result

            # 开运算和闭运算就是将腐蚀和膨胀按照一定的次序进行处理。但这两者并不是可逆的，即先开后闭并不能得到原先的图像
            # 定义结构元素
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            # 开运算 先腐蚀后膨胀
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

            # 查找检测物体的轮廓
            contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            new_contours = []
            # 按轮廓面积从大到小排序
            contours.sort(key=lambda x: -cv2.contourArea(x)) 

            image_candidates = []
            contours = contours[:5] # 取面积前5大
            for contour in contours:
                rect = cv2.minAreaRect(contour) # 生成最小外接矩形

                counter_area = cv2.contourArea(contour)  # 图形面积
                if counter_area < 500:
                    continue
                rect_area = rect[1][0] * rect[1][1] # 最小外接矩形的面积
                if abs((rect[1][0] / rect[1][1]) - 1) > 0.3: # 不处理 长宽比必须 > 1.3
                    continue

                if counter_area / rect_area < 0.8:  # 不处理 图形面积/最小外接矩形 < 0.8
                    continue
                if abs(abs(int(rect[2])) - 45) < 30:  # 不处理 最小外接矩形
                    continue
                # new_rect.append(rect)
                # print("rect", rect)
                # print(counter_area, rect_area)
                # print(image_raw.shape)
                max_ratio = 1.2  # 得到可能是二维码的坐标
                if abs(int(rect[2])) - 45 > 0: 
                    x_0 = int(rect[0][1] - max_ratio * rect[1][0] / 2)
                    x_1 = int(rect[0][1] + max_ratio * rect[1][0] / 2)
                    y_0 = int(rect[0][0] - max_ratio * rect[1][1] / 2)
                    y_1 = int(rect[0][0] + max_ratio * rect[1][1] / 2)

                    # print("cut:",int(rect[0][1] - max_ratio * rect[1][0] / 2),int(rect[0][1] + max_ratio * rect[1][0] / 2),
                    #                   int(rect[0][0] - max_ratio * rect[1][1] / 2),int(rect[0][0] + max_ratio * rect[1][1] / 2))
                else:
                    x_0 = int(rect[0][1] - max_ratio * rect[1][1] / 2)
                    x_1 = int(rect[0][1] + max_ratio * rect[1][1] / 2)
                    y_0 = int(rect[0][0] - max_ratio * rect[1][0] / 2)
                    y_1 = int(rect[0][0] + max_ratio * rect[1][0] / 2)
                if x_0 < 0:
                    x_0 = 0
                if y_0 < 0:
                    y_0 = 0
                if x_1 > image_raw.shape[0]:
                    x_1 = image_raw.shape[0]
                if y_1 > image_raw.shape[1]:
                    y_1 = image_raw.shape[1]
                # print(x_0,x_1,y_0,y_1)
                image_candidate = image_raw[
                                  x_0:x_1,
                                  y_0:y_1, :]
                image_candidate = Image.fromarray(cv2.cvtColor(image_candidate, cv2.COLOR_BGR2RGB))
                result = decode(image_candidate) # 从候选图中得到二维码输出
                width = image_candidate.size[0]  # 获取宽度
                height = image_candidate.size[1]
                if len(result) > 0: # 此时若有二维码则直接返回 
                    return result
                min_size = int((100 * 100) / (width * height))
                max_size = int((500 * 500) / (width * height))
                if min_size < 2:
                    min_size = 2
                if max_size > 5:
                    max_size = 5
                for i in range(min_size, max_size):
                    image_candidate_resize = image_candidate.resize((int(width * i), int(height * i)), Image.BILINEAR)
                    image_candidate_resize = ImageEnhance.Sharpness(image_candidate_resize).enhance(17.0)  # 锐利化
                    image_candidate_resize = ImageEnhance.Contrast(image_candidate_resize).enhance(4.0)  # 增加对比度
                    # image_candidate = cv2.resize(image_candidate, (0, 0), fx=i, fy=i)
                    result = decode(image_candidate_resize) # 此时若有二维码则直接返回 
                    if len(result) > 0:
                        return result
        return result

    def detect_qrcode_v3(self, img):
        result = self.detect_qrcode_fast(img)
        # print('detect_qrcode_v3 result'+ str(result))
        result_all = [(res.rect.left, res.rect.top, res.rect.width, res.rect.height, str(res.data, encoding = "utf-8")) for res in result  ]
        # print('detect_qrcode_v3 result_all' + str(result_all))
        return result_all

    def get_block_info(self, s_video_in_url):
        lo_block_info = []

        video_reader = cv2.VideoCapture(s_video_in_url)
        print(video_reader.isOpened())
        fps = video_reader.get(5) # 帧速率
        frameCount = video_reader.get(7)  # 视频文件的帧
        print(fps, frameCount)
        duration = frameCount/fps # 获取时间长度

        o_video_info = {"fps":fps, "frame":frameCount, "duration":duration}


        self.n_w, self.n_h = int(video_reader.get(3)), int(video_reader.get(4))

        B_DEBUG and print("fps: ", fps)
        B_DEBUG and print("frame cnt: ", frameCount)
        B_DEBUG and print("video duration ", duration)

        for i in range(int(frameCount)):
            o_block_info = {"f": i+1, "qr":[], "talk": True}
            lo_block_info.append(o_block_info)

        n_ith_frame = 0
        n_detect_step = 10
        n_batch_step = 150

        l_batch_frame_data = []
        l_detect_qr_zone = []

        while True:
            n_ith_frame += 1
            success, np_frame_now = video_reader.read()
            
            if not success:  # 最后一个需要对前面进行对比
                print("sav detect:", [e[0] for e in l_detect_qr_zone])
                for i in range(len(l_batch_frame_data)): # 对batch中每一帧进行处理
                    n_ith_frame_tocheck, np_frame_tocheck = l_batch_frame_data[i]
                    # 每个位置算SSIM，每个位置的SSIM大于阈值认为等价于有二维码

                    
                    ln_qr_tmp = []
                    for qr_detect, np_frame_qr in l_detect_qr_zone:
                        

                        margin=1
                        x0 = qr_detect[1] - margin if (qr_detect[1] - margin) > 0 else qr_detect[1] # 
                        x1 = (qr_detect[1] + qr_detect[3] + margin) if (qr_detect[1] + qr_detect[3] + margin) < self.n_h else qr_detect[1] + qr_detect[3]
                        y0 = qr_detect[0] - margin if (qr_detect[0] - margin) > 0 else qr_detect[0]
                        y1 = (qr_detect[0] + qr_detect[2] + margin) if (qr_detect[0] + qr_detect[2]) < self.n_w else qr_detect[0] + qr_detect[2]

                        for qr_tmp in ln_qr_tmp:
                            if abs(qr_tmp[0]-qr_detect[0])<=15 or abs(qr_tmp[1]-qr_detect[1])<=15:
                                continue
                        
                        # cv2.imwrite("tocheck.jpg", np_frame_tocheck[x0:x1,y0:y1])
                        # cv2.imwrite("detect.jpg", np_frame_qr)

                        sim, _ = compare_ssim(np_frame_tocheck[x0:x1,y0:y1], np_frame_qr, full=True)
                        # print(x1-x0, y1-y0, np_frame_qr.shape)
                        # print("cmp now ",n_ith_frame_tocheck, sim)
                        if sim>0.90:
                            ln_qr_tmp.append(qr_detect)  
                    lo_block_info[n_ith_frame_tocheck - 1]["qr"] = ln_qr_tmp   
                l_batch_frame_data = []
                video_reader.release()
                break
              
            np_frame_now = cv2.cvtColor(np_frame_now, cv2.COLOR_BGR2GRAY)
            l_batch_frame_data.append((n_ith_frame, np_frame_now))

            if n_ith_frame % n_detect_step == 0: # 每隔10帧进行侦测   
                ln_qr_detect = self.detect_qrcode_v2(np_frame_now[:, :int(np_frame_now.shape[1] * 3 / 4)])
                
                for qr_detect in ln_qr_detect:
                    if qr_detect[2] < 10 or qr_detect[3] < 10: # 太小的不是二维码区间
                        continue

                    b_add = True # 已有的不需要加入
                    for qr_sav, _ in l_detect_qr_zone:
                        if abs(qr_sav[0]-qr_detect[0])<15 or abs(qr_sav[1]-qr_detect[1])<15:
                            b_add = False
                            break
                    
                    margin=1
                    x0 = qr_detect[1] - margin if (qr_detect[1] - margin) > 0 else qr_detect[1] # 
                    x1 = (qr_detect[1] + qr_detect[3] + margin) if (qr_detect[1] + qr_detect[3] + margin) < self.n_h else qr_detect[1] + qr_detect[3]
                    y0 = qr_detect[0] - margin if (qr_detect[0] - margin) > 0 else qr_detect[0]
                    y1 = (qr_detect[0] + qr_detect[2] + margin) if (qr_detect[0] + qr_detect[2]) < self.n_w else qr_detect[0] + qr_detect[2]

                    b_add and l_detect_qr_zone.append((qr_detect, np_frame_now[x0:x1,y0:y1]))


            if n_ith_frame % n_batch_step == 0: # 每间100帧进行批处理
                # print("sav detect:", [e[0] for e in l_detect_qr_zone])

                for i in range(len(l_batch_frame_data)): # 对batch中每一帧进行处理
                    n_ith_frame_tocheck, np_frame_tocheck = l_batch_frame_data[i]
                    # 每个位置算SSIM，每个位置的SSIM大于阈值认为等价于有二维码

                    ln_qr_tmp = []
                    for qr_detect, np_frame_qr in l_detect_qr_zone:
                        for qr_tmp in ln_qr_tmp: # 已有的不需要重复侦测
                            if abs(qr_tmp[0]-qr_detect[0])<=15 or abs(qr_tmp[1]-qr_detect[1])<=15:
                                continue

                        margin=1
                        x0 = qr_detect[1] - margin if (qr_detect[1] - margin) > 0 else qr_detect[1] # 
                        x1 = (qr_detect[1] + qr_detect[3] + margin) if (qr_detect[1] + qr_detect[3] + margin) < self.n_h else qr_detect[1] + qr_detect[3]
                        y0 = qr_detect[0] - margin if (qr_detect[0] - margin) > 0 else qr_detect[0]
                        y1 = (qr_detect[0] + qr_detect[2] + margin) if (qr_detect[0] + qr_detect[2]) < self.n_w else qr_detect[0] + qr_detect[2]

                        sim, _ = compare_ssim(np_frame_tocheck[x0:x1,y0:y1], np_frame_qr, full=True)
                        # print("cmp", n_ith_frame_tocheck, sim)
                        if sim>0.90:
                            ln_qr_tmp.append(qr_detect)  
                    lo_block_info[n_ith_frame_tocheck - 1]["qr"] = ln_qr_tmp   
                l_batch_frame_data = []
                # print()
                
            if n_ith_frame % 50 == 0:
                print(n_ith_frame, lo_block_info[n_ith_frame - 1])

        for e in lo_block_info:
            print(e)

        print("len lo_block_info ", len(lo_block_info))
        print("frame cnt: ", frameCount)
        print("memery size: ", sys.getsizeof(lo_block_info))
        return 0, lo_block_info, o_video_info


    def block_img(self, np_img_raw, o_block_info):
        for qr_r in o_block_info["qr"]:
            x0=qr_r[1]-1 if (qr_r[1]-1)>0 else qr_r[1]
            x1=(qr_r[1]+qr_r[3]+1) if (qr_r[1]+qr_r[3]+1)<np_img_raw.shape[0] else qr_r[1]+qr_r[3]
            y0=qr_r[0]-1 if (qr_r[0]-1)>0 else qr_r[0]
            y1=(qr_r[0]+qr_r[2]+1) if (qr_r[0]+qr_r[2])<np_img_raw.shape[1] else qr_r[0]+qr_r[2]

            np_qr_template_in = cv2.resize(self.np_qr_template_raw, (y1-y0,x1-x0), fx=1, fy=1,
                                           interpolation=cv2.INTER_AREA)
            # print(np_qr_template_in.shape,np_img_raw.shape)
            # print(x0,x1,y0,y1,x1-x0,y1-y0)
            # cv2.imwrite("./log/frme.png",np_img_raw)
            # cv2.imwrite("./log/zone.png",np_img_raw[x0:x1,y0:y1, :])
            # cv2.imwrite("./log/test2.png",np_qr_template_in)
            np_img_raw[ x0:x1,y0:y1, :] = np_qr_template_in

        return np_img_raw


    def block_video_faster(self, s_video_in_url, s_video_out_url):
        n_ret, s_run_msg = 0, "Success"
        t_start = time.time()
        # 得到每一帧需频闭的信息
        t1 = time.time()
        n_ret, lo_block_info, o_video_info = self.get_block_info(s_video_in_url)
        t2 = time.time()
        print("get block info ", t2 - t1)

        # 对每一帧图像进行处理
        if n_ret == 0: 
            # get all frame and process each frame
            # fourcc_h264 = cv2.VideoWriter_fourcc('H','2','6','4')
            t1 = time.time()
            s_video_out_tmp_url = s_video_out_url.replace(".mp4", "_tmp.mp4")
            print(s_video_out_tmp_url)

            # try:
            video_reader = cv2.VideoCapture(s_video_in_url)
            video_writer = cv2.VideoWriter(s_video_out_tmp_url, self.fourcc_mp4, o_video_info["fps"], (self.n_w, self.n_h))  
            n_ith_frame = 0
            while True:
                n_ith_frame += 1
                success, np_frame_now = video_reader.read()  
                if not success:
                    break
                if n_ith_frame == 20:
                    cv2.imwrite('ppt/qr.jpg', np_frame_now)
                np_frame_out = self.block_img(np_frame_now, lo_block_info[n_ith_frame - 1])
                if n_ith_frame == 20:
                    cv2.imwrite('ppt/qr_block.jpg', np_frame_out)
                video_writer.write(np_frame_out)
            video_reader.release()
            video_writer.release()
            # except Exception as e:
            #     s_run_msg = str(e)
            #     n_ret = -1

        # get audio from input video   
        t1 = time.time()
        s_aac_url = ""
        if n_ret == 0: 
            try:
                s_aac_url = s_video_in_url.replace(".mp4", ".aac")
                if os.path.exists(s_aac_url):
                    os.remove(s_aac_url)
                s_ffmpeg_cmd = "%s -i %s -vn -y -acodec copy %s" % (self.s_ffmpeg , s_video_in_url, s_aac_url)
                # s_ffmpeg_cmd = "%s -i %s -vn -y -acodec copy %s  -loglevel quiet" % (self.s_ffmpeg , s_video_in_url, s_aac_url)
                print(s_ffmpeg_cmd) 
                subprocess.call(s_ffmpeg_cmd, shell=True) # 调用命令行进行转换
            except Exception as e:
                s_run_msg = str(e)
                n_ret = -1
        t2 = time.time()
        print("get audio time ", t2 - t1)
       

        # mix audio to video
        t1 = time.time()
        if n_ret == 0:
            try:
                if os.path.exists(s_video_out_url):
                    os.remove(s_video_out_url)
                # s_ffmpeg_cmd = "%s -i %s -i %s -vcodec h264  -bsf:a aac_adtstoasc -strict -2 %s" % (self.s_ffmpeg , s_video_out_tmp_url, s_aac_url, s_video_out_url)
                s_ffmpeg_cmd = "%s -i %s -i %s -vcodec h264 -acodec copy %s" % (self.s_ffmpeg , s_video_out_tmp_url, s_aac_url, s_video_out_url)
                
                print(s_ffmpeg_cmd) 
                subprocess.call(s_ffmpeg_cmd, shell=True) # 调用命令行进行转换
            except Exception as e:
                s_run_msg = str(e)
                n_ret = -1
        t2 = time.time()
        print("get new video time ", t2 - t1)

        if os.path.exists(s_video_out_tmp_url): 
            os.remove(s_video_out_tmp_url)

        if os.path.exists(s_aac_url): 
            os.remove(s_aac_url)

        t_end = time.time()
        print("duration ", o_video_info["duration"])
        print("total cost time ", t_end - t_start)
        print()


        o_video_info["cost_time"] = t_end - t_start
        return n_ret, s_run_msg, o_video_info

    def get_qr(self, img):
        '''
        detect qrcode of an image
        input: img: numpy array with shape [h,w,3] or [h,w]
        outputs: success code and a list of qrcode position and url in qrcode: 0, [(x1,y1,w1,h1,url1) (x2,y2,w2,h2,url2), ...]
        0 for success and -1 for failed
        x, y for the upper left corner of the rect
        '''
        try:
            res = self.detect_qrcode_v3(img)
            res = self.unique_qr_zone(res)
        except Exception as e:
            print(e)
            return -1, []
        return 0, res

    def cover_img(self, np_img_raw, np_block_img_raw, qrs):
        '''
        cover multiple area of an image with a logo image
        inputs: 
            np_img_raw, np_block_img_raw: image to cover and logo image are numpy array with same channels
            qrs: a list of positions to cover
        outputs: success code and blocked image, 0 for success and -1 for failed
        '''
        self.np_qr_template_raw = np_block_img_raw
        try:
            for qr in qrs:
                np_img_raw = self.block_img(np_img_raw, {'qr': [qr]})
        except Exception as e:
            print(e)
            return -1, np_img_raw
        return 0, np_img_raw
