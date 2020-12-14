import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision
import numpy as np
import cv2
import random
import net
import numpy
from torchvision import transforms
from utils import *
import matplotlib.image as img
import matplotlib.pyplot as plt
import pytesseract



def init_weights(m):
   
   if type(m) == nn.modules.conv.Conv2d:
      print("Weights initialized for:", m)
      torch.nn.init.xavier_uniform(m.weight)
      m.bias.data.fill_(0.01)


def enhance(img_path, scale):

   SRNet = net.SRNet().cuda()
   SRNet.apply(init_weights)

   criterion = nn.L1Loss().cuda()

   optimizer = torch.optim.Adam(SRNet.parameters(), lr=0.001)

   SRNet.train()

   image = img.imread(img_path)
   hr_fathers_sources = [image]

   scale_factors = np.array([[1.0, 1.5], [1.5, 1.0], [1.5, 1.5], [1.5, 2.0], [2.0, 1.5], [2.0, 2.0]])
   back_projection_iters = np.array([6, 6, 8, 10, 10, 12])
   learning_rate_change_iter_nums = [0]

   rec_mse = []
   steps_mse = []

   
   for sf_ind, scale in enumerate(scale_factors):

      for i in range(10000):

         hr_father = random_augment(ims=hr_fathers_sources,
                              base_scales = [1.0] + list(scale_factors),
                              leave_as_is_probability=0.05,
                              no_interpolate_probability=0.45,
                              min_scale=0.5,
                              max_scale=([1.0]+list(scale_factors))[len(hr_fathers_sources)-1],
                              allow_rotation=True,
                              scale_diff_sigma=0.25,
                              shear_sigma=0.1,
                              crop_size=128
                              )

         lr_son = father_to_son(hr_father, scale)
         lr_son_interpolated = imresize(lr_son, scale, hr_father.shape, "cubic")

         hr_father = torch.from_numpy(hr_father).unsqueeze(0).cuda().permute(0,3,1,2).float()
         lr_son_interpolated = torch.from_numpy(lr_son_interpolated).unsqueeze(0).cuda().permute(0,3,1,2).float()

         sr_son = SRNet(lr_son_interpolated)

         loss = criterion(sr_son, hr_father)

         if(not i % 50):
            son_out = father_to_son(image, scale)
            son_out_inter = imresize(son_out, scale, image.shape, "cubic")
            son_out_inter = torch.from_numpy(son_out_inter).unsqueeze(0).cuda().permute(0,3,1,2).float()            
            sr_son_out = SRNet(son_out_inter).permute(0,2,3,1).squeeze().data.cpu().numpy()
            sr_son_out = np.clip(np.squeeze(sr_son_out), 0, 1)
            rec_mse.append(np.mean(np.ndarray.flatten(np.square(image - sr_son_out))))
            steps_mse.append(i)

         lr_policy(i, optimizer, learning_rate_change_iter_nums, steps_mse, rec_mse)

         #curr_lr = 100
         for param_group in optimizer.param_groups:
            #if param_group['lr'] < 9e-6:
            curr_lr = param_group['lr']
            break




         optimizer.zero_grad()
         loss.backward()
         optimizer.step()      

         if i%10 == 0:
            print("Iteration:", i, "Loss:",loss.item())

         if curr_lr < 9e-6:
            break
      

       ### Evaluation the result

      lr_img = img.imread(img_path)
      
      interpolated_lr_img = imresize(lr_img, scale, None, "cubic")
      interpolated_lr_img = torch.from_numpy(interpolated_lr_img).unsqueeze(0).cuda().permute(0,3,1,2).float()
      
      sr_img = infer(lr_img, scale, sf_ind, SRNet, back_projection_iters) #SRNet(interpolated_lr_img)

      save_img = torch.from_numpy(sr_img).unsqueeze(0).permute(0,3,1,2)
      torchvision.utils.save_image((save_img),img_path.split(".")[0]+'SR.'+ img_path.split(".")[1], normalize=False)
      torchvision.utils.save_image((interpolated_lr_img),img_path.split(".")[0]+'LR.'+img_path.split(".")[1] , normalize=False)

      hr_fathers_sources.append(sr_img)
      print("Optimization done for scale", scale)



def infer(input_img, scale, sf_ind, SRNet, back_projection_iters):
   
   outputs = []

   for k in range(0, 1+7, 1+int(scale[0] != scale[1])):
      test_img = np.rot90(input_img, k) if k < 4 else np.fliplr(np.rot90(input_img,k))
      interpolated_test_img = imresize(test_img, scale, None, "cubic")
      interpolated_test_img = torch.from_numpy(interpolated_test_img).unsqueeze(0).cuda().permute(0,3,1,2).float()
      tmp_output = SRNet(interpolated_test_img)
      tmp_output = tmp_output.permute(0,2,3,1).squeeze().data.cpu().numpy()
      tmp_output = np.clip(np.squeeze(tmp_output), 0, 1)

      tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), k)

      for bp_iter in range(back_projection_iters[sf_ind]):
         tmp_output = back_projection(tmp_output, input_img, "cubic", "cubic", scale)

      outputs.append(tmp_output)


   outputs_pre = np.median(outputs, 0)

   for bp_iter in range(back_projection_iters[sf_ind]):
      outputs_pre = back_projection(outputs_pre, input_img, "cubic", "cubic", scale)

   return outputs_pre


def lr_policy(iters, optimizer, learning_rate_change_iter_nums, mse_steps, mse_rec):

   if ((not (1 + iters) % 60) and (iters - learning_rate_change_iter_nums[-1] > 256)):
      [slope, _], [[var,_],_] = np.polyfit(mse_steps[(-256//50):], mse_rec[(-256//50):], 1, cov=True)

      std = np.sqrt(var)

      print('Slope:', slope, "STD:", std)

      if -1.5*slope < std:
         for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.8
         print("Learning Rate Updated:", param_group['lr'])
         learning_rate_change_iter_nums.append(iters)
      

def otsu_thres(image): 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #gray 로 변환
    g_blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0) #blurring 필수, noise 제거
    _, thresh_img = cv2.threshold(g_blur, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # return null, thresh_img
    thresh_img = cv2.bitwise_not(thresh_img) # 흑백 변환(tesseract는 검은바탕에 흰 숫자를 찾음)
    return thresh_img


def adaptive_thres(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #gray 로 변환
    g_blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0) #blurring 필수, noise 제거
    thresh_img = cv2.adaptiveThreshold(
        g_blur, # grayscale image
        maxValue=255.0, # maxValue – 임계값
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # adaptiveMethod – thresholding value를 결정하는 계산 방법
        thresholdType=cv2.THRESH_BINARY_INV, # thresholdType – threshold type
        blockSize=19, # blockSize – thresholding을 적용할 영역 사이즈 (최대 size = 19)
        C=9 # C – 평균이나 가중평균에서 차감할 값
    )
    return thresh_img


def num_thres(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    ret, thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh_img

def draw_contour(thresh_img):
    mode = cv2.RETR_LIST # 모든 컨투어 라인을 찾기
    method = cv2.CHAIN_APPROX_SIMPLE #컨투어 라인을 그릴 수 있는 포인트만 반환    
    contours, _ = cv2.findContours(thresh_img, mode, method) #contour 찾기

    contour_image = np.zeros((height, width), dtype=np.uint8) # contour그려진 이미지
    contourIdx = -1 # 컨투어 라인 번호
    color = (255,255,255) #white
    cv2.drawContours(contour_image, contours, contourIdx, color) #contour 그리기

    return contours, contour_image


def rect_contours(contours):
    rect_contour = [] #list 형태
    rect_image = np.zeros((height, width), dtype=np.uint8) # 사각형 
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) # contour를 둘러싸는 사각형 구하기 # 사각형 왼쪽 끝 point (x,y) , w,h 
        cv2.rectangle(rect_image, (x, y), (x+w, y+h), (255, 255, 255), thickness=2) # 이미지, 왼쪽 위 , 오른쪽 아래, 흰색, 선 두께 
    
        # dict_contour 추가 하기 
        rect_contour.append({'contour': contour, 'x': x,  'y': y, 'w': w, 'h': h, 
                             'cx': x + (w / 2), 'cy': y + (h / 2)  }) # cx, cy  = 사각형의 중심좌표
    
    return rect_contour, rect_image



def choice_1(rect_contour):
    # 실험값
    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0  # 가로/세로 비율 값
    
    candidate1 = []
    index = 0
    for d in rect_contour:
        area = d['w'] * d['h'] # 넓이
        ratio = d['w'] / d['h'] # 가로/세로의 비율 
    
        if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = index # index도 추가 해준다
            index += 1
            candidate1.append(d) #넓이 , 너비, 높이, 비율 기준 통과한 사각형을 candidate list에 추가 



    candidate1_image = np.zeros((height, width), dtype=np.uint8) #candidate1 image
    for d in candidate1:
        cv2.rectangle(candidate1_image, (d['x'], d['y']), (d['x']+d['w'], d['y']+d['h']), (255, 255, 255), thickness=2)
    
    return candidate1, candidate1_image



def dist(x,y,nx,ny): # sqrt( (x-nx)^2 + (y-ny)^2), 거리 구하기
    diff = x-nx
    diff = diff*diff
    dif = y -ny
    dif = dif*dif
    return np.sqrt(dif+diff)

def choice_2_idx(candidate1):
    #실험 값
    MAX_DAKAK_MULTIPLYER = 5 # box안의 대각선 길이와 box와 box사이의 거리는 5배가 넘지 않는다.
    MAX_ANGLE_DIFF = 12.0 # box와 box사이의 각도가 12도를 넘지 않는다.
    MAX_AREA_DIFF = 0.5 # box와 box 넓이 차이가 0.5배 이상이다.
    MAX_WIDTH_DIFF = 0.8 # box와 box 너비 차이가 0.8배 이상이다.
    MAX_HEIGHT_DIFF = 0.2 # box와 box 높이 차이가 0.2배 이상이다.
    MIN_N_MATCHED = 4 # 위의 조건을 만족하는 box가 4개 이상이여야 한다.
    
    candidate2_idx = [] #조건을 만족하는 index들을 저장한다.
    for d1 in candidate1:
        satisfy = [] # 조건을 만족하는 index를 임시로 저장할 list
        for d2 in candidate1:
            if d1['idx'] == d2['idx']: #같으면 continue
                continue
            
            over_check = False
            for c2_idx in candidate2_idx: # 중복 있는지 검사 
                if d1['idx'] in c2_idx:
                    over_check = True
                    break
                if d2['idx'] in c2_idx:
                    over_check = True
                    break
            if over_check is True: # 중복된게 있으면 스킵
                continue

            #중십값들의 차이 cdx,cdy
            cdx = abs(d1['cx'] - d2['cx'])
            cdy = abs(d1['cy'] - d2['cy']) 
            
            dakak_length = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2) # 대각선 길이 = sqrt(w^2 + h^2)
            distance = dist(d1['cx'], d1['cy'],d2['cx'], d2['cy']) # box와 box사이 거리
            if distance > dakak_length * MAX_DAKAK_MULTIPLYER: # 거리가 대각선 * 5 보다 더 크면 스킵
                continue
                
            if cdx == 0: # cdx 가 0인경우 예외처리
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(cdy / cdx)) # tan-1(cdy/cdx) , radian 을 degree로 바꾼다.
                
            if(angle_diff >= MAX_ANGLE_DIFF): # 각도가 크면 무시
                continue
                
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h']) # 넓이의 비 
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if  area_diff < MAX_AREA_DIFF and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF: # 모든 조건 통과
                satisfy.append(d2['idx'])

        # d2 for문 종료
        satisfy.append(d1['idx']) #d1도 추가
        satisfy_cnt = len(satisfy)
        if satisfy_cnt < MIN_N_MATCHED: # box가 4개 미만이면 무시
            continue

        candidate2_idx.append(satisfy) #모든 조건을 만족한 index들 candidate2_idx에 추가
        


    return candidate2_idx



def choice_2(candidate2_idx):
    
    candidate2 = [] # 조건을 만족하는 box들의 dictionary 정보
    candidate2_image = np.zeros((height, width), dtype=np.uint8)
   
    for index in candidate2_idx:
        candidate2.append(np.take(candidate1, index)) # index 정보를 통해 조건을 만족하는 dictionary 형태의 정보를 저장
    
    for candi in candidate2:
        for d in candi:
            cv2.rectangle(candidate2_image, (d['x'], d['y']), (d['x']+d['w'], d['y']+d['h']), (255, 255, 255), thickness=2)

    
    return candidate2, candidate2_image


def find_plate(candidate2): # plate = 번호판
    
    #실험 값
    PLATE_WIDTH_PADDING = 1.3 # plate 너비
    PLATE_HEIGHT_PADDING = 1.5 # plate 높이
    MIN_PLATE_RATIO = 3  # plate 가로/세로 비
    MAX_PLATE_RATIO = 10
    
    plate_images = [] 
    plate_infos = []

    for candi in candidate2:
        sorted_candi = sorted(candi, key=lambda x: x['cx']) # center 점을 기준으로 정렬한다.(왼쪽부터 순서대로)
    
        # 번호판의 센터점
        plate_cx = (sorted_candi[0]['cx'] + sorted_candi[-1]['cx']) / 2 # 가장 왼쪽 cx와 가장 오른쪽 cx의 가운데
        plate_cy = (sorted_candi[0]['cy'] + sorted_candi[-1]['cy']) / 2 # 가장 왼쪽 cy와 가장 오른쪽 cy의 가운데
    
        plate_width = (sorted_candi[-1]['x'] + sorted_candi[-1]['w'] - sorted_candi[0]['x']) * PLATE_WIDTH_PADDING # plate 너비
        # padding 붙이는 이유 ? 
    
        sum_height = 0
        for d in sorted_candi:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_candi) * PLATE_HEIGHT_PADDING) # plate 높이
    
        # 벌어진 각도에 따라 삼각형을 그릴수 있다.
        triangle_height = sorted_candi[-1]['cy'] - sorted_candi[0]['cy'] # 삼각형의 높이
        triangle_dakak = dist(sorted_candi[0]['cx'], sorted_candi[0]['cy'],sorted_candi[-1]['cx'], sorted_candi[-1]['cy']) #삼각형의 대각선 길이
    
        angle = np.degrees(np.arcsin(triangle_height / triangle_dakak)) # sin-1(h/dakak)
    
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0) #회전 행렬을 구한다
    
        rotate_image = cv2.warpAffine(thresh_img, M=rotation_matrix, dsize=(width, height)) # Affine 변형(여기서는 벌어진 만큼 회전)
    
        plate_size=(int(plate_width), int(plate_height))
        plate_center=(int(plate_cx), int(plate_cy))
    
        plate_image = cv2.getRectSubPix(rotate_image,plate_size, plate_center) #회전된 이미지 에서 번호판을 얻는다(아직 후보)
        plate_h, plate_w = plate_image.shape
        
        if plate_w / plate_h < MIN_PLATE_RATIO or plate_w / plate_h  > MAX_PLATE_RATIO: # 번호판의 가로/세로 비 검사
            continue
    
        plate_images.append(plate_image) #조건을 만족하는 번호판 이미지 저장
        
        plate_infos.append({             
            'x': int(plate_cx - plate_width / 2), #번호판 왼쪽 위 끝 point(x,y)
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        }) #조건을 만족하는 번호판의 정보를 저장
    
    return plate_images, plate_infos



def choose_plate(plate_images):
    # 실험값
    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0  # 가로/세로 비율 값
    
    
    max_len = 0 # 가장 긴 문자를 찾기위해 -> 수정 해보자
    answer = '' # 정답
    answer_idx = 0 # 정답 plate index
    idx =0
    length = len(plate_images)
    for plate in plate_images: # 후보 plate에서 contour를 찾아 본다, 문자열만 추리기 위해서
        
        plate = cv2.resize(plate, dsize=(0, 0), fx=1.6, fy=1.6) # ????
        _, plate = cv2.threshold(plate, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU) # thres
        contours ,_ = cv2.findContours(plate, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE) #contour
        
        #문자열 만 추리기 위해서 contour들의 min,max (x,y)를 각각 찾는다
        plate_min_x, plate_min_y = plate.shape[1], plate.shape[0] 
        plate_max_x, plate_max_y = 0, 0
    
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
        
            area = w * h
            ratio = w / h
            if area > MIN_AREA and w > MIN_WIDTH and h > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO: #문자 박스의 크기를 본다
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h
                
        img_result = plate[plate_min_y:plate_max_y, plate_min_x:plate_max_x] # 번호판 이미지 생성(문자열만 추려진)
        #print(img_result.shape)
        img_result = cv2.resize(img_result, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        #print(img_result.shape)
        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0) #노이즈 제거,필수
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU) #thres
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    #     tesseract가 잘 인식 할 수 있도록 경계를 만들어 준다
#         img_result = cv2.bitwise_not(img_result) # 흑백 변환(tesseract는 검은바탕에 흰 숫자를 찾음)
#         cv2.imwrite('./dataset/{}_{}.jpg'.format(inum,idx), img_result) 
        
        plt.figure(figsize=(8, 6))
        plt.subplot(length, 1, idx+1)
        plt.imshow(img_result, cmap='gray')
        chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0') #tesseract를 통해 이미지를 문자로 변환
        # psm 7 글이 한줄로 연결되어 있다는 가정
        # oem 1 옛날 버전(번호판에는 문맥이 없으므로 rnn을 사용하지 않은 예전 버젼을 이용)
        string = ''
        number_find = False
        char_find = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit(): #한글 or 숫자
                if c.isdigit(): #숫자가 포함이 되어있는지
                    number_find = True
                else: # 한글이 포함이 되어있는지
                    char_find = True
                string += c
    
        if len(string) > max_len and number_find and char_find: #숫자,한글이 포함되어있고, 가장 긴 문자열이 가장 높은 확률로 정답
            answer = string
            max_len = len(string)
            answer_idx = idx # answer_plate의 index
        idx += 1
        
    return answer, answer_idx









if __name__ == '__main__':
   ## First argument is the image that you want to upsample with ZSSR. 
        ## Second argument is the scale with which you want to resize. Currently only scale = 2 supported. For other scales, change the variable 'scale_factors' accordingly.
  
    enhance('images/dddd/26.png', 2)
    image = cv2.imread('images/dddd/26SR.png') 
    #image = cv2.resize(image, dsize=(826, 464), interpolation=cv2.INTER_AREA)
    height, width, _ = image.shape
    #print(height,width)
    #thresh_img = otsu_thres(image)
    thresh_img = adaptive_thres(image)
    #thresh_img = num_thres(image)
    contours, contour_image = draw_contour(thresh_img) # contour 그리기

    rect_contour, rect_image = rect_contours(contours) # contour -> 사각형

    candidate1, candidate1_image = choice_1(rect_contour) # 후보 1 선택
    candidate2_idx = choice_2_idx(candidate1) # 조건을 만족하는 contour후보의 idx가 저장됨
    candidate2, candidate2_image = choice_2(candidate2_idx) #후보2 선택
    plate_images, plate_infos = find_plate(candidate2) #번호판 후보 찾기

    # length = len(plate_images) 
    # for i in range(length): #번호판 후보 보기
    #     plt.subplot(length, 1, i+1)
    #     plt.imshow(plate_images[i], cmap='gray')
    
    answer, answer_idx = choose_plate(plate_images) #정답 찾기
    print(answer)
    
    d = plate_infos[answer_idx] #번호판 후보중 정답 idx로 정답 plate에 접근
    res_image = image.copy()
    cv2.rectangle(res_image, (d['x']-10, d['y']-10), (d['x']+d['w']+10, d['y']+d['h']+10), (255,0,0), thickness=2) # 정답 plate에 빨간 박스

    plt.figure(figsize=(12, 10))
    plt.imshow(res_image)

     