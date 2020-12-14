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
	
	image = image.reshape(image.shape[0],image.shape[1],1)
	#print(image.shape)
	hr_fathers_sources = [image]

	scale_factors = np.array([[1.0, 1.5], [1.5, 1.0], [1.5, 1.5], [1.5, 2.0], [2.0, 1.5], [2.0, 2.0]])
	back_projection_iters = np.array([6, 6, 8, 10, 10, 12])
	learning_rate_change_iter_nums = [0]

	rec_mse = []
	steps_mse = []

	
	for sf_ind, scale in enumerate(scale_factors):
		losses = list()

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
			hr_father = np.reshape(hr_father,(hr_father.shape[0],hr_father.shape[1],1))
			#print(hr_father.shape)

			lr_son = father_to_son(hr_father, scale)
			#print(lr_son.shape)

			lr_son_interpolated = imresize(lr_son, scale, hr_father.shape, "cubic")
			#print(lr_son_interpolated.shape)

			
			hr_father = torch.from_numpy(hr_father).unsqueeze(0).cuda().permute(0,3,1,2).float()
			#print(hr_father.shape)
			
			lr_son_interpolated = torch.from_numpy(lr_son_interpolated).unsqueeze(0).cuda().permute(0,3,1,2).float()
			#print(lr_son_interpolated .shape)
			

			sr_son = SRNet(lr_son_interpolated)
			#print(sr_son.shape)

			loss = criterion(sr_son, hr_father)
		

			if(not i % 50):
				son_out = father_to_son(image, scale)
				son_out_inter = imresize(son_out, scale, image.shape, "cubic")
				son_out_inter = torch.from_numpy(son_out_inter).unsqueeze(0).cuda().permute(0,3,1,2).float()
				#print(son_out_inter .shape)				
				sr_son_out = SRNet(son_out_inter).permute(0,2,3,1).squeeze().data.cpu().numpy()
				sr_son_out = np.clip(np.squeeze(sr_son_out), 0, 1)
				sr_son_out = sr_son_out.reshape(sr_son_out.shape[0],sr_son_out.shape[1],1) 
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
			
			losses.append(loss.item())
					

			if i%1000 == 0:
				print("Iteration:", i, "Loss:",loss.item())

			if curr_lr < 9e-6:
				break
		

	    ### Evaluation the result

		lr_img = img.imread(img_path) 
		interpolated_lr_img = imresize(lr_img, scale, None, "cubic")
		interpolated_lr_img = interpolated_lr_img.reshape(interpolated_lr_img.shape[0],interpolated_lr_img.shape[1],1)
		interpolated_lr_img = torch.from_numpy(interpolated_lr_img).unsqueeze(0).cuda().permute(0,3,1,2).float()
		
    

		sr_img = infer(lr_img, scale, sf_ind, SRNet, back_projection_iters) #SRNet(interpolated_lr_img)
		plt.figure(figsize=(12, 10))
		plt.imshow(sr_img, cmap='gray')
		sr_img = sr_img.reshape(sr_img.shape[0],sr_img.shape[1],1)
		save_img = torch.from_numpy(sr_img).unsqueeze(0).permute(0,3,1,2)
		torchvision.utils.save_image((save_img),img_path.split(".")[0]+'SR.'+ img_path.split(".")[1], normalize=False)
		torchvision.utils.save_image((interpolated_lr_img),img_path.split(".")[0]+'LR.'+img_path.split(".")[1] , normalize=False)

		hr_fathers_sources.append(sr_img)
		print("Optimization done for scale", scale)
		losses = np.asarray(losses)
		print(losses.shape)
		fig = plt.figure()
		plt.plot(range(losses.shape[0]), losses[:], 'r--')
		plt.xlabel('ITERATION')
		plt.ylabel('LOSS')
		plt.show()
    

    



def infer(input_img, scale, sf_ind, SRNet, back_projection_iters):
	
	outputs = []

	for k in range(0, 1+7, 1+int(scale[0] != scale[1])):
		test_img = np.rot90(input_img, k) if k < 4 else np.fliplr(np.rot90(input_img,k))
		interpolated_test_img = imresize(test_img, scale, None, "cubic")
		interpolated_test_img = interpolated_test_img.reshape(interpolated_test_img.shape[0],interpolated_test_img.shape[1],1)
    
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
		











if __name__ == '__main__':
  print("hi")
  img_ori = cv2.imread("images/(%d).png" % 37)

  height, width, channel = img_ori.shape
  plt.figure(figsize=(12, 10))
  plt.imshow(img_ori, cmap='gray')

  gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

  plt.figure(figsize=(12, 10))
  plt.imshow(gray, cmap='gray')

  img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

  img_thresh = cv2.adaptiveThreshold(
    img_blurred, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19, 
    C=9
  )

  plt.figure(figsize=(12, 10))
  plt.imshow(img_thresh, cmap='gray')

  contours, _ = cv2.findContours(
    img_thresh, 
    mode=cv2.RETR_LIST, 
    method=cv2.CHAIN_APPROX_SIMPLE
  )

  temp_result = np.zeros((height, width, channel), dtype=np.uint8)

  cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

  plt.figure(figsize=(12, 10))
  plt.imshow(temp_result)

  temp_result = np.zeros((height, width, channel), dtype=np.uint8)

  contours_dict = []

  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
    
    # insert to dict
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

  plt.figure(figsize=(12, 10))
  plt.imshow(temp_result, cmap='gray')

  MIN_AREA = 80
  MIN_WIDTH, MIN_HEIGHT = 2, 8
  MIN_RATIO, MAX_RATIO = 0.25, 1.0

  possible_contours = []

  cnt = 0
  for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']
    
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)
        
  # visualize possible contours
  temp_result = np.zeros((height, width, channel), dtype=np.uint8)

  for d in possible_contours:
#     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

  plt.figure(figsize=(12, 10))
  plt.imshow(temp_result, cmap='gray')

  MAX_DIAG_MULTIPLYER = 5 # 5
  MAX_ANGLE_DIFF = 12.0 # 12.0
  MAX_AREA_DIFF = 0.5 # 0.5
  MAX_WIDTH_DIFF = 0.8
  MAX_HEIGHT_DIFF = 0.2
  MIN_N_MATCHED = 3 # 3

  def find_chars(contour_list):
    matched_result_idx = []
    
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # append this contour
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        
        # recursive
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx
    
  result_idx = find_chars(possible_contours)

  matched_result = []
  for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

  # visualize possible contours
  temp_result = np.zeros((height, width, channel), dtype=np.uint8)

  for r in matched_result:
    for d in r:
  #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

  plt.figure(figsize=(12, 10))
  plt.imshow(temp_result, cmap='gray')

  PLATE_WIDTH_PADDING = 1.3 # 1.3
  PLATE_HEIGHT_PADDING = 1.5 # 1.5
  MIN_PLATE_RATIO = 3
  MAX_PLATE_RATIO = 10

  plate_imgs = []
  plate_infos = []

  for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
    
    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )
    
    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    
    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
    
    img_cropped = cv2.getRectSubPix(
        img_rotated, 
        patchSize=(int(plate_width), int(plate_height)), 
        center=(int(plate_cx), int(plate_cy))
    )
    
    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        continue
    
    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })
    
    plt.subplot(len(matched_result), 1, i+1)
    plt.imshow(img_cropped, cmap='gray')

  longest_idx, longest_text = -1, 0
  plate_chars = []

  for i, plate_img in enumerate(plate_imgs):
    plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # find contours again (same as above)
    contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    
    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        area = w * h
        ratio = w / h

        if area > MIN_AREA \
        and w > MIN_WIDTH and h > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h
                
    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
    
    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

    chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
    
    result_chars = ''
    has_digit = False
    for c in chars:
        if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
            if c.isdigit():
                has_digit = True
            result_chars += c
    
    print(result_chars)
    plate_chars.append(result_chars)

    if has_digit and len(result_chars) > longest_text:
        longest_idx = i

    plt.subplot(len(plate_imgs), 1, i+1)
    plt.imshow(img_result, cmap='gray')
    cv2.imwrite("images/%d_c.png" % 37, img_result)

  info = plate_infos[longest_idx]
  chars = plate_chars[longest_idx]

  print(chars)

  img_out = img_ori.copy()

  cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(255,0,0), thickness=2)

  plt.figure(figsize=(12, 10))
  plt.imshow(img_out)

  enhance("images/%d_c.png" % 37, 2)
  img_crop = cv2.imread("images/(%d_cSR).png" % 37)
  chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
  print(chars)

	#enhance('images/1_gray.png', 2)

