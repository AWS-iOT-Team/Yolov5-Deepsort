import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

import numpy as np
import threading
import pyrealsense2 as rs

# 쓰레드종료 알림 변수 (기본False, 종료요청True)
stopThread_flag = False

# 테스트용 전역변수
target_xval = 0.5
distance_val = 0.0
obs_val = 0

# 파이프라인fifo 쓰레드함수
def fifoThread ():
    global stopThread_flag
    global target_xval
    global distance_val
    global obs_val

    print("fifo thread on")
    if not os.path.exists('/tmp/from_yolo_fifo'):
        os.mkfifo("/tmp/from_yolo_fifo", 0o777)
    if not os.path.exists('/tmp/to_yolo_fifo'):
        os.mkfifo("/tmp/to_yolo_fifo", 0o777)

    fd_from_yolo = os.open("/tmp/from_yolo_fifo", os.O_RDWR)
    fd_to_yolo = os.open("/tmp/to_yolo_fifo", os.O_RDWR)

    while True:
        # 타겟의 x좌표의 오른쪽에있고, 오른쪽으로 회전하기 위한값을 fifo로전달
        if target_xval > 0.9:
            buff_a = 'A'
        elif target_xval > 0.8:
            buff_a = 'B'
        elif target_xval > 0.7:
            buff_a = 'C'
        elif target_xval > 0.6:
            buff_a = 'D'

        # 타겟의 x좌표가 왼쪽에있고, 왼쪽으로 회전하기 위한값을 fifo로전달
        elif target_xval < 0.4:
            buff_a = 'E'
        elif target_xval < 0.3:
            buff_a = 'F'
        elif target_xval < 0.2:
            buff_a = 'G'
        elif target_xval < 0.1:
            buff_a = 'H'

        # 타겟의 x좌표가 중앙 0.5에 있을때
        else:
            # 타겟과의 거리가 멀리있을때 전진
            if distance_val > 80.0:
                buff_a = 'c'
            # 타겟과의 거리가 가까이있을때 후진 # 후진할필요없으면 주석처리
            # elif distance_val < 0.5:
            #     buff_a = 'd'
            # 타겟과의 거리가 적당거리일떄 멈춤
            else:
                buff_a = 'i'
	
        print(buff_a)
        # 파일에 fifo로 쓰기 문자열은 .encode()해서 보내야함
        os.write(fd_from_yolo, buff_a.encode())
        time.sleep(1)

        if stopThread_flag == True: # 종료문
            break

    print("fifo thread off")

# 물체의 좌우 끝 좌표를 받아 그 물체 의 거리값을 가져와주는 함수 
def location_to_depth(grayimg, loc1, loc2, depth_data):
    if loc1[0] < loc2[0]:
        arr = []
        start = loc1[0]
        end = loc2[0]

        for i in range(start, end+1):
            arr.append(grayimg[loc1[1], i])
        np_arr = np.array(arr)
        index_arr = np.argmax(np_arr)

        target_depth = depth_data.get_distance( loc1[0] + index_arr, loc1[1] )

    return round(100 * target_depth,2)

def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or '1' or '2' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    global stopThread_flag # 쓰레드 종료명령
    global target_xval # 쓰레드 사물의x좌표 전달 변수
    global distance_val # 쓰레드 거리데이터 전달 변수

    # 추적할 person타겟 초기화 
    targeting_pers = 1 

    # initialize deepsort 초기화
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        # read class object with enum ,enumerate(class)

    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    # fifo쓰레드 생성
    t1 = threading.Thread(target = fifoThread, args=())
    # 쓰레드 생성실패시
    if t1 == 0: 
         os.exit()
    else: # 성공시
         t1.start()

    # datsset.py로부터 class를 통해 영상들 받아오는곳
    for frame_idx, (path, img, im0s, vid_cap, depth_img, depth_im0s, depth_data) in enumerate(dataset):
        #print("frame_idx:",frame_idx,"path:",path,"vid_cap",vid_cap)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            #depth
            dep_img = depth_im0s[i].copy() 
            dm = cv2.flip(dep_img, 1)
            depth_alpha = cv2.convertScaleAbs(dm, alpha=0.15)
            dm0 = depth_colorImg = cv2.applyColorMap(depth_alpha, cv2.COLORMAP_JET)

            depth_grayimg = cv2.cvtColor(dm0,cv2.COLOR_RGBA2GRAY)
            # 영상 이진화 OTSU
            _, OTSU_binary = cv2.threshold(depth_grayimg, 0, 255, cv2.THRESH_OTSU)
            # 가우시안 블러와 OTSU로 노이즈제거
            OTSU_blur = cv2.GaussianBlur(OTSU_binary, (5, 5), 0)
            cv2.imshow("OTSU_blur", OTSU_blur)
            ret, OTSU_gaubin = cv2.threshold(OTSU_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imshow('src_gaubin', OTSU_gaubin)

            # STEP 5. 외곽선 검출
            # cv2 contours 외곽선 추출함수
            contours, _ = cv2.findContours(OTSU_gaubin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours_images = np.zeros((480, 640, 3), np.uint8) # 검은판 하나 생성
            # 모든 객체 외곽선
            for Line_idx in range(len(contours)):
                color = (0,0,255) 
                cv2.drawContours(contours_images, contours, Line_idx, color, 1, cv2.LINE_AA)

            # 장애물 외곽선 연결선
            box_cnt = 0 # 박스번호
            for contour in contours:
                # convexHull 나머지 모든점을 포함하는 다각형을 만들어내는 알고리즘 = 장애물 외곽선
                conhull = cv2.convexHull(contour)
                hull = np.max(conhull, axis=1)
                maxbox = np.max(hull, axis=1)
                # 최대값x와 최소값x을 뺀 절대값이 (노란박스크기가)작은건 안그려지게한다
                if abs(max(maxbox) - min(maxbox)) > 90:  
                    cv2.drawContours(contours_images, [conhull], 0, (0, 255, 255), 3)  

                    # 한 박스(hull) 마다  맨 좌측값, 맨우측값 가져오기
                    max_x = np.array([24, 0]) # y축 맨오른쪽좌표가 들어갈 변수
                    min_x = np.array([639, 0])  # y축 맨왼쪽좌표가 들어갈 변수
                    for count in hull:
                        if max_x[0] < count[0]:
                            max_x = count 
                        if min_x[0] > count[0]:
                            min_x = count
                    cv2.circle(contours_images, max_x, 3, (255, 0, 0), 2, cv2.LINE_AA)  # 최우측 좌표에 파란
                    cv2.circle(contours_images, min_x, 3, (255, 255, 0), 2, cv2.LINE_AA)  # 최좌측 좌표에 청록
                    obs_depth = location_to_depth(depth_grayimg, min_x, max_x, depth_data)
                    if obs_depth < 300.00: #최소 장애물과의 거리
                        # 오른쪽에서 시작된 장애물 (맨오른쪽좌표가 화면끝과 연결됨)
                        if max_x[0] >= 638:
                            obs_val = float(min_x[0])
                        # 왼쪽에서 시작된 장애물 (맨왼쪽좌표가 화면끝과 연결됨)
                        elif min_x[0] <= 28:
                            obs_val = float(max_x[0])                  
                        # 중앙 장애물
                        else:
                            obs_val = float(max_x[0]+min_x[0])

                        print(box_cnt,"번째 장애물의 거리 = {} cm // x위치값 = {}".format(obs_depth,obs_val))
                    box_cnt = box_cnt + 1
                
                else: # 장애물이 없는상태
                    obs_val = 0

            cv2.imshow("src", contours_images)

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            # box
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            annotator2 = Annotator(dm0, line_width=2, pil=not ascii)            

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]                       
                 		
                        
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        w = 640
                        h = 480
                        depth_w = (w-20) * 23 // 35
                        depth_h = h * 2 // 3
                        
                        
                        start_left =  depth_w * (output[0] / w) + (w-20) // 7 # 왼쪽위 x좌표
                        start_top =  depth_h * (output[1] / h) + h // 6 # 왼쪽위 y좌표
                        end_left =  depth_w * (output[2] / w) + (w-20) // 7 # 오른쪽아래 x좌표
                        end_top =  depth_h * (output[3] / h) + h // 6 # 오른쪽아래 y좌표  

                        if output[0] >= 20:
                            bboxes2 = np.array([start_left, start_top, end_left, end_top])  
                        else:
                            bboxes2 = np.array([(w-20) // 7, start_top, end_left, end_top])
                        annotator2.box_label(bboxes2, label, color=colors(c, True))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                               f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                        #print((start_left+end_left))
                        #print("start_left", start_left)
                        #print(type(start_left))
                        dep_x, dep_y = int((start_left+end_left)//2),int((start_top+end_top)//2)
                        target_distacne = depth_data.get_distance(dep_x, dep_y)
                        cv2.circle(dm0,(dep_x, dep_y),3,(0,0,255),1,cv2.LINE_AA)

                        # 타겟의 거리가 적당한것만 (오류제외)
                        if 0.01 < target_distacne < 7.0:
                            # 내가 원하는 타겟 person
                            if targeting_pers == id:
                                # 타켓person의 x축위치를 fifo쓰레드로 전달
                                target_xval = ((start_left + end_left) / 2) / w
                                # 타켓person의 거리를 fifo쓰레드로 전달 
                                distance_val = int(target_distacne * 100)
                                print(id,":타겟 person과의 거리 = {:.2f} cm".format(target_distacne * 100))
                            # 타겟외에 다른 person
                            else:
                                print(id,":person과의 거리 = {:.2f} cm".format(target_distacne * 100))

                        # class의 target : person 0 id -> 2,3,4,5을 바꿔가며 쫒아가는
                        

            else:
                deepsort.increment_ages()


          

            # 객체A 검출에 걸리는 시간 (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            im0 = annotator.result()
            dm0 = annotator2.result()
            if show_vid:
                cv2.imshow("deteciton", im0)
                cv2.imshow("dm0", dm0)

                inputKey = cv2.waitKey(1)
                if inputKey == ord('q') or inputKey == 27:  # q to quit
                    # 쓰레드 종료명령
                    stopThread_flag = True
                    raise StopIteration

            # 영상 저장 (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print("fps:",fps, "w:", w, "h:",h)
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
    # 로그 txt파일
    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    # 전체 작동시간fps
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    
    with torch.no_grad():
        detect(args)


