import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import yaml

def write_instruction(im,text,color):
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    # Calculate text size
    org = (int(im.shape[1]/25) ) , (int(im.shape[0]*0.9))
    im = cv2.putText(im, text, org, font, 0.8, color, font_thickness)
    return im
def calculate_iou_tensor(boxA, my_queue):
    c=0
    for boxesB in my_queue:
        ious=[]
        for boxB in boxesB:
            # Ensure the tensors are floats, as integer division might cause issues
            boxA = boxA.float()
            boxB = boxB.float()
            
            # Determine the (x, y)-coordinates of the intersection rectangle
            xA = torch.max(boxA[0], boxB[0])
            yA = torch.max(boxA[1], boxB[1])
            xB = torch.min(boxA[2], boxB[2])
            yB = torch.min(boxA[3], boxB[3])
            
            # Compute the area of intersection rectangle
            intersection = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
            
            # Compute the area of both the prediction and ground-truth rectangles
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            
            # Compute the area of the union
            union = boxAArea + boxBArea - intersection
            
            # Compute the IoU by dividing the intersection by the union
            iou = intersection / union
            ious.append(iou.item())
        ious = max(ious)
        if ious>0.9:
            c+=1
    return c

def add_to_queue(element,my_queue):
    # Add the new element to the end of the queue
    my_queue.append(element)  
    # If the length of the queue exceeds 10, remove the oldest elements
    if len(my_queue) > 25:
        # Calculate how many elements to remove
        num_to_remove = len(my_queue) - 25
        # Remove the oldest elements (from the beginning of the list)
        my_queue = my_queue[num_to_remove:]
    return my_queue

@smart_inference_mode()
def run(
        weights='best.pt',  # model path or triton URL
        source='6.mp4',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(1280,1280),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        xmin=0,
        ymin=1/3,
        xmax=1,
        ymax=3/4,
        instruction="instruction.yaml"
):
    # read the data file and the instruction file
    with open("data.yaml", 'r') as file:
        data_yaml = yaml.safe_load(file)
    with open("instruction.yaml", 'r') as file:
        instruction_content = yaml.safe_load(file)
    variables = {}
    instruction=[data_yaml["names"].index(i) for i in instruction_content["sequence"].values()]
    for i,ii in enumerate(instruction):
        variables[f"seq{i}"] = [ii,False]

    xmin,ymin,xmax,ymax = float(xmin),float(ymin),float(xmax),float(ymax)
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    counter=0
    class_list=[v[0] for v in variables.values()]
    first_task = data_yaml["names"][class_list[0]]
    actual_text = f'Start by taking the {first_task}'
    actual_color = (0,255,255)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    my_queue = []
    for path, im, im0s, vid_cap, s in dataset:
        counter +=1
        # skip specific frames
        if counter<270:
            continue
        #if counter>700:
        #    break
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        start_the_process = False
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                counter_of_previous_class=0
                previous_classes = {}
                my_queue = add_to_queue(det[:,:4],my_queue)
                for *xyxy, conf, cls in reversed(det):
                    first_object_c = 0
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    img_test = im0.copy()
                    
                        #im0 = write_instruction(im0,actual_text,color=(0,255,255))
                    #if xyxy[0].item() >= 0 and xyxy[1].item()>=0 and xyxy[2].item() <= int(img_test.shape[1]/2) and xyxy[3].item()<=int(img_test.shape[0]*3/3): #int(im0.shape[0]/5)
                    classes_inside_the_area = [i[-1].item() for  i in det[:,:] if i[0].item() >= int(img_test.shape[1]*xmin) and i[1].item()>=int(img_test.shape[0]*ymin)  and i[2].item() <= int(img_test.shape[1]*xmax) and i[3].item()<=int(img_test.shape[0]*ymax)]
                    if len(my_queue) >= 10:
                        start_the_process = True
                    if start_the_process and xyxy[0].item() >= int(img_test.shape[1]*xmin) and xyxy[1].item()>=int(img_test.shape[0]*ymin)  and xyxy[2].item() <= int(img_test.shape[1]*xmax) and xyxy[3].item()<=int(img_test.shape[0]*ymax) : #int(im0.shape[0]/5)
                        # this statement ensure that in order to consider an object as a part of the sequence, it must be seen at least 20 times in the last 25 frames in the same area
                        if calculate_iou_tensor(det[:,:4][i],my_queue)>20:
                            keys_list = [v[1] for v in variables.values()]
                            class_list = [v[0] for v in variables.values()]
                            try:
                                index_of_seq=keys_list.index(False)
                            except:
                                index_of_seq = None
                            if index_of_seq is not None:
                                name_of_target_class = data_yaml["names"][class_list[index_of_seq]]

                                
                                if int(cls.item()) in [v[0] for v in variables.values() if v[1]==True] and counter_of_previous_class<= [v[0] for v in variables.values() if v[1]==True].count(int(cls.item())):
                                    counter_of_previous_class+=1
                                    previous_classes[f'cls{int(cls.item())}']=counter_of_previous_class
                                    name_of_next_class = data_yaml["names"][class_list[index_of_seq]]
                                    actual_text = f"Cool, now take the {name_of_next_class}"
                                    actual_color = (0,255,0)
                                    annotator.box_label(xyxy, label, color=(0,255,0))
                                elif int(cls.item()) == class_list[index_of_seq]:
                                    annotator.box_label(xyxy, label, color=(0,255,0))
                                    variables[f"seq{index_of_seq}"][1] = True
                                    if index_of_seq+1 < len(class_list):
                                        name_of_next_class = data_yaml["names"][class_list[index_of_seq+1]]
                                        actual_text = f"Cool, now take the {name_of_next_class}"
                                        actual_color = (0,255,0)
                                        break
                                    else:
                                        actual_text = "Process finished"
                                        actual_color = (255,255,0)
                                        break
                                else:
                                    actual_text = f"No, take the {name_of_target_class}"
                                    actual_color = (0,0,255)
                                    if name_of_target_class=="pliers":
                                        print(" ")
                                    annotator.box_label(xyxy, label, color=(0,0,255))
                                    break
            # Stream results
            im0 = write_instruction(im0,actual_text,color=actual_color)
            im0 = annotator.result()
            # draw the area where we are looking at objects
            im0  = cv2.rectangle(im0,(int(im0.shape[1]*xmin),int(im0.shape[0]*ymin)),(int(im0.shape[1]*xmax),int(im0.shape[0]*ymax)),(255,0,0),2) #int(im0.shape[0]/5)
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    #im0  = cv2.rectangle(im0.copy(),(0,int(im0.shape[0]/5)),(int(im0.shape[1]/2),int(im0.shape[0]*3/3)),(255,0,0),2)
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', type=str, default="instruction.yaml", help='The instruction file')
    parser.add_argument('--xmin', type=str, default=0, help='xmin coordinate of the zone')
    parser.add_argument('--xmax', type=str, default=1, help='xmax coordinate of the zone')
    parser.add_argument('--ymin', type=str, default=0.33, help='ymin coordinate of the zone')
    parser.add_argument('--ymax', type=str, default=0.75, help='ymax coordinate of the zone')
    parser.add_argument('--weights', nargs='+', type=str, default="best.pt", help='model path or triton URL')
    parser.add_argument('--source', type=str, default='6.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))
    #run()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
