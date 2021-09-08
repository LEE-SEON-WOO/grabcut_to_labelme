import cv2
import numpy as np
import sys
import os
import json

COLOR_BG = (255,0,0)
COLOR_FG = (0,255,0)

def mask2color(mask):
    r,c = mask.shape[:2]
    color = np.zeros((r,c,3),np.uint8)
    color[np.where((mask==0)|(mask==2))] = COLOR_BG
    color[np.where((mask==1)|(mask==3))] = COLOR_FG
    return color

def color2mask(color):
    r,c = color.shape[:2]
    mask = np.zeros((r,c),np.uint8)
    mask[np.where((color==COLOR_BG).all(axis=2))] = 1
    mask[np.where((color==COLOR_FG).all(axis=2))] = 0
    return mask

def on_mouse(event,x,y,flags,param):
    param.mouse_cb(event,x,y,flags)

def nothing(x):
    pass

class InteractiveImageSegmentation:
    def __init__(self):
        self.winname = "InteractiveImageSegmentation"
        self.img = np.zeros((0))
        self.mask = np.zeros((0))
        self.left_mouse_down = False
        self.right_mouse_down = False
        self.radius = 3
        self.max_radius = 40
        self.use_prev_mask = False
        self.cur_mouse = (-1,-1)
        self.draw_color = 0
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, on_mouse, self)
        cv2.createTrackbar('brush size',self.winname,self.radius,self.max_radius,nothing)

    def mouse_cb(self,event,x,y,flags):
        self.cur_mouse = (x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.left_mouse_down = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.right_mouse_down = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.right_mouse_down = False
        if (self.left_mouse_down or self.right_mouse_down) and self.mask.size>0 and self.img.size>0:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                cv2.circle(self.img, (x,y), self.radius, (COLOR_BG if self.left_mouse_down else tuple([k/3 for k in COLOR_BG])), -1)
                cv2.circle(self.mask, (x,y), self.radius, (cv2.GC_BGD if self.left_mouse_down else cv2.GC_PR_BGD), -1)
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                cv2.circle(self.img, (x,y), self.radius, (COLOR_FG if self.left_mouse_down else tuple([k/3 for k in COLOR_FG])), -1)
                cv2.circle(self.mask, (x,y), self.radius, (cv2.GC_FGD if self.left_mouse_down else cv2.GC_PR_FGD), -1)
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags<0:
                diff_k = int(np.clip(self.radius*0.4,1,5))
                self.radius+=diff_k
            elif flags>0:
                diff_k = int(np.clip(self.radius*0.4,1,5))
                self.radius-=diff_k
            self.radius = np.clip(self.radius, 1, self.max_radius)
            cv2.setTrackbarPos('brush size', self.winname, self.radius)

    def __init_mask(self, mask):
        mask[:] = cv2.GC_PR_FGD
        mask[:10,:] = cv2.GC_PR_BGD

    def process(self, img):
        self.img = np.copy(img)
        if self.use_prev_mask==False or self.mask.shape[:2]!=self.img.shape[:2]:
            self.mask = np.zeros(img.shape[:2],'uint8')
            self.__init_mask(self.mask)
        self.bgdModel = np.zeros((1,65),np.float64)
        self.fgdModel = np.zeros((1,65),np.float64)
        cv2.grabCut(img, self.mask, None, self.bgdModel, self.fgdModel, 1, cv2.GC_INIT_WITH_MASK)

        while True:
            self.radius = cv2.getTrackbarPos('brush size',self.winname)
            color = mask2color(self.mask)
            
            alpha = 0.5 if self.draw_color==0 else (1 if self.draw_color==1 else 0)
            show_img = (self.img*alpha + color*(1-alpha)).astype('uint8')
            cv2.circle(show_img, self.cur_mouse, self.radius, (200,200,200), (2 if self.left_mouse_down else 1))
            cv2.imshow(self.winname,show_img)
            cv2.imshow('color',color)
            key = cv2.waitKey(100)
            if key == ord('c'):
                self.img = np.copy(img)
                self.__init_mask(self.mask)
            elif key == ord('q') or key == 27 or key==ord('s') or \
                key==ord('p') or key==ord('n') or key==ord('d') or key == 10:
                break
            elif key == ord('w'):
                self.draw_color = (self.draw_color+1)%3
            elif key == ord('a') or key == 32:
                cv2.putText(show_img, 'segmenting...', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
                cv2.imshow(self.winname,show_img)
                cv2.waitKey(1)
                # mask enum
                # GC_BGD    = 0,  //배경
                # GC_FGD    = 1,  //전경
                # GC_PR_BGD = 2,  //가능한 배경
                # GC_PR_FGD = 3   //가능한 잠재 전경
                hist, _ = np.histogram(self.mask,[0,1,2,3,4])
                if hist[0]+hist[2]!=0 and hist[1]+hist[3]!=0:
                    cv2.grabCut(img, self.mask, None, self.bgdModel, self.fgdModel, 1, cv2.GC_INIT_WITH_MASK)
                    self.img = np.copy(img)
                    '''
                    img: 입력 이미지
                    mask: 입력 이미지와 크기가 같은 1 채널 배열, 배경과 전경을 구분하는 값을 저장 
                    (cv2.GC_BGD: 확실한 배경(0), 
                    cv2.GC_FGD: 확실한 전경(1), 
                    cv2.GC_PR_BGD: 아마도 배경(2), 
                    cv2.GC_PR_FGD: 아마도 전경(3))
                    rect: 전경이 있을 것으로 추측되는 영역의 사각형 좌표, 
                    튜플 (x1, y1, x2, y2)
                    bgdModel, fgdModel: 함수 내에서 사용할 임시 배열 버퍼 
                    (재사용할 경우 수정하지 말 것)
                    iterCount: 반복 횟수
                    mode(optional): 
                    동작 방법 
                    (cv2.GC_INIT_WITH_RECT: rect에 지정한 좌표를 기준으로 그랩컷 수행, 
                    cv2.GC_INIT_WITH_MASK: mask에 지정한 값을 기준으로 그랩컷 수행, 
                    cv2.GC_EVAL: 재시도)
                    '''
        return key
    
def floodfillmask(tire_mask):
    #tire_mask = cv2.cvtColor(tire_mask, cv2.COLOR_BGR2GRAY)
    des = cv2.bitwise_not(tire_mask)
    contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(des,[cnt],0,255,-1)

    gray = cv2.bitwise_not(des)
    
    return gray
    '''
    ret, im_th = cv2.threshold(tire_mask, 0, 1, cv2.THRESH_BINARY+\
                                +cv2.THRESH_OTSU)
    im_floodfill = im_th.copy()

    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
    im_out = np.where(im_out==255,1, 0)
    '''
    return im_out

def get_json(segm_result, json_path):
    
    json_name = json_path.split('.')[0]
    
    if segm_result is not None:
        segs = [segm for segm_list in segm_result for segm in segm_list]
    else:
        segs = None
    
    labels = np.array([0, 1])
    classes = ['background', 'tire']
    
    # img_arr = cv2.imread(json_path)
    ret_json = dict()
    ret_json["version"] = "4.5.9"
    ret_json["flags"] = dict()
    ret_json["shapes"] = list()
    ret_json["imagePath"] = json_path.split(os.sep)[-1]
    ret_json["imageData"] = None
    ret_json["imageHeight"] = int(segm_result.shape[0])
    ret_json["imageWidth"] = int(segm_result.shape[1])
    
    
    for i in range(1):
        obj = dict()
        obj['label'] = 'tire'
        points_lst = []
        # if segs is not None:
            
        #     mask = segs[i]
        #     conto = np.zeros_like(mask, dtype = np.uint8)
            
        #     conto[mask == i] = 255
        
        # _, imthres = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
        
        contour2, _ = cv2.findContours(segm_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cord in contour2[0]:
            points_lst.append([float(cord[0][0]), float(cord[0][1])])
        obj['points'] = points_lst
        obj['group_id'] = None
        obj['shape_type'] = "polygon"
        obj['flags'] = dict()
        ret_json['shapes'].append(obj)
        
    with open(json_name + ".json", 'w') as f:
        json.dump(ret_json, f, indent=4)
        
    #return ret_json

if __name__ == '__main__':
    if(len(sys.argv)!=3):
        print('Usage: interactive_image_segmentation.py [img_dir] [save_dir]')
        exit()

    img_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print('%s not exists, create it.'%save_dir)

    print("================= Interactive Image Segmentation =================")
    print("CTRL+left mouse button: select certain background pixels ")
    print("SHIFT+left mouse button: select certain foreground pixels ")
    print("CTRL+right mouse button: select possible background pixels ")
    print("SHIFT+right mouse button: select possible foreground pixels ")
    print("'a'/SPACE: run sengementation again")
    print("'p': prev image       'n': next image")
    print("'s'/ENTER: save label        'q'/ESC: exit")

    iis = InteractiveImageSegmentation()
    iis.use_prev_mask = True
    fimglist = sorted([x for x in os.listdir(img_dir) if '.png' in x or '.jpg' in x])
    idx = 0
    while idx<len(fimglist) and os.path.exists(os.path.join(save_dir,fimglist[idx])):
        idx += 1

    while idx<len(fimglist):
        fimg = fimglist[idx]
        ori_fimg = cv2.imread(os.path.join(img_dir,fimg))
        fimg = np.copy(ori_fimg)
        h, w = ori_fimg.shape[:2]
        if h > w:
            fimg = cv2.rotate(ori_fimg, cv2.ROTATE_90_CLOCKWISE)
        print('process %s'%fimglist[idx])
        if os.path.exists(os.path.join(save_dir,fimglist[idx])):
            iis.mask = color2mask(fimg)
        key = iis.process(fimg)
        if key == ord('s') or key == 10:
            saveimg = os.path.join(save_dir, fimglist[idx])
            if h > w:
                fimg = cv2.rotate(ori_fimg, cv2.ROTATE_90_COUNTERCLOCKWISE)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            res = cv2.morphologyEx(mask2color(iis.mask),cv2.MORPH_OPEN,kernel)
            cv2.imwrite(saveimg, res)
            #cv2.imwrite(saveimg, mask2color(floodfillmask(color2mask(mask2color(iis.mask)))))
            print('save label %s.'%saveimg)
            idx += 1
        elif key == ord('p') and idx>0:
            idx -= 1
        elif key == ord('d') or key == 10:
            saveimg = os.path.join(save_dir, fimglist[idx])
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            res = cv2.morphologyEx(mask2color(iis.mask), cv2.MORPH_OPEN,kernel)
            if h > w:
                res = cv2.rotate(res, cv2.ROTATE_90_COUNTERCLOCKWISE)
            get_json(color2mask(res), json_path=os.path.join(img_dir, fimglist[idx]))
            # cv2.imwrite(saveimg, mask2color(floodfillmask(color2mask(mask2color(iis.mask)))))
            print('save json label %s.'%saveimg)
            idx += 1
        elif key == ord('n') or key == 32:
            idx += 1
        elif key == ord('q') or key == 27:
            break
        iis.mask[np.where(iis.mask==cv2.GC_BGD)]=cv2.GC_PR_BGD
        iis.mask[np.where(iis.mask==cv2.GC_FGD)]=cv2.GC_PR_FGD
