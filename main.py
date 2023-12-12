import numpy as np
import cv2, sys
import time
from src.tracking.byte_track import BYTETracker
from src.detector.Yolo_detect import Detector
from src.MMC.color_recognition import Color_Recognitiion
from PIL import Image
from src.OCR.text_recognizer import TextRecognizer
import time
import math
import warnings
import argparse
from src.detector.predictor_v8 import Detect_v8

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True, help="video path to test")
opt = parser.parse_args()

target_num_points = 14
all_points = [None] * target_num_points

state = 0
frame = None
crop = False

CLASS_NAME3 = ["car", "bike", "bus", "truck", "coach"]  # VDS


def get_area_detect(img, points):
    # points = points.reshape((-1, 1, 2))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dts = cv2.bitwise_and(img, img, mask=mask)
    return dts


def on_mouse(event, x, y, flags, userdata):
    global state, all_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if state < target_num_points:
            all_points[state] = [x, y]
            state += 1
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        all_points = [None] * target_num_points
        state = 0


def is_non_decreasing(lst):
    """
    Hàm này kiểm tra xem tọa độ của xe có phải là đang tăng không.
    Nếu có thì có nghĩa nó đang đi ngược chiều
    """
    for i in range(len(lst) - 1):
        if lst[i] >= lst[i + 1]:
            return False
    return True


def check_traffic_light(img, box):
    global traffic_light
    """
    Hàm này kiểm tra đèn màu gì
    """
    # box = [602.3,   0, 620.52,  50.92] # xmin, ymin, xmax, ymax
    x0 = int(box[0])
    y0 = int(box[1])
    x1 = int(box[2])
    y1 = int(box[3])

    light = img[y0:y1, x0:x1]
    light_hsv = cv2.cvtColor(light, cv2.COLOR_BGR2HSV)

    mask_red_1 = cv2.inRange(light_hsv, (0, 100, 100), (10, 255, 255))
    mask_red_2 = cv2.inRange(light_hsv, (160, 100, 100), (180, 255, 255))
    mask_red = cv2.add(mask_red_1, mask_red_2)
    mask_green = cv2.inRange(light_hsv, (40, 50, 50), (90, 255, 255))
    mask_yellow = cv2.inRange(light_hsv, (15, 150, 150), (35, 255, 255))

    num_red = np.count_nonzero(mask_red == 255)
    num_green = np.count_nonzero(mask_green == 255)
    num_yellow = np.count_nonzero(mask_yellow == 255)

    if num_red > num_green and num_red > num_yellow:
        traffic_light = "red"
    elif num_green > num_red and num_green > num_yellow:
        traffic_light = "green"
    elif num_yellow > num_red and num_yellow > num_green:
        traffic_light = "yellow"
    else:
        traffic_light = "unknown"

    cv2.imwrite("b3_test_result/traffic_light_cut.png", light)

    return traffic_light


def check_id_in_dict(diction, id):
    """
    Hàm này nhận vào dict (bao gồm: keys là các class_id, values là list chứa các id) và một id mà mình cần xét.
    Kiểm tra xem, id mình đang xét hiện đang thuộc class_id nào.
    Nếu chưa thuộc class_id nào thì trả về -1
    Nếu đã tồn tại thì trả về class_id chứa id mình đang xét
    """
    for key, value in diction.items():
        if id in value:
            return key
    return -1


def update_new_id(diction, ele, min_negative_id):
    """
    Hàm này nhận vào dict (bao gồm: keys là các class_id, values là list chứa các id),
    và ele (bao gồm: id cần xét, class_id của id cần xét),
    và min_negative_id: được sử dụng để cập nhật lại id, nếu id đã bị sử dụng cho class khác
    """
    id_old = ele[0]
    class_id_new = ele[1]
    class_id_old = check_id_in_dict(diction, id_old)
    if class_id_old == -1:  # id chưa từng được dùng trước đó
        return id_old

    else:
        if (
            class_id_old == class_id_new
        ):  # id đã được dùng trước đó, nhưng class_id vẫn thế, nên không cần cập nhật
            return id_old
        else:  # id đã được dùng trước đó, class_id có sự thay đổi, chứng tỏ bị nhảy class, nên cần cập nhật
            min_negative_id -= 1
            return min_negative_id


class Args:
    def __init__(self) -> None:
        self.track_thresh = 0.4
        self.track_buffer = 30
        self.match_thresh = 0.7
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = True
        self.tsize = None
        self.exp_file = None


class System_KH:
    def __init__(self) -> None:
        super(System_KH, self).__init__()
        args = Args()
        self.tracker = BYTETracker(args, frame_rate=22)
        self.vehicle_detector = Detect_v8(model_path="weights/yolov8s_best.onnx")
        self.plate_detector = Detector("weights/detect_plate.onnx")
        self.hetmet_trafficlight = Detector(model="weights/helmet.onnx")
        self.color_recog = Color_Recognitiion("weights/color_recognition.pth")
        self.lp_recog = TextRecognizer(model_dir="weights/en_PP-OCRv3_rec_infer/")
        self.test_size = (640, 640)

    def __four_points_transform(self, image):
        H, W, _ = image.shape
        rect = np.asarray([[0, 0], [W, 0], [W, H], [0, H]], dtype="float32")
        width = 195
        height = 50
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(image, M, (width, height))

        return warp

    def __norm_plate(self, iplate, classID=None):
        if iplate is None:
            return []
        h, w, _ = iplate.shape
        if w / h > 2.5:
            iplate_transform = self.__four_points_transform(iplate)

            return iplate_transform
        else:
            iplate_1 = iplate[0 : int(h / 2), 0:w]
            iplate_2 = iplate[int(h / 2) : h, 0:w]
            _iplate_1 = cv2.resize(iplate_1, (165, 50))
            _iplate_2 = cv2.resize(iplate_2, (165, 50))
            iplate_concat = cv2.hconcat([_iplate_1, _iplate_2])
            iplate_blur = cv2.GaussianBlur(iplate_concat, (7, 7), 0)

            return iplate_blur

    def __rule(self, text):
        text_new = ""
        text = "".join(char for char in text if char.isalnum() or char.isalpha())
        text = text.upper()
        arr = list(text)
        if len(arr) == 9 and arr[3].isalpha() and not arr[2].isalpha():
            arr = arr[1:]
        if len(arr) > 6:
            if arr[2] in "8":
                arr = arr[:2] + ["B"] + arr[2 + 1 :]
            if arr[0] in "B":
                arr = ["8"] + arr[1:]
            if arr[1] in "B":
                arr = arr[:1] + ["8"] + arr[1 + 1 :]
            # 7 Z
            if arr[0] in "Z":
                arr = ["7"] + arr[1:]
            if arr[1] in "Z":
                arr = arr[:1] + ["7"] + arr[1 + 1 :]
            # 0 D
            if arr[2] in "0":
                arr = arr[:2] + ["D"] + arr[2 + 1 :]
            # A 4
            if arr[2] in "4":
                arr = arr[:2] + ["A"] + arr[2 + 1 :]
            # 7 V
            if arr[0] in "V":
                arr = ["7"] + arr[1:]
            if arr[1] in "V":
                arr = arr[:1] + ["7"] + arr[1 + 1 :]
            # O D
            if arr[3] in "O":
                arr = arr[:3] + ["D"] + arr[3 + 1 :]
            text_new = "".join(str(elem) for elem in arr)

        return text_new

    def color_recognition(self, img, box):
        """
        Nhận diện màu xe ,cắt các vùng ảnh chứa xe r cho vào nhận diện
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img)
        car_img_crop = None
        # img=Image.fromarray(img)
        box = list(map(int, box))
        # print(box)

        car_img_crop = img[box[1] : box[3], box[0] : box[2]]

        h, w = car_img_crop.shape[:2]

        if h > 0 and w > 0:
            cv2.cvtColor(car_img_crop, cv2.COLOR_BGR2RGB)
            PIL_img = Image.fromarray(car_img_crop)
            color = self.color_recog.infer(PIL_img)
            return color
        return "no recog"

    def plate_recognition(self, img, box):
        """
        Phát hiện phương tiện phải chạy liên tục
        sau đó đến điểm cần detect biển thì chạy detect biển
        """
        box = list(map(int, box))
        vehicle_img = img[box[1] : box[3], box[0] : box[2]]
        output, box_lp = self.plate_detector.detect(vehicle_img)
        txt_list = []
        # print(box_lp)
        if box_lp.size > 0:
            b = list(map(int, box_lp[0]))

            if box[0] > 5:
                iplate = vehicle_img[
                    abs(b[1]) : abs(b[3] + 3), abs(b[0] - 5) : abs(b[2] + 5)
                ]
            iplate = vehicle_img[abs(b[1]) : abs(b[3]), abs(b[0]) : abs(b[2])]
            res = []
            text_pred = ""
            norm_plate = self.__norm_plate(iplate)
            if norm_plate is None:
                print("error")
            try:
                text = self.lp_recog.recognizer(norm_plate)
                res.append(text[0][0])
            except:
                text = ""
                res.append(text)
            _text = list(map(lambda text: str(text), res))
            _text_pred = "".join(_text)
            text_pred = "".join(
                char for char in _text_pred if char.isalnum() or char.isalpha()
            ).upper()
            txt = self.__rule(text_pred)
            txt_list.append(txt)
            return txt_list
        return None

    def hemet_detector(self, img, bbox_person):
        """
        Phát hiện người
        sau đó crop vùng người trong ảnh r cho vào mô hình
        """
        cls_name = ["helmet", "without_helmet"]
        cls = None
        output, bbox = self.hetmet_trafficlight.detect(img)
        cls = output[:, 5]
        if len(cls.size()) > 0:
            class_name = [cls_name[int(x)] for x in cls]
            return class_name

    def trafic_detection(self, img, cls_traffic_light):
        """
        Phát hiện trên từng frame hình
        đưa ra nhãn
        """
        pass

    def tracking(self, img: np.ndarray):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        # create filter class
        ratio = 1
        outputs, bbox = self.vehicle_detector.detect(img)
        # print("img shape :",img.shape)
        scores = outputs[:, 4]
        cls = outputs[:, 5]
        img_info["ratio"] = ratio
        filter_class = [0, 1, 2, 3, 4, 5, 6]
        list_count = [0, 0, 0, 0]
        if outputs is not None:
            online_targets = self.tracker.update(
                outputs, [height, width], self.test_size, filter_class
            )
            online_tlwhs = []
            online_ids = []
            for t, cl in zip(online_targets, cls):
                tlwh = t.tlwh
                tid = t.track_id
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        return bbox, cls, online_ids, scores


if __name__ == "__main__":
    video = opt.video
    cap = cv2.VideoCapture(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img = np.zeros((1280, 720, 3), np.uint8)
    track = System_KH()
    count = 0
    check = 0
    MIN = 40
    list_count_left = [0, 0, 0, 0]
    list_count_right = [0, 0, 0, 0]
    start_point = {}
    check_point = {}
    M_left = {
        "bus": [],
        "car": [],
        "lane": [],
        "person": [],
        "trailer": [],
        "truck": [],
        "bike": [],
    }
    M_right = {
        "bus": [],
        "car": [],
        "lane": [],
        "person": [],
        "trailer": [],
        "truck": [],
        "bike": [],
    }
    Max_contours = 0
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", on_mouse)
    result = cv2.VideoWriter(
        "file.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 720)
    )

    # 1. Dem xe 1
    considered_id = []
    vehicles_dict = dict()
    for idx in range(len(CLASS_NAME3)):
        vehicles_dict[idx] = []

    # 2. Check nguoc chieu 1
    multi_coor_y = dict()
    opposite_direction = set()

    # 3. Check running a red light 1
    running_red_set = set()

    # 6. Xử lý nhảy class 1
    # min_negative_id = 0

    # 4. Wrong lane: car lane
    wrong_car_lane_set = set()

    # 5. prohibiting right turn
    id_belong_region = dict()
    prohibiting_turn_set = set()

    while True:
        update_point = {}
        _, frame = cap.read()
        c3_new = []
        count += 1
        if count > length:
            break
        frame = cv2.resize(frame, (1280, 720))
        frame_shape = frame.shape
        img = frame.copy()
        start_time = time.time()

        # for single_point in all_points:
        #     img = cv2.circle(img, single_point, radius=3, color=(0, 0, 255), thickness=-1)

        # Cropping image
        if all(p is not None for p in all_points):
            pts = np.array(all_points[:4], np.int32)
            # crop frame
            img_croped = img.copy()
            cv2.polylines(img, [pts], True, (0, 0, 142), 3)
            img_copy = img_croped.copy()
            # Tracking
            bbox, cls, id_list, scores = track.tracking(img_croped)

            # 3.Check running a red light 2
            traffic_light = check_traffic_light(
                img,
                [
                    all_points[4][0],
                    all_points[4][1],
                    all_points[5][0],
                    all_points[5][1],
                ],
            )
            cv2.rectangle(
                img,
                np.array(all_points[4], dtype=np.int32),
                np.array(all_points[5], dtype=np.int32),
                (0, 0, 142),
                3,
            )

            area_running_red = np.array(
                [all_points[0], all_points[1], all_points[6], all_points[7]],
                dtype=np.int32,
            )
            cv2.polylines(img, [area_running_red], True, (0, 221, 186), 3)

            arr_point = all_points[:4]
            arr_point.sort(key=lambda x: x[1])
            arr_point2 = all_points[:4]
            arr_point2.sort()

            point_1 = [
                int((arr_point[0][0] + arr_point[1][0]) / 2),
                int((arr_point[0][1] + arr_point[1][1]) / 2),
            ]
            point_2 = [
                int((arr_point[2][0] + arr_point[3][0]) / 2),
                int((arr_point[2][1] + arr_point[3][1]) / 2),
            ]
            center = [point_1, point_2]
            """
            Viết phương trình đường thẳng  
            """
            # Phương trình đường thẳng AB với A=p1_2 , B=p2_2
            x1, y1 = arr_point2[0]
            x2, y2 = arr_point2[1]
            a_AB = ((y1 - y2)) / (x1 - x2)
            b_AB = (y2 * x1 - y1 * x2) / (x1 - x2)
            y_M = max(y1, y2) * 0.6
            x_M = (y_M - b_AB) / a_AB
            K = [int(x_M), int(y_M)]
            
            # Phương trình đường thẳng CD với C=p3_2 D=p4_2
            x3, y3 = arr_point2[2]
            x4, y4 = arr_point2[3]
            a_CD = ((y3 - y4)) / (x3 - x4)
            b_CD = (y4 * x3 - y3 * x4) / (x3 - x4)
            y_N = max(y3, y4) * 0.6
            x_N = (y_N - b_CD) / a_CD
            N = [int(x_N), int(y_N)]
            area_Goal2 = np.array([K, N, arr_point[2], arr_point[3]], np.int32)
            area_Goal = np.array(
                [
                    [arr_point[3][0], arr_point[3][1] - 150],
                    [arr_point[2][0], arr_point[2][1] - 150],
                    arr_point[2],
                    arr_point[3],
                ],
                dtype=np.int32,
            )
            cv2.polylines(img, [area_Goal], True, (0, 125, 125), 4)

            interested_large_area = np.array(all_points[:4], dtype=np.int32)
            car_lane_area = np.array(
                [all_points[8], all_points[1], all_points[2], all_points[9]],
                dtype=np.int32,
            )
            cv2.polylines(img, [car_lane_area], True, (0, 125, 125), 4)

            right_turn_area = np.array(all_points[10:14], dtype=np.int32)
            cv2.polylines(img, [right_turn_area], True, (0, 125, 125), 4)

            for box, cl, id, sc in zip(bbox, cls, id_list, scores):
                # pass
                box = list(map(int, box))

                mid_point = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                results_goal = cv2.pointPolygonTest(area_Goal, mid_point, False)
                running_red_goal = cv2.pointPolygonTest(
                    area_running_red, mid_point, False
                )
                interested_large_goal = cv2.pointPolygonTest(
                    interested_large_area, mid_point, False
                )
                car_lane_goal = cv2.pointPolygonTest(car_lane_area, mid_point, False)
                right_turn_goal = cv2.pointPolygonTest(
                    right_turn_area, mid_point, False
                )

                # 6. Xử lý nhảy class 2
                # id = update_new_id(vehicles_dict, (id, cl), min_negative_id)
                # min_negative_id = id

                # 4. Wrong lane
                if car_lane_goal >= 0 and cl != 0:
                    wrong_car_lane_set.add(id)

                # 0. Show all class/ id
                if interested_large_goal >= 0:
                    cv2.rectangle(
                        img,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (255, 0, 125),
                        2,
                    )
                    cv2.putText(
                        img,
                        str(int(cl)) + "/" + str(id),
                        (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        (0, 0, 255),
                        1,
                    )

                # 3. Consider running red
                if running_red_goal >= 0 and traffic_light == "red":
                    running_red_set.add(id)

                # 5. prohibiting turn
                id_belong_region[id] = id_belong_region.get(id, [-1] * 5)
                if interested_large_goal > 0:
                    temp_region = 1
                elif right_turn_goal > 0:
                    temp_region = 2
                else:
                    temp_region = -1
                id_belong_region[id].append(temp_region)
                id_belong_region[id].pop(0)
                if id_belong_region[id][0] == 1 and id_belong_region[id][-1] == 2:
                    prohibiting_turn_set.add(id)

                if results_goal >= 0:
                    # 1. Dem xe 2
                    if id not in considered_id:
                        considered_id.append(id)
                        vehicles_dict[int(cl)].append(id)

                    # 2. Check nguoc chieu 2
                    new_coor_y = (box[1] + box[3]) / 2
                    multi_coor_y[id] = multi_coor_y.get(id, [frame_shape[0]] * 5)
                    multi_coor_y[id].append(new_coor_y)
                    multi_coor_y[id].pop(0)
                    if is_non_decreasing(multi_coor_y[id]):
                        opposite_direction.add(id)
                        print(multi_coor_y[id])

                    if int(cl) == 1:
                        # cv2.putText(img,"Person : Found ",(10,130),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                        # cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,125),2)
                        # cv2.putText(img,str(int(cl)) + "/" + str(id),(int(box[0]),int(box[1])-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                        img_person = img_copy[box[1] : box[3], box[0] : box[2]]
                        result_helmet = track.hemet_detector(img_person, box)
                        # print("result_helmet",result_helmet)
                    else:
                        img_car = img[
                            box[0] : box[0] + box[2], box[1] : box[1] + box[3]
                        ]
                        cv2.rectangle(
                            img,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (255, 0, 125),
                            2,
                        )
                        # cv2.putText(img,str(int(cl)),(int(box[0]),int(box[1])-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                        result_car = track.color_recognition(img_copy, box)
                        cv2.putText(
                            img,
                            str(result_car),
                            (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1,
                            (0, 0, 255),
                            1,
                        )
                        # print("color",result_car)
                        """
                        xác định biển số xe
                        """
                        txt_list = track.plate_recognition(img_copy, box)
                        # print(txt_list)

                    # if  int(cl)!=2 :
                    #     cv2.putText(img,CLASS_NAME2[int(cl)],(int(box[2]),int(box[3])-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                    #     cv2.putText(img,str(id),(int(box[0]),int(box[1])-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                    #     mid_point=(int((box[0]+box[2])/2),int((box[1]+box[3])/2))
                    #     if str(id) not in start_point.keys():
                    #         start_point[str(id)]=[mid_point,time.perf_counter()]
                    #         # print("Add object to dict")
                    #     update_point[str(id)]=[mid_point,time.perf_counter()]

                    #     '''
                    #     Phân chia làn
                    #     '''
                    #     if mid_point[0]<max(point_1[0],point_2[0]):
                    #         if str(id) not in M_left[CLASS_NAME2[int(cl)]] :
                    #             M_left[CLASS_NAME2[int(cl)]].append(str(id))
                    #         print(" Đối tượng đang ở Làn 1 ")
                    #         list_count_left=[len(M_left["car"]),len(M_left["bus"]),len(M_left["trailer"]),len(M_left["truck"])]
                    #     else :
                    #         if str(id) not in M_right[CLASS_NAME2[int(cl)]] :
                    #             M_right[CLASS_NAME2[int(cl)]].append(str(id))
                    #         print("Đối tượng đang ở làn 2")
                    #         list_count_right=[len(M_right["car"]),len(M_right["bus"]),len(M_right["trailer"]),len(M_right["truck"])]

                    #     mid_point_t=start_point[str(id)][0]
                    #     t=start_point[str(id)][1]
                    #     mid_point_t_1=update_point[str(id)][0]

                    #     t_1=update_point[str(id)][1]-t

                    #     distance_pixel=math.hypot(abs(mid_point_t[0]-mid_point_t_1[0]),abs(mid_point_t[1]-mid_point_t_1[1]))
                    #     if distance_pixel<MIN and t_1>5:
                    #         cv2.putText(img,"Stopped",(int(box[0]),int(box[1]+10)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,12,244),1)
                    #     results_goal=cv2.pointPolygonTest(area_Goal,mid_point,False)
                    #     if results_goal>=0:
                    #         mid_point_prev=start_point[str(id)][0]
                    #         start_time=start_point[str(id)][1]
                    #         distance_pixel=math.hypot(abs(mid_point[0]-mid_point_prev[0]),abs(mid_point[1]-mid_point_prev[1]))
                    #         end_time=update_point[str(id)][1]-start_time
                    #         print("Time",end_time)
                    #         print("Đối tượng {} chuẩn bị thoát ra khỏi vùng kiểm soát".format(id))
                    #         distance_const=distance_pixel*0.3 # m
                    #         velocity=(distance_const/end_time)*3.6
                    #         print("velocity : ",velocity)
                    #         print("Khoảng cách đối tượng di chuyển trong vùng quan sát là :",distance_const)
                    #         cv2.putText(img,"{:.2f} km/h".format(velocity),(int(box[0]),int(box[1]+10)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,12,14),1)
            # result.write(img)
            #

            vehicles_stats = {}
            for idx in range(len(CLASS_NAME3)):
                vehicles_stats[CLASS_NAME3[idx]] = len(vehicles_dict[idx])
            print(f"1. Dem xe: {vehicles_stats}")
            print(f"2. Nguoc chieu: {opposite_direction}")
            print(f"3. Vuot den do - {traffic_light}: {running_red_set}")
            print(f"4. Sai lan: {wrong_car_lane_set}")
            print(f"5. Cam re: {prohibiting_turn_set}")
            print()
        cv2.imshow("frame", img)

        ##### Clear dict update_point_t1 sau khi đã sử dụng ###########

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break