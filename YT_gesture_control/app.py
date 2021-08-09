import sys
from collections import Counter
from collections import deque

import mediapipe as mp
from PyQt5.Qt import Qt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow

from model import MouseClassifier
# models
from model import PointHistoryClassifier
from utils import CvFpsCalc
from utils.func import *


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Window size
        self.WIDTH = 60
        self.HEIGHT = 60
        self.resize(self.WIDTH, self.HEIGHT)

        # Widget
        self.centralwidget = QWidget(self)
        self.centralwidget.resize(self.WIDTH, self.HEIGHT)

        # Menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.right_menu)

        # Initial
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.3)

        self.radius = 30
        self.centralwidget.setStyleSheet(
            """
            background:rgb(255, 255, 255);
            border-radius:{0}px;
            """.format(self.radius)
        )

        self.runButton = QPushButton(self)
        self.runButton.setText("Run")  # text
        self.runButton.setGeometry(10, 10, 40, 20)
        self.work = WorkThread()
        self.runButton.clicked.connect(self.execute)
        self.check_worked = False

    def execute(self):
        # 启动线程
        if self.check_worked == False:
            self.check_worked = True
            self.work.start()
            # 线程自定义信号连接的槽函数
            self.work.trigger.connect(self.display)
            self.runButton.setText('Stop')
            self.centralwidget.setStyleSheet(
                """
                background:rgb(255, 200, 255);
                border-radius:{0}px;
                """.format(self.radius))

        else:
            self.check_worked = False
            self.work.stop()
            self.runButton.setText('Run')

    def display(self, int):
        if int == 0:
            self.centralwidget.setStyleSheet(
                """
                background:rgb(255, 0, 0);
                border-radius:{0}px;
                """.format(self.radius)
            )
        if int == 1:
            self.centralwidget.setStyleSheet(
                """
                background:rgb(0, 255, 0);
                border-radius:{0}px;
                """.format(self.radius)
            )
        if int == 2:
            self.centralwidget.setStyleSheet(
                """
                background:rgb(0, 0, 255);
                border-radius:{0}px;
                """.format(self.radius)
            )

    def right_menu(self, pos):
        menu = QMenu()

        # Add menu options
        exit_option = menu.addAction('Exit')

        # Menu option events
        exit_option.triggered.connect(lambda: sys.exit())

        # Position
        menu.exec_(self.mapToGlobal(pos))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.moveFlag = True
            self.movePosition = event.globalPos() - self.pos()
            self.setCursor(QCursor(Qt.OpenHandCursor))
            event.accept()

    def mouseMoveEvent(self, event):
        if Qt.LeftButton and self.moveFlag:
            self.move(event.globalPos() - self.movePosition)
            event.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.moveFlag = False
        self.setCursor(Qt.CrossCursor)


class WorkThread(QThread):
    trigger = pyqtSignal(int)

    def __int__(self, parent=None):
        # 初始化函数
        super(WorkThread, self).__init__(parent)
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def run(self):
        # 重写线程执行的run函数
        # 触发自定义信号
        self.stop_flag = False
        # Argument parsing #################################################################
        args = get_args()

        cap_device = args.device
        cap_width = args.width
        cap_height = args.height

        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence

        use_brect = True

        # Camera preparation ###############################################################
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        # Model load #############################################################
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        mouse_classifier = MouseClassifier(invalid_value=2, score_th=0.4)
        point_history_classifier = PointHistoryClassifier()

        # Read labels ###########################################################

        with open(
                'model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]
        with open(
                'model/mouse_classifier/mouse_classifier_label.csv', encoding='utf-8-sig') as f:
            mouse_classifier_labels = csv.reader(f)
            mouse_classifier_labels = [
                row[0] for row in mouse_classifier_labels
            ]
        # FPS Measurement ########################################################
        cvFpsCalc = CvFpsCalc(buffer_len=3)

        # Coordinate history #################################################################
        history_length = 16
        point_history = deque(maxlen=history_length)

        # Finger gesture history ################################################
        finger_gesture_history = deque(maxlen=history_length)
        mouse_id_history = deque(maxlen=30)
        m_id_history = deque(maxlen=6)

        # 靜態手勢最常出現參數初始化 =========

        # ========= 使用者自訂姿勢、指令區 =========

        # ========= 按鍵前置作業 =========
        mode = 0
        presstime = resttime = time.time()

        mode_change = False
        detect_mode = 0
        what_mode = 'Sleep'
        landmark_list = 0
        pyautogui.PAUSE = 0
        i = 0

        # ========= 滑鼠前置作業 =========
        wScr, hScr = pyautogui.size()
        frameR = 100
        smoothening = 10
        plocX, plocY = 0, 0
        clocX, clocY = 0, 0
        # 關閉 滑鼠移至角落啟動保護措施
        pyautogui.FAILSAFE = False

        # ========= 主程式運作 =========
        while self.stop_flag == False:
            mouse_id = -1
            fps = cvFpsCalc.get()

            # Process Key (ESC: end)
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)

            # Camera capture
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            #  ####################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    # print(landmark_list)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                    # # 靜態手勢資料預測

                    mouse_id = mouse_classifier(pre_processed_landmark_list)
                    # print(mouse_id)

                    # 手比one 觸發動態資料抓取
                    if mouse_id == 0:
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    # 動態手勢資料預測
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                    # print(finger_gesture_id) # 0 = stop, 1 = clockwise, 2 = counterclockwise, 3 = move,偵測出現的動態手勢

                    # 動態手勢最常出現id #########################################
                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()

                    # 滑鼠的deque
                    mouse_id_history.append(mouse_id)
                    most_common_ms_id = Counter(mouse_id_history).most_common()
                    # print(most_common_ms_id)

                    m_id_history.append(mouse_id)
                    m_id = Counter(m_id_history).most_common(2)
                    mouse_id = m_id[0][0] if m_id[0][1] >= 4 else 2

                    # print(f'm_id {m_id}\n, mouse_id {mouse_id}')
                    # ===== 偵測到手時，重製紀錄時間 ==============================
                    resttime = time.time()

                    ###############################################################

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        mouse_classifier_labels[mouse_id],
                        point_history_classifier_labels[most_common_fg_id[0][0]]
                    )

            else:
                point_history.append([0, 0])

            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, fps, mode, number)

            # 偵測是否有手勢 #########################################
            if mouse_id > -1:
                # change mode Gesture six changes to the different mode
                if most_common_ms_id[0][0] == 3 and most_common_ms_id[0][1] == 30:
                    if time.time() - presstime > 2:
                        detect_mode = (detect_mode + 1) % 3
                        mode_change = True
                        presstime = time.time()

                    # control keyboard
                elif detect_mode == 2:
                    # 靜態手勢控制
                    presstime = control_keyboard(mouse_id, 1, 'K', presstime)
                    presstime = control_keyboard(mouse_id, 8, 'C', presstime)
                    presstime = control_keyboard(mouse_id, 5, 'up', presstime)
                    presstime = control_keyboard(mouse_id, 6, 'down', presstime)
                    # presstime = control_keyboard(most_common_keypoint_id, 0, 'right', presstime)
                    # presstime = control_keyboard(most_common_keypoint_id, 7, 'left', presstime)

                    if mouse_id == 4:
                        # print(i, time.time() - presstime)
                        if i == 3 and time.time() - presstime > 0.3:
                            pyautogui.press('right')
                            i = 0
                            presstime = time.time()
                        elif i == 3 and time.time() - presstime > 0.25:
                            pyautogui.press('right')
                            presstime = time.time()
                        elif time.time() - presstime > 1:
                            pyautogui.press('right')
                            i += 1
                            presstime = time.time()

                    if mouse_id == 7:
                        # print(i, time.time() - presstime)
                        if i == 3 and time.time() - presstime > 0.3:
                            pyautogui.press('left')
                            i = 0
                            presstime = time.time()
                        elif i == 3 and time.time() - presstime > 0.25:
                            pyautogui.press('left')
                            presstime = time.time()
                        elif time.time() - presstime > 1:
                            pyautogui.press('left')
                            i += 1
                            presstime = time.time()

                    # 動態手勢控制
                    if most_common_fg_id[0][0] == 1 and most_common_fg_id[0][1] > 9:
                        if time.time() - presstime > 1.5:
                            pyautogui.hotkey('shift', '>')
                            print('speed up')
                            presstime = time.time()

                    elif most_common_fg_id[0][0] == 2 and most_common_fg_id[0][1] > 12:
                        if time.time() - presstime > 1.5:
                            pyautogui.hotkey('shift', '<')
                            print('speed down')
                            presstime = time.time()


                elif detect_mode == 1:
                    if mouse_id == 0:  # Point gesture
                        x1, y1 = landmark_list[8]
                        cv.rectangle(debug_image, (50, 30), (cap_width - 50, cap_height - 170),
                                     (255, 0, 255), 2)
                        # 座標轉換
                        # x軸: 鏡頭上50~(cap_width - 50)轉至螢幕寬0~wScr
                        # y軸: 鏡頭上30~(cap_height - 170)轉至螢幕長0~hScr
                        x3 = np.interp(x1, (50, (cap_width - 50)), (0, wScr))
                        y3 = np.interp(y1, (30, (cap_height - 170)), (0, hScr))

                        # 6. Smoothen Values
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening

                        # 7. Move Mouse
                        pyautogui.moveTo(clocX, clocY)
                        cv.circle(debug_image, (x1, y1), 15, (255, 0, 255), cv.FILLED)
                        plocX, plocY = clocX, clocY

                    elif mouse_id == 1:
                        # 10. Click mouse if distance short
                        if time.time() - presstime > 0.5:
                            pyautogui.click()
                            presstime = time.time()

                    if mouse_id == 5:
                        pyautogui.scroll(-20)

                    if mouse_id == 6:
                        pyautogui.scroll(20)

                    if mouse_id == 7:
                        if time.time() - presstime > 1.5:
                            pyautogui.click(clicks=2)
                            presstime = time.time()

                    if mouse_id == 8:
                        if time.time() - presstime > 2:
                            pyautogui.hotkey('alt', 'left')
                            presstime = time.time()

                # 比讚 從休息模式 換成 鍵盤模式
                elif detect_mode == 0:
                    if mouse_id == 5:
                        i += 1
                        if i == 1 or time.time() - presstime > 3:
                            presstime = time.time()
                        elif time.time() - presstime > 2:
                            detect_mode = 2
                            mode_change = True
                            i = 0

            # 距離上次監測到手的時間大於30秒、切回休息模式 =========================
            if time.time() - resttime > 30:
                if detect_mode != 0:
                    detect_mode = 0
                    mode_change = True

            # 檢查模式有沒有更動 ========================
            if mode_change:
                if detect_mode == 0:
                    what_mode = 'Sleep'
                elif detect_mode == 2:
                    what_mode = 'Keyboard'
                elif detect_mode == 1:
                    what_mode = 'Mouse'

                mode_change = False
                print('Mode has changed')
                print(f'Current mode => {what_mode}')

            cv.putText(debug_image, what_mode, (480, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            # Screen reflection ###################################JL##########################
            cv.imshow('Hand Gesture Recognition', debug_image)

            self.trigger.emit(detect_mode)

        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
