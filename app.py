from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
import os
import time
from threading import Thread
from imutils.video import VideoStream
import RPi.GPIO as GPIO
from wifi import Cell
import requests
import socket
import subprocess
import shutil
import netifaces

# Khai báo số chân GPIO mà bạn muốn sử dụng
gpio_pin = 17
duckdns_domain = "doan2.duckdns.org"
api_token = "971323af-a564-4ce2-bb5a-4a1ec5318403"


GPIO.setmode(GPIO.BCM)
GPIO.setup(gpio_pin, GPIO.OUT)

def e_dist(pA, pB):
    return np.linalg.norm(pA - pB)

def eye_ratio(eye):
    d_V1 = e_dist(eye[1], eye[5])
    d_V2 = e_dist(eye[2], eye[4])
    d_H = e_dist(eye[0], eye[3])
    eye_ratio_val = (d_V1 + d_V2) / (2.0 * d_H)
    return eye_ratio_val

class Webcam():
    def __init__(self):
        self.vid = cv2.VideoCapture(0)
        #self.face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        #self.landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        face_cascade_file = os.path.join(current_dir, "haarcascade_frontalface_default.xml")
        self.face_detect = cv2.CascadeClassifier(face_cascade_file)
        # Xác định đường dẫn tuyệt đối đến file "shape_predictor_68_face_landmarks.dat"
        landmarks_file = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
        self.landmark_detect = dlib.shape_predictor(landmarks_file)
        self.left_eye_start, self.left_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.right_eye_start, self.right_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.eye_ratio_threshold = 0.25
        self.max_sleep_frames = 16
        self.sleep_frames = 0
        time.sleep(1.0)

    def detect_sleeping(self, frame):
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmark = self.landmark_detect(gray, rect)
            landmark = face_utils.shape_to_np(landmark)
            leftEye = landmark[self.left_eye_start:self.left_eye_end]
            rightEye = landmark[self.right_eye_start:self.right_eye_end]
            left_eye_ratio = eye_ratio(leftEye)
            right_eye_ratio = eye_ratio(rightEye)
            eye_avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

            left_eye_bound = cv2.convexHull(leftEye)
            right_eye_bound = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [left_eye_bound], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_bound], -1, (0, 255, 0), 1)

            if eye_avg_ratio < self.eye_ratio_threshold:
                self.sleep_frames += 1
                if self.sleep_frames >= self.max_sleep_frames:
                    GPIO.output(gpio_pin, GPIO.HIGH)
                    cv2.putText(frame, "NGU GAT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.sleep_frames = 0
                GPIO.output(gpio_pin, GPIO.LOW)
                cv2.putText(frame, "EAR: {:.3f}".format(eye_avg_ratio), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return frame

    def get_frame(self):
        if not self.vid.isOpened():
            return

        while True:
            _, img = self.vid.read()
            img = imutils.resize(img, width=450)
            frame_with_alert = self.detect_sleeping(img)

            # Encoding the frame in JPG format and yielding the result
            yield cv2.imencode('.jpg', frame_with_alert)[1].tobytes()

def is_valid_wifi_config(ssid, password):
    # Kiểm tra xem có mạng WiFi trong môi trường hay không
    wifi_list = Cell.all('wlan0')
    available_ssids = [cell.ssid for cell in wifi_list]
    if ssid not in available_ssids:
        return False

    # Kiểm tra mật khẩu đúng cho mạng WiFi
    # (Ở đây tôi giả sử mật khẩu là ít nhất 8 ký tự)
    if len(password) < 8:
        return False

    return True


def change_wifi_config(ssid, password):
    if not is_valid_wifi_config(ssid, password):
        return False, "Invalid Wi-Fi configuration."

    # Xác nhận thông tin Wi-Fi cũ
    current_ssid, _ = get_current_wifi_config()
    print(f"Current Wi-Fi: {current_ssid}")

    # Định nghĩa nội dung cấu hình mới cho WiFi
    wifi_config = f"""\
country=VN
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={{
    ssid="{ssid}"
    psk="{password}"
}}
"""

    try:
        # Đọc nội dung của file wpa_supplicant.conf vào biến backup_content
        with open('/etc/wpa_supplicant/wpa_supplicant.conf', 'rt') as file:
            backup_content = file.read()

        # Cấp quyền truy cập cho tệp wpa_supplicant.conf (quyền 666)
        subprocess.run(['sudo', 'chmod', '666', '/etc/wpa_supplicant/wpa_supplicant.conf'])

        # Lưu nội dung mới vào tệp cấu hình WiFi
        with open('/etc/wpa_supplicant/wpa_supplicant.conf', 'wt') as file:
            file.write(wifi_config)

        # Xác nhận kết nối Wi-Fi mới
        success = test_wifi_connection(ssid)
        if success:
            current_ssid, _ = get_current_wifi_config()
            print(f"Wi-Fi configuration changed successfully. New Wi-Fi: {ssid}")
            return True, ""
        else:
            error_message = "Failed to connect to the new Wi-Fi. Restoring the original Wi-Fi configuration..."
            # Phục hồi nội dung từ biến backup_content
            with open('/etc/wpa_supplicant/wpa_supplicant.conf', 'wt') as file:
                file.write(backup_content)
            # Áp dụng lại cấu hình gốc
            subprocess.run(['sudo', 'wpa_cli', '-i', 'wlan0', 'reconfigure'])
            return False, error_message

    except PermissionError as e:
        error_message = f"Error occurred: {e}"
        return False, error_message

    # Áp dụng cấu hình mới ngay lập tức
    subprocess.run(['sudo', 'wpa_cli', '-i', 'wlan0', 'reconfigure'])
    update_ip()
    return True, ""





def get_current_wifi_config():
    try:
        result = subprocess.run(['sudo', 'iwgetid', '-r'], capture_output=True, text=True)
        current_ssid = result.stdout.strip()
        return current_ssid, True
    except subprocess.CalledProcessError:
        return None, False

def test_wifi_connection(ssid):
    try:
        # Thử kết nối với Wi-Fi mới
        subprocess.run(['sudo', 'wpa_cli', '-i', 'wlan0', 'reconfigure'])
        # Chờ 10 giây để xem kết nối có thành công hay không
        time.sleep(30)
        # Kiểm tra xem Raspberry Pi đã kết nối với Wi-Fi mới hay chưa
        current_ssid, success = get_current_wifi_config()
        update_ip()
        if success and current_ssid == ssid:
            return True
        else:
            return False
    except subprocess.CalledProcessError:
        return False


app = Flask(__name__)

webcam = Webcam()


def get_internal_ip():
    try:
        # Lấy danh sách giao diện mạng
        interfaces = netifaces.interfaces()

        # Lặp qua các giao diện mạng để tìm địa chỉ IP nội bộ
        for iface in interfaces:
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:
                for addr_info in addrs[netifaces.AF_INET]:
                    ip_address = addr_info['addr']
                    if not ip_address.startswith('127.'):
                        return ip_address

    except:
        pass

    return None



# Cập nhật địa chỉ IP của DuckDNS
def update_ip():
    internal_ip = get_internal_ip()
    if internal_ip:
        try:
            response = requests.get(f"https://www.duckdns.org/update?domains={duckdns_domain}&token={api_token}&ip={internal_ip}")
            if response.status_code == 200:
                print(f"Update IP success. DuckDNS IP: {internal_ip}")
            else:
                print(f"Failed to update IP. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error occurred: {e}")
    else:
        print("Unable to get internal IP.")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        new_ssid = request.form["ssid"]
        new_password = request.form["password"]
        result, message = change_wifi_config(new_ssid, new_password)
        if result:
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "message": message})
    return render_template("index.html")


def read_from_webcam():
    while True:
        # Đọc ảnh từ class Webcam
        image = next(webcam.get_frame())

        # Nhận diện qua model YOLO


        # Trả về cho web bằng lệnh yeild
        yield b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n--frame\r\n'


@app.route("/image_feed")
def image_feed():
    return Response( read_from_webcam(), mimetype="multipart/x-mixed-replace; boundary=frame" )


if __name__=="__main__":
#   app.run(host='0.0.0.0', debug=False)
    update_ip()
    app.run(host='0.0.0.0', port=5000)
