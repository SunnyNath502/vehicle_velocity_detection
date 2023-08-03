from flask import Flask, render_template, Response, request,redirect
import cv2
import numpy as np
import time
import math
import os

app = Flask(__name__)

speed_factor=1  # change this value to get a optimum factor for exact speed
line_height = 450
car_counter = 0
total_speed = 0
num_cars = 0
prev_position = None
prev_frame_time = None

# external video
def estimate_vehicle_speed(video_path):
    line_height = 450
    car_counter = 0
    total_speed = 0
    num_cars = 0
    def get_centroid(x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx, cy

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    prev_position = None
    prev_frame_time = None

    while ret:
        d = cv2.absdiff(frame1, frame2)
        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3)))
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# _ --> hierarchy
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            contour_valid = (w >= 40) and (h >= 40)

            if not contour_valid:
                continue

            centroid = get_centroid(x, y, w, h)
            cy = centroid[1]

            if (cy < line_height + 10) and (cy > line_height - 10):
                car_counter += 0.27

                current_position = centroid

                if prev_position is not None and prev_frame_time is not None:
                    distance = math.sqrt((current_position[0] - prev_position[0]) ** 2 +
                                         (current_position[1] - prev_position[1]) ** 2)
                    time_elapsed = time.time() - prev_frame_time

                    if time_elapsed != 0:
                        speed = distance / time_elapsed
                        total_speed += speed
                        num_cars += 1

                prev_position = current_position
                prev_frame_time = time.time()

        cv2.putText(frame1, "Vehicle Count: " + str(math.floor(car_counter)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if num_cars != 0:
            average_speed = total_speed / num_cars
            speed_text = "Average Speed: {:.2f} kmph".format(average_speed )
            cv2.putText(frame1, speed_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Vehicle Detection", cv2.resize(frame1, (700, 480)))

        if cv2.waitKey(40) == ord('q'):
            break

        frame1 = frame2
        ret, frame2 = cap.read()

    cv2.destroyAllWindows()
    cap.release()

    return speed_text


@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return "No video file found.", 400

    video = request.files['video']
    if video.filename == '':
        return "No selected video file.", 400

    video_path = "uploads/" + video.filename
    video.save(video_path)

    speed_text = estimate_vehicle_speed(video_path)

    return redirect("/")

# camera
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

def process_frame(frame1, frame2):
    global car_counter, total_speed, num_cars, prev_position, prev_frame_time

    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        contour_valid = (w >= 40) and (h >= 40)

        if not contour_valid:
            continue

        centroid = get_centroid(x, y, w, h)
        cy = centroid[1]

        if (cy < line_height + 10) and (cy > line_height - 10):
            car_counter += 0.27

            current_position = centroid

            if prev_position is not None and prev_frame_time is not None:
                distance = math.sqrt((current_position[0] - prev_position[0]) ** 2 +
                                     (current_position[1] - prev_position[1]) ** 2)
                time_elapsed = time.time() - prev_frame_time

                if time_elapsed != 0:
                    speed = distance / time_elapsed
                    total_speed += speed
                    num_cars += 1

            prev_position = current_position
            prev_frame_time = time.time()

    cv2.putText(frame1, "Vehicle Count: " + str(math.floor(car_counter)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if num_cars != 0:
        average_speed = total_speed / num_cars
        speed_text = "Average Speed: {:.2f} kmph".format(average_speed*speed_factor)
        cv2.putText(frame1, speed_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame1

def generate_frames():
    global car_counter, total_speed, num_cars, prev_position, prev_frame_time

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening camera")
        exit()

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while ret:
        frame = process_frame(frame1, frame2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # \r\n\r\n -- empty line
        frame1 = frame2
        ret, frame2 = cap.read()

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap, prev_gray, prev_pts, total_distance, speed_values, window_size, start_time
    cap = cv2.VideoCapture(0)  # Use the appropriate video source, e.g., 0 for webcam
    prev_gray = None
    prev_pts = None
    total_distance = 0
    speed_values = []
    window_size = 30
    start_time = time.time()
    return 'Camera started'

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap
    if cap is not None:
        cap.release()
    cap = None
    return 'Camera stopped'

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
