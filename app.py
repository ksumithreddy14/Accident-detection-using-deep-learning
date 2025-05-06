import os
import MySQLdb
import smtplib
import random
import string
from datetime import datetime
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash, send_file, Response, jsonify
from database import db_connect, vcact2, upload, ins_loginact, inc_reg
import base64
import io
import json
import re
import tensorflow as tf
from collections import namedtuple
from io import StringIO
from PIL import Image
import numpy as np
import winsound
from geopy.geocoders import Nominatim
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.io import gfile
from urllib.request import urlopen  # <-- Added this import


# Enable eager execution for TensorFlow 2.x
tf.compat.v1.enable_eager_execution()

global filename
global detectionGraph
global msg

app = Flask(__name__)
app.secret_key = os.urandom(24)


def get_location_name(latitude, longitude):
    geolocator = Nominatim(user_agent="location_lookup")
    location = geolocator.reverse((latitude, longitude), language='en')
    return location.address


@app.route("/")
def FUN_root():
    return render_template("index.html")


@app.route("/inceregact", methods=['GET', 'POST'])
def inceregact():
    if request.method == 'POST':
        status = inc_reg(request.form['username'], request.form['password'], request.form['email'], request.form['mobile'], request.form['address'])

        if status == 1:
            return render_template("login.html", m1="success")
        else:
            return render_template("reg.html", m1="failed")


@app.route("/inslogin", methods=['GET', 'POST'])
def inslogin():
    if request.method == 'POST':
        status = ins_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:
            session['username'] = request.form['username']
            return render_template("uhome.html", m1="success")
        else:
            return render_template("login.html", m1="Login Failed")


@app.route("/admin.html")
def admin():
    return render_template("admin.html")


@app.route("/index.html")
def index():
    return render_template("index.html")


@app.route("/login.html")
def login():
    return render_template("login.html")


@app.route("/reg.html")
def reg():
    return render_template("reg.html")


@app.route("/uhome.html")
def uhome():
    return render_template("uhome.html")


@app.route("/vc.html")
def vc():
    data1 = vcact2()
    print(data1)
    return render_template("vc.html", data1=data1)


def generate_random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))


def rectArea(xmax, ymax, xmin, ymin):
    x = np.abs(xmax - xmin)
    y = np.abs(ymax - ymin)
    return x * y


def extract_street_name(location_name):
    # Use regular expression to find the street name pattern
    match = re.search(r'\b(\d+-\d+,?\s)?([\w\s]+),', location_name)
    if match:
        return match.group(2)
    else:
        return None


def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    return dx * dy


# Removed @tf.function decorator to handle eager execution for TensorFlow
def calculateCollision(boxes, classes, scores, image_np):
    global msg
    boxes_np = boxes  # No need to call .numpy() since it's already a NumPy array
    for i, b in enumerate(boxes_np[0]):  # Now iterating over numpy array
        if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
            if scores[0][i] > 0.5:
                for j, c in enumerate(boxes_np[0]):  # Iterate over numpy array
                    if (i != j) and (classes[0][j] == 3 or classes[0][j] == 6 or classes[0][j] == 8) and scores[0][j] > 0.5:
                        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
                        ra = Rectangle(boxes_np[0][i][3], boxes_np[0][i][2], boxes_np[0][i][1], boxes_np[0][i][3])
                        rb = Rectangle(boxes_np[0][j][3], boxes_np[0][j][2], boxes_np[0][j][1], boxes_np[0][j][3])
                        ar = rectArea(boxes_np[0][i][3], boxes_np[0][i][1], boxes_np[0][i][2], boxes_np[0][i][3])
                        col_threshold = 0.6 * np.sqrt(ar)
                        area(ra, rb)
                        if (area(ra, rb) < col_threshold):
                            print('accident')
                            msg = 'ACCIDENT'
                            beep()
                            return True
                        else:
                            return False


@app.route("/showact", methods=['GET', 'POST'])
def showact():
    c1 = request.args.get('c')
    d1 = request.args.get('d')

    return render_template("show.html", c1=c1, d1=d1)


def beep():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)


import cv2


@app.route('/detect')
def detect():
    global msg
    msg = ''
    count = 0
    global detectionGraph
    detectionGraph = tf.Graph()

    try:
        with detectionGraph.as_default():
            od_graphDef = GraphDef()
            with tf.io.gfile.GFile('model/frozen_inference_graph.pb', 'rb') as file:
                serializedGraph = file.read()
                od_graphDef.ParseFromString(serializedGraph)
                tf.import_graph_def(od_graphDef, name='')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return "Error: Model could not be loaded!"

    cap = cv2.VideoCapture(0)  # Use the default camera (0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return "Error: Could not access webcam"

    with detectionGraph.as_default():
        with tf.compat.v1.Session(graph=detectionGraph) as sess:
            while True:
                ret, image_np = cap.read()
                if not ret:
                    print("Error: Could not read frame from webcam.")
                    break

                print("Processing frame...")  # Debugging line

                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detectionGraph.get_tensor_by_name('image_tensor:0')
                boxes = detectionGraph.get_tensor_by_name('detection_boxes:0')
                scores = detectionGraph.get_tensor_by_name('detection_scores:0')
                classes = detectionGraph.get_tensor_by_name('detection_classes:0')
                num_detections = detectionGraph.get_tensor_by_name('num_detections:0')

                boxes, scores, classes, num_detections = sess.run([boxes, scores, classes, num_detections],
                                                                   feed_dict={image_tensor: image_np_expanded})

                calculateCollision(boxes, classes, scores, image_np)

                cv2.putText(image_np, msg, (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('Accident Detection', image_np)

                if msg == "ACCIDENT" and count == 0:
                    random_string = generate_random_string(4)
                    cv2.imwrite("static/uploads/" + random_string + ".jpg", image_np)
                    image_path = "static/uploads/" + random_string + ".jpg"
                    location = json.loads(urlopen('http://ipinfo.io/json').read())
                    print(location)
                    latitude, longitude = map(float, location['loc'].split(','))
                    location_name = get_location_name(latitude, longitude)
                    street_name = extract_street_name(location_name)

                    if street_name:
                        print("Street Name:", street_name)
                    else:
                        print("Street Name not found.")

                    print("Location Name:", location_name)
                    upload(image_path, "ameerpet", street_name)
                    print("Accident detected! Image saved.")
                    count = 1

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

    return render_template("admin.html")


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
