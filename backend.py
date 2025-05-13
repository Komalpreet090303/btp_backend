from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import shutil
import numpy as np
import cv2
import ast
from scipy.signal import medfilt, find_peaks

app = Flask(__name__)
CORS(app)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def match_vectors(lastPr, Pr, delta_max):
    shift = 0
    last_diff = np.inf
    s2 = len(Pr)
    for delta in range(1, delta_max + 1):
        if delta < s2:
            curr_diff = np.mean(np.abs(lastPr[delta:] - Pr[:s2 - delta]))
            if curr_diff < last_diff:
                shift = delta - 1
                last_diff = curr_diff
            curr_diff = np.mean(np.abs(lastPr[:s2 - delta] - Pr[delta:]))
            if curr_diff < last_diff:
                shift = -(delta - 1)
                last_diff = curr_diff
    return shift

def fix_image_size(img, size=(224, 224)):
    return cv2.resize(img, size)

def calculate_tilt_score(frame):
    edges = cv2.Canny(frame, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None:
        return 0
    total_weighted_orientation = 0
    total_length = 0
    for line in lines:
        for rho, theta in line:
            angle = theta - np.pi / 2
            if np.abs(angle) <= np.pi / 4:
                length = np.abs(rho)
                total_weighted_orientation += angle * length
                total_length += length
    if total_length > 0:
        Stilt = np.abs(total_weighted_orientation / total_length) / (np.pi / 4)
    else:
        Stilt = 0
    return Stilt

def process_frame(frame):
    if frame.ndim == 3:
        frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_cvt = frame
    image2 = fix_image_size(frame_cvt)
    blur_map = cv2.Laplacian(frame_cvt, cv2.CV_64F)
    blurr_score = np.var(blur_map)
    contrast_score = np.std(image2)
    brightness_score = np.mean(image2)
    N_i = image2.size
    burnt_pixels = np.logical_or(image2 <= 10, image2 >= 245)
    N_b = np.sum(burnt_pixels)
    burnt_score = 1 - N_b / (0.25 * N_i) if (N_b / (0.25 * N_i)) < 1 else 0
    tilt_score = calculate_tilt_score(image2)
    return tilt_score, blurr_score, contrast_score, brightness_score, burnt_score



# Motion analysis
def motion_analysis(video_path, std_width=720, std_height=480):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    delta_max = 64

    ret, first_frame = cap.read()
    if not ret:
        return None

    first_frame_resized = cv2.resize(first_frame, (std_width, std_height))
    first_gray = cv2.cvtColor(first_frame_resized, cv2.COLOR_BGR2GRAY)

    s1, s2 = first_gray.shape
    lastPrX = np.mean(first_gray, axis=0)
    lastPrY = np.mean(first_gray, axis=1)

    acc_pan = np.zeros(nFrames)
    acc_tilt = np.zeros(nFrames)
    pan = np.zeros(nFrames)
    tilt = np.zeros(nFrames)

    for frameNum in range(1, nFrames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (std_width, std_height))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        PrX = np.mean(gray, axis=0)
        PrY = np.mean(gray, axis=1)

        pan[frameNum] = match_vectors(lastPrX, PrX, delta_max)
        tilt[frameNum] = match_vectors(lastPrY, PrY, delta_max)

        acc_pan[frameNum] = acc_pan[frameNum - 1] + pan[frameNum]
        acc_tilt[frameNum] = acc_tilt[frameNum - 1] + tilt[frameNum]

        lastPrX = PrX
        lastPrY = PrY

    cap.release()
    return pan, tilt, acc_pan, acc_tilt, nFrames

# Alam shakiness measure
def alam(acc_pan, acc_tilt, nFrames):
    acc_pan_f = medfilt(acc_pan, 15)
    acc_tilt_f = medfilt(acc_tilt, 15)

    shakiness_temp = (0.5 * np.abs(acc_pan - acc_pan_f) + 0.5 * np.abs(acc_tilt - acc_tilt_f)) / (300 / 100)
    shakiness_temp[shakiness_temp > 1] = 1

    return np.sum(shakiness_temp > 0.1) / nFrames

# Main shakiness score calculator
def psiv(input_video):
    std_width = 720
    std_height = 480
    pan, tilt, acc_pan, acc_tilt, nFrames = motion_analysis(input_video, std_width, std_height)
    acc_pan_f = medfilt(acc_pan, 15)
    acc_tilt_f = medfilt(acc_tilt, 15)
    diff_pan = np.abs(acc_pan_f - acc_pan)
    diff_tilt = np.abs(acc_tilt_f - acc_tilt)

    MPF = (1 / nFrames) * np.sum(diff_pan)
    MTF = (1 / nFrames) * np.sum(diff_tilt)
    alam_vedi = alam(acc_pan, acc_tilt, nFrames)

    peaks, _ = find_peaks(np.sqrt(diff_pan**2 + diff_tilt**2), height=np.sqrt(8), width=0)
    PEAKS = (1 / nFrames) * len(peaks)

    linear_reg_shakiness = 0.046782 + 0.44424 * alam_vedi + 3.0626 * PEAKS

    if linear_reg_shakiness > 1:
        shakiness_score = 1
    elif linear_reg_shakiness < 0:
        shakiness_score = 0
    else:
        shakiness_score = linear_reg_shakiness

    return shakiness_score


def normalize(value, min_val, max_val):
    return max(0, min(1, (value - min_val) / (max_val - min_val)))



@app.route("/api/predict", methods=["POST"])
def predict():
    if "video" not in request.files or "features" not in request.form:
        return jsonify({"error": "Missing file or features"}), 400

    video = request.files["video"]
    features = request.form["features"]

    try:
        features = ast.literal_eval(features)
        if not isinstance(features, list):
            features = [features]
    except:
        features = [features]

    requested_features = [f.strip().lower() for f in features]

    temp_video_path = "temp_uploaded_video.mp4"
    video.save(temp_video_path)

    results = {}

    try:
        if "shakinessscore" in requested_features:
            score = psiv(temp_video_path)
            results["shakinessScore"] = round(float(score),3)
        if any(f in requested_features for f in ["tiltscore", "blurrinessscore", "contrast", "brightness", "burnt_pixel", "burnt_pixels"]):
            cap = cv2.VideoCapture(temp_video_path)
            tilt_scores, blurr_scores, contrast_scores, brightness_scores, burnt_scores = [], [], [], [], []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                tilt, blur, contrast, brightness, burnt = process_frame(frame)
                tilt_scores.append(tilt)
                blurr_scores.append(blur)
                contrast_scores.append(contrast)
                brightness_scores.append(brightness)
                burnt_scores.append(burnt)
            cap.release()

            if "tiltscore" in requested_features:
                results["tiltScore"] = round(float(np.mean(tilt_scores)),3)
            if "blurrinessscore" in requested_features:
                raw_blurriness = float(np.mean(blurr_scores))
                results["blurrinessScore"] = round(1 - normalize(raw_blurriness, 0, 2000),3)
            if "contrast" in requested_features:
                raw_contrast = float(np.mean(contrast_scores))
                results["contrastScore"] = round(normalize(raw_contrast, 20, 70),3)
            if "brightness" in requested_features:
                results["brightnessScore"] = round(float(np.mean(brightness_scores)),3)
            if "burnt_pixels" in requested_features or "burnt_pixel" in requested_features:
                results["burntPixelScore"] = round(float(np.mean(burnt_scores)),3)

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
