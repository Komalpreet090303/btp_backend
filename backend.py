# backend/main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.signal import medfilt
import cv2
import ast
import uvicorn


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# class TransformerBlock(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
#         super(TransformerBlock, self).__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.ff_dim = ff_dim
#         self.rate = rate
#         self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = tf.keras.Sequential([
#             tf.keras.layers.Dense(ff_dim, activation="relu"),
#             tf.keras.layers.Dense(embed_dim),
#         ])
#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = tf.keras.layers.Dropout(rate)
#         self.dropout2 = tf.keras.layers.Dropout(rate)

#     def call(self, inputs, training=False, **kwargs):
#         attn_output = self.att(inputs, inputs)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output)

#     def get_config(self):
#         config = super(TransformerBlock, self).get_config()
#         config.update({
#             "embed_dim": self.embed_dim,
#             "num_heads": self.num_heads,
#             "ff_dim": self.ff_dim,
#             "rate": self.rate,
#         })
#         return config

def motion_analysis(video_path, std_width=720, std_height=480):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    delta_max = 64
    ret, first_frame = cap.read()
    if not ret:
        return None
    first_frame_resized = cv2.resize(first_frame, (std_width, std_height))
    first_gray = cv2.cvtColor(first_frame_resized, cv2.COLOR_BGR2GRAY)
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
    return pan, tilt, acc_pan, acc_tilt, nFrames, fps

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

def psiv(input_video, std_width=720, std_height=480):
    result = motion_analysis(input_video, std_width, std_height)
    if result is None:
        return None
    pan, tilt, acc_pan, acc_tilt, nFrames, fps = result
    acc_pan_f = medfilt(acc_pan, 15)
    acc_tilt_f = medfilt(acc_tilt, 15)
    diff_pan = np.abs(acc_pan_f - acc_pan)
    diff_tilt = np.abs(acc_tilt_f - acc_tilt)
    MPF = (1 / nFrames) * np.sum(diff_pan)
    MTF = (1 / nFrames) * np.sum(diff_tilt)
    return MPF, MTF, diff_pan, diff_tilt, nFrames, fps

# def process_video(video_path, max_seq_length, std_width=720, std_height=480):
#     result = psiv(video_path, std_width, std_height)
#     if result is None:
#         raise ValueError("Error processing video: " + video_path)
#     _, _, pan, tilt, nFrames, fps = result
#     pan_padded = pad_sequences([pan], maxlen=max_seq_length, padding='post', dtype='float32')
#     tilt_padded = pad_sequences([tilt], maxlen=max_seq_length, padding='post', dtype='float32')
#     sequence_input = np.stack((pan_padded, tilt_padded), axis=-1)
#     return sequence_input

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# with keras.utils.custom_object_scope({'TransformerBlock': TransformerBlock}):
#     loaded_model = keras.models.load_model("model1.keras")

# intermediate_model = keras.Model(
#     inputs=loaded_model.input,
#     outputs=loaded_model.output
# )

max_seq_length = 2263

@app.post("/api/predict")
async def predict(video: UploadFile = File(...), features: str = Form(...)):
    if isinstance(features, str):
        try:
            parsed_features = ast.literal_eval(features)
            if isinstance(parsed_features, list):
                features = parsed_features
            else:
                features = [features]  # Just a single string
        except Exception:
            features = [features]  # If parsing fails, treat as single string

    # Now 'features' is definitely a list
    requested_features = [f.strip().lower() for f in features]
    print(requested_features)
    temp_video_path = "temp_uploaded_video.mp4"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    results = {}

    try:
        # if "shakinessscore" in requested_features:
        #     input_data = process_video(temp_video_path, max_seq_length)
        #     prediction = intermediate_model.predict(input_data)
        #     results["shakinessScore"] = float(prediction[0][0])

        if any(f in requested_features for f in ["tiltscore", "blurrinessscore", "contrast", "brightness", "burnt_pixel"]):
            
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
                results["tiltScore"] = float(np.mean(tilt_scores))
            if "blurrinessscore" in requested_features:
                results["blurrinessScore"] = float(np.mean(blurr_scores))
            if "contrast" in requested_features:
                results["contrastScore"] = float(np.mean(contrast_scores))
            if "brightness" in requested_features:
                results["brightnessScore"] = float(np.mean(brightness_scores))
            if "burnt_pixels" in requested_features:
                results["burntPixelScore"] = float(np.mean(burnt_scores))

    except Exception as e:
        return JSONResponse(content={"error": f"Processing failed: {str(e)}"}, status_code=500)
    finally:
        os.remove(temp_video_path)

    return JSONResponse(content=results)


