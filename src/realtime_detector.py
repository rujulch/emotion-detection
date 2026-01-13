"""
Real-time facial emotion detection using webcam.
Features: Temporal smoothing, confidence bars, FPS counter, graceful error handling.
"""

import os
import cv2
import numpy as np
from collections import deque
import time
import argparse
import tensorflow as tf

from config import (
    setup_gpu, CNN_MODEL_PATH, EFFICIENTNET_MODEL_PATH,
    CNN_IMG_SIZE, EFFICIENTNET_IMG_SIZE, HAAR_CASCADE_PATH,
    EMOTION_LABELS, NUM_CLASSES, SMOOTHING_WINDOW, MIN_CONFIDENCE,
    FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE,
    DISPLAY_WIDTH, DISPLAY_HEIGHT, CNN_CHANNELS
)


class EmotionDetector:
    """Real-time emotion detection with temporal smoothing."""
    
    def __init__(self, model_type='cnn', smoothing_window=SMOOTHING_WINDOW):
        """
        Initialize the emotion detector.
        
        Args:
            model_type: 'cnn' or 'efficientnet'
            smoothing_window: Number of frames to average predictions
        """
        self.model_type = model_type
        self.smoothing_window = smoothing_window
        
        # Setup GPU
        setup_gpu()
        
        # Load model
        self.model = self._load_model()
        
        # Load face detector (ONCE, not per frame!)
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if self.face_cascade.empty():
            # Try OpenCV's built-in path
            cv2_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cv2_cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")
        
        # Set image size based on model type
        if model_type == 'cnn':
            self.img_size = CNN_IMG_SIZE
            self.color_mode = 'gray'
        else:
            self.img_size = EFFICIENTNET_IMG_SIZE
            self.color_mode = 'rgb'
        
        # Prediction history for temporal smoothing
        self.prediction_history = deque(maxlen=smoothing_window)
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        
        # Emotion colors (BGR format)
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 0),    # Green
            'fear': (128, 0, 128),     # Purple
            'happy': (0, 255, 255),    # Yellow
            'neutral': (128, 128, 128), # Gray
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 165, 255)  # Orange
        }
        
        print(f"\n[OK] Emotion Detector initialized ({model_type.upper()} model)")
        print(f"  Image size: {self.img_size}x{self.img_size}")
        print(f"  Smoothing window: {smoothing_window} frames")
    
    def _load_model(self):
        """Load the appropriate model."""
        if self.model_type == 'cnn':
            model_path = CNN_MODEL_PATH
        else:
            model_path = EFFICIENTNET_MODEL_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                f"Please train the model first using: python train.py --model {self.model_type}"
            )
        
        print(f"Loading model from: {model_path}")
        try:
            # Try loading normally first
            model = tf.keras.models.load_model(model_path, compile=False)
        except (TypeError, ValueError) as e:
            # If that fails due to version mismatch, rebuild the model and load weights
            print(f"Model format mismatch, rebuilding architecture and loading weights...")
            from model import create_cnn_model, create_efficientnet_model
            
            if self.model_type == 'cnn':
                model = create_cnn_model(
                    input_shape=(CNN_IMG_SIZE, CNN_IMG_SIZE, CNN_CHANNELS),
                    num_classes=NUM_CLASSES
                )
            else:
                model = create_efficientnet_model(num_classes=NUM_CLASSES)
            
            # Load weights only
            model.load_weights(model_path)
        
        return model
    
    def preprocess_face(self, face_img):
        """
        Preprocess a face image for the model.
        
        Args:
            face_img: BGR face image from OpenCV
        
        Returns:
            Preprocessed image ready for model input
        """
        if self.color_mode == 'gray':
            # Convert to grayscale
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            # Resize
            face_img = cv2.resize(face_img, (self.img_size, self.img_size))
            # Normalize
            face_img = face_img.astype('float32') / 255.0
            # Add channel and batch dimensions
            face_img = np.expand_dims(face_img, axis=-1)
            face_img = np.expand_dims(face_img, axis=0)
        else:
            # RGB mode for EfficientNet
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (self.img_size, self.img_size))
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
    
    def predict(self, face_img):
        """
        Predict emotion from a face image with temporal smoothing.
        
        Args:
            face_img: BGR face image from OpenCV
        
        Returns:
            emotion: Predicted emotion label
            confidence: Prediction confidence
            all_probs: Smoothed probabilities for all emotions
        """
        # Preprocess
        processed = self.preprocess_face(face_img)
        
        # Predict
        probs = self.model.predict(processed, verbose=0)[0]
        
        # Add to history for smoothing
        self.prediction_history.append(probs)
        
        # Average predictions over history (temporal smoothing)
        smoothed_probs = np.mean(self.prediction_history, axis=0)
        
        # Get prediction
        emotion_idx = np.argmax(smoothed_probs)
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = smoothed_probs[emotion_idx]
        
        return emotion, confidence, smoothed_probs
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame.
        
        Args:
            frame: BGR image from OpenCV
        
        Returns:
            List of (x, y, w, h) face bounding boxes
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE
        )
        return faces
    
    def draw_results(self, frame, faces, emotions, confidences, all_probs_list):
        """
        Draw detection results on the frame.
        
        Args:
            frame: BGR image to draw on
            faces: List of face bounding boxes
            emotions: List of predicted emotions
            confidences: List of confidence scores
            all_probs_list: List of probability arrays for all emotions
        """
        for (x, y, w, h), emotion, confidence, probs in zip(faces, emotions, confidences, all_probs_list):
            # Get color for this emotion
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label with confidence
            if confidence >= MIN_CONFIDENCE:
                label = f"{emotion.upper()}: {confidence*100:.1f}%"
            else:
                label = "Uncertain"
                color = (128, 128, 128)
            
            # Background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x, y-text_size[1]-10), (x+text_size[0]+5, y), color, -1)
            cv2.putText(frame, label, (x+2, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw confidence bars on the side
            self._draw_confidence_bars(frame, probs, x + w + 10, y)
        
        # Draw FPS
        self._draw_fps(frame)
        
        return frame
    
    def _draw_confidence_bars(self, frame, probs, x, y, bar_width=100, bar_height=15):
        """Draw confidence bars for all emotions."""
        for i, (emotion_idx, emotion_name) in enumerate(EMOTION_LABELS.items()):
            prob = probs[emotion_idx]
            color = self.emotion_colors.get(emotion_name, (255, 255, 255))
            
            # Bar position
            bar_y = y + i * (bar_height + 5)
            
            # Check if bar is within frame
            if bar_y + bar_height > frame.shape[0] or x + bar_width > frame.shape[1]:
                continue
            
            # Background bar
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Probability bar
            prob_width = int(prob * bar_width)
            cv2.rectangle(frame, (x, bar_y), (x + prob_width, bar_y + bar_height), color, -1)
            
            # Border
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (200, 200, 200), 1)
            
            # Label
            label = f"{emotion_name[:3].upper()}"
            cv2.putText(frame, label, (x - 35, bar_y + bar_height - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _draw_fps(self, frame):
        """Draw FPS counter on frame."""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time
        self.fps_history.append(fps)
        
        avg_fps = np.mean(self.fps_history)
        fps_text = f"FPS: {avg_fps:.1f}"
        
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
    
    def run(self, camera_id=0):
        """
        Run real-time emotion detection.
        
        Args:
            camera_id: Camera device ID (default: 0)
        """
        print(f"\nStarting webcam (camera {camera_id})...")
        print("Press 'q' to quit, 'r' to reset smoothing\n")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            # Try alternative camera
            cap = cv2.VideoCapture(camera_id + 1)
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam")
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Process each face
            emotions = []
            confidences = []
            all_probs_list = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = frame[y:y+h, x:x+w]
                
                # Skip if face region is too small
                if face_img.size == 0 or w < 20 or h < 20:
                    continue
                
                # Predict emotion
                try:
                    emotion, confidence, probs = self.predict(face_img)
                    emotions.append(emotion)
                    confidences.append(confidence)
                    all_probs_list.append(probs)
                except Exception as e:
                    print(f"Prediction error: {e}")
                    continue
            
            # Draw results
            if len(faces) > 0 and len(emotions) > 0:
                frame = self.draw_results(frame, faces[:len(emotions)], emotions, confidences, all_probs_list)
            else:
                # Draw "No face detected" message
                cv2.putText(frame, "No face detected", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self._draw_fps(frame)
            
            # Show frame
            cv2.imshow('Facial Emotion Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset smoothing
                self.prediction_history.clear()
                print("Smoothing reset")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nDetection stopped.")


def main():
    parser = argparse.ArgumentParser(description='Real-time Facial Emotion Detection')
    parser.add_argument('--model', type=str, choices=['cnn', 'efficientnet'],
                       default='cnn', help='Model to use for detection')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--smoothing', type=int, default=SMOOTHING_WINDOW,
                       help='Number of frames for temporal smoothing')
    
    args = parser.parse_args()
    
    try:
        detector = EmotionDetector(
            model_type=args.model,
            smoothing_window=args.smoothing
        )
        detector.run(camera_id=args.camera)
    except FileNotFoundError as e:
        print(f"\n[ERROR]: {e}")
        print("\nPlease train the model first using:")
        print(f"  python train.py --model {args.model}")
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        raise


if __name__ == "__main__":
    main()


