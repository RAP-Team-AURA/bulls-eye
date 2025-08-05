

import cv2
import torch
import numpy as np
import os
import json
from collections import defaultdict, deque
from datetime import datetime
import face_recognition
from deepface import DeepFace
import pickle

class VideoProcessor:
    def __init__(self, video_source=0, output_dir="output"):
        self.video_source = video_source
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        self.person_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.tracked_people = {}
        self.next_id = 0
        self.max_disappeared = 30
        self.face_encodings = {} 
        self.person_face_mapping = {}  
        
        self.known_faces = {}
        self.face_similarity_threshold = 0.6
        
        self.cleanup_counter = 0
        
        self.tracking_history = defaultdict(lambda: deque(maxlen=30)) 
        self.colors = self.generate_colors(100)
        
        self.frame_count = 0
        self.detection_log = []
        
    def ensure_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, "frames"))
            os.makedirs(os.path.join(self.output_dir, "faces"))
    
    def generate_colors(self, n):
        colors = []
        for i in range(n):
            hue = (i * 137.508) % 360 
            rgb = self.hsv_to_rgb(hue, 0.7, 0.9)
            colors.append(rgb)
        return colors
    
    def hsv_to_rgb(self, h, s, v):
        h = h / 360.0
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if i % 6 == 0:
            r, g, b = v, t, p
        elif i % 6 == 1:
            r, g, b = q, v, p
        elif i % 6 == 2:
            r, g, b = p, v, t
        elif i % 6 == 3:
            r, g, b = p, q, v
        elif i % 6 == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
            
        return (int(b * 255), int(g * 255), int(r * 255))
    
    def cleanup_known_faces(self):
        self.known_faces = {k: v for k, v in self.known_faces.items() if v is not None}
    
    def detect_persons(self, frame):
        results = self.person_detector(frame)
        detections = []
        
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0 and conf > 0.5: 
                x1, y1, x2, y2 = map(int, box)
                if (x2 - x1) > 50 and (y2 - y1) > 100:
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })
        
        return detections
    
    def detect_faces(self, frame, person_bbox):
        x1, y1, x2, y2 = person_bbox
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return []
        
        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_detections = []
        for (fx, fy, fw, fh) in faces:
            face_bbox = (x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh)
            face_roi = frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
            
            if face_roi.size > 0:
                face_detections.append({
                    'bbox': face_bbox,
                    'roi': face_roi
                })
        
        return face_detections
    
    def extract_face_encoding(self, face_roi):
        try:
            if face_roi is None or face_roi.size == 0:
                return None
                
            if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                return None
                
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(face_rgb)
            if encodings:
                return encodings[0]
        except Exception as e:
            if "No faces found" not in str(e):
                print(f"Error extracting face encoding: {e}")
        return None
    
    def reidentify_person(self, face_encoding):
        if face_encoding is None:
            return None, 0.0
        
        best_match_id = None
        best_similarity = 0.0
        
        for person_id, known_encoding in self.known_faces.items():
            try:
                if known_encoding is None:
                    continue
                    
                similarity = 1 - face_recognition.face_distance([known_encoding], face_encoding)[0]
                if similarity > best_similarity and similarity > self.face_similarity_threshold:
                    best_similarity = similarity
                    best_match_id = person_id
            except Exception as e:
                if "NoneType" not in str(e):
                    print(f"Error in face comparison: {e}")
        
        return best_match_id, best_similarity
    
    def track_persons(self, current_detections):
        used_detections = set()
        new_tracked_people = {}
        
        for person_id, track_info in self.tracked_people.items():
            prev_bbox = track_info['bbox']
            disappeared_count = track_info['disappeared']
            
            best_match = None
            best_score = 0
            best_idx = -1
            
            for idx, detection in enumerate(current_detections):
                if idx in used_detections:
                    continue
                
                iou_score = self.calculate_iou(prev_bbox, detection['bbox'])
                dist = self.calculate_center_distance(prev_bbox, detection['bbox'])
                
                score = iou_score - (dist / 1000)
                
                if score > best_score and iou_score > 0.3:
                    best_score = score
                    best_match = detection
                    best_idx = idx
            
            if best_match is not None:
                new_tracked_people[person_id] = {
                    'bbox': best_match['bbox'],
                    'confidence': best_match['confidence'],
                    'disappeared': 0,
                    'faces': best_match.get('faces', []),
                    'face_encoding': best_match.get('face_encoding')
                }
                used_detections.add(best_idx)
            elif disappeared_count < self.max_disappeared:
                new_tracked_people[person_id] = {
                    'bbox': prev_bbox,
                    'confidence': track_info['confidence'],
                    'disappeared': disappeared_count + 1,
                    'faces': track_info.get('faces', []),
                    'face_encoding': track_info.get('face_encoding')
                }
        
        for idx, detection in enumerate(current_detections):
            if idx not in used_detections:
                new_tracked_people[self.next_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'disappeared': 0,
                    'faces': detection.get('faces', []),
                    'face_encoding': detection.get('face_encoding')
                }
                self.next_id += 1
        
        self.tracked_people = new_tracked_people
        return new_tracked_people
    
    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        if xB <= xA or yB <= yA:
            return 0.0
        
        interArea = (xB - xA) * (yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        return interArea / float(boxAArea + boxBArea - interArea)

    def calculate_center_distance(self, boxA, boxB):
        centerA = ((boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2)
        centerB = ((boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2)
        return np.sqrt((centerA[0] - centerB[0])**2 + (centerA[1] - centerB[1])**2)

    def visualize_results(self, frame, tracked_people):
        for person_id, track_info in tracked_people.items():
            bbox = track_info['bbox']
            disappeared = track_info['disappeared']
            confidence = track_info['confidence']
            faces = track_info.get('faces', [])
            
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            self.tracking_history[person_id].append(center)
            
            if disappeared > 0:
                color = (0, 165, 255)  
                alpha = 0.5
            else:
                color = self.colors[person_id % len(self.colors)]
                alpha = 1.0
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            if len(self.tracking_history[person_id]) > 1:
                points = list(self.tracking_history[person_id])
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)
            
            for face in faces:
                face_bbox = face['bbox']
                cv2.rectangle(frame, (face_bbox[0], face_bbox[1]), 
                            (face_bbox[2], face_bbox[3]), (255, 0, 0), 2)
            
            label = f'Person {person_id} ({confidence:.2f})'
            if faces:
                label += f' [Face: {len(faces)}]'
            
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (bbox[0], bbox[1] - label_height - 10), 
                         (bbox[0] + label_width, bbox[1]), color, -1)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        active_count = len([p for p in tracked_people.values() if p['disappeared'] == 0])
        cv2.putText(frame, f'Active People: {active_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Total Tracks: {len(tracked_people)}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Frame: {self.frame_count}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def export_results(self, frame, tracked_people):
        frame_path = os.path.join(self.output_dir, "frames", f"frame_{self.frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_detections = []
        for person_id, track_info in tracked_people.items():
            if track_info['disappeared'] == 0:  
                detection_info = {
                    'frame': self.frame_count,
                    'person_id': person_id,
                    'bbox': track_info['bbox'],
                    'confidence': track_info['confidence'],
                    'faces': []
                }
                
                for i, face in enumerate(track_info.get('faces', [])):
                    face_path = os.path.join(self.output_dir, "faces", 
                                           f"person_{person_id}_frame_{self.frame_count}_face_{i}.jpg")
                    cv2.imwrite(face_path, face['roi'])
                    detection_info['faces'].append(face_path)
                
                frame_detections.append(detection_info)
        
        self.detection_log.append({
            'frame': self.frame_count,
            'timestamp': datetime.now().isoformat(),
            'detections': frame_detections
        })
    
    def save_detection_log(self):
        log_path = os.path.join(self.output_dir, "detection_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.detection_log, f, indent=2)
        
        cleaned_faces = {k: v for k, v in self.known_faces.items() if v is not None}
        
        encodings_path = os.path.join(self.output_dir, "face_encodings.pkl")
        with open(encodings_path, 'wb') as f:
            pickle.dump(cleaned_faces, f)
    
    def process_video(self):
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            return
        
        print("Starting video processing pipeline...")
        print("Press 'q' to quit, 's' to save current state")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            self.frame_count += 1
            
            person_detections = self.detect_persons(frame)
            
            for detection in person_detections:
                faces = self.detect_faces(frame, detection['bbox'])
                detection['faces'] = faces
                
                if faces:
                    face_encoding = self.extract_face_encoding(faces[0]['roi'])
                    detection['face_encoding'] = face_encoding
                    
                    matched_id, similarity = self.reidentify_person(face_encoding)
                    if matched_id is not None:
                        detection['reidentified_id'] = matched_id
                        detection['similarity'] = similarity
                    elif face_encoding is not None:
                        self.known_faces[self.next_id] = face_encoding
            
            tracked_people = self.track_persons(person_detections)
            
            self.visualize_results(frame, tracked_people)
            
            if self.frame_count % 30 == 0:
                self.export_results(frame, tracked_people)
            
            self.cleanup_counter += 1
            if self.cleanup_counter % 100 == 0:
                self.cleanup_known_faces()
            
            cv2.imshow("Video Processing Pipeline", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_detection_log()
                print(f"Saved detection log to {self.output_dir}")

        cap.release()
        cv2.destroyAllWindows()
        
        self.save_detection_log()
        print(f"Processing complete. Results saved to {self.output_dir}")

def main():
    video_source = "/Users/kirthika/Downloads/oo.mp4"
    
    processor = VideoProcessor(video_source=video_source, output_dir="output")
    processor.process_video()

if __name__ == "__main__":
    main()