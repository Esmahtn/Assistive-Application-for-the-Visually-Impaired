import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import time


engine = pyttsx3.init()


model_path = "ourmodel.pt"
video_path = "inference/Adsız tasarım (2).mp4"


class_names = []
with open("coco-classes.txt", "r") as f:
    class_names = f.read().strip().split("\n")

# sadece istediğimiz sınıfların numaraları 
vehicles = [0, 1, 2, 3,4, 5, 6, 7, 9,10, 11,]

cap = cv2.VideoCapture(video_path)
model = YOLO(model_path)

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
thickness = 2
font_scale = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX

# İç içe iki poligon tanımlayın
outer_polygon_points = (350, 717), (480, 545), (798, 543), (894, 714)

# Create the outer_polygon as a NumPy array
outer_polygon = np.array(outer_polygon_points, np.int32)


inner_polygon_points = (475, 713), (550, 599), (757, 592), (822, 715)


inner_polygon = np.array(inner_polygon_points, np.int32)





width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("video.avi", fourcc, 10.0, (width, height))

# Dış poligon ve iç poligon için uyarı verilen nesnelerin kimliklerini saklamak için bayraklar
outer_polygon_warning_id = None
inner_polygon_warning_id = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, persist=True, verbose=False)[0]
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")
    
    # Dış poligona giren en yakın nesne için uyarı
    closest_outer_distance = float('inf')
    closest_outer_box = None
    
    for box in bboxes:
        # Algılama bilgisi eksikse, geç
        if len(box) != 7:
            continue
        
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        if class_id in vehicles:
            distance_to_outer_polygon = cv2.pointPolygonTest(outer_polygon, (cx, cy), True)
            if distance_to_outer_polygon >= 0 and distance_to_outer_polygon < closest_outer_distance:
                closest_outer_distance = distance_to_outer_polygon
                closest_outer_box = box

    if closest_outer_box is not None:
        x1, y1, x2, y2, track_id, score, class_id = closest_outer_box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        class_name = class_names[class_id]
        
        if outer_polygon_warning_id != track_id:
            engine.say(f"Warning: Closest {class_name} detected inside the outer polygon.")
            engine.runAndWait()
            outer_polygon_warning_id = track_id
    
    if outer_polygon_warning_id is not None:
        cx_outer, cy_outer = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if cv2.pointPolygonTest(outer_polygon, (cx_outer, cy_outer), False) < 0:
            outer_polygon_warning_id = None
    
    # İç poligona giren en yakın nesne için uyarı
    closest_inner_distance = float('inf')
    closest_inner_box = None

    for box in bboxes:
        # Algılama bilgisi eksikse, geç
        if len(box) != 7:
            continue
        
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        if class_id in vehicles:
            distance_to_inner_polygon = cv2.pointPolygonTest(inner_polygon, (cx, cy), True)
            if distance_to_inner_polygon >= 0 and distance_to_inner_polygon < closest_inner_distance:
                closest_inner_distance = distance_to_inner_polygon
                closest_inner_box = box

    if closest_inner_box is not None:
        x1, y1, x2, y2, track_id, score, class_id = closest_inner_box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        class_name = class_names[class_id]
        
        if inner_polygon_warning_id != track_id:
            engine.say(f"Warning: Closest {class_name} detected inside the inner polygon.")
            engine.runAndWait()
            inner_polygon_warning_id = track_id
    
    if inner_polygon_warning_id is not None:
        cx_inner, cy_inner = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if cv2.pointPolygonTest(inner_polygon, (cx_inner, cy_inner), False) < 0:
            inner_polygon_warning_id = None

    # İç içe iki poligonu çiz
    cv2.polylines(frame, [outer_polygon], isClosed=True, color=green, thickness=2)
    cv2.polylines(frame, [inner_polygon], isClosed=True, color=red, thickness=2)
    
    # Her bir kutuyu, sınıf bilgisini, kimlik numarasını ve orta noktayı çiz
    for box in bboxes:
        # Algılama bilgisi eksikse, geç
        if len(box) != 7:
            continue
        
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if class_id in vehicles:
            class_name = class_names[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=blue, thickness=thickness)
            cv2.putText(frame, f"{class_name}", (x1, y1 - 10), font, font_scale, blue, thickness)
            cv2.circle(frame, (cx, cy), radius=5, color=yellow, thickness=-1)  # Orta noktayı işaretle
    
    writer.write(frame)
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()



































