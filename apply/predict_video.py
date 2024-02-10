import os

from ultralytics import YOLO
import cv2


# VIDEOS_DIR = os.path.join('.', 'videos')

video_path = "C:\\Users\\zzy\\Desktop\\project\\BicycleDetection-main\\1.mp4"
video_path_out = "C:\\Users\\zzy\\Desktop\\project\\BicycleDetection-main\\o_1_3.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# model_path = "C:\\Users\\zzy\\Desktop\\project\\train-yolov8\\local_env\\runs\\detect\\train10\\weights\\best.pt"
model_path = "C:\\Users\\zzy\\Desktop\\project\\try\\runs\\detect\\train2\\weights\\best.pt"

# Load a model
model = YOLO(model_path)  # load a custom model

threshold_l = 0.1
threshold_h = 0.7

while ret:

    results = model(frame)[0]

    cnt = 0

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        print(f"class_id: {class_id}, score: {score}")

        if (score > threshold_l and int(class_id) == 0) or (score > threshold_h and int(class_id) == 1) :
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cnt = cnt + 1
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"{score:.2f}", (int(x1 - 50), int(y1-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
    print("cnt: ", cnt)
    cv2.putText(frame, f"{cnt}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
