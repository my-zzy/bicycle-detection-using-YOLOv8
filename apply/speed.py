import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('best.pt')

# 打开视频文件
video_path = "bike1.mp4"
cap = cv2.VideoCapture(video_path)

# 创建输出视频编写器
output_file = "D:\desk\IT-DL\output_video1.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编解码器
out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# 遍历视频帧
x1=[];y1=[];x2=[];y2=[];v=0
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 在该帧上运行YOLOv8推理
        results = model(frame)
        for r in results:
            annotated_frame = r.plot()
            box = r.boxes.xywh  #save
            x = box[:,0]; y = box[:,1]
            for i in range(len(x)):
                for j in x1:
                    if abs(x[i-1]-j)<30 :
                        v = abs(x[i-1]-j)
                       
                       

                        # 在帧上绘制两行文本
                        #text_confidence = f"Confidence: {r.boxes.conf[i]:.2f}"
                        text_v = f"v: {v:.2f}"

                        # 设置文本显示位置
                        #text_location_confidence = (int(box[i, 0]), int(box[i, 1]) - 10)
                        
                        text_location_v = (int(x[i-1]),int(y[i-1])-30 )

                        # 在图像上绘制文本
                       
                        #cv2.putText(annotated_frame, text_confidence, text_location_confidence, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(annotated_frame, text_v, text_location_v, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                        out.write(annotated_frame)
            x1=x;y1=y

        # 显示带注释的帧
        cv2.imshow("YOLOv8推理", annotated_frame)

        # 如果按下'q'则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()