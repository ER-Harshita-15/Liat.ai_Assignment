# Description: Handles player detection and tracking using YOLO + Deep SORT
# ==========================================

import cv2
import numpy as np
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

def track_players(model, video_cap, source_name="video"):
    deepsort = DeepSort(max_age=30)
    player_tracks = defaultdict(list)
    frame_idx = 0

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    out_path = f"results/{source_name}_with_ids.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        detections = []
        results = model(frame)
        for box in results[0].boxes:
            cls_id = int(box.cls)
            if cls_id == 0:  # Player
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, None))

        tracks = deepsort.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            player_tracks[track_id].append({
                'frame': frame_idx,
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2)//2, (y1 + y2)//2)
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    video_cap.release()
    out.release()
    return player_tracks