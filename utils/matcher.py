
# Description: Matches players across two videos using spatial + visual features and visualizes matched pairs
# ==========================================

import os
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
import torch

def match_players(broadcast_tracks, tacticam_tracks, broadcast_video, tacticam_video, use_middle_frame=False):
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    resnet.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def extract_embedding(frame, bbox):
        h, w, _ = frame.shape
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            print("⚠️ Empty or invalid crop detected")
            return None
        try:
            input_tensor = preprocess(to_pil_image(crop)).unsqueeze(0)
            with torch.no_grad():
                embedding = resnet(input_tensor).squeeze().numpy()
            return embedding
        except Exception as e:
            print(f"❌ Embedding extraction failed: {e}")
            return None

    def get_frame(video_path, frame_number):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"⚠️ Failed to read frame {frame_number} from {video_path}")
        return frame if ret else None

    def resize_keep_aspect(img, target_height):
        h, w = img.shape[:2]
        scale = target_height / h
        return cv2.resize(img, (int(w * scale), target_height))

    mapping = {}
    os.makedirs("results/matches", exist_ok=True)

    for tid, t_track in tacticam_tracks.items():
        print(f"\n🔄 Matching Tacticam ID: {tid}")
        best_match = None
        best_score = float('inf')
        if not t_track:
            print("🚫 Empty tacticam track")
            continue

        t_data = t_track[len(t_track) // 2] if use_middle_frame else t_track[0]
        t_center = t_data['center']
        t_frame = get_frame(tacticam_video, t_data['frame'])
        if t_frame is None:
            continue
        t_embedding = extract_embedding(t_frame, t_data['bbox'])
        if t_embedding is None:
            print("🚫 Skipping due to failed embedding")
            continue

        for bid, b_track in broadcast_tracks.items():
            if not b_track:
                continue
            b_data = b_track[len(b_track) // 2] if use_middle_frame else b_track[0]
            b_center = b_data['center']
            b_frame = get_frame(broadcast_video, b_data['frame'])
            if b_frame is None:
                continue
            b_embedding = extract_embedding(b_frame, b_data['bbox'])
            if b_embedding is None:
                continue

            spatial_dist = np.linalg.norm(np.array(t_center) - np.array(b_center))
            visual_dist = cosine(t_embedding, b_embedding)
            score = 0.6 * visual_dist + 0.4 * spatial_dist

            print(f"🔁 Tacticam {tid} vs Broadcast {bid} → Score: {score:.4f}")

            if score < best_score:
                best_score = score
                best_match = bid

        if best_match is not None:
            print(f"✅ Best match for Tacticam {tid} → Broadcast {best_match} with score {best_score:.4f}")
            mapping[tid] = best_match

            # Visualization side-by-side
            b_img = get_frame(broadcast_video, broadcast_tracks[best_match][0]['frame'])
            t_img = get_frame(tacticam_video, t_track[0]['frame'])

            if b_img is not None and t_img is not None:
                bx1, by1, bx2, by2 = broadcast_tracks[best_match][0]['bbox']
                tx1, ty1, tx2, ty2 = t_track[0]['bbox']

                h1, w1, _ = b_img.shape
                h2, w2, _ = t_img.shape
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w1, bx2), min(h1, by2)
                tx1, ty1 = max(0, tx1), max(0, ty1)
                tx2, ty2 = min(w2, tx2), min(h2, ty2)

                b_crop = b_img[by1:by2, bx1:bx2]
                t_crop = t_img[ty1:ty2, tx1:tx2]

                print("📏 b_crop:", b_crop.shape if b_crop is not None else None, "/ t_crop:", t_crop.shape if t_crop is not None else None)

                if (
                    b_crop is not None and t_crop is not None and
                    b_crop.size > 0 and t_crop.size > 0 and
                    b_crop.ndim == 3 and t_crop.ndim == 3 and
                    b_crop.shape[2] == t_crop.shape[2]
                ):
                    b_crop_resized = resize_keep_aspect(b_crop, target_height=128)
                    t_crop_resized = resize_keep_aspect(t_crop, target_height=128)
                    combined = cv2.hconcat([b_crop_resized, t_crop_resized])
                    cv2.imwrite(f"results/matches/t_{tid}_b_{best_match}.jpg", combined)
                else:
                    print("⚠️ Skipped image save due to incompatible crop dimensions or empty crop")

    return mapping