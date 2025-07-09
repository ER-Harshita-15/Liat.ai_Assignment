
# ==========================================
# File: utils/matcher.py
# Description: Matches players across two videos using visual + spatial features. Now selects best frame, applies padded crops, and logs debug matches.
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
        pad = 10  # padding around box
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
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

    def select_valid_frame(track, video_path):
        # First try largest bbox frame
        sorted_track = sorted(track, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse=True)
        for t in sorted_track:
            frame = get_frame(video_path, t['frame'])
            if frame is not None:
                h, w, _ = frame.shape
                x1, y1, x2, y2 = t['bbox']
                if x2 > x1 and y2 > y1 and x2 <= w and y2 <= h:
                    return t, frame
        print("⚠️ No valid frame found for track")
        return None, None

    mapping = {}
    os.makedirs("results/matches", exist_ok=True)

    for tid, t_track in tacticam_tracks.items():
        print(f"\n🔄 Matching Tacticam ID: {tid}")
        best_match = None
        best_score = float('inf')
        if not t_track:
            print("🚫 Empty tacticam track")
            continue

        t_data, t_frame = select_valid_frame(t_track, tacticam_video)
        if t_data is None or t_frame is None:
            continue

        t_embedding = extract_embedding(t_frame, t_data['bbox'])
        b_img_debug, t_img_debug = None, None

        for bid, b_track in broadcast_tracks.items():
            if not b_track:
                continue
            b_data, b_frame = select_valid_frame(b_track, broadcast_video)
            if b_data is None or b_frame is None:
                continue

            b_embedding = extract_embedding(b_frame, b_data['bbox'])
            if t_embedding is None or b_embedding is None:
                continue

            spatial_dist = np.linalg.norm(np.array(t_data['center']) - np.array(b_data['center']))
            visual_dist = cosine(t_embedding, b_embedding)
            score = 0.6 * visual_dist + 0.4 * spatial_dist

            print(f"🔁 Tacticam {tid} vs Broadcast {bid} → Score: {score:.4f}")

            if score < best_score:
                best_score = score
                best_match = bid
                b_img_debug = b_frame
                t_img_debug = t_frame
                best_b_data = b_data

        if best_match is not None:
            print(f"✅ Best match for Tacticam {tid} → Broadcast {best_match} with score {best_score:.4f}")
            mapping[tid] = best_match

            bx1, by1, bx2, by2 = best_b_data['bbox']
            tx1, ty1, tx2, ty2 = t_data['bbox']

            h1, w1 = b_img_debug.shape[:2]
            h2, w2 = t_img_debug.shape[:2]
            pad = 10
            bx1, by1 = max(0, bx1 - pad), max(0, by1 - pad)
            bx2, by2 = min(w1, bx2 + pad), min(h1, by2 + pad)
            tx1, ty1 = max(0, tx1 - pad), max(0, ty1 - pad)
            tx2, ty2 = min(w2, tx2 + pad), min(h2, ty2 + pad)

            b_crop = b_img_debug[by1:by2, bx1:bx2]
            t_crop = t_img_debug[ty1:ty2, tx1:tx2]

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

        elif t_embedding is not None:
            # Save debug crop even if no match was found
            tx1, ty1, tx2, ty2 = t_data['bbox']
            tx1, ty1 = max(0, tx1 - 10), max(0, ty1 - 10)
            tx2, ty2 = min(t_frame.shape[1], tx2 + 10), min(t_frame.shape[0], ty2 + 10)
            t_crop = t_frame[ty1:ty2, tx1:tx2]
            if t_crop is not None and t_crop.size > 0:
                t_crop_resized = resize_keep_aspect(t_crop, target_height=128)
                cv2.imwrite(f"results/matches/t_{tid}_unmatched.jpg", t_crop_resized)

    return mapping