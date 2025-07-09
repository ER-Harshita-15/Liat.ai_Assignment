# Description: Main runner script to track and match players
# ==========================================

import cv2
import os
import csv
import pickle
from ultralytics import YOLO
from utils.tracker import track_players
from utils.matcher import match_players

# Create results folder if not exists
os.makedirs("results", exist_ok=True)

# Load YOLO model
model = YOLO("best.pt")

# OPTION A: Run tracking if no saved tracks exist
if not os.path.exists("results/broadcast_tracks.pkl") or not os.path.exists("results/tacticam_tracks.pkl"):
    # Load videos
    broadcast_cap = cv2.VideoCapture("Assignment Materials/broadcast.mp4")
    tacticam_cap = cv2.VideoCapture("Assignment Materials/tacticam.mp4")

    # Track players and save results
    broadcast_tracks = track_players(model, broadcast_cap, source_name="broadcast")
    tacticam_tracks = track_players(model, tacticam_cap, source_name="tacticam")

    # Save tracks for future reuse
    with open("results/broadcast_tracks.pkl", "wb") as f:
        pickle.dump(broadcast_tracks, f)
    with open("results/tacticam_tracks.pkl", "wb") as f:
        pickle.dump(tacticam_tracks, f)

else:
    # OPTION B: Load precomputed tracking results
    print("Loading saved tracking data...")
    with open("results/broadcast_tracks.pkl", "rb") as f:
        broadcast_tracks = pickle.load(f)
    with open("results/tacticam_tracks.pkl", "rb") as f:
        tacticam_tracks = pickle.load(f)

# Debug print track IDs
print("Broadcast IDs:", list(broadcast_tracks.keys()))
print("Tacticam IDs:", list(tacticam_tracks.keys()))

# Match player identities between views
mapping_dict = match_players(
    broadcast_tracks,
    tacticam_tracks,
    "Assignment Materials/broadcast.mp4",
    "Assignment Materials/tacticam.mp4",
    use_middle_frame=True
)

# Save mapping results to CSV
with open("player_mappings.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tacticam_id", "broadcast_id"])
    for tacticam_id, broadcast_id in mapping_dict.items():
        writer.writerow([tacticam_id, broadcast_id])

print("Player ID mapping completed and saved to player_mappings.csv")
