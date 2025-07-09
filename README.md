# Player Re-Identification Assignment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-green)
![DeepSORT](https://img.shields.io/badge/DeepSORT-Tracking-orange)

> **Football Player Re-Identification across Multiple Camera Views**  
> Assignment for LIAT.AI &mdash; by Harshita Pandey

---

## Overview

This project demonstrates **Player Re-Identification (ReID)** in football videos, matching player identities across two distinct camera angles using state-of-the-art object detection, tracking, and deep visual embeddings.

## Project Structure

```
Assignment_liat.ai/
├── detect_and_match.py        # Main runner script
├── best.pt                    # Pretrained YOLOv8 model (not in repo)
├── utils/
│   ├── tracker.py             # Player and ball tracking logic
│   └── matcher.py             # Player matching logic using embeddings
├── results/
│   ├── matches/               # Debug match images
│   ├── broadcast_tracks.pkl   # Pickled detection+tracking results
│   └── tacticam_tracks.pkl
├── Assignment Materials/
│   ├── broadcast.mp4
│   └── tacticam.mp4
├── player_mappings.csv        # Final ID mapping results
└── resume.pdf                 # Author's resume
```

## Features

- **Player Detection:** Custom YOLOv8 model trained for football players and ball.
- **Tracking:** DeepSORT for robust multi-object tracking.
- **Cross-View Matching:** Visual (ResNet18 embeddings) + spatial similarity.
- **Debug Outputs:** Side-by-side match images for verification.
- **Easy Reproducibility:** Cached results to avoid recomputation.
- **Extensible:** Modular code for further research or improvements.

## Quick Start

1. **Clone the Repository**
    ```bash
    git clone https://github.com/ER-Harshita-15/Liat.ai_Assignment
    cd Assignment_liat.ai
    ```

2. **Set Up Environment**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Linux/Mac:
    source venv/bin/activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    <sub>If `requirements.txt` is missing:</sub>
    ```bash
    pip install opencv-python torch torchvision ultralytics deep_sort_realtime
    ```

4. **Add Model Weights**  
   Place the provided `best.pt` YOLOv8 model in the project root.

5. **Run the Pipeline**
    ```bash
    python detect_and_match.py
    ```

## Methodology

1. **Detection & Tracking:** Players are detected and tracked in both videos using YOLOv8 + DeepSORT.
2. **Frame Selection:** For each player, the frame with the largest bounding box is chosen.
3. **Embedding Extraction:** Cropped player images are passed through ResNet18 to obtain deep embeddings.
4. **Matching:** Cosine similarity (visual) and spatial distance are combined for matching.
5. **Results:** Matched pairs are saved as images in `results/matches/`. Final ID mappings are exported to `player_mappings.csv`.

## Output Files

| File/Folder                | Description                                 |
|----------------------------|---------------------------------------------|
| `player_mappings.csv`      | Matched player IDs between both views       |
| `results/matches/*.jpg`    | Side-by-side images of matched players      |
| `.pkl` files               | Cached detection/tracking outputs           |

## Notes

- The YOLOv8 model is custom-trained for football players and the ball.
- Only valid bounding boxes are used for embedding extraction.
- All preprocessing (resizing, normalization) follows ImageNet standards.
- The code is modular and easy to extend for research or production.

## Potential Improvements

- Integrate OCR for jersey number recognition.
- Use temporal consistency (track overlap) for more robust matching.
- Apply L2-normalization to embeddings.
- Save and analyze matching scores for each pair.

## About the Author

Prepared by **Harshita Pandey** for the LIAT.AI assignment.  
For questions or collaboration, please refer to the attached `Resume.docx`.

## License

This project is for educational and demonstration purposes only.


