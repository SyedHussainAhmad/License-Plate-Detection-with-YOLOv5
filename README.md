# License Plate Detection with YOLOv5 + EasyOCR

A complete solution for detecting and reading license plates from images and videos using YOLOv5 for object detection and EasyOCR for text recognition.

## Overview

This project combines two powerful technologies:
- **YOLOv5**: For accurate license plate detection and localization
- **EasyOCR**: For optical character recognition to extract text from detected plates

The system can process both single images and video files, outputting results with bounding boxes and extracted text.

## Features

- 🎯 **Accurate Detection**: Uses YOLOv5 for robust license plate detection
- 📖 **Text Recognition**: EasyOCR extracts text from detected license plates
- 🖼️ **Image Processing**: Process single images with visualization
- 🎥 **Video Processing**: Process entire videos frame by frame
- 📊 **CSV Export**: Save results to CSV files for further analysis
- 🎨 **Visual Output**: Generate annotated images and videos with bounding boxes and text

## Requirements

```
torch
opencv-python
easyocr
pandas
matplotlib
pathlib
```

## Installation

1. Clone the YOLOv5 repository:
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install easyocr
```

3. Prepare your custom dataset in the required format with `data.yaml` configuration file.

## Training Custom YOLOv5 Model

To train your own license plate detection model:

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --data ../license-plate-dataset/data.yaml \
  --weights yolov5s.pt \
  --name license-plate-detector
```

### Training Parameters:
- `--img 640`: Input image size
- `--batch 16`: Batch size (adjust based on GPU memory)
- `--epochs 50`: Number of training epochs
- `--data`: Path to your dataset configuration file
- `--weights yolov5s.pt`: Pre-trained weights to start from
- `--name`: Name for your training run

## Usage

### Loading the Trained Model

```python
import torch

# Load your custom trained model
model = torch.hub.load('yolov5', 'custom', path='path/to/best.pt', source='local')
```

### Image Processing

```python
import easyocr
import cv2
import pandas as pd
from matplotlib import pyplot as plt

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Run detection on image
results = model('test.jpg')
results.render()

# Extract bounding boxes and perform OCR
img = results.ims[0]
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
boxes = results.xyxy[0].cpu().numpy()

ocr_results = []
for *xyxy, conf, cls in boxes:
    x1, y1, x2, y2 = map(int, xyxy)
    cropped = img[y1:y2, x1:x2]
    
    # OCR on cropped license plate
    text = reader.readtext(cropped, detail=0)
    text = ' '.join(text)
    
    ocr_results.append({
        "box": [x1, y1, x2, y2],
        "text": text
    })
    
    # Draw text on image
    cv2.putText(img, text, (x1, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Save results
df = pd.DataFrame(ocr_results)
df.to_csv("license_plate_results.csv", index=False)
```

### Video Processing

```python
import cv2

# Initialize video capture
cap = cv2.VideoCapture('input_video.mp4')

# Setup video writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

ocr_results = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process frame with YOLO + OCR
    results = model(frame)
    results.render()
    frame_with_boxes = results.ims[0]
    frame_bgr = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)
    
    # Extract and process bounding boxes
    boxes = results.xyxy[0].cpu().numpy()
    for *xyxy, conf, cls in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        cropped = frame[y1:y2, x1:x2]
        
        text = reader.readtext(cropped, detail=0)
        text = ' '.join(text)
        
        ocr_results.append({
            "frame": frame_count,
            "box": [x1, y1, x2, y2],
            "text": text
        })
        
        cv2.putText(frame_bgr, text, (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    out.write(frame_bgr)

cap.release()
out.release()

# Save results
df = pd.DataFrame(ocr_results)
df.to_csv('license_plate_video_results.csv', index=False)
```

## Output Files

### CSV Results Format

The system generates CSV files with the following structure:

**For Images:**
| Column | Description |
|--------|-------------|
| box | Bounding box coordinates [x1, y1, x2, y2] |
| text | Extracted license plate text |

**For Videos:**
| Column | Description |
|--------|-------------|
| frame | Frame number |
| box | Bounding box coordinates [x1, y1, x2, y2] |
| text | Extracted license plate text |

### Visual Outputs

- **Images**: Annotated images with bounding boxes and OCR text
- **Videos**: Processed videos with real-time detection and text overlay

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for faster processing
2. **Batch Processing**: Increase batch size if you have sufficient GPU memory
3. **Image Resolution**: Balance between accuracy and speed by adjusting input image size
4. **OCR Languages**: Specify appropriate languages in EasyOCR for better accuracy

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch size or image resolution
2. **Poor OCR Results**: Ensure license plates are clearly visible and properly cropped
3. **Slow Processing**: Consider using smaller YOLOv5 models (yolov5n.pt) for faster inference

### Tips for Better Results:

- Use high-quality training images with diverse lighting conditions
- Ensure proper annotation of license plates in training data
- Fine-tune confidence thresholds based on your specific use case
- Consider data augmentation techniques during training

## Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Submitting pull requests

## License

This project uses:
- YOLOv5 (GPL-3.0 License)
- EasyOCR (Apache 2.0 License)

Please ensure compliance with respective licenses when using this code.

---

For questions or support, please open an issue in the repository.
