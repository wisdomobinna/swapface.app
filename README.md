# Face Swap App

AI-powered face swapping using InsightFace. This tool provides both a web interface (Streamlit) and a batch processing CLI for automated face swapping.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [Batch Processing (CLI)](#batch-processing-cli)
- [Step-by-Step Guide](#step-by-step-guide)
- [CSV Format](#csv-format)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Web Interface**: Interactive Streamlit app for single face swaps
- **Batch Processing**: CLI tool for processing multiple face swaps automatically
- **Automatic Model Download**: Models are downloaded automatically on first run
- **High Quality**: Uses InsightFace's state-of-the-art face swapping technology

---

## Installation

### 1. Set Up Virtual Environment

First, activate the virtual environment (recommended to avoid dependency conflicts):

```bash
source venv/bin/activate
```

If the virtual environment doesn't exist, create it first:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- streamlit
- opencv-python
- insightface
- onnxruntime
- Pillow
- numpy

### 3. Model Download

The face swap model (~554MB) will be **automatically downloaded** the first time you run either the web app or batch tool. No manual download needed!

The model is stored in: `~/.insightface/models/inswapper_128.onnx`

---

## Usage

### Web Interface

Run the interactive Streamlit app:

```bash
streamlit run app.py
```

Then open your browser to the URL shown (usually http://localhost:8501)

### Batch Processing (CLI)

Process multiple face swaps automatically using a CSV file:

```bash
python batch_face_swap.py face_swap_batch.csv
```

---

## Step-by-Step Guide

### Step 1: Prepare Your Images

1. **Input Images** (source faces): Place in the `input/` folder
   - These are the faces you want to swap FROM
   - Example: `input/image1.jpg`, `input/image2.jpg`, etc.

2. **Target Images** (destination): Place in the `target/` folder
   - These are the images where faces will be replaced
   - Example: `target/target1.jpg`, `target/target2.jpg`, etc.

3. **Output Folder**: Results will be saved to `output/`
   - Created automatically if it doesn't exist

### Step 2: Create the CSV File

Create a CSV file (e.g., `face_swap_batch.csv`) with three columns:

```csv
face_image,target_image,output
/full/path/to/input/image1.jpg,/full/path/to/target/target1.jpg,/full/path/to/output/result1.jpg
/full/path/to/input/image2.jpg,/full/path/to/target/target2.jpg,/full/path/to/output/result2.jpg
```

**Important**: Use **absolute paths** (full paths) for all images.

### Step 3: Run the Batch Script

```bash
python batch_face_swap.py face_swap_batch.csv
```

The script will:
1. Check if the face swap model exists
2. Download it automatically if needed (~554MB, first time only)
3. Initialize the face analysis model
4. Process each row in your CSV
5. Save results to the output paths specified
6. Show a summary of successful/failed swaps

### Example Output

```
Checking for face swap model...
✅ Model found!
Initializing face analysis model...
Loading face swapper model...

Processing CSV: face_swap_batch.csv
------------------------------------------------------------

[1/3] Processing:
  Face: /Users/kcwis/Sites/vision_apps/swapface.app/input/image1.jpg
  Target: /Users/kcwis/Sites/vision_apps/swapface.app/target/target1.jpg
  Output: /Users/kcwis/Sites/vision_apps/swapface.app/output/result1.jpg
  ✅ Saved to: /Users/kcwis/Sites/vision_apps/swapface.app/output/result1.jpg

[2/3] Processing:
  Face: /Users/kcwis/Sites/vision_apps/swapface.app/input/image2.jpg
  Target: /Users/kcwis/Sites/vision_apps/swapface.app/target/target2.jpg
  Output: /Users/kcwis/Sites/vision_apps/swapface.app/output/result2.jpg
  ✅ Saved to: /Users/kcwis/Sites/vision_apps/swapface.app/output/result2.jpg

============================================================
SUMMARY
============================================================
Total: 3
Successful: 3
Failed: 0
============================================================
```

---

## CSV Format

The CSV must have exactly three columns:

| Column | Description | Example |
|--------|-------------|---------|
| `face_image` | Path to source face image | `/path/to/input/face1.jpg` |
| `target_image` | Path to target image | `/path/to/target/person1.jpg` |
| `output` | Path where result will be saved | `/path/to/output/result1.jpg` |

**Tips:**
- Use absolute paths (full paths from root)
- Supported formats: JPG, JPEG, PNG
- Ensure images contain visible faces
- Output directories will be created automatically

### Example CSV Template

```csv
face_image,target_image,output
/Users/username/swapface.app/input/image1.jpg,/Users/username/swapface.app/target/target1.jpg,/Users/username/swapface.app/output/result1.jpg
/Users/username/swapface.app/input/image2.jpg,/Users/username/swapface.app/target/target2.jpg,/Users/username/swapface.app/output/result2.jpg
/Users/username/swapface.app/input/image3.jpg,/Users/username/swapface.app/target/target3.jpg,/Users/username/swapface.app/output/result3.jpg
```

---

## Troubleshooting

### No face detected in image
- Ensure the image contains a clear, visible face
- Try using a higher resolution image
- Make sure the face is well-lit and not obscured

### Model download fails
If automatic download fails, manually download:
1. Go to: https://huggingface.co/ezioruan/inswapper_128.onnx
2. Click 'Files and versions'
3. Download `inswapper_128.onnx`
4. Place in: `~/.insightface/models/` (Mac/Linux) or `%USERPROFILE%\.insightface\models\` (Windows)

### File not found errors
- Double-check all paths in your CSV are correct
- Use absolute paths (full paths)
- Ensure files exist at the specified locations

### Memory errors
- Process images in smaller batches
- Reduce image resolution before processing
- Close other memory-intensive applications

---

## Requirements

- Python 3.7+
- 2GB+ RAM recommended
- ~1GB disk space for models
- Internet connection (first run only, for model download)

---

## License

This project uses InsightFace for face swapping. Please refer to InsightFace's license for commercial usage terms.