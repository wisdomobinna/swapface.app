#!/usr/bin/env python3
"""
Batch Face Swap CLI Tool

Usage:
    python batch_face_swap.py input.csv

CSV Format:
    face_image,target_image,output
    /path/to/face1.jpg,/path/to/target1.jpg,/path/to/output1.jpg
    /path/to/face2.jpg,/path/to/target2.jpg,/path/to/output2.jpg
"""

import argparse
import csv
import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
import urllib.request


def download_model_if_needed():
    """Download the inswapper model if not present"""
    model_path = os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'inswapper_128.onnx')

    if not os.path.exists(model_path):
        print("🔄 Downloading face swap model (one-time, ~554MB)... This may take a few minutes.")
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Try multiple mirrors
            mirrors = [
                'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx',
                'https://huggingface.co/Aitrepreneur/insightface/resolve/main/inswapper_128.onnx'
            ]

            for mirror in mirrors:
                try:
                    print(f"Trying mirror: {mirror}")
                    urllib.request.urlretrieve(mirror, model_path)
                    print("✅ Model downloaded successfully!")
                    return model_path
                except Exception as e:
                    print(f"Failed to download from {mirror}: {e}")
                    continue

            print("""
            ❌ Automatic download failed. Please manually download:

            1. Go to: https://huggingface.co/ezioruan/inswapper_128.onnx
            2. Click 'Files and versions'
            3. Download 'inswapper_128.onnx'
            4. Place it in: ~/.insightface/models/ (Mac/Linux) or %USERPROFILE%\\.insightface\\models\\ (Windows)
            """)
            return None
        except Exception as e:
            print(f"Download error: {str(e)}")
            return None

    return model_path


def perform_face_swap(face_image_path, target_image_path, output_path, app, swapper):
    """
    Perform face swap using InsightFace

    Args:
        face_image_path: Path to the source face image
        target_image_path: Path to the target image
        output_path: Path where the result should be saved
        app: FaceAnalysis instance
        swapper: Swapper model instance

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load images
        face_image = Image.open(face_image_path)
        target_image = Image.open(target_image_path)

        # Convert PIL images to cv2 format
        face_cv2 = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
        target_cv2 = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)

        # Get faces from both images
        face_faces = app.get(face_cv2)
        target_faces = app.get(target_cv2)

        if not face_faces:
            print(f"  ❌ No face detected in face image: {face_image_path}")
            return False

        if not target_faces:
            print(f"  ❌ No face detected in target image: {target_image_path}")
            return False

        # Get the face to swap (source face)
        source_face = face_faces[0]

        # Perform face swap on all faces in target image
        result = target_cv2.copy()
        for target_face in target_faces:
            result = swapper.get(result, target_face, source_face, paste_back=True)

        # Convert back to RGB for PIL
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_rgb)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save the result
        result_image.save(output_path)
        print(f"  ✅ Saved to: {output_path}")
        return True

    except FileNotFoundError as e:
        print(f"  ❌ File not found: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error during face swap: {str(e)}")
        return False


def process_csv(csv_path):
    """
    Process a CSV file containing face swap tasks

    Args:
        csv_path: Path to the CSV file
    """
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)

    # Download model if needed
    print("Checking for face swap model...")
    model_path = download_model_if_needed()
    if model_path is None:
        print("❌ Cannot proceed without the model")
        sys.exit(1)

    # Initialize InsightFace (do this once for all swaps)
    print("Initializing face analysis model...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Initialize the swapper model (do this once for all swaps)
    print("Loading face swapper model...")
    swapper = insightface.model_zoo.get_model(model_path, download=False)

    # Read and process CSV
    print(f"\nProcessing CSV: {csv_path}")
    print("-" * 60)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        # Validate CSV columns
        required_columns = {'face_image', 'target_image', 'output'}
        if not required_columns.issubset(reader.fieldnames or []):
            print(f"❌ CSV must have columns: {', '.join(required_columns)}")
            print(f"   Found columns: {', '.join(reader.fieldnames or [])}")
            sys.exit(1)

        # Process each row
        rows = list(reader)
        total = len(rows)
        successful = 0
        failed = 0

        for idx, row in enumerate(rows, 1):
            face_image = row['face_image'].strip()
            target_image = row['target_image'].strip()
            output = row['output'].strip()

            print(f"\n[{idx}/{total}] Processing:")
            print(f"  Face: {face_image}")
            print(f"  Target: {target_image}")
            print(f"  Output: {output}")

            success = perform_face_swap(face_image, target_image, output, app, swapper)
            if success:
                successful += 1
            else:
                failed += 1

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Batch face swap tool using InsightFace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example CSV format:
  face_image,target_image,output
  /path/to/face1.jpg,/path/to/target1.jpg,/path/to/output1.jpg
  /path/to/face2.jpg,/path/to/target2.jpg,/path/to/output2.jpg

Example usage:
  python batch_face_swap.py input.csv
        """
    )
    parser.add_argument('csv_file', help='Path to CSV file containing face swap tasks')

    args = parser.parse_args()

    process_csv(args.csv_file)


if __name__ == "__main__":
    main()
