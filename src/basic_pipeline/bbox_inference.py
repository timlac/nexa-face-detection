import glob
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json


def get_frame_width_from_images(args):
    """Extracts frame width from the first available image."""
    flist = sorted(glob.glob(os.path.join(args.savePath, 'pyframes', '*.jpg')))

    if not flist:
        raise ValueError("No frames found in the specified directory.")

    image = cv2.imread(flist[0])  # Read the first frame
    frame_width = image.shape[1]  # Extract width (shape: height, width, channels)

    print(f"Detected Frame Width: {frame_width}")
    return frame_width


def do_side_by_side_inference(args):
    """Processes bounding boxes and determines their position relative to frame width."""
    dets = pickle.load(open(os.path.join(args.savePath, 'faces.pckl'), 'rb'))
    frame_width = get_frame_width_from_images(args)

    results = []

    for detection in dets:

        left = []
        right = []

        for face in detection:
            bbox = face["bbox"]
            x_centroid = (bbox[0] + bbox[2]) / 2  # (x_min + x_max) / 2
            position = "RIGHT" if x_centroid > (frame_width / 2) else "LEFT"

            item = {
                "frame": face["frame"],
                "centroid_x": round(x_centroid, 2),
                "position": position,
                "bbox": bbox,
                "confidence": round(face["conf"], 4)
            }


            if position == "LEFT":
                left.append(item)
            else:
                right.append(item)

        results.append({"left": left, "right": right})

    # Save results
    output_file = os.path.join(args.savePath, "face_inference.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")

    return results


def count_left_right_faces(results):
    # TODO: Need to improve this
    left_numbers = []
    right_numbers = []
    for i in results:
        left_numbers.append(len(i["left"]))

        right_numbers.append(len(i["right"]))

    left_arr = np.array(left_numbers)
    right_arr = np.array(right_numbers)

    print(np.unique(left_arr, return_counts=True))
    print(np.unique(right_arr, return_counts=True))



def count_faces_per_frame(args):
    dets = pickle.load(open(os.path.join(args.savePath, 'faces.pckl'), 'rb'))

    """Counts the number of detected faces per frame and returns a list."""
    face_counts = [len(frame_faces) for frame_faces in dets]
    return face_counts



def plot_centroid_positions(results):
    """Plots centroid values for LEFT and RIGHT positions over time."""
    frames = [res["frame"] for res in results]
    centroids = [res["centroid_x"] for res in results]
    positions = [res["position"] for res in results]

    # Split left/right
    left_centroids = [c if p == "LEFT" else None for c, p in zip(centroids, positions)]
    right_centroids = [c if p == "RIGHT" else None for c, p in zip(centroids, positions)]

    plt.figure(figsize=(10, 5))
    plt.scatter(frames, left_centroids, color='blue', label="Left", alpha=0.7)
    plt.scatter(frames, right_centroids, color='red', label="Right", alpha=0.7)

    plt.axhline(xmin=0, xmax=max(frames), linestyle="--", color="gray", alpha=0.5, label="Midpoint")

    plt.xlabel("Frame Number")
    plt.ylabel("Centroid X-Position")
    plt.title("Centroid Positions Over Time")
    plt.legend()
    plt.grid(True)

    plt.show()
