import os
import json
import glob
from constants import ROOT_DIR
from src.faceDetector.s3fd import S3FD
from src.basic_pipeline.pipe import (
    extract_frames,
    scene_detect,
    inference_video, get_video_metadata,
)

import warnings

warnings.filterwarnings("ignore")


from src.basic_pipeline.bbox_inference import do_side_by_side_inference


def process_videos(input_dir, output_dir):
    """Processes all video files in input_dir and saves results to output_dir."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))  # Adjust extension if needed

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(output_dir, video_name)  # Unique folder for each video

        os.makedirs(save_path, exist_ok=True)  # Ensure subfolder exists

        print(f"Processing {video_path}...")

        args = type('Args', (object,), {})()  # Simple object for args
        args.input_video = video_path
        args.savePath = save_path
        args.facedetScale = 0.25
        args.extractionFrameRate = 5

        get_video_metadata(args)

        extract_frames(args.input_video, args.savePath, args.extractionFrameRate)
        scene_list = scene_detect(args.input_video, args.savePath)

        dets = inference_video(args)

        # Run inference
        res = do_side_by_side_inference(args)


if __name__ == '__main__':
    input_directory = os.path.join(ROOT_DIR, "data/videos")
    output_directory = os.path.join(ROOT_DIR, "data/out")

    process_videos(input_directory, output_directory)
