import sys, time, os, argparse, glob, subprocess, warnings, cv2, pickle, numpy, json

from extract_frames_scenes import extract_frames, scene_detect, rename_frames
from face_tracking import track_faces, inference_video
from pckl2json import convert_pickles_to_json


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Scene & Face Detection")
parser.add_argument('--input_video', type=str, required=True, help='Path to the input video file')
parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder')
parser.add_argument('--facedetScale', type=float, default=0.25, help='Scale factor for face detection')
parser.add_argument('--minTrack', type=int, default=10, help='Min frames for each shot')
parser.add_argument('--numFailedDet', type=int, default=10, help='Missed detections before tracking stops')
parser.add_argument('--minFaceSize', type=int, default=1, help='Minimum face size in pixels')
parser.add_argument('--cropScale', type=float, default=0.40, help='Scale bounding box')
parser.add_argument('--start', type=int, default=0, help='Start time of the video')
parser.add_argument('--duration', type=int, default=0, help='Duration of the video')
parser.add_argument('--frameStep', type=int, default=1, help='Skip frames during extraction')
args = parser.parse_args()

args.videoPath = args.input_video
args.savePath = args.output_folder


def main():
    os.makedirs(args.savePath, exist_ok=True)
    os.makedirs(os.path.join(args.savePath, 'pyframes'), exist_ok=True)

    pyframes_path = os.path.join(args.savePath, 'pyframes')

    # Extract frames
    frame_path = extract_frames(args.videoPath, args.savePath, args.frameStep)
    if args.frameStep > 1:
        rename_frames(pyframes_path, args.frameStep)

    # Scene detection
    scene_list = scene_detect(args.videoPath, args.savePath)

    # Face tracking
    faces = inference_video(args)

    tracks = track_faces(args, faces)

    # Convert pickles to JSON
    json_path = os.path.join(args.savePath, 'out_json')
    os.makedirs(json_path, exist_ok=True)
    convert_pickles_to_json(args.savePath, json_path)

if __name__ == '__main__':
    main()

