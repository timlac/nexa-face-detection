import sys, time, os, argparse, glob, subprocess, warnings, cv2, pickle, numpy, json
from scenedetect import VideoStreamCv2, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector

from src.basic_pipeline.bbox_inference import do_side_by_side_inference, plot_centroid_positions, count_faces_per_frame
from src.faceDetector.s3fd import S3FD
from constants import ROOT_DIR
from src.utils import save_data
import numpy as np

def extract_frames(video_path, save_path):
    pyframes_path = os.path.join(save_path, 'pyframes')
    os.makedirs(pyframes_path, exist_ok=True)  # Ensures directory exists

    frame_path = os.path.join(pyframes_path, '%06d.jpg')

    # -qscale:v 2 = audio quality
    # -r 5 = frame rate
    # -async 1 = audio sync
    # -y = overwrite output files
    command = f"ffmpeg -y -i {video_path} -qscale:v 2 -r 5 -async 1 {frame_path}"

    subprocess.call(command, shell=True)

    return frame_path


def scene_detect(video_path, save_path):
    video = VideoStreamCv2(video_path)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())

    scene_manager.detect_scenes(frame_source=video)  # Directly use video
    scene_list = scene_manager.get_scene_list(start_in_scene=True)

    save_data(scene_list, os.path.join(save_path, 'scene'))

    return scene_list


def inference_video(args):
    DET = S3FD(device='cuda')
    flist = sorted(glob.glob(os.path.join(args.savePath, 'pyframes', '*.jpg')))
    dets = []

    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        image_numpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = DET.detect_faces(image_numpy, conf_th=0.9, scales=[args.facedetScale])
        dets.append([{'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]} for bbox in bboxes])
    save_data(dets, os.path.join(args.savePath, 'faces'))
    return dets


def main():
    video_path = os.path.join(ROOT_DIR, "data/videos/test_video.mp4")
    save_path = os.path.join(ROOT_DIR, "data/out/temp")

    args = argparse.Namespace()
    args.input_video = video_path
    args.savePath = save_path
    args.facedetScale = 0.25

    # extract_frames(args.input_video, args.savePath)
    # scene_list = scene_detect(args.input_video, args.savePath)
    #
    # dets = inference_video(args)

    res = do_side_by_side_inference(args)

    output_path = os.path.join(args.savePath, "face_inference.json")
    with open(output_path, "w") as f:
        json.dump(res, f, indent=4)  # Pretty-print for readability
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()