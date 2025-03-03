import os, subprocess, glob
from scenedetect import VideoStreamCv2, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector

from utils import save_data


def extract_frames(video_path, save_path):
    pyframes_path = os.path.join(save_path, 'pyframes')
    os.makedirs(pyframes_path, exist_ok=True)  # Ensures directory exists

    frame_path = os.path.join(pyframes_path, '%06d.jpg')

    subprocess.call(f"ffmpeg -y -i {video_path} -qscale:v 2 -r 5 {frame_path}", shell=True)

    return frame_path


def scene_detect(video_path, save_path):
    video = VideoStreamCv2(video_path)  # No need to open()
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())

    scene_manager.detect_scenes(frame_source=video)  # Directly use video
    scene_list = scene_manager.get_scene_list(start_in_scene=True)

    save_data(scene_list, os.path.join(save_path, 'scene'))

    return scene_list


def main():
    video_path = '/media/tim/TIMS-DISK/kosmos/snippets/output_downsampled.mp4'
    save_path = '../data/out/test_snippet2'

    extract_frames(video_path, save_path)

    scene_list = scene_detect(video_path, save_path)

    print('List of scenes obtained:')
    for i, scene in enumerate(scene_list):
        print(f'Scene {i}: Start Frame {scene[0].get_frames()}, End Frame {scene[1].get_frames()}')

if __name__ == '__main__':
    main()
