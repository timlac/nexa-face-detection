import os, subprocess, glob

from scenedetect import VideoStreamCv2, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector

from utils import save_data


def extract_frames(video_path, pyframes_path, frame_step=5):
    """
    Extracts every Nth frame from a video and saves them as images.

    Args:
        video_path (str): Path to the input video.
        pyframes_path: Path to the directory where frames are saved.
        frame_step (int): Extract every Nth frame (default is 5).

    Returns:
        str: Path where frames are saved.
    """

    frame_path = os.path.join(pyframes_path, '%06d.jpg')

    # Use 'select' filter to extract every Nth frame
    subprocess.call(
        f"ffmpeg -y -i {video_path} -qscale:v 2 -vf \"select='not(mod(n\,{frame_step}))'\" -vsync vfr {frame_path}",
        shell=True
    )

    return pyframes_path


def rename_frames(pyframes_path, frame_step):
    """
    Renames extracted frames to reflect their original video frame indices, avoiding overwrites.

    Args:
        pyframes_path (str): Path where frames are stored.
        frame_step (int): Step size used in frame extraction.

    Returns:
        None
    """
    frame_files = sorted(os.listdir(pyframes_path))  # Ensure correct order

    # Step 1: Rename files to a temporary name first
    temp_names = {}
    for i, filename in enumerate(frame_files):
        old_path = os.path.join(pyframes_path, filename)
        temp_filename = f"temp_{i:06d}.jpg"
        temp_path = os.path.join(pyframes_path, temp_filename)
        os.rename(old_path, temp_path)
        temp_names[temp_path] = filename  # Store old names

    # Step 2: Rename files to their final corrected names
    for i, temp_path in enumerate(sorted(temp_names.keys())):
        new_filename = f"{(i * frame_step + 1):06d}.jpg"
        new_path = os.path.join(pyframes_path, new_filename)
        os.rename(temp_path, new_path)
        print(f"Renamed {temp_path} to {new_path}")



#
# def extract_frames(video_path, save_path):
#     pyframes_path = os.path.join(save_path, 'pyframes')
#     os.makedirs(pyframes_path, exist_ok=True)  # Ensures directory exists
#
#     frame_path = os.path.join(pyframes_path, '%06d.jpg')
#
#     subprocess.call(f"ffmpeg -y -i {video_path} -qscale:v 2 -r 5 {frame_path}", shell=True)
#
#     return frame_path


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
    video_path = '/media/tim/TIMS-DISK/kosmos/snippets/output2.mp4'
    save_path = '../data/out/test_snippet_timestamps_2'

    pyframes_path = os.path.join(save_path, 'pyframes')
    os.makedirs(pyframes_path, exist_ok=True)

    frame_step = 5

    extract_frames(video_path, pyframes_path, frame_step)

    if frame_step > 1:
        rename_frames(pyframes_path, frame_step)

    scene_list = scene_detect(video_path, save_path)

    print('List of scenes obtained:')
    for i, scene in enumerate(scene_list):
        print(f'Scene {i}: Start Frame {scene[0].get_frames()}, End Frame {scene[1].get_frames()}')


if __name__ == '__main__':
    main()
