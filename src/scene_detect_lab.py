from scenedetect import VideoStreamCv2, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector


def check_video(video_path):
    video = VideoStreamCv2(video_path)
    frame_count = 0
    while True:
        frame = video.read()
        if frame is None:
            break
        frame_count += 1
    print(f"Total frames: {frame_count}")


def scene_detect(video_path):
    video = VideoStreamCv2(video_path)  # No need to open()
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())

    scene_manager.detect_scenes(frame_source=video)  # Directly use video
    scene_list = scene_manager.get_scene_list(start_in_scene=True)

    print('List of scenes obtained:')
    for i, scene in enumerate(scene_list):
        print(f'Scene {i}: Start Frame {scene[0].get_frames()}, End Frame {scene[1].get_frames()}')
    return scene_list

def main():
    from scenedetect import detect, ContentDetector

    path = '/media/tim/TIMS-DISK/kosmos/snippets/output2.mp4'

    # scene_list = detect(path, ContentDetector())
    # print(scene_list)
    #
    # for i, scene in enumerate(scene_list):
    #     print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
    #         i + 1,
    #         scene[0].get_timecode(), scene[0].get_frames(),
    #         scene[1].get_timecode(), scene[1].get_frames(),))

    # check_video(path)
    scene_detect(path)

if __name__ == '__main__':
    main()
