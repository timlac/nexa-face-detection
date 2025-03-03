import sys, time, os, argparse, glob, subprocess, warnings, cv2, pickle, numpy, json
from scipy import signal
from shutil import rmtree
from scipy.interpolate import interp1d
from scenedetect import VideoStreamCv2, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from faceDetector.s3fd import S3FD

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
args = parser.parse_args()

args.videoPath = args.input_video
args.savePath = args.output_folder

def save_data(data, path):
    with open(path + '.pckl', 'wb') as fil:
        import pickle
        pickle.dump(data, fil)


def scene_detect(args):
    video = VideoStreamCv2(args.videoPath)  # No need to open()
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())

    scene_manager.detect_scenes(frame_source=video)  # Directly use video
    scene_list = scene_manager.get_scene_list(start_in_scene=True)

    print('List of scenes obtained:')
    for i, scene in enumerate(scene_list):
        print(f'Scene {i}: Start Frame {scene[0].get_frames()}, End Frame {scene[1].get_frames()}')
    save_data(scene_list, os.path.join(args.savePath, 'scene'))

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


def bb_intersection_over_union(boxA, boxB):
    x_a, y_a = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    x_b, y_b = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    box_b_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(box_a_area + box_b_area - inter_area)


def track_shot(args, scene_faces):
    iou_thres = 0.5
    tracks = []
    while True:
        track = []
        for frame_faces in scene_faces:
            for face in frame_faces:
                if not track:
                    track.append(face)
                    frame_faces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    if bb_intersection_over_union(face['bbox'], track[-1]['bbox']) > iou_thres:
                        track.append(face)
                        frame_faces.remove(face)
        if not track:
            break
        elif len(track) > args.minTrack:
            frame_num = numpy.array([f['frame'] for f in track])
            bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
            frame_i = numpy.arange(frame_num[0], frame_num[-1] + 1)
            bboxes_i = numpy.stack([interp1d(frame_num, bboxes[:, ij])(frame_i) for ij in range(4)], axis=1)
            if max(numpy.mean(bboxes_i[:, 2] - bboxes_i[:, 0]),
                   numpy.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > args.minFaceSize:
                tracks.append({'frame': frame_i, 'bbox': bboxes_i})
    return tracks


def track_faces(args, faces):
    all_tracks = []
    scene_list = pickle.load(open(os.path.join(args.savePath, 'scene.pckl'), 'rb'))

    for shot in scene_list:
        if shot[1]['frame_num'] - shot[0]['frame_num'] >= args.minTrack:
            all_tracks.extend(track_shot(args, faces[shot[0]['frame_num']:shot[1]['frame_num']]))

    save_data(all_tracks, os.path.join(args.savePath, 'tracks'))
    return all_tracks


def main():
    os.makedirs(args.savePath, exist_ok=True)
    os.makedirs(os.path.join(args.savePath, 'pyframes'), exist_ok=True)

    # Extract frames
    frame_path = os.path.join(args.savePath, 'pyframes', '%06d.jpg')
    subprocess.call(f"ffmpeg -y -i {args.videoPath} -qscale:v 2 -r 25 {frame_path}", shell=True)

    # Scene detection
    scene_detect(args)

    # Face detection
    faces = inference_video(args)

    # Face tracking
    track_faces(args, faces)

if __name__ == '__main__':
    main()
