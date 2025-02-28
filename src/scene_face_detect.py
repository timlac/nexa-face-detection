import sys, time, os, argparse, glob, subprocess, warnings, cv2, pickle, numpy, json
from scipy import signal
from shutil import rmtree
from scipy.interpolate import interp1d
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
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
parser.add_argument('--output_format', type=str, choices=['pckl', 'json'], default='pckl', help='Output format for results (pckl or json)')
args = parser.parse_args()

args.videoPath = args.input_video
args.savePath = args.output_folder

def save_data(data, path, output_format):
    if output_format == 'json':
        with open(path + '.json', 'w') as fil:
            json.dump(data, fil, default=str, indent=4)
    else:
        with open(path + '.pckl', 'wb') as fil:
            import pickle
            pickle.dump(data, fil)


def scene_detect(args):
    videoManager = VideoManager([args.videoPath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    save_data(sceneList, os.path.join(args.savePath, 'scene'), args.output_format)
    return sceneList


def inference_video(args):
    DET = S3FD(device='cuda')
    flist = sorted(glob.glob(os.path.join(args.savePath, 'pyframes', '*.jpg')))
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
        dets.append([{'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]} for bbox in bboxes])
    save_data(dets, os.path.join(args.savePath, 'faces'), args.output_format)
    return dets


def bb_intersection_over_union(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def track_shot(args, sceneFaces):
    iouThres = 0.5
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if not track:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    if bb_intersection_over_union(face['bbox'], track[-1]['bbox']) > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
        if not track:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f['frame'] for f in track])
            bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = numpy.stack([interp1d(frameNum, bboxes[:, ij])(frameI) for ij in range(4)], axis=1)
            if max(numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                   numpy.mean(bboxesI[:, 3] - bboxesI[:, 1])) > args.minFaceSize:
                tracks.append({'frame': frameI, 'bbox': bboxesI})
    return tracks


def track_faces(args, faces):
    allTracks = []
    sceneList = json.load(open(os.path.join(args.savePath, 'scene.json'))) if args.output_format == 'json' \
        else pickle.load(open(os.path.join(args.savePath, 'scene.pckl'), 'rb'))

    for shot in sceneList:
        if shot[1]['frame_num'] - shot[0]['frame_num'] >= args.minTrack:
            allTracks.extend(track_shot(args, faces[shot[0]['frame_num']:shot[1]['frame_num']]))

    save_data(allTracks, os.path.join(args.savePath, 'tracks'), args.output_format)
    return allTracks


def main():
    os.makedirs(args.savePath, exist_ok=True)
    os.makedirs(os.path.join(args.savePath, 'pyframes'), exist_ok=True)

    # Extract frames
    framePath = os.path.join(args.savePath, 'pyframes', '%06d.jpg')
    subprocess.call(f"ffmpeg -y -i {args.videoPath} -qscale:v 2 -r 25 {framePath}", shell=True)

    # Scene detection
    scene_detect(args)

    # Face detection
    faces = inference_video(args)

    # Face tracking
    track_faces(args, faces)

if __name__ == '__main__':
    main()
