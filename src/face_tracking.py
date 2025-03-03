import sys, time, os, argparse, glob, subprocess, warnings, cv2, pickle, numpy, json
from scipy import signal
from shutil import rmtree
from scipy.interpolate import interp1d
from scenedetect import VideoStreamCv2, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from faceDetector.s3fd import S3FD

from utils import save_data


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
        start_frame, end_frame = shot[0].get_frames(), shot[1].get_frames()
        if end_frame - start_frame >= args.minTrack:
            scene_faces = faces[start_frame:end_frame]
            all_tracks.extend(track_shot(args, scene_faces))
    save_data(all_tracks, os.path.join(args.savePath, 'tracks'))
    return all_tracks


def main():
    args = argparse.Namespace(
        input_video="/home/tim/Work/nexa/nexa-face-detection/data/out/test_snippet2",
        output_folder="/home/tim/Work/nexa/nexa-face-detection/data/out/test_snippet2",
        facedetScale=0.5,
        minTrack=15,
        numFailedDet=5,
        minFaceSize=50,
        cropScale=0.5,
        start=10,
        duration=60
    )

    args.videoPath = args.input_video
    args.savePath = args.output_folder

    # faces = inference_video(args)

    # load faces
    faces = pickle.load(open(os.path.join(args.savePath, 'faces.pckl'), 'rb'))

    track_faces(args, faces)


if __name__ == '__main__':
    main()






