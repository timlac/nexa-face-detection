
# Nexa Face Detection 

## Components 

### Scene detection

Scene detect is run on the raw video, it will output a list of `scenes` that has one start frame and one end frame.

### Frame Extraction

Is currently accomplished with a subprocess call to ffmpeg. There are options to run this with a skip rate parameters,
where only, for example, every 5:th frame is extracted.

### Face detection

The face detection is accomplished with a custom version of S3FD. 

For each extracted frame, we obtain a list of `faces`, with a bounding box for each face detected, along with confidence scores.

### Face Tracking 

The face tracking is accomplished using Intersection over Union (IoU) between the bounding boxes of the faces.

The result of this operation is a number of `tracks`, where each track is a list of frames, and bboxes, 
where the same face is detected.

## Optimization

Since it's desirable for the algorithm to be fast and not consume too much disk space, 
it seems like an obvious improvement to add a `skip_rate` parameter to the face detection.

This comes with several problems in itself such as: 

- **We need to know the exact time stamp of each frame extracted**, this should in theory be trivial to obtain using the video metadata, such as duration and frame rate.
- We need to **harmonize the reduced frame rate with the scene detectio**n, as well as the IOU face tracking.
  - The start and end scene frames are given in the original frame rate, so we need to convert scenes to the reduced frame rate.
  - The IOU tracking needs to be able to handle the reduced frame rate, in the sense that the `numFailedDet` parameters 
need to adjust dynamically to the reduced frame rate. A `skip_rate` of 5, along with a `numFailedDet` of 5, should be equivalent to a `skip_rate` of 1, and a `numFailedDet` of 1.

## Issues

- Face detection gives "false positives" where there are:
  - Dogs or other animals they may be mistaken for faces.
  - Objects (e.g. paintings on the wall) that are not faces, but are detected as faces.
  - Multiple people in the same frame (should not happen often in the case of therapy videos).

The problems can potentially be handled by using face tracking and face clustering, however this:

- Cumbersome to implement 
- Preliminary tests show that face clustering is not 100% reliable, prompting the need to manual inspection of the results.