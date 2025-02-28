
There is an error with the json output:

frame=46800 fps=3752 q=2.0 Lsize=N/A time=00:31:11.96 bitrate=N/A dup=0 drop=9358 speed= 150x    
VideoManager is deprecated and will be removed.
`base_timecode` argument is deprecated and has no effect.
Traceback (most recent call last):
  File "/home/tim/Work/nexa/nexa-face-detection/src/scene_face_detect.py", line 132, in <module>
    main()
  File "/home/tim/Work/nexa/nexa-face-detection/src/scene_face_detect.py", line 129, in main
    track_faces(args, faces)
  File "/home/tim/Work/nexa/nexa-face-detection/src/scene_face_detect.py", line 107, in track_faces
    if shot[1]['frame_num'] - shot[0]['frame_num'] >= args.minTrack:
TypeError: string indices must be integers
