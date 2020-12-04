#####################################################
##               Read bag from file                ##
#####################################################


import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
import shutil
from mmcv import ProgressBar

def parse_args():
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("input", type=str, help="Path to the bag file. Set to None to get from Camera")
    parser.add_argument("--depth-resolution", choices=['qvga','vga', 'xga'], default='xga', 
                        help="Depth resolution: qvga(320x240), vga(640x480), xga (1024x768).\
                            If data is loaded from a bag-file, it must be consistent with recording condition.")
    parser.add_argument("--rgb-resolution", choices=['vga', 'xga', 'hd', 'fhd'], default='hd', 
                        help="RGB resolution: vga(960x540), xga(1024x768), hd(1280x720), fhd(1920x1080).\
                            If data is loaded from a bag-file, it must be consistent with recording condition.")
    parser.add_argument("--color-mode", choices=['rgb8','bgr8'], default='rgb8', 
                        help="The color mode when collecting data.\
                            If data is loaded from a bag-file, it must be consistent with recording condition.")                        
    parser.add_argument("--align", choices=['to_depth','to_color'], default='to_color', 
                        help="Align Depth and RGB images together: to_depth(RGB is align to Depth image), and vice versa.\
                            If data is loaded from a bag-file, it must be consistent with recording condition.")
    parser.add_argument("--threshold-distance", type=float, default=None, 
                        help="clipping distance (meter) for visualization. Objects farther than this distance is ignored.")
    parser.add_argument("--num_frames", type=int, default=None, 
                        help="Number of frames to stream. Will stream all data if None.")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory")
    parser.add_argument("--show", action='store_true', help='show the video')

    # Parse the command line arguments to an object
    args = parser.parse_args()
    return args

def setup_pipeline(args):
    # Check if the given file have bag extension
    assert os.path.isfile(args.input), f'file {args.input} does not exit' 
    assert os.path.splitext(args.input)[1] == ".bag", "Only .bag files are accepted"

     # Create pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()

    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)
    # Configure the pipeline to stream the depth & color 
    resolution = {
        'qvga':(320, 240),
        'vga':(640, 480),
        'xga':(1024, 768),
        'hd':(1280, 720),
        'fhd':(1920, 1080)
    }

    depth_resolution = resolution[args.depth_resolution]
    rgb_resolution = resolution[args.rgb_resolution]

    config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, 30)
    color_mode = rs.format.rgb8 if args.color_mode=='rgb8' else rs.format.bgr8
    config.enable_stream(rs.stream.color, rgb_resolution[0], rgb_resolution[1], color_mode, 30)

    # Start streaming from file
    try:
        profile = pipeline.start(config)
    except ValueError:
        print("The config does not match with bag-file recorded conditiion.\
             Please check your depth-resolution,rgb-resolution or color mode,etc")

    clipping_distance=None
    if args.threshold_distance:
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)
        clipping_distance = args.clipping_distance_in_meters / depth_scale
    # Create an align object
    align_to = rs.stream.depth if args.align=='to_depth' else rs.stream.color
    align = rs.align(align_to)

    # Get camera intrinsics
    intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()

    return pipeline, align, clipping_distance, intrinsics

if __name__ == "__main__":
    args = parse_args()

    # Setup camera
    pipeline, align, clipping_distance, intrinsics = setup_pipeline(args)

    if args.outdir:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir,exist_ok=True)

        if os.path.exists(args.outdir):
            shutil.rmtree(args.outdir)

        os.makedirs(args.outdir,exist_ok=True)
        os.makedirs(os.path.join(args.outdir,'depth'),exist_ok=True)
        os.makedirs(os.path.join(args.outdir,'rgb'),exist_ok=True)
        association_file = open(os.path.join(args.outdir,'associations.txt'), 'w+')

        # Write intrinsics file
        intrinsic_file = open(os.path.join(args.outdir,'intrinsics.txt'), 'w+')
        intrinsic_file.write(f'{intrinsics.fx} {intrinsics.fy} {intrinsics.ppx} {intrinsics.ppy}')
        intrinsic_file.close()
        
    # Streaming loop
    frame_i=0
    # progress = ProgressBar(task_num=args.num_frames if args.num_frames else 1e3)
    progress = ProgressBar(task_num=1e1)
    while True:
        frames = pipeline.wait_for_frames()

        # Align the depth frame and color frame
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame() 
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if args.color_mode=='rgb8':
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        if args.outdir:
            filename_depth=os.path.join(args.outdir,f'depth/{frame_i}.png')
            filename_rgb=os.path.join(args.outdir,f'rgb/{frame_i}.png')
            cv2.imwrite(filename_depth,depth_image)
            cv2.imwrite(filename_rgb,color_image)
            association_file.write(str(frame_i) + ' depth/' + str(frame_i) + '.png ' + str(frame_i) + ' rgb/' + str(frame_i) + '.png\n')
            # import pdb; pdb.set_trace()
            # check_depth=cv2.imread(filename_depth,cv2.IMREAD_UNCHANGED)
            # print((depth_image-check_depth).abs().sum())

        if args.show:
            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            mask = (depth_image_3d <0) 
            if clipping_distance:
                mask = mask | (depth_image_3d > clipping_distance)
            bg_removed = np.where(mask, grey_color, color_image)

            # Render images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))
            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1000)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

        frame_i+=1
        if args.num_frames and frame_i > args.num_frames:
            break
        if frame_i % 1e1 == 0:
            # Reset progress bar
            progress.completed=0
            progress.file.flush()
        progress.update()

    association_file.close()