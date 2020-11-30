#####################################################
##               Read bag from file                ##
#####################################################


import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
# from mmcv import ProgressBar

def parse_args():
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("input", type=str, help="Path to the bag file")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory")
    parser.add_argument("--num_frames", type=int, default=None, 
                        help="Number of frames to stream. Will stream all data if None.")
    parser.add_argument("--show", action='store_true', help='show the video')

    # Parse the command line arguments to an object
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Check if the given file have bag extension
    assert os.path.isfile(args.input), f'file {args.input} does not exit' 
    assert os.path.splitext(args.input)[1] == ".bag", "Only .bag files are accepted"

    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)
    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

    # Start streaming from file
    pipeline.start(config)

    # Create colorizer object
    colorizer = rs.colorizer()

    # Streaming loop
    frame_i=0
    if args.outdir:
        os.makedirs(args.outdir,exist_ok=True)
    
    # Create opencv window to render image in
    if args.show:
        cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

    
    # progress = ProgressBar(task_num=args.num_frames if args.num_frames else 1e5)
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()

        
        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        print(frame_i, depth_color_image.max())
        if args.outdir:
            filename=os.path.join(args.outdir,f'{frame_i}.png')
            cv2.imwrite(filename,depth_color_image)

        if args.show:
            # Render image in opencv window
            cv2.imshow("Depth Stream", depth_color_image)
            key = cv2.waitKey(1)
            # if pressed escape exit program
            if key == 27:
                cv2.destroyAllWindows()
                break

        frame_i+=1
        if args.num_frames and frame_i > args.num_frames:
            break
        # progress.update()