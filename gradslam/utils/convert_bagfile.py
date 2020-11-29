#####################################################
##               Read bag from file                ##
#####################################################


import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os


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
    if os.path.isfile(args.input) and os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)
    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # Start streaming from file
    pipeline.start(config)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    
    # Create colorizer object
    colorizer = rs.colorizer()

    # Streaming loop
    frame_i=0
    if args.outdir:
        os.makedirs(args.outdir,exist_ok=True)
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

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