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
import mmcv 

def parse_args():
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("input", type=str, help="Path to the bag file. Set to None to get from Camera")
    parser.add_argument("--depth-resolution", choices=['qvga','vga', 'xga'], default='xga', 
                        help="Depth resolution: qvga(320x240), vga(640x480), xga (1024x768).\
                            If data is loaded from a bag-file, it must be consistent with recording condition.")
    parser.add_argument("--rgb-resolution", choices=['vga', 'wga', 'xga', 'hd', 'fhd'], default='hd', 
                        help="RGB resolution: vga(640x480), wga(960x540), xga(1024x768), hd(1280x720), fhd(1920x1080).\
                            If data is loaded from a bag-file, it must be consistent with recording condition.")
    parser.add_argument("--color-mode", choices=['rgb8','bgr8'], default='rgb8', 
                        help="The color mode when collecting data.\
                            If data is loaded from a bag-file, it must be consistent with recording condition.")
    parser.add_argument("--pointcloud", '-pcd', action='store_true', help='extract point cloud (pcd)')                                        
    parser.add_argument("--align", choices=['to_depth','to_color'], default='to_color', 
                        help="Align Depth and RGB images together: to_depth(RGB is aligned to Depth image), and vice versa.\
                            If data is loaded from a bag-file, it must be consistent with recording condition.")
    parser.add_argument("--threshold-distance", '-dth', type=float, default=None, 
                        help="clipping distance (meter) for visualization. Objects farther than this distance are ignored.")
    parser.add_argument("--num-frames", '-nf', type=int, default=None, 
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
        'wga':(960, 540),
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

    # Intrinsic parameters
    def get_intrinsics(stream_type):
        sensor_profile = rs.video_stream_profile(profile.get_stream(stream_type))
        sensor_intrinsics = sensor_profile.get_intrinsics()
        attr_list=['coeffs','fx','fy','height', 'width', 'ppx','ppy']
        return {attr:getattr(sensor_intrinsics,attr,'None') for attr in attr_list}
        
    intrinsics = get_intrinsics(align_to)
    color_intrinsics = get_intrinsics(rs.stream.color)
    depth_intrinsics = get_intrinsics(rs.stream.depth)
    if args.outdir:
        mmcv.dump(intrinsics, os.path.join(args.outdir,'intrinsics.json'),)
        mmcv.dump(color_intrinsics, os.path.join(args.outdir,'color_intrinsics.json'))
        mmcv.dump(depth_intrinsics, os.path.join(args.outdir,'depth_intrinsics.json'))
    return pipeline, align, clipping_distance 

if __name__ == "__main__":
    args = parse_args()

    if args.outdir:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir,exist_ok=True)

        if os.path.exists(args.outdir):
            shutil.rmtree(args.outdir)

        os.makedirs(args.outdir,exist_ok=True)
        os.makedirs(os.path.join(args.outdir,'depth'),exist_ok=True)
        os.makedirs(os.path.join(args.outdir,'rgb'),exist_ok=True)
        os.makedirs(os.path.join(args.outdir,'pcd'),exist_ok=True)
        associations = open(os.path.join(args.outdir,'associations.txt'), 'w+')

    # Setup camera
    pipeline, align, clipping_distance = setup_pipeline(args)
    pc = rs.pointcloud()

    # Streaming loop
    frame_i=0
    progress = ProgressBar(task_num=args.num_frames if args.num_frames else 1e3)
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

        if args.pointcloud:
            # convert depth to point cloud
            points = pc.calculate(depth_frame)
            pc.map_to(depth_frame)

        if args.color_mode=='rgb8':
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        if args.outdir:
            filename_depth=os.path.join(args.outdir,f'depth/{frame_i}.png')
            filename_rgb=os.path.join(args.outdir,f'rgb/{frame_i}.png')
            cv2.imwrite(filename_depth, depth_image)
            cv2.imwrite(filename_rgb, color_image)
            if args.pointcloud:
                # Save as ply extension is very heavy
                # filename_pcd=os.path.join(args.outdir,f'pcd/{frame_i}.ply')
                # points.export_to_ply(filename_pcd, depth_frame)
                # Save Pointcloud data as numpy arrays
                filename_pcd=os.path.join(args.outdir,f'pcd/{frame_i}.npy')
                v, t = points.get_vertices(), points.get_texture_coordinates()
                verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
                texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
                with open(filename_pcd, 'wb') as f:
                    np.save(f,verts)
                    np.save(f,texcoords)

            associations.write(str(frame_i) + ' depth/' + str(frame_i) + '.png ' + str(frame_i) + ' rgb/' + str(frame_i) + '.png\n')


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

    associations.close()