gradslam.odometry
=================================

.. currentmodule:: gradslam.odometry


gradslam.odometry.base
-------------------------------
.. autoclass:: gradslam.odometry.base.Odometry
	:members:

gradslam.odometry.gradicp
-------------------------------
.. autoclass:: GradICPOdometry
	:members:


gradslam.odometry.groundtruth
-------------------------------
.. autoclass:: GroundTruthOdometry
	:members:


gradslam.odometry.icp
-------------------------------
.. autoclass:: ICPOdometry
	:members:


gradslam.odometry.pcl_odom_utils
-------------------------------
.. autofunction:: gradslam.odometry.pcl_odom_utils.solve_linear_system
.. autofunction:: gradslam.odometry.pcl_odom_utils.gauss_newton_solve
.. autofunction:: gradslam.odometry.pcl_odom_utils.point_to_plane_ICP
.. autofunction:: gradslam.odometry.pcl_odom_utils.point_to_plane_gradICP
.. autofunction:: gradslam.odometry.pcl_odom_utils.downsample_pointclouds
.. autofunction:: gradslam.odometry.pcl_odom_utils.downsample_rgbdimages
