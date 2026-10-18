from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # right_lidar_body_filter = Node(
    #     package='pcl_ros',
    #     executable='filter_crop_box_node',
    #     name='right_lidar_body_filter',
    #     parameters=[
    #         {
    #             'min_x': -0.2,
    #             'max_x': 1.7,
    #             'min_y': -0.9,
    #             'max_y': 0.5,
    #             'min_z': -0.8,
    #             'max_z': 0.3,
    #             'negative': True,
    #         }
    #     ],
    #     remappings=[
    #         ('input', '/ouster_right/points'),
    #         ('output', '/ouster_right/points_filtered'),
    #     ],
    # )

    outter_lidar_body_filter = Node(
        package='pcl_ros',
        executable='filter_crop_box_node',
        name='outter_lidar_body_filter',
        parameters=[
            {
                'min_y': -2.0,
                'max_y': 2.0,
                'min_x': -3.0,
                'max_x': 1.7,
                'min_z': -3.0,
                'max_z': 3.0,
                'negative': True,
            }
        ],
        remappings=[
            ('input', '/os/merged_cloud'),
            ('output', '/os/merged_cloud_filtered_temp'),
        ],
    )

    bug_lidar_body_filter = Node(
        package='pcl_ros',
        executable='filter_crop_box_node',
        name='bug_lidar_body_filter',
        parameters=[
            {
                'min_y': -0.9,
                'max_y': 0.9,
                'min_x': 2.3,
                'max_x': 3.5,
                'min_z': -0.2,
                'max_z': 0.6,
                'negative': True,
            }
        ],
        remappings=[
            ('input', '/os/merged_cloud_filtered_temp'),
            ('output', '/os/merged_cloud_filtered'),
        ],
    )

    return LaunchDescription([
        left_lidar_body_filter,
        bug_lidar_body_filter,
    ])