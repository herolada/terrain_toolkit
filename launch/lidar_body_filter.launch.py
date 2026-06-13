from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    lidar_body_filter = Node(
        package='pcl_ros',
        executable='filter_crop_box_node',
        name='lidar_body_filter',
        parameters=[
            {
                'min_x': -0.2,
                'max_x': 2.0,
                'min_y': -0.85,
                'max_y': 1.0,
                'min_z': -0.7,
                'max_z': 0.8,
                'negative': True,
            }
        ],
        remappings=[
            ('input', '/all_lidar'),
            ('output', '/all_lidar/filtered'),
        ],
    )

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

    # left_lidar_body_filter = Node(
    #     package='pcl_ros',
    #     executable='filter_crop_box_node',
    #     name='left_lidar_body_filter',
    #     parameters=[
    #         {
    #             'min_x': -0.2,
    #             'max_x': 1.7,
    #             'min_y': -0.5,
    #             'max_y': 0.5,
    #             'min_z': -0.5,
    #             'max_z': 0.3,
    #             'negative': True,
    #         }
    #     ],
    #     remappings=[
    #         ('input', '/ouster_left/points'),
    #         ('output', '/ouster_left/points_filtered'),
    #     ],
    # )

    return LaunchDescription([
        lidar_body_filter,
    ])