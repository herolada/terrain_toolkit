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
                'min_x': -15.0,
                'max_x': 15.0,
                'min_y': -15.0,
                'max_y': 15.0,
                'min_z': -1.5,
                'max_z': 0.5,
                'negative': False,
                'input_frame': "base_link",
                'output_frame': "base_link",
            }
        ],
        remappings=[
            ('input', '/os2/points'),
            ('output', '/os2/points_filtered'),
        ],
    )

    return LaunchDescription([
        outter_lidar_body_filter,
        # right_lidar_body_filter,
    ])