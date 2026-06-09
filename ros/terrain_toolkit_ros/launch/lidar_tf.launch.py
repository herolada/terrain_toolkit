from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Quaternion order for static_transform_publisher: qx qy qz qw

    left_lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_ouster_left_lidar',
        arguments=[
            '--frame-id', 'base_link',
            '--child-frame-id', 'ouster_left_lidar',
            '--x',  '1.2',
            '--y',  '0.75',
            '--z',  '0.95',
            '--qx', '0.1622300095848031',
            '--qy', '0.31177915461393213',
            '--qz', '0.9331045499755749',
            '--qw', '-0.07609915606112987',
        ],
    )

    right_lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_ouster_right_lidar',
        arguments=[
            '--frame-id', 'base_link',
            '--child-frame-id', 'ouster_right_lidar',
            '--x',  '-1.2',
            '--y',  '0.75',
            '--z',  '0.95',
            '--qx', '-0.27243070771376066',
            '--qy', '-0.24008908729445147',
            '--qz', '-0.041314299636547376',
            '--qw', '0.9308232207579688',
        ],
    )

    return LaunchDescription([
        left_lidar_tf,
        right_lidar_tf,
    ])