import os
import subprocess
from pathlib import Path
from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource as Source
from launch.substitutions import LaunchConfiguration as C
from launch_ros.actions import Node


def generate_launch_description():
    assets = share('wpr_simulation2')
    pkg = share('home_explore')
    # 直接 ros2 launch 时也校验 TF2，确保所有子进程使用同一修复库。
    roots = [*Path(__file__).resolve().parents, *Path(pkg).parents]
    project = next((p for p in roots if (p / 'scripts/tf2_overlay.py').is_file()), None)
    if project is None:
        raise RuntimeError('找不到项目TF2校验脚本；请通过 scripts/run_all.sh 启动')
    fixed = subprocess.check_output(
        ['/usr/bin/python3', str(project / 'scripts/tf2_overlay.py'), 'check', str(project / 'work/tf2fix')],
        text=True,
    ).strip()
    libraries = os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)
    libraries = [p for p in libraries if p and p != fixed]
    if fixed:
        libraries.insert(0, fixed)
    library_path = os.pathsep.join(libraries)
    description = Path(assets + '/models/wpb_home_explore.urdf').read_text().split('?>', 1)[-1]
    remaps = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    specs = [
        ('nav2_controller', 'controller_server', [('cmd_vel', 'cmd_vel_nav')]),
        ('nav2_smoother', 'smoother_server', []),
        ('nav2_planner', 'planner_server', []),
        ('nav2_behaviors', 'behavior_server', [('cmd_vel', '/vln/cmd_vel_raw')]),
        ('nav2_bt_navigator', 'bt_navigator', []),
        ('nav2_waypoint_follower', 'waypoint_follower', []),
        (
            'nav2_velocity_smoother',
            'velocity_smoother',
            [('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', '/vln/cmd_vel_raw')],
        ),
    ]
    navigation = [
        Node(
            package=p,
            executable=n,
            name=n,
            parameters=[pkg + '/config/nav2.yaml']
            + (
                [
                    {
                        'default_nav_to_pose_bt_xml': pkg + '/config/navigate_forward.xml',
                        'default_nav_through_poses_bt_xml': pkg + '/config/navigate_through_forward.xml',
                    }
                ]
                if n == 'bt_navigator'
                else []
            ),
            remappings=remaps + r,
            output='screen',
        )
        for p, n, r in specs
    ]
    navigation += [
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            parameters=[{'use_sim_time': True, 'autostart': True, 'node_names': [n for _, n, _ in specs]}],
            output='screen',
        )
    ]
    return LaunchDescription(
        [
            SetEnvironmentVariable('LD_LIBRARY_PATH', library_path),
            LogInfo(msg='TF2 runtime: ' + (fixed or 'system version >= 0.25.24')),
            DeclareLaunchArgument('gui', default_value='true'),
            DeclareLaunchArgument('rviz', default_value='true'),
            DeclareLaunchArgument('results_dir', default_value='results'),
            DeclareLaunchArgument(
                'perception_backend', default_value=os.environ.get('EXPLORE_PERCEPTION_BACKEND', 'yolo')
            ),
            SetEnvironmentVariable(
                'GAZEBO_MODEL_PATH', str(Path(assets).parent) + ':' + os.environ.get('GAZEBO_MODEL_PATH', '')
            ),
            SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
            IncludeLaunchDescription(
                Source(share('gazebo_ros') + '/launch/gzserver.launch.py'),
                launch_arguments={'world': assets + '/worlds/home_explore.world', 'verbose': 'false'}.items(),
            ),
            IncludeLaunchDescription(
                Source(share('gazebo_ros') + '/launch/gzclient.launch.py'), condition=IfCondition(C('gui'))
            ),
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                parameters=[{'robot_description': description, 'use_sim_time': True}],
            ),
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-topic',
                    'robot_description',
                    '-entity',
                    'wpb_home',
                    '-x',
                    '-6.0',
                    '-y',
                    '-0.5',
                    '-Y',
                    '0.0',
                ],
                output='screen',
            ),
            Node(
                package='home_explore',
                executable='guard',
                parameters=[{'use_sim_time': True}],
                output='screen',
            ),
            Node(
                package='home_explore',
                executable='scan_adapter',
                parameters=[{'use_sim_time': True}],
                output='screen',
            ),
            Node(
                package='home_explore',
                executable='obstacle_cloud',
                parameters=[{'use_sim_time': True}],
                output='screen',
            ),
            TimerAction(
                period=8.0,
                actions=[
                    Node(
                        package='home_explore',
                        executable='map_adapter',
                        parameters=[{'use_sim_time': True}],
                        output='screen',
                    ),
                    Node(
                        package='slam_toolbox',
                        executable='async_slam_toolbox_node',
                        name='slam_toolbox',
                        parameters=[pkg + '/config/slam.yaml'],
                        remappings=[('/map', '/map_raw')],
                        output='screen',
                    ),
                    *navigation,
                    Node(
                        package='home_explore',
                        executable='agent',
                        parameters=[
                            pkg + '/config/behavior.yaml',
                            {
                                'use_sim_time': True,
                                'results_dir': C('results_dir'),
                                'perception_backend': C('perception_backend'),
                                'model': os.environ.get('VLN_MODEL', 'qwen3.8-flash'),
                            },
                        ],
                        output='screen',
                    ),
                    Node(
                        package='rviz2',
                        executable='rviz2',
                        arguments=['-d', pkg + '/config/explore.rviz'],
                        parameters=[{'use_sim_time': True}],
                        condition=IfCondition(C('rviz')),
                    ),
                ],
            ),
        ]
    )
