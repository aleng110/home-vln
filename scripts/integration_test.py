#!/usr/bin/python3
"""Gazebo 集成验证：检查真实运动记录，仅清理本次启动的进程。"""
import argparse
import shutil
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from home_explore.image_utils import decode_ros_image

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {'SUCCEEDED', 'FAILED', 'CANCELLED'}


def same_goal(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    return (
        all(key in a and key in b for key in ('x', 'y'))
        and math.hypot(a['x'] - b['x'], a['y'] - b['y']) < 0.01
        and a.get('kind') == b.get('kind')
    )


def task_motion_metrics(events):
    """独立检查行为轨迹，不以任务状态或模型回复代替运动证据。"""
    translations = []
    adjustments = {}
    coarse_turns = {}
    coarse_episodes = {}
    step = 0
    boundary = 0
    old_sweeps = 0
    positions = {}
    inspection_pans = 0
    for index, event in enumerate(events):
        name = event.get('event')
        if name == 'state' and event.get('state') == 'TARGET_STEP':
            step = event.get('step_index', step + 1)
        if name == 'state' and event.get('state') == 'INSPECTION_ALIGNING':
            inspection_pans += 1
        if event.get('waypoint') == 'local_view_sweep':
            old_sweeps += 1
        if name == 'visual_adjustment':
            key = str(event.get('step_index', step))
            entry = adjustments.setdefault(key, {'count': 0, 'total_angle_rad': 0.0})
            entry['count'] += 1
            entry['total_angle_rad'] += abs(float(event['delta_rad']))
        if name == 'arrival_facing_adjustment':
            key = str(event.get('step_index', step))
            coarse_turns[key] = coarse_turns.get(key, 0) + 1
            episode = key + ':' + str(event.get('arrival_episode', 0))
            coarse_episodes[episode] = coarse_episodes.get(episode, 0) + 1
        if name != 'navigation_translation_start':
            continue
        goal = event.get('goal')
        if goal and goal.get('kind') == 'dock':
            key = str(step)
            positions[key] = positions.get(key, 0) + 1
        window = events[boundary:index]
        look_indices = [
            i
            for i, e in enumerate(window)
            if e.get('event') == 'departure_look' and same_goal(e.get('goal'), goal)
        ]
        ordered = False
        observation_stamp = None
        for look_index in reversed(look_indices):
            observations = [
                (i, e)
                for i, e in enumerate(window)
                if i > look_index
                and e.get('event') == 'departure_observation'
                and e.get('observed') is True
                and isinstance(e.get('stamp'), (int, float))
                and math.isfinite(e['stamp'])
                and e['stamp'] > 0
                and same_goal(e.get('goal'), goal)
            ]
            for observation_index, observation in observations:
                ready = [
                    e
                    for e in window[observation_index + 1 :]
                    if e.get('event') == 'departure_ready' and same_goal(e.get('goal'), goal)
                ]
                if ready:
                    ordered = True
                    observation_stamp = observation.get('stamp')
                    break
            if ordered:
                break
        translations.append(
            {
                'goal': goal,
                'look_observe_ready_ordered': ordered,
                'observation_stamp': observation_stamp,
                'pose': event.get('pose'),
            }
        )
        boundary = index + 1
    for entry in adjustments.values():
        entry['total_angle_deg'] = math.degrees(entry['total_angle_rad'])
    return {
        'translations': translations,
        'translation_count': len(translations),
        'all_departures_observed': all(e['look_observe_ready_ordered'] for e in translations),
        'visual_adjustments_by_step': adjustments,
        'old_local_view_sweeps': old_sweeps,
        'coarse_arrival_turns_by_step': coarse_turns,
        'coarse_arrival_turns_by_episode': coarse_episodes,
        'coarse_budget_ok': all(count <= 1 for count in coarse_episodes.values()),
        'visual_budget_ok': all(
            e['count'] <= 1 and e['total_angle_deg'] <= 20.0 + 1e-5 for e in adjustments.values()
        ),
        'docking_moves_by_step': positions,
        'inspection_moves': sum(e['goal'].get('kind') == 'inspection' for e in translations),
        'inspection_pans': inspection_pans,
        'inspection_has_no_post_return': not any(
            e['goal'].get('kind') == 'inspection_return' for e in translations
        ),
        'successful_verifications': [
            e for e in events if e.get('event') == 'verification' and e.get('success')
        ],
    }


def safe_log_tail(path, lines=25):
    if not path.exists():
        return []
    selected = path.read_text(errors='replace').splitlines()[-lines:]
    return [
        re.sub(
            r'(?i)(?:sk-[a-z0-9_-]+|bearer\s+\S+|(?:api[_-]?key|authorization)\s*[=:]\s*\S+)',
            '[REDACTED]',
            line,
        )
        for line in selected
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', action='store_true')
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument(
        '--observe-raw-images',
        action='store_true',
        help='Diagnostic only: raw image subscriptions can affect transport.',
    )
    parser.add_argument(
        '--probe-delay', type=float, default=10.0, help='Start passive observers after simulator startup.'
    )
    parser.add_argument('--direct-launch', action='store_true', help='Bypass run_all.sh for comparison only.')
    parser.add_argument('--gui', action='store_true', help='Exercise the normal Gazebo client launch path.')
    parser.add_argument('--rviz', action='store_true', help='Exercise the normal RViz launch path.')
    parser.add_argument(
        '--start-delay', type=float, default=0.0, help='Idle before instruction, wall seconds.'
    )
    parser.add_argument(
        '--failure-hold', type=float, default=0.0, help='Keep owned stopped scene briefly for diagnostics.'
    )
    parser.add_argument(
        '--cancel-after-motion',
        action='store_true',
        help='Cancel after forward motion; verify continued stop.',
    )
    parser.add_argument('--instruction', default='找到茶几，到茶几附近，检查茶几上有什么。')
    parser.add_argument(
        '--repeat-instruction', help='A different remembered target can test look-before-move.'
    )
    parser.add_argument('--seconds', type=float, default=180.0)
    parser.add_argument('--domain', type=int, default=66)
    parser.add_argument('--gazebo-uri', default='http://127.0.0.1:11386')
    args = parser.parse_args()
    if not 0 <= args.probe_delay <= 60:
        parser.error('probe-delay must be between 0 and 60 seconds')
    if args.repeat and not args.cloud:
        parser.error('--repeat requires --cloud')
    if args.repeat and args.cancel_after_motion:
        parser.error('repeat and cancellation checks are separate runs')
    os.environ['EXPLORE_ROS_DOMAIN_ID'] = str(args.domain)
    os.environ['EXPLORE_GAZEBO_MASTER_URI'] = args.gazebo_uri
    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    os.environ['GAZEBO_MASTER_URI'] = args.gazebo_uri
    folder = ROOT / 'results' / ('integration_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(parents=True)
    report = dict(
        type='real_wpb_slam_nav2',
        cloud_requested=args.cloud,
        prior_map=False,
        fixed_viewpoints=False,
        ros_domain_id=args.domain,
        gazebo_uri=args.gazebo_uri,
        gui=args.gui,
        rviz=args.rviz,
        start_delay_s=args.start_delay,
        raw_image_observer=args.observe_raw_images,
        probe_delay_s=args.probe_delay,
        launcher='direct' if args.direct_launch else 'run_all.sh',
    )
    log = (folder / 'bringup.log').open('w')
    proc = None
    node = None
    data, poses, states, velocities, task_runs = {}, [], [], [], []

    def keep(key, msg):
        data[key] = (
            json.loads(msg.data) if key in ('cloud_status', 'semantic_status', 'semantic_map') else msg
        )
        if key == 'vel':
            velocities.append([time.monotonic(), msg.linear.x, msg.angular.z])

    def odom(msg):
        data['odom'] = msg
        poses.append([time.time(), msg.pose.pose.position.x, msg.pose.pose.position.y])

    def status(msg):
        value = json.loads(msg.data)
        value['semantic_frames'] = data.get('semantic_status', {}).get('processed_frames', 0)
        data['status'] = value
        if not states or (states[-1]['state'], states[-1].get('run_id')) != (
            value['state'],
            value.get('run_id'),
        ):
            states.append(value.copy())
            print('STATE', value['state'], value['message'], flush=True)

    def wait(predicate, seconds, message):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            if predicate():
                return
            if proc and proc.poll() is not None:
                raise RuntimeError('Bringup exited')
        raise TimeoutError(message)

    def cancel_task():
        client = node.create_client(Trigger, '/vln/cancel')
        assert client.wait_for_service(timeout_sec=4.0), 'Cancel service missing'
        future = client.call_async(Trigger.Request())
        wait(lambda: future.done(), 5.0, 'Cancel timeout')
        wait(lambda: data['status']['state'] == 'CANCELLED', 12.0, 'No cancelled terminal state')

    def stopped():
        return (
            'vel' in data
            and abs(data['vel'].linear.x) < 1e-5
            and abs(data['vel'].angular.z) < 1e-5
            and 'lease' in data
            and data['lease'].data is False
        )

    def terminal_safety():
        wait(stopped, 5.0, 'Terminal velocity nonzero or motion lease still enabled')
        begin = time.monotonic()
        while time.monotonic() - begin < (3.0 if args.cancel_after_motion else 0.6):
            rclpy.spin_once(node, timeout_sec=0.05)
            assert stopped(), 'Motion resumed after terminal stop'
        return True

    def finish_task():
        wait(lambda: data['status']['state'] in TERMINAL, args.seconds, 'Task wait timeout')
        terminal = data['status'].copy()
        terminal_safety()
        run = terminal.get('run_id')
        wait(lambda: (folder / 'tasks' / run / 'summary.json').exists(), 5.0, 'Task summary not finalized')
        path = folder / 'tasks' / run
        events = [
            json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines() if line.strip()
        ]
        metrics = task_motion_metrics(events)
        summary = json.loads((path / 'summary.json').read_text())
        task_runs.append(
            dict(
                run_id=run,
                terminal=terminal,
                summary=summary,
                motion=metrics,
                zero_velocity=True,
                motion_lease_disabled=True,
            )
        )
        assert terminal['state'] != 'FAILED', 'Task failed: ' + str(terminal.get('message'))
        if args.cancel_after_motion:
            assert terminal['state'] == 'CANCELLED', 'Cancellation did not reach terminal state'
        elif args.cloud:
            assert terminal['state'] == 'SUCCEEDED', 'Cloud task failed: ' + str(terminal.get('message'))
            assert metrics[
                'all_departures_observed'
            ], 'Translation occurred without processed pre-departure observation'
            assert metrics['visual_budget_ok'], 'Per-step visual correction budget exceeded'
            assert metrics['coarse_budget_ok'], 'Repeated coarse arrival facing turns'
            assert not any(
                e.get('goal', {}).get('kind') == 'inspection_return' for e in metrics['translations']
            ), 'Unnecessary return after inspection'
            assert metrics['old_local_view_sweeps'] == 0, 'Old alternating local_view_sweep is still active'
            assert metrics['successful_verifications'], 'No successful fresh visual verification recorded'
            if summary.get('inspection_question'):
                assert summary.get('inspection_complete') is True and summary.get(
                    'inspection_evidence'
                ), 'Inspection declared success without a complete visible region'
                wait(lambda: 'inspection' in data, 3.0, 'Complete inspection image was not published')
                proof = json.loads((path / 'inspection.json').read_text())
                image = data['inspection']
                image_stamp = image.header.stamp.sec + image.header.stamp.nanosec * 1e-9
                assert (
                    abs(image_stamp - proof['stamp']) < 1e-6
                ), 'Inspection image does not match verified evidence'
                task_runs[-1]['inspection_image_received'] = True
            assert all(
                e['distance_to_observed_surface'] <= 1.25 + 1e-6 for e in metrics['successful_verifications']
            ), 'Arrival was too far from actual fresh surface'
            assert all(
                e.get('independent_range')
                and e.get('range_stamps', [1, 0])[1] > e.get('range_stamps', [1, 0])[0]
                for e in metrics['successful_verifications']
            ), 'Missing independent two-frame range proof'
        frames = data['semantic_status']['processed_frames']
        wait(
            lambda: data['semantic_status'].get('processed_frames', 0) > frames,
            10.0,
            'Semantic mapping stopped after task terminal state',
        )
        task_runs[-1]['semantic_after_terminal'] = dict(data['semantic_status'])
        inspections = [
            x['semantic_frames']
            for x in states
            if x.get('run_id') == run and x['state'] == 'INSPECTION_VERIFYING'
        ]
        if inspections:
            assert data['semantic_status']['processed_frames'] > min(
                inspections
            ), 'Mapping did not continue through inspection'
            task_runs[-1]['inspection_mapping_progress'] = data['semantic_status']['processed_frames'] - min(
                inspections
            )
        return events

    try:
        command = (
            ['ros2', 'launch', 'home_explore', 'bringup.launch.py']
            if args.direct_launch
            else ['bash', str(ROOT / 'scripts/run_all.sh')]
        )
        command += [
            'gui:=' + str(args.gui).lower(),
            'rviz:=' + str(args.rviz).lower(),
            'results_dir:=' + str(folder / 'tasks'),
        ]
        proc = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        time.sleep(args.probe_delay)
        rclpy.init()
        node = Node('explore_integration_probe')
        for key, topic, typ, qos in [
            (
                'semantic_status',
                '/vln/semantic_map/status',
                String,
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            ),
            (
                'semantic_map',
                '/vln/semantic_map',
                String,
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            ),
            (
                'cloud_status',
                '/vln/obstacle_cloud/status',
                String,
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            ),
            (
                'inspection',
                '/vln/inspection/image',
                Image,
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            ),
            ('scan', '/scan', LaserScan, qos_profile_sensor_data),
            ('map', '/map', OccupancyGrid, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
            ('vel', '/cmd_vel', Twist, 10),
            ('lease', '/vln/motion_enabled', Bool, 10),
        ]:
            node.create_subscription(typ, topic, lambda msg, key=key: keep(key, msg), qos)
        if args.observe_raw_images:
            for key, topic in [('rgb', '/camera/image_raw'), ('depth', '/camera/depth/image_raw')]:
                node.create_subscription(Image, topic, lambda m, key=key: keep(key, m), QoSProfile(depth=1))
        node.create_subscription(Odometry, '/odom', odom, 10)
        node.create_subscription(
            String, '/vln/status', status, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )

        required = ['scan', 'map', 'odom', 'status', 'cloud_status', 'semantic_status', 'semantic_map'] + (
            ['rgb', 'depth'] if args.observe_raw_images else []
        )
        wait(
            lambda: all(key in data for key in required) and data['cloud_status'].get('ready') is True,
            110.0,
            'Sensors/SLAM/agent/obstacle cloud not ready',
        )
        report['launch_pid'] = proc.pid
        if args.start_delay:
            print('IDLE', args.start_delay, flush=True)
            idle = time.monotonic()
            while time.monotonic() - idle < args.start_delay:
                rclpy.spin_once(node, timeout_sec=0.1)
        health = data['cloud_status']
        report.update(
            rgb=[health['width'], health['height'], health['frame']],
            depth=[health['width'], health['height']],
            image_metadata_source='CameraInfo checked against depth in obstacle_cloud',
            obstacle_cloud_start=health,
            map_start=[data['map'].info.width, data['map'].info.height],
            scan_samples=len(data['scan'].ranges),
        )
        assert (
            health['depth_received'] > 0 and health['published'] >= 2
        ), 'Obstacle cloud not actually produced'
        if args.observe_raw_images:
            cv2.imwrite(str(folder / 'camera_start.jpg'), decode_ros_image(data['rgb']))
            assert [data['rgb'].width, data['rgb'].height] == report['depth'], 'RGB-D dimensions mismatch'
        np.save(str(folder / 'initial_scan_ranges.npy'), np.array(data['scan'].ranges))
        before_semantic = data['semantic_status']['processed_frames']
        wait(
            lambda: data['semantic_status'].get('processed_frames', 0) >= before_semantic + 2,
            15.0,
            'Semantic mapping did not update while idle',
        )
        report['semantic_idle_start'] = dict(data['semantic_status'])
        first = list(poses[-1][1:])
        pub = node.create_publisher(String, '/vln/instruction', 10)
        wait(lambda: pub.get_subscription_count() > 0, 8.0, 'No instruction subscriber')
        before = data['status'].get('run_id')
        pub.publish(String(data=args.instruction if args.cloud else '__explore__'))
        wait(lambda: data['status'].get('run_id') != before, 10.0, 'Task not accepted')
        if args.cancel_after_motion:
            wait(
                lambda: 'vel' in data and data['vel'].linear.x > 0.02,
                100.0,
                'No forward movement before cancellation',
            )
            cancel_task()
        began = time.monotonic()
        wait(
            lambda: data['status']['state'] in TERMINAL or time.monotonic() - began > args.seconds,
            args.seconds + 10.0,
            'Task wait timeout',
        )
        if data['status']['state'] not in TERMINAL:
            cancel_task()
        finish_task()
        report['terminal'] = data['status'].copy()
        if args.repeat:
            before = data['status']['run_id']
            repeat_instruction = args.repeat_instruction or args.instruction
            report['repeat_instruction'] = repeat_instruction
            pub.publish(String(data=repeat_instruction))
            wait(lambda: data['status'].get('run_id') != before, 10.0, 'Repeat task not accepted')
            try:
                repeat_events = finish_task()
            except TimeoutError:
                cancel_task()
                raise
            report['repeat_terminal'] = data['status'].copy()
            report['repeat_initial_scan'] = any(
                event.get('state') == 'INITIAL_SCAN' for event in repeat_events
            )
            assert not report['repeat_initial_scan'], 'Second task unnecessarily repeated initial scan'
        terminal_safety()
        if args.observe_raw_images:
            cv2.imwrite(str(folder / 'camera_end.jpg'), decode_ros_image(data['rgb']))
        elif task_runs:
            first_images = sorted((folder / 'tasks' / task_runs[0]['run_id']).glob('image_*.jpg'))
            last_images = sorted((folder / 'tasks' / task_runs[-1]['run_id']).glob('image_*.jpg'))
            if first_images:
                shutil.copy2(first_images[0], folder / 'camera_start.jpg')
            if last_images:
                shutil.copy2(last_images[-1], folder / 'camera_end.jpg')
            report['camera_snapshots_source'] = (
                'Agent-saved actual frames; test does not subscribe to raw images'
            )
        report['obstacle_cloud_end'] = data['cloud_status']
        report['semantic_end'] = data['semantic_status']
        report['semantic_confirmed_objects'] = [
            {k: o.get(k) for k in ('id', 'label', 'confirmed', 'voxel_count', 'stable_voxel_count', 'state')}
            for o in data['semantic_map']['objects']
        ]
        arr = np.array(data['map'].data, dtype=np.int16).reshape(
            data['map'].info.height, data['map'].info.width
        )
        image = np.full(arr.shape, 140, np.uint8)
        image[(arr >= 0) & (arr < 25)] = 245
        image[arr >= 65] = 0
        cv2.imwrite(str(folder / 'online_map.png'), np.flipud(image))
        report.update(
            max_displacement_m=max(np.linalg.norm(np.array(pose[1:]) - first) for pose in poses),
            map_end=[data['map'].info.width, data['map'].info.height],
            known_cells=int(np.sum(arr >= 0)),
            zero_velocity=True,
            motion_lease_disabled=True,
        )
        np.save(str(folder / 'odom.npy'), np.array(poses))
        if not args.cancel_after_motion:
            assert report['max_displacement_m'] > 0.35, 'Robot did not move sufficiently'
        report['cancel_after_motion'] = args.cancel_after_motion
        minimum = min((sample[1] for sample in velocities), default=0.0)
        negatives = sum(sample[1] < -1e-4 for sample in velocities)
        report.update(min_linear_x=minimum, negative_linear_x_samples=negatives)
        assert negatives == 0, 'Negative forward velocity observed: ' + str(minimum)
        report['passed'] = True
    except Exception as exc:
        report.update(passed=False, error=str(exc))
        if data.get('status', {}).get('state') not in TERMINAL:
            try:
                cancel_task()
                terminal_safety()
            except Exception as cancel_error:
                report['cleanup_cancel_error'] = str(cancel_error)
        raise
    finally:
        report.update(
            states=states,
            task_runs=task_runs,
            min_linear_x=min((sample[1] for sample in velocities), default=None),
            negative_linear_x_samples=sum(sample[1] < -1e-4 for sample in velocities),
            velocity_sample_count=len(velocities),
        )
        if not report.get('passed'):
            log.flush()
            report['last_status_messages'] = states[-10:]
            report['bringup_log_tail'] = safe_log_tail(folder / 'bringup.log')
        (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print('REPORT', folder / 'report.json', flush=True)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if not report.get('passed') and args.failure_hold:
            print('DIAGNOSTIC_HOLD', min(60.0, args.failure_hold), flush=True)
            begin = time.monotonic()
            while time.monotonic() - begin < min(60.0, args.failure_hold):
                rclpy.spin_once(node, timeout_sec=0.1)
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=8.0)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        log.close()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
