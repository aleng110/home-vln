#!/bin/bash
set -euo pipefail
VLN_SETUP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec /usr/bin/python3 - "$VLN_SETUP_ROOT" "$@" <<'PY'
"""配置离线模型和本机路径；仅在指定 --create-yolo 时创建环境。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

root = Path(sys.argv[1]).resolve()
parser = argparse.ArgumentParser(description='配置本机路径并校验离线模型。')
parser.add_argument('--assets-from', type=Path, help='已有模型目录或项目目录，不下载权重。')
parser.add_argument('--credentials-file', type=Path, help='已有凭据文件，仅保存文件路径。')
parser.add_argument('--create-yolo', action='store_true', help='创建缺失的 yolo 环境，需要联网。')
args = parser.parse_args(sys.argv[2:])

def fail(message):
    raise SystemExit('配置失败：' + message)

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

missing = []
for name in ('nav2_bringup', 'slam_toolbox', 'gazebo_ros', 'robot_state_publisher', 'rmw_cyclonedds_cpp'):
    if not (Path('/opt/ros/humble/share') / name / 'package.xml').exists():
        missing.append(name)
if not Path('/opt/ros/humble/setup.bash').exists() or missing:
    fail('需要 Ubuntu 22.04 / ROS 2 Humble 及包：' + ', '.join(missing or ['ros-humble-desktop'])
         + '。先安装 ROS/Nav2/SLAM/Gazebo 依赖，再运行本脚本。')
if not shutil.which('gzserver') or not shutil.which('colcon'):
    fail('缺少 gzserver 或 colcon。请安装 Gazebo 11 与 python3-colcon-common-extensions。')

settings = {}
local_env = root / 'config/local.env'
if local_env.exists():
    # 只解析白名单路径，不执行旧配置或读取凭据内容。
    for line in local_env.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if len(words) == 1 and '=' in words[0]:
            key, value = words[0].split('=', 1)
            if key in {'YOLO_PYTHON', 'VLN_CREDENTIALS_FILE'}:
                settings[key] = value

credentials = args.credentials_file
if credentials is not None:
    credentials = credentials.expanduser().resolve()
    if not credentials.is_file():
        fail('--credentials-file 文件不存在。不要把 Key 写进命令行参数。')
    settings['VLN_CREDENTIALS_FILE'] = str(credentials)
elif settings.get('VLN_CREDENTIALS_FILE') and not Path(settings['VLN_CREDENTIALS_FILE']).is_file():
    fail('上次保存的凭据路径已失效；使用 --credentials-file 指定文件，或删除 config/local.env 中该路径。')

candidates = [os.environ.get('YOLO_PYTHON'), settings.get('YOLO_PYTHON'),
              str(Path.home() / 'anaconda3/envs/yolo/bin/python'),
              str(Path.home() / 'miniconda3/envs/yolo/bin/python')]
interpreter = next((Path(path).expanduser().absolute() for path in candidates
                    if path and Path(path).expanduser().is_file()
                    and os.access(Path(path).expanduser(), os.X_OK)), None)
if interpreter is None and args.create_yolo:
    conda_candidates = [os.environ.get('CONDA_EXE'), str(Path.home() / 'anaconda3/bin/conda'),
                        str(Path.home() / 'miniconda3/bin/conda'), shutil.which('conda')]
    conda = next((path for path in conda_candidates if path and Path(path).is_file()), None)
    if conda is None:
        fail('未找到 Conda；请先安装 Anaconda/Miniconda，或设置 YOLO_PYTHON 指向已有 yolo 环境。')
    # 保留已有环境，仅显式创建缺失的环境。
    print('显式创建 yolo 环境；按 config/environment.yml 安装，需要网络。', flush=True)
    clean = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'AMENT_PREFIX_PATH',
                'CMAKE_PREFIX_PATH', 'COLCON_PREFIX_PATH', 'DASHSCOPE_API_KEY', 'CLOUD_AI_API_KEY', 'OPENAI_API_KEY'):
        clean.pop(key, None)
    subprocess.run([conda, 'env', 'create', '--file', str(root / 'config/environment.yml')],
                   cwd=root, env=clean, check=True)
    information = json.loads(subprocess.check_output([conda, 'env', 'list', '--json'], env=clean))
    matches = [Path(path) / 'bin/python' for path in information['envs'] if Path(path).name == 'yolo']
    interpreter = next((path for path in matches if path.is_file()), None)
if interpreter is None:
    fail('未找到 yolo 环境。设置 YOLO_PYTHON，或显式运行 bash scripts/setup_local.sh --create-yolo。')
settings['YOLO_PYTHON'] = str(interpreter)

manifest = json.loads((root / 'config/model_assets.json').read_text())
assets = root / 'work/yolo/assets'
assets.mkdir(parents=True, exist_ok=True)
source = args.assets_from.expanduser().resolve() if args.assets_from else assets
if (source / 'work/yolo/assets').is_dir():
    source = source / 'work/yolo/assets'
for item in manifest['assets']:
    name, expected = item['name'], item['sha256']
    target = assets / name
    if target.is_file() and digest(target) == expected:
        print('模型校验通过：' + name)
        continue
    if target.exists():
        fail(name + ' 已存在但 SHA256 不匹配；请移走该文件后重新配置，不自动覆盖。')
    original = source / name
    if not original.is_file():
        fail('缺少离线模型 ' + name + '；用 --assets-from 指向已下载资源目录。官方来源：' + item['source_url'])
    if original.stat().st_size != item['bytes'] or digest(original) != expected:
        fail(name + ' 来源文件大小或 SHA256 不匹配，已拒绝使用。')
    fd, temporary_name = tempfile.mkstemp(prefix='.' + name + '.', dir=assets)
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        try:
            os.link(original, temporary)
        except OSError:
            shutil.copy2(original, temporary)
        if digest(temporary) != expected:
            fail(name + ' 拷贝后的 SHA256 不匹配。')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print('离线模型已安装并校验：' + name)

local_env.write_text('# 仅保存本机路径；Git 和源码包均排除此文件。\n'
                     + ''.join(key + '=' + shlex.quote(value) + '\n' for key, value in sorted(settings.items())))
local_env.chmod(0o600)
probe = r'''
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
import torch
import yaml

# 从同一份环境文件读取版本，避免安装与检查各维护一份清单。
environment = yaml.safe_load(Path(sys.argv[2]).read_text())
requirements = next(d['pip'] for d in environment['dependencies'] if isinstance(d, dict) and 'pip' in d)
problems = []
for item in requirements:
    if '==' not in item:
        continue
    name, expected = item.split('==', 1)
    try:
        actual = metadata.version(name)
        if actual != expected:
            problems.append(name + '=' + actual + '，需要 ' + expected)
    except metadata.PackageNotFoundError:
        problems.append('缺少 ' + name)
if problems:
    raise SystemExit('yolo 核心版本不匹配：' + '；'.join(problems))
clip_requirement = next(item for item in requirements if item.startswith('clip @ '))
commit = clip_requirement.rsplit('@', 1)[-1]
clip_source = json.loads(metadata.distribution('clip').read_text('direct_url.json') or '{}')
if clip_source.get('vcs_info', {}).get('commit_id') != commit:
    raise SystemExit('CLIP 来源不匹配，请按 config/environment.yml 安装。')
scene = yaml.safe_load(Path(sys.argv[1]).read_text())
if str(scene.get('device', '0')) != 'cpu' and not torch.cuda.is_available():
    raise SystemExit('CUDA 不可用，请检查驱动或显式配置 device: cpu。')
print('yolo 核心版本与设备检查通过；Python '+sys.version.split()[0])
'''
try:
    subprocess.run(['bash', str(root / 'scripts/yolo.sh'), 'python', '-c', probe,
                    str(root / 'config/yolo_scene_classes.yaml'), str(root / 'config/environment.yml')], check=True)
except subprocess.CalledProcessError:
    fail('yolo 环境检查未通过；参考 config/environment.yml，未自动修改已有环境。')
if not (root / '.env.local').exists() and not settings.get('VLN_CREDENTIALS_FILE'):
    print('云端凭据尚未指定：复制 .env.example 为本项目 .env.local 并在本机填写。')
print('配置完成。下一步：bash scripts/build.sh；bash scripts/test.sh；bash scripts/run_all.sh')
PY
