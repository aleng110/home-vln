#!/usr/bin/python3
"""校验并加载项目内的 TF2 修复库，保留系统软件包。"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

FIXED_VERSION = '0.25.24'
PACKAGES = ('ros-humble-tf2', 'ros-humble-tf2-ros')
LIBRARIES = ('opt/ros/humble/lib/libtf2.so', 'opt/ros/humble/lib/libtf2_ros.so')


def command(args, **kwargs):
    env = dict(os.environ, LC_ALL='C')
    return subprocess.check_output(args, env=env, text=True, stderr=subprocess.STDOUT, **kwargs).strip()


def upstream(version):
    return version.split(':')[-1].split('-')[0]


def system_fixed():
    for name in PACKAGES:
        try:
            version = command(['dpkg-query', '-W', '-f=${Version}', name])
        except subprocess.CalledProcessError:
            return False
        if subprocess.run(
            ['dpkg', '--compare-versions', version, 'ge', FIXED_VERSION], check=False
        ).returncode:
            return False
    return True


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_overlay(target, architecture):
    target = Path(target)
    if target.is_symlink():
        raise ValueError('TF2 overlay must not be an external symlink')
    manifest = json.loads((target / 'manifest.json').read_text())
    if manifest.get('format') != 1 or manifest.get('fixed_version') != FIXED_VERSION:
        raise ValueError('Invalid TF2 overlay manifest format/version')
    records = manifest.get('packages', {})
    if set(records) != set(PACKAGES):
        raise ValueError('TF2 overlay must contain both verified packages')
    for name, record in records.items():
        if upstream(record.get('version', '')) != FIXED_VERSION or record.get('architecture') != architecture:
            raise ValueError('TF2 overlay package version/architecture mismatch: ' + name)
        if not re.fullmatch('[0-9a-f]{64}', record.get('deb_sha256', '')):
            raise ValueError('TF2 overlay source package digest is missing')
    files = manifest.get('libraries', {})
    if set(files) != set(LIBRARIES):
        raise ValueError('TF2 overlay library manifest is incomplete')
    for name, expected in files.items():
        path = target / name
        if (
            not path.resolve().is_relative_to(target.resolve())
            or not path.is_file()
            or sha256(path) != expected
        ):
            raise ValueError('TF2 overlay library missing or changed: ' + name)
    return target / 'opt/ros/humble/lib'


def metadata(name, version):
    values = {}
    for line in command(['apt-cache', 'show', name + '=' + version]).split('\n\n')[0].splitlines():
        if ': ' in line and not line.startswith(' '):
            key, value = line.split(': ', 1)
            values[key] = value
    return values


def setup(target, architecture):
    target = Path(target)
    if system_fixed():
        print('System TF2 is >= ' + FIXED_VERSION + '; no local overlay needed.')
        return
    if (target / 'manifest.json').is_file():
        try:
            verify_overlay(target, architecture)
        except (OSError, ValueError, TypeError):
            pass
        else:
            print('Verified existing project-local TF2 ' + FIXED_VERSION + ': ' + str(target))
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError('Refusing to replace a symlinked TF2 overlay')
    # 下载指定版本，解包前按本机已认证的 apt 索引核验 SHA256。
    with tempfile.TemporaryDirectory(prefix='.tf2fix-', dir=target.parent) as folder:
        temporary = Path(folder)
        prefix = temporary / 'prefix'
        prefix.mkdir()
        records = {}
        for name in PACKAGES:
            policy = command(['apt-cache', 'policy', name])
            match = re.search(r'^\s*Candidate:\s*(\S+)', policy, re.M)
            if match is None or upstream(match.group(1)) != FIXED_VERSION:
                raise ValueError(
                    name
                    + ' candidate must be '
                    + FIXED_VERSION
                    + '; update ROS apt metadata or use a fixed system TF2'
                )
            version = match.group(1)
            expected = metadata(name, version)
            if (expected.get('Package'), expected.get('Version'), expected.get('Architecture')) != (
                name,
                version,
                architecture,
            ):
                raise ValueError('Unexpected apt package metadata: ' + name)
            expected_hash = expected.get('SHA256', '')
            if not re.fullmatch('[0-9a-f]{64}', expected_hash):
                raise ValueError('Missing apt SHA256: ' + name)
            download = temporary / name
            download.mkdir()
            command(['apt-get', 'download', name + '=' + version], cwd=download)
            archives = list(download.glob('*.deb'))
            if len(archives) != 1 or sha256(archives[0]) != expected_hash:
                raise ValueError('Downloaded package does not match authenticated apt metadata: ' + name)
            archive = archives[0]
            fields = command(['dpkg-deb', '-f', str(archive), 'Package', 'Version', 'Architecture'])
            actual = dict(line.split(': ', 1) for line in fields.splitlines())
            if (actual.get('Package'), actual.get('Version'), actual.get('Architecture')) != (
                name,
                version,
                architecture,
            ):
                raise ValueError('Downloaded .deb metadata mismatch: ' + name)
            command(['dpkg-deb', '-x', str(archive), str(prefix)])
            records[name] = dict(version=version, architecture=architecture, deb_sha256=expected_hash)
        manifest = dict(
            format=1,
            fixed_version=FIXED_VERSION,
            packages=records,
            libraries={name: sha256(prefix / name) for name in LIBRARIES},
        )
        (prefix / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        verify_overlay(prefix, architecture)
        backup = temporary / 'previous'
        if target.exists():
            target.rename(backup)
        try:
            prefix.rename(target)
        except Exception:
            if backup.exists():
                backup.rename(target)
            raise
        print('Installed verified project-local TF2 ' + FIXED_VERSION + ': ' + str(target))


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ('setup', 'check'):
        raise ValueError('Usage: tf2_overlay.py setup|check PROJECT/work/tf2fix')
    architecture = command(['dpkg', '--print-architecture'])
    target = Path(sys.argv[2])
    if sys.argv[1] == 'setup':
        setup(target, architecture)
    elif system_fixed():
        return
    else:
        try:
            print(verify_overlay(target, architecture))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(
                'TF2 0.25.23 can deadlock Nav2. Run bash scripts/setup_tf2.sh; overlay check failed: '
                + str(exc)
            ) from exc


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as exc:
        print('TF2 setup/check: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
