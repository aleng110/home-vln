#!/usr/bin/python3
"""打包可迁移源码，排除凭据、模型权重和本机生成文件。"""
from pathlib import Path
import os
import shlex
import zipfile

root = Path(__file__).resolve().parents[1]
destination = root.parent / (root.name + '_source.zip')
skip_parts = {
    'build',
    'install',
    'log',
    'results',
    'deliverables',
    '.git',
    '__pycache__',
    'work',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    '.venv',
    '.conda',
}
skip_names = {'.env', '.env.local', 'local.env', '.DS_Store'}
excluded_extensions = {
    '.pt',
    '.pth',
    '.onnx',
    '.engine',
    '.ts',
    '.ckpt',
    '.safetensors',
    '.pem',
    '.key',
    '.pyc',
    '.pyo',
}
credential_paths = []
if os.environ.get('VLN_CREDENTIALS_FILE'):
    credential_paths.append(Path(os.environ['VLN_CREDENTIALS_FILE']).expanduser().resolve())
local_config = root / 'config/local.env'
if local_config.is_file():
    for line in local_config.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if len(words) == 1 and words[0].startswith('VLN_CREDENTIALS_FILE='):
            credential_paths.append(Path(words[0].split('=', 1)[1]).expanduser().resolve())
with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as package:
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if (
            not path.is_file()
            or set(rel.parts) & skip_parts
            or path.name in skip_names
            or path.resolve() in credential_paths
            or path.name.startswith('.env.')
            and path.name != '.env.example'
            or path.suffix in excluded_extensions
        ):
            continue
        # 拒绝打包指向项目外部的符号链接。
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise ValueError('拒绝打包外部符号链接：' + str(rel))
        package.write(path, Path(root.name) / rel)
print(destination)
