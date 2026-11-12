import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import os
import subprocess
import shutil
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/tf2_overlay.py'
spec = importlib.util.spec_from_file_location('tf2_overlay', SCRIPT)
tf2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tf2)


class TF2OverlayTests(unittest.TestCase):
    def overlay(self, folder):
        root = Path(folder) / 'tf2fix'
        root.mkdir()
        files = {}
        for name in tf2.LIBRARIES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = dict(
            format=1,
            fixed_version='0.25.24',
            libraries=files,
            packages={
                name: dict(version='0.25.24-1jammy.test', architecture='amd64', deb_sha256='1' * 64)
                for name in tf2.PACKAGES
            },
        )
        (root / 'manifest.json').write_text(json.dumps(manifest))
        return root

    def test_verified_pair_provides_project_library_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.overlay(folder)
            self.assertEqual(tf2.verify_overlay(root, 'amd64'), root / 'opt/ros/humble/lib')

    def test_changed_or_missing_runtime_library_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.overlay(folder)
            path = root / tf2.LIBRARIES[0]
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                tf2.verify_overlay(root, 'amd64')
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                tf2.verify_overlay(root, 'amd64')

    def test_wrong_architecture_or_one_package_only_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.overlay(folder)
            with self.assertRaisesRegex(ValueError, 'architecture mismatch'):
                tf2.verify_overlay(root, 'arm64')
            manifest = json.loads((root / 'manifest.json').read_text())
            del manifest['packages'][tf2.PACKAGES[1]]
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'both verified packages'):
                tf2.verify_overlay(root, 'amd64')

    def test_local_configuration_cannot_override_verified_library_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'scripts').mkdir()
            shutil.copy2(SCRIPT.parent / 'env.sh', root / 'scripts/env.sh')
            (root / 'scripts/tf2_overlay.py').write_text('print("/verified/tf2/lib")\n')
            (root / '.env.local').write_text('export LD_LIBRARY_PATH=/configuration/override\n')
            process = subprocess.run(
                [
                    '/bin/bash',
                    '-c',
                    'source "$1"; printf "%s" "$LD_LIBRARY_PATH"',
                    '_',
                    str(root / 'scripts/env.sh'),
                ],
                env=dict(os.environ),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(process.stdout, '/verified/tf2/lib:/configuration/override')

    def test_external_library_symlink_is_rejected_even_if_digest_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.overlay(folder)
            path = root / tf2.LIBRARIES[0]
            external = Path(folder) / 'external.so'
            external.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(external)
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                tf2.verify_overlay(root, 'amd64')


if __name__ == '__main__':
    unittest.main()
