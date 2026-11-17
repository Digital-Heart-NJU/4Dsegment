"""Route UVC/fiber requests without invoking the segmentation workflow."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run_pipline as pipeline


class UVCPipelineDispatchTests(unittest.TestCase):
    def fixture(self, root, subjects):
        for script in ('code_by_stage/DMACS.py',
                       'model_vol_septum/generate_open_volmesh.py',
                       'model_vol_septum/generate_UVC_fiber.py'):
            path = root / script
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for number in subjects:
            subject = root / 'data' / f'subject_{number}'
            subject.mkdir(parents=True)
            (subject / 'example.nii').touch()

    def run_request(self, root, request, subjects):
        with patch.object(pipeline, '__file__', str(root / 'run_pipline.py')), \
                patch.object(pipeline.shutil, 'which', return_value='/usr/bin/conda'), \
                patch.object(pipeline.subprocess, 'run') as runner, \
                contextlib.redirect_stdout(io.StringIO()):
            result = pipeline.main(['--request', request, '--subject', subjects])
        self.assertEqual(result, 0)
        return runner.call_args_list

    def test_uvc_fiber_subject_range_routes_only_to_current_generator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root, (11, 12))
            calls = self.run_request(root, 'uvc_fiber', '11-12')
            self.assertEqual(len(calls), 2)
            for number, call in zip((11, 12), calls):
                self.assertEqual(call.args[0], [
                    '/usr/bin/conda', 'run', '--no-capture-output', '-n', 'cme-dt-env',
                    'python', '-B', '-u', str(root / 'model_vol_septum/generate_UVC_fiber.py'),
                    '--data-dir', str(root / 'data'), '--subject', f'subject_{number}',
                    '--uvc-fiber', '--frames', 'ED,ES', '--carp-bin-dir', '/opt/openCARP/usr/bin',
                ])
                self.assertEqual(call.kwargs, {'cwd': root, 'check': True})

    def test_all_runs_uvc_after_volumetric_and_refreshes_its_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root, (11,))
            calls = self.run_request(root, 'all', '11')
            self.assertEqual(len(calls), 4)
            scripts = [Path(call.args[0][8]).name for call in calls]
            self.assertEqual(scripts, ['DMACS.py', 'DMACS.py',
                                       'generate_open_volmesh.py', 'generate_UVC_fiber.py'])
            self.assertIn('--force', calls[-1].args[0])
            self.assertIn('--uvc-fiber', calls[-1].args[0])


if __name__ == '__main__':
    unittest.main()
