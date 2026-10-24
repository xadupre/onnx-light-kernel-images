# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Session-local registration must not change decoding in other evaluators."""

import os
import subprocess
import sys
import unittest

import numpy as np
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_kernel_images.onnx_py._imgpykernels import register_image_kernels
from test_image_formats import TIFF_PACKBITS_DATA, _make_image_decoder_model


class TestSessionRegistration(unittest.TestCase):
    def test_context_and_evaluator_registration(self):
        encoded = np.frombuffer(TIFF_PACKBITS_DATA, dtype=np.uint8)
        for use_context in (False, True):
            with self.subTest(use_context=use_context):
                session = ReferenceEvaluator(_make_image_decoder_model("RGB"))
                context = session._ctx if use_context else session
                register_image_kernels(context=context)
                register_image_kernels(context)
                for _ in range(2):
                    (image,) = session.run(None, {"encoded": encoded})
                    np.testing.assert_array_equal(image, [[[255, 0, 0], [0, 255, 0]]])

    def test_invalid_context(self):
        for context in (object(), 0, "session"):
            with (
                self.subTest(context=context),
                self.assertRaisesRegex(TypeError, "RuntimeContext or ReferenceEvaluator"),
            ):
                register_image_kernels(context)

        class InvalidEvaluator:
            _ctx = object()

        with self.assertRaisesRegex(TypeError, "context._ctx must be a RuntimeContext"):
            register_image_kernels(InvalidEvaluator())

    def test_registration_does_not_leak(self):
        # A fresh process keeps global registrations in other tests from hiding leaks.
        code = """
import numpy as np
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_kernel_images.onnx_py._imgpykernels import register_image_kernels
from test_image_formats import TIFF_PACKBITS_DATA, _make_image_decoder_model

model = _make_image_decoder_model("RGB")
feeds = {"encoded": np.frombuffer(TIFF_PACKBITS_DATA, dtype=np.uint8)}
before = ReferenceEvaluator(model)
expected = before.run(None, feeds)[0].copy()
local = ReferenceEvaluator(model)
register_image_kernels(local)
np.testing.assert_array_equal(local.run(None, feeds)[0], [[[255, 0, 0], [0, 255, 0]]])
np.testing.assert_array_equal(before.run(None, feeds)[0], expected)
np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, feeds)[0], expected)
assert local._ctx.unregister_custom_kernel("ai.onnx", "ImageDecoder")
assert not before._ctx.unregister_custom_kernel("ai.onnx", "ImageDecoder")

# Local registration must not consume the one-time global registration.
register_image_kernels(None)
np.testing.assert_array_equal(
    ReferenceEvaluator(model).run(None, feeds)[0], [[[255, 0, 0], [0, 255, 0]]]
)
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
