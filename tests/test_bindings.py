import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
import torch

from chroma import Program, compile
from chroma._src.program import Array


class BindingTests(unittest.TestCase):
    def test_validation_rebuild_and_runtime_only_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            x = torch.arange(4, dtype=torch.float32)
            layer = torch.nn.Linear(4, 4).eval().requires_grad_(False)
            with compile(torch.export.export(layer, (x,)), directory, blas=False) as first:
                expected = layer(x).numpy()
                np.testing.assert_allclose(first(x), expected, rtol=2e-5, atol=2e-6)
                out = np.empty(4, dtype=np.float32)
                unaligned = np.ndarray((4,), dtype=np.float32, buffer=bytearray(17), offset=1)
                readonly = out.copy()
                readonly.setflags(write=False)
                model = first._model
                assert model is not None
                for inputs, outputs in (
                    ([], [out]),
                    ([x.numpy()], []),
                    ([x.numpy().astype(np.float64)], [out]),
                    ([x.numpy().astype(">f4")], [out]),
                    ([unaligned], [out]),
                    ([x.numpy()[:2]], [out]),
                    ([x.numpy()], [readonly]),
                    ([x.numpy()], [unaligned]),
                    ([x.numpy()], [x.numpy()]),
                ):
                    with self.assertRaises(ValueError):
                        model.run(cast(list[Array], inputs), cast(list[Array], outputs))
                with self.assertRaises(TypeError):
                    model.run(cast(list[Array], [[0, 1, 2, 3]]), [out])
                replacement = torch.nn.Linear(4, 2).eval().requires_grad_(False)
                with compile(torch.export.export(replacement, (x,)), directory, blas=False) as second:
                    np.testing.assert_allclose(second(x), replacement(x).numpy(), rtol=2e-5, atol=2e-6)
                    np.testing.assert_allclose(first(x), expected, rtol=2e-5, atol=2e-6)
                    with Program(directory) as loaded:
                        np.testing.assert_array_equal(loaded(x), second(x))
                    # Recompiling replaces the previous library and weights instead of accumulating them.
                    self.assertEqual(len(list(Path(directory).glob("_chroma_*"))), 1)
                    self.assertEqual(len(list(Path(directory).glob("weights-*.bin"))), 1)
                    np.save(Path(directory) / "expected.npy", second(x))
            code = """
import importlib.abc
import inspect
import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from types import ModuleType
import numpy as np
class RuntimeOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None) -> ModuleSpec | None:
        if fullname.split('.')[0] in {'torch', 'pybind11'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, RuntimeOnly())
from chroma import Program, compile
inspect.signature(compile)
inspect.signature(Program._array)
inspect.signature(Program.__call__)
with Program(sys.argv[1]) as model:
    np.testing.assert_array_equal(model(np.arange(4, dtype=np.float32)), np.load(sys.argv[1] + '/expected.npy'))
"""
            subprocess.run([sys.executable, "-c", code, directory], check=True)
            path = Path(directory) / "model.json"
            manifest = json.loads(path.read_text())
            for incompatible in ({"version": 1}, {"python_abi": "different"}):
                path.write_text(json.dumps(manifest | incompatible))
                with self.assertRaisesRegex(ValueError, "recompile"):
                    Program(directory)


if __name__ == "__main__":
    unittest.main()
