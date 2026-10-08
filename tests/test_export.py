import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from chroma import compile


class ExportTests(unittest.TestCase):
    def test_example_cli_seed_and_native_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            states = []
            for index, seed in enumerate((7, 7, 8)):
                output = root / str(index)
                result = subprocess.run(
                    [sys.executable, "scripts/export.py", "--example", "--output", str(output), "--num-tokens", "5", "--seed", str(seed)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertIn("Input: torch.float32 [5]", result.stdout)
                self.assertIn("matches eager PyTorch on 2 inputs", result.stdout)
                self.assertFalse((output / "config.json").exists())
                self.assertEqual(json.loads((output / "export.json").read_text())["seed"], seed)
                states.append(load_file(output / "weights.safetensors"))
            for key in states[0]:
                torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
            self.assertFalse(torch.equal(states[0]["weight"], states[2]["weight"]))
            program = torch.export.load(root / "0/example.pt2")
            reference = load_file(root / "0/reference.safetensors")
            with compile(program, root / "native", blas=False) as native:
                torch.testing.assert_close(torch.from_numpy(native(reference["tokens"])), reference["logits"])
                changed = reference["tokens"].flip(0)
                torch.testing.assert_close(torch.from_numpy(native(changed)), program.module()(changed))


if __name__ == "__main__":
    unittest.main()
