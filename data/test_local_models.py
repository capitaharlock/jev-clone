"""Tests de `data.local_models` para #T-host-qwen-ladder.

Nunca llaman a Ollama: las etiquetas se inyectan. Lo que fijan:

* QWEN_MODEL y QWEN_VOLUME_MODEL ganan sobre la lista instalada.
* Sin env, el profesor prefiere 3.8 mlx y el volumen un 8B si esta.
* Si no hay 8B, el volumen cae al profesor.
* Un tag Atlas no se elige si hay un 3.8 limpio.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from data import local_models as LM

_CLEAR = {"QWEN_MODEL": "", "QWEN_VOLUME_MODEL": ""}


class Resolve(unittest.TestCase):
    def test_env_teacher_wins(self):
        with patch.dict(os.environ, {"QWEN_MODEL": "forced:teacher"}):
            self.assertEqual(LM.teacher_model(["qwen3:8b"]), "forced:teacher")

    def test_env_volume_wins(self):
        with patch.dict(os.environ, {"QWEN_VOLUME_MODEL": "forced:vol"}):
            self.assertEqual(LM.volume_model(["qwen3:8b"]), "forced:vol")

    def test_teacher_picks_38_mlx_over_atlas_q8(self):
        tags = ["qwen3.6:27b-q8_0", "qwen3.8:27b-mlx"]
        with patch.dict(os.environ, _CLEAR, clear=False):
            self.assertEqual(LM.teacher_model(tags), "qwen3.8:27b-mlx")

    def test_volume_picks_8b_when_present(self):
        tags = ["qwen3.6:27b-q8_0", "qwen3:8b"]
        with patch.dict(os.environ, _CLEAR, clear=False):
            self.assertEqual(LM.volume_model(tags), "qwen3:8b")

    def test_volume_falls_back_to_teacher(self):
        tags = ["qwen3.6:27b-q8_0"]
        with patch.dict(os.environ, _CLEAR, clear=False):
            self.assertEqual(LM.volume_model(tags), "qwen3.6:27b-q8_0")

    def test_empty_tags_keep_canonical_teacher_name(self):
        with patch.dict(os.environ, _CLEAR, clear=False):
            self.assertEqual(LM.teacher_model([]), "qwen3.8:27b-mlx")


if __name__ == "__main__":
    unittest.main()
