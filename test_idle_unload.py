import os
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QWEN3_TTS_IDLE_UNLOAD_SECONDS", "180")

import openai_server


class IdleUnloadTests(unittest.TestCase):
    def setUp(self):
        openai_server._loaded_models.clear()
        if hasattr(openai_server, "_last_model_access_at"):
            openai_server._last_model_access_at.clear()

    def tearDown(self):
        openai_server._loaded_models.clear()
        if hasattr(openai_server, "_last_model_access_at"):
            openai_server._last_model_access_at.clear()

    def test_idle_timeout_defaults_to_three_minutes_from_env(self):
        self.assertEqual(openai_server.get_idle_unload_seconds(), 180)

    def test_unload_idle_models_removes_expired_models_and_clears_cuda_cache(self):
        model = Mock()
        openai_server._loaded_models[("CustomVoice", "1.7B")] = model
        openai_server._last_model_access_at[("CustomVoice", "1.7B")] = 100.0

        with patch.object(openai_server, "time") as fake_time, \
             patch.object(openai_server, "gc") as fake_gc, \
             patch.object(openai_server.torch.cuda, "is_available", return_value=True), \
             patch.object(openai_server.torch.cuda, "empty_cache") as empty_cache:
            fake_time.time.return_value = 281.0
            unloaded = openai_server.unload_idle_models(now=281.0)

        self.assertEqual(unloaded, [("CustomVoice", "1.7B")])
        self.assertNotIn(("CustomVoice", "1.7B"), openai_server._loaded_models)
        fake_gc.collect.assert_called()
        empty_cache.assert_called_once()

    def test_get_model_reloads_after_idle_unload(self):
        first = Mock(name="first")
        second = Mock(name="second")

        with patch.object(openai_server, "_load_model", side_effect=[first, second]) as load_model:
            with patch.object(openai_server, "time") as fake_time:
                fake_time.time.return_value = 100.0
                self.assertIs(openai_server.get_model("CustomVoice", "1.7B"), first)
                fake_time.time.return_value = 281.0
                openai_server.unload_idle_models(now=281.0)
                self.assertIs(openai_server.get_model("CustomVoice", "1.7B"), second)

        self.assertEqual(load_model.call_count, 2)


if __name__ == "__main__":
    unittest.main()
