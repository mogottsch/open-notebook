from unittest.mock import AsyncMock, patch

import pytest

from open_notebook.podcasts.models import SpeakerProfile, with_podcast_tts_capabilities


def test_vibevoice_capability_is_preserved_during_profile_config_injection():
    config = with_podcast_tts_capabilities(
        "microsoft/VibeVoice-1.5B", {"base_url": "http://tts/v1"}
    )

    assert config["single_pass_multi_speaker"] is True


@pytest.mark.asyncio
async def test_vibevoice_enables_single_pass_multispeaker_config():
    profile = SpeakerProfile(
        name="VibeVoice",
        description="Native multi-speaker test",
        voice_model="model:vibevoice",
        speakers=[
            {"name": "Host", "voice_id": "Alice", "backstory": "", "personality": ""},
            {"name": "Expert", "voice_id": "Frank", "backstory": "", "personality": ""},
        ],
    )

    with patch(
        "open_notebook.podcasts.models._resolve_model_config",
        new=AsyncMock(
            return_value=(
                "openai_compatible",
                "microsoft/VibeVoice-1.5B",
                {"base_url": "http://tts/v1"},
            )
        ),
    ):
        provider, model, config = await profile.resolve_tts_config()

    assert provider == "openai_compatible"
    assert model == "microsoft/VibeVoice-1.5B"
    assert config == {
        "base_url": "http://tts/v1",
        "single_pass_multi_speaker": True,
    }
