"""Tests for Pydantic models."""

import json

import pytest

from pathfinder_sdk.models import (
    CandidateRecommendation,
    ConfigurationError,
    FetchError,
    ModelLoadError,
    ModelNotFoundError,
    RankingResult,
)


class TestExceptions:
    def test_model_not_found_error(self):
        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError("model missing")

    def test_configuration_error(self):
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("bad config")

    def test_fetch_error(self):
        with pytest.raises(FetchError):
            raise FetchError("network down")

    def test_model_load_error(self):
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("corrupted file")


class TestCandidateRecommendation:
    def test_basic_creation(self):
        cand = CandidateRecommendation(
            rank=1,
            href="https://example.com/privacy",
            text="Privacy Policy",
            score=0.89,
            context_snippet="See our privacy policy",
        )
        assert cand.rank == 1
        assert cand.score == 0.89

    def test_score_bounds(self):
        with pytest.raises(ValueError):
            CandidateRecommendation(rank=1, href="x", text="x", score=1.5)


class TestRankingResult:
    def test_basic_creation(self):
        result = RankingResult(
            task_description="Find privacy policy",
            source_url="https://example.com",
            candidates=[],
            latency_ms=123.4,
            total_links_analyzed=100,
            total_links_after_filter=95,
            model_tier="default",
        )
        assert result.total_links_analyzed == 100
        assert result.model_tier == "default"

    def test_to_json(self):
        result = RankingResult(
            task_description="Find privacy policy",
            source_url="https://example.com",
            candidates=[
                CandidateRecommendation(
                    rank=1,
                    href="https://example.com/privacy",
                    text="Privacy Policy",
                    score=0.89,
                )
            ],
            latency_ms=123.4,
            total_links_analyzed=10,
            total_links_after_filter=8,
            model_tier="default",
        )
        data = json.loads(result.to_json())
        assert data["task_description"] == "Find privacy policy"
        assert len(data["candidates"]) == 1

    def test_to_dict(self):
        result = RankingResult(
            task_description="Find privacy policy",
            source_url="https://example.com",
            candidates=[],
            latency_ms=0.0,
            total_links_analyzed=0,
            total_links_after_filter=0,
            model_tier="default",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["model_tier"] == "default"
