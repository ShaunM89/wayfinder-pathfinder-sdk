"""Tests for progress bar control during model download."""

from unittest.mock import MagicMock, patch

from pathfinder_sdk.ranker import BiEncoderRanker


class TestQuietMode:
    @patch("pathfinder_sdk.ranker.snapshot_download")
    @patch("pathfinder_sdk.ranker.SentenceTransformer")
    def test_quiet_true_suppresses_progress(self, mock_st, mock_snapshot):
        mock_snapshot.return_value = "/fake/model"
        mock_st.return_value = MagicMock()

        ranker = BiEncoderRanker(model_tier="default", quiet=True)
        _ = ranker.model

        call_kwargs = mock_snapshot.call_args[1]
        tqdm_class = call_kwargs.get("tqdm_class")
        assert tqdm_class is not None
        # No-op tqdm should not print anything
        with patch("sys.stdout") as mock_stdout:
            inst = tqdm_class(total=10)
            inst.update(5)
            inst.close()
            mock_stdout.write.assert_not_called()

    @patch("pathfinder_sdk.ranker.snapshot_download")
    @patch("pathfinder_sdk.ranker.SentenceTransformer")
    def test_quiet_false_allows_progress(self, mock_st, mock_snapshot):
        mock_snapshot.return_value = "/fake/model"
        mock_st.return_value = MagicMock()

        ranker = BiEncoderRanker(model_tier="default", quiet=False)
        _ = ranker.model

        call_kwargs = mock_snapshot.call_args[1]
        tqdm_class = call_kwargs.get("tqdm_class")
        # quiet=False passes tqdm_class=None -> snapshot_download uses default
        assert tqdm_class is None

    @patch("pathfinder_sdk.ranker.snapshot_download")
    @patch("pathfinder_sdk.ranker.SentenceTransformer")
    def test_default_quiet_is_false(self, mock_st, mock_snapshot):
        mock_snapshot.return_value = "/fake/model"
        mock_st.return_value = MagicMock()

        ranker = BiEncoderRanker(model_tier="default")
        _ = ranker.model

        call_kwargs = mock_snapshot.call_args[1]
        tqdm_class = call_kwargs.get("tqdm_class")
        # Default quiet=False passes tqdm_class=None
        assert tqdm_class is None


class TestPathfinderQuiet:
    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_pathfinder_passes_quiet_to_ranker(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker_class.return_value = mock_ranker

        from pathfinder_sdk.core import Pathfinder

        pf = Pathfinder(quiet=True)
        assert pf._ranker is mock_ranker

        call_kwargs = mock_ranker_class.call_args[1]
        assert call_kwargs.get("quiet") is True
