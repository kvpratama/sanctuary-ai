"""Tests for the ensure_dataset function."""

from unittest.mock import MagicMock, patch

import pytest
from langsmith.schemas import Dataset, Example

from src.eval.dataset import DATASET_NAME, ensure_dataset, examples, get_content_hash


def _make_dataset() -> Dataset:
    """Create a minimal mock Dataset."""
    ds = MagicMock(spec=Dataset)
    ds.id = "test-dataset-id"
    ds.name = DATASET_NAME
    return ds


def _make_example_object(external_id: str, content_hash: str | None = None) -> Example:
    """Create a minimal mock Example object with metadata."""
    metadata: dict = {"external_id": external_id}
    if content_hash:
        metadata["content_hash"] = content_hash
    ex = MagicMock(spec=Example)
    ex.id = f"example-id-{external_id}"
    ex.metadata = metadata
    return ex


def test_ensure_dataset_creates_dataset_when_missing() -> None:
    """Creates the dataset if it does not already exist."""
    mock_client = MagicMock()
    mock_client.read_dataset.side_effect = Exception("Dataset not found")
    mock_client.create_dataset.return_value = _make_dataset()
    mock_client.list_examples.return_value = []

    result = ensure_dataset(client=mock_client)

    assert result == DATASET_NAME
    mock_client.create_dataset.assert_called_once_with(dataset_name=DATASET_NAME)


def test_ensure_dataset_uses_existing_dataset() -> None:
    """Does not create the dataset if it already exists."""
    mock_client = MagicMock()
    mock_client.read_dataset.return_value = _make_dataset()
    mock_client.list_examples.return_value = []

    result = ensure_dataset(client=mock_client)

    assert result == DATASET_NAME
    mock_client.create_dataset.assert_not_called()


def test_ensure_dataset_creates_new_examples() -> None:
    """Creates examples that do not exist remotely."""
    mock_client = MagicMock()
    mock_client.read_dataset.return_value = _make_dataset()
    mock_client.list_examples.return_value = []

    ensure_dataset(client=mock_client)

    assert mock_client.create_example.call_count == len(examples)
    mock_client.update_example.assert_not_called()
    mock_client.delete_example.assert_not_called()


def test_ensure_dataset_updates_changed_examples() -> None:
    """Updates an example when the local content differs from remote."""
    mock_client = MagicMock()
    mock_client.read_dataset.return_value = _make_dataset()

    # Simulate one existing example with a different hash
    stale_hash = "stale-hash-value"
    existing = _make_example_object("id_1", stale_hash)
    mock_client.list_examples.return_value = [existing]

    # Derive expected IDs/counts from the fixture data
    local_ids = {ex["metadata"]["external_id"] for ex in examples}
    existing_ids = {e.metadata["external_id"] for e in [existing] if e.metadata}
    expected_creates = local_ids - existing_ids
    expected_deletes = existing_ids - local_ids
    expected_updates = existing_ids & local_ids  # hash differs (stale vs current)

    ensure_dataset(client=mock_client)

    assert mock_client.update_example.call_count == len(expected_updates)
    assert mock_client.create_example.call_count == len(expected_creates)
    assert mock_client.delete_example.call_count == len(expected_deletes)


def test_ensure_dataset_leaves_unchanged_examples_alone() -> None:
    """Does not update or create when content hashes match."""
    mock_client = MagicMock()
    mock_client.read_dataset.return_value = _make_dataset()

    # Simulate all three examples already existing with correct hashes
    existing_list = []
    for ex in examples:
        content_hash = get_content_hash(ex["inputs"], ex["outputs"])
        existing_list.append(
            _make_example_object(ex["metadata"]["external_id"], content_hash)
        )
    mock_client.list_examples.return_value = existing_list

    ensure_dataset(client=mock_client)

    mock_client.update_example.assert_not_called()
    mock_client.create_example.assert_not_called()
    mock_client.delete_example.assert_not_called()


def test_ensure_dataset_deletes_remote_examples_no_longer_local() -> None:
    """Deletes remote examples whose external_id is not in the local list."""
    mock_client = MagicMock()
    mock_client.read_dataset.return_value = _make_dataset()

    # Simulate an extra remote example that doesn't exist locally
    orphaned = _make_example_object("id_orphaned", "some-hash")
    mock_client.list_examples.return_value = [orphaned]

    ensure_dataset(client=mock_client)

    mock_client.delete_example.assert_called_once_with(example_id=orphaned.id)


def test_ensure_dataset_deletes_multiple_orphaned_examples() -> None:
    """Deletes all remote examples not present in local list."""
    mock_client = MagicMock()
    mock_client.read_dataset.return_value = _make_dataset()

    # Simulate multiple orphaned remote examples
    orphaned_1 = _make_example_object("remote_only_1", "hash-1")
    orphaned_2 = _make_example_object("remote_only_2", "hash-2")
    mock_client.list_examples.return_value = [orphaned_1, orphaned_2]

    ensure_dataset(client=mock_client)

    assert mock_client.delete_example.call_count == 2
    mock_client.delete_example.assert_any_call(example_id=orphaned_1.id)
    mock_client.delete_example.assert_any_call(example_id=orphaned_2.id)


def test_ensure_dataset_prints_summary_with_deletion_count(
    capsys: pytest.CaptureFixture,
) -> None:
    """Prints a summary including the number of deleted examples."""
    mock_client = MagicMock()
    mock_client.read_dataset.return_value = _make_dataset()

    # One orphaned example
    orphaned = _make_example_object("remote_only", "hash")
    mock_client.list_examples.return_value = [orphaned]

    ensure_dataset(client=mock_client)

    captured = capsys.readouterr()
    assert "deleted 1 examples" in captured.out


def test_ensure_dataset_creates_and_updates_and_deletes_together() -> None:
    """Handles create, update, and delete in a single sync call."""
    mock_client = MagicMock()
    mock_client.read_dataset.return_value = _make_dataset()

    # Pick one local example to exist with a stale hash (should update)
    stale_external_id = examples[0]["metadata"]["external_id"]
    stale_hash = "wrong-hash"
    stale_example = _make_example_object(stale_external_id, stale_hash)

    # Pick another local example to exist with the correct hash (unchanged)
    unchanged_idx = 1
    unchanged_external_id = examples[unchanged_idx]["metadata"]["external_id"]
    unchanged_hash = get_content_hash(
        examples[unchanged_idx]["inputs"], examples[unchanged_idx]["outputs"]
    )
    unchanged_example = _make_example_object(unchanged_external_id, unchanged_hash)

    # Add an orphaned remote example (should delete)
    orphaned = _make_example_object("remote_extra", "some-hash")
    mock_client.list_examples.return_value = [
        stale_example,
        unchanged_example,
        orphaned,
    ]

    # Derive expected sets from fixture data
    local_ids = {ex["metadata"]["external_id"] for ex in examples}
    existing_ids_from_mock = {
        e.metadata["external_id"]
        for e in mock_client.list_examples.return_value
        if e.metadata
    }
    expected_creates = local_ids - existing_ids_from_mock
    # Only stale_example should update (it's in both sets and has a different hash)
    expected_updates = {stale_external_id}

    ensure_dataset(client=mock_client)

    assert mock_client.update_example.call_count == len(expected_updates)
    assert mock_client.create_example.call_count == len(expected_creates)
    mock_client.delete_example.assert_called_once_with(example_id=orphaned.id)
