
"""Tests for downloading and verifying the AI4I dataset."""

from io import BytesIO
from zipfile import ZipFile

import pytest

from maintain import download_data


@pytest.fixture
def original_dataset():
    """Use the dataset already verified in this project."""

    return download_data.DEFAULT_PATH.read_bytes()


def make_archive(files):
    """Build a small ZIP archive without contacting UCI."""

    buffer = BytesIO()

    with ZipFile(buffer, "w") as archive:
        for filename, contents in files.items():
            archive.writestr(filename, contents)

    return buffer.getvalue()


def test_original_dataset_passes_verification(original_dataset):
    """The expected dataset must match our recorded reference."""

    details = download_data.verify_dataset(original_dataset)

    assert details["sha256"] == download_data.EXPECTED_SHA256
    assert details["rows"] == 10000
    assert details["columns"] == 14
    assert details["failures"] == 339


def test_existing_dataset_is_not_replaced(
    original_dataset, tmp_path, monkeypatch
):
    """An existing valid file should be checked, not downloaded again."""

    destination = tmp_path / "ai4i2020.csv"
    destination.write_bytes(original_dataset)

    def unexpected_download():
        raise AssertionError("Existing dataset should not be downloaded.")

    monkeypatch.setattr(
        download_data, "download_dataset", unexpected_download
    )

    result = download_data.prepare_dataset(destination)

    assert result == destination
    assert destination.read_bytes() == original_dataset


def test_new_dataset_is_downloaded(
    original_dataset, tmp_path, monkeypatch
):
    """A missing dataset should be downloaded and saved after verification."""

    destination = tmp_path / "raw" / "ai4i2020.csv"

    monkeypatch.setattr(
        download_data,
        "download_dataset",
        lambda: original_dataset,
    )

    result = download_data.prepare_dataset(destination)

    assert result == destination
    assert destination.read_bytes() == original_dataset


def test_corrupted_existing_dataset_is_rejected(tmp_path):
    """A corrupted local file must not be silently replaced."""

    destination = tmp_path / "ai4i2020.csv"
    corrupted_contents = b"This is not the original dataset."
    destination.write_bytes(corrupted_contents)

    with pytest.raises(ValueError, match="checksum"):
        download_data.prepare_dataset(destination)

    assert destination.read_bytes() == corrupted_contents


def test_invalid_download_is_never_saved(tmp_path, monkeypatch):
    """Downloaded bytes must pass verification before saving."""

    destination = tmp_path / "raw" / "ai4i2020.csv"

    monkeypatch.setattr(
        download_data,
        "download_dataset",
        lambda: b"Incorrect dataset contents",
    )

    with pytest.raises(ValueError, match="checksum"):
        download_data.prepare_dataset(destination)

    assert not destination.exists()


def test_download_extracts_correct_csv(
    original_dataset, monkeypatch
):
    """The downloader must extract the correct file from the ZIP."""

    archive_bytes = make_archive({
        "documentation.txt": b"Dataset description",
        "ai4i2020.csv": original_dataset,
    })

    def fake_urlopen(url, timeout):
        assert url == download_data.DATA_URL
        return BytesIO(archive_bytes)

    monkeypatch.setattr(
        download_data, "urlopen", fake_urlopen
    )

    result = download_data.download_dataset()

    assert result == original_dataset


def test_download_rejects_missing_csv(monkeypatch):
    """An archive without the expected CSV must fail."""

    archive_bytes = make_archive({
        "other_data.csv": b"Wrong dataset",
    })

    monkeypatch.setattr(
        download_data,
        "urlopen",
        lambda url, timeout: BytesIO(archive_bytes),
    )

    with pytest.raises(ValueError, match="expected CSV"):
        download_data.download_dataset()


def test_download_rejects_duplicate_csv(monkeypatch):
    """An ambiguous archive must not silently select a file."""

    archive_bytes = make_archive({
        "folder_a/ai4i2020.csv": b"First version",
        "folder_b/ai4i2020.csv": b"Second version",
    })

    monkeypatch.setattr(
        download_data,
        "urlopen",
        lambda url, timeout: BytesIO(archive_bytes),
    )

    with pytest.raises(ValueError, match="expected CSV"):
        download_data.download_dataset()
