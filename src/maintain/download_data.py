
"""Download and verify the AI4I predictive maintenance dataset."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from maintain.data import PROJECT_ROOT


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

EXPECTED_SHA256 = (
    "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"
)

DEFAULT_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


def verify_dataset(contents):
    """Check the dataset's identity and basic structure."""

    actual_hash = sha256(contents).hexdigest()

    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            "Dataset checksum does not match the original experiment."
        )

    data = pd.read_csv(BytesIO(contents))

    if (
        len(data) != 10000
        or len(data.columns) != 14
        or "Machine failure" not in data.columns
        or int(data["Machine failure"].sum()) != 339
        or int(data.isna().sum().sum()) != 0
    ):
        raise ValueError(
            "Dataset structure does not match the original experiment."
        )

    return {
        "sha256": actual_hash,
        "rows": len(data),
        "columns": len(data.columns),
        "failures": int(data["Machine failure"].sum()),
    }


def download_dataset():
    """Retrieve the original CSV from the UCI archive."""

    print("Downloading AI4I 2020 from the UCI repository...")

    with urlopen(DATA_URL, timeout=60) as response:
        archive_bytes = response.read()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        matches = [
            filename
            for filename in archive.namelist()
            if Path(filename).name.lower() == "ai4i2020.csv"
        ]

        if len(matches) != 1:
            raise ValueError(
                "The UCI archive does not contain the expected CSV."
            )

        return archive.read(matches[0])


def prepare_dataset(destination=DEFAULT_PATH):
    """Verify an existing dataset or download it if missing."""

    destination = Path(destination)

    if destination.exists():
        if not destination.is_file():
            raise ValueError(
                f"Dataset path is not a regular file: {destination}"
            )

        print("Existing dataset found. Verifying...")
        contents = destination.read_bytes()
        downloaded = False
    else:
        contents = download_dataset()
        downloaded = True

    # Verify before writing anything to disk.
    details = verify_dataset(contents)

    if downloaded:
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Never silently replace a dataset created by someone else.
        with destination.open("xb") as file:
            file.write(contents)

    print("\nDATASET VERIFICATION")
    print("-" * 55)
    print(f"SHA-256: {details['sha256']}")
    print(f"Rows: {details['rows']:,}")
    print(f"Columns: {details['columns']}")
    print(f"Machine failures: {details['failures']}")
    print(f"Location: {destination}")
    print(f"Status: {'Downloaded' if downloaded else 'Already present'}")
    print("\nDataset verification passed.")

    return destination


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify the AI4I 2020 dataset."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PATH,
        help="Optional destination for the dataset CSV.",
    )

    args = parser.parse_args()
    prepare_dataset(args.output)


if __name__ == "__main__":
    main()
