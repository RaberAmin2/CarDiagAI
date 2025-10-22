"""Utility helpers for managing the application's semantic version."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

VERSION_FILE = Path(__file__).resolve().parent / "VERSION"

VersionPart = Literal["major", "minor", "patch"]


@dataclass(frozen=True)
class Version:
    """Simple representation of a semantic version number."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, raw: str) -> "Version":
        """Create a version instance from a string like ``"1.2.3"``."""
        try:
            major, minor, patch = (int(part) for part in raw.strip().split("."))
        except ValueError as exc:  # pragma: no cover - defensive programming
            raise ValueError(
                "VERSION file must contain a semantic version like '1.2.3'."
            ) from exc
        return cls(major, minor, patch)

    def bump(self, part: VersionPart) -> "Version":
        """Return a bumped version without mutating the current instance."""
        if part == "major":
            return Version(self.major + 1, 0, 0)
        if part == "minor":
            return Version(self.major, self.minor + 1, 0)
        if part == "patch":
            return Version(self.major, self.minor, self.patch + 1)
        raise ValueError(f"Unknown version part: {part}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def read_version() -> Version:
    """Read the project's current version from the ``VERSION`` file."""
    if not VERSION_FILE.exists():
        raise FileNotFoundError(
            "Could not find VERSION file. Run the version bump script first."
        )
    return Version.parse(VERSION_FILE.read_text(encoding="utf-8"))


def write_version(version: Version) -> None:
    """Persist the provided version to disk."""
    VERSION_FILE.write_text(f"{version}\n", encoding="utf-8")


def bump_version(part: VersionPart) -> Version:
    """Bump the version by the selected ``part`` and persist the change."""
    current = read_version()
    updated = current.bump(part)
    write_version(updated)
    return updated


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Bump the project's semantic version and persist it."
    )
    parser.add_argument(
        "part",
        choices=("major", "minor", "patch"),
        help="Which part of the semantic version to bump.",
    )

    args = parser.parse_args()
    updated = bump_version(args.part)
    print(f"Version updated to {updated}")


if __name__ == "__main__":
    main()
