"""Thin CLI entry point.

Contains no business logic; it only wires user input to the SDK facade
(expanded in later phases). Excluded from coverage by design.
"""

from airllm_lab.shared.version import __version__


def main() -> None:
    """Print the version banner (placeholder until the SDK CLI lands)."""
    print(f"airllm-lab v{__version__}")


if __name__ == "__main__":
    main()
