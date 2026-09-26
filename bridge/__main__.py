"""WB-WAM deployment entrypoint: python -m bridge."""

from bridge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
