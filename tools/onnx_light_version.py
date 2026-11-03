# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Read the pinned onnx-light release used by build and CI tooling."""

import argparse
import os
import re
import tomllib
from pathlib import Path


def onnx_light_version(pyproject):
    with Path(pyproject).open("rb") as stream:
        metadata = tomllib.load(stream)
    versions = []
    for requirements in (
        metadata["project"]["dependencies"],
        metadata["build-system"]["requires"],
    ):
        pins = [
            requirement
            for requirement in requirements
            if re.match(r"onnx-light(?:[<>=!~\s;\[]|$)", requirement)
        ]
        if len(pins) != 1 or not re.fullmatch(r"onnx-light==[0-9]+(?:\.[0-9]+)+", pins[0]):
            raise ValueError("Expected one exact onnx-light release pin in each dependency list.")
        versions.append(pins[0].split("==")[1])
    if versions[0] != versions[1]:
        raise ValueError("Build and runtime onnx-light pins must match.")
    return versions[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject", type=Path, default=Path(__file__).resolve().parents[1] / "pyproject.toml"
    )
    parser.add_argument("--github-env", action="store_true")
    args = parser.parse_args()
    version = onnx_light_version(args.pyproject)
    if args.github_env:
        with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as stream:
            stream.write(f"ONNX_LIGHT_VERSION={version}\n")
            stream.write(
                "PIP_FIND_LINKS=https://github.com/xadupre/onnx-light/"
                f"releases/expanded_assets/{version}\n"
            )
    else:
        print(version)


if __name__ == "__main__":
    main()
