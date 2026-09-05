import argparse
import glob
import json
import os
import shutil
from pathlib import Path

RESULTS_GLOB = "results/**/*.per_image.jsonl"


def find_matching_images(word: str, result_files: list[str]) -> set[str]:
    word = word.lower()
    matches = set()
    for fp in result_files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rep = d.get("details", {}).get("representation", "")
                if word in rep.lower():
                    matches.add(d["image_path"])
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Find images whose vibe representation mentions a given word, and copy them to a directory."
    )
    parser.add_argument("word", help="Word to search for in vibe representations (case-insensitive)")
    parser.add_argument(
        "--result-files",
        nargs="+",
        default=None,
        help="Specific .per_image.jsonl files to search (default: all files under results/)",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination directory to copy matching images into (default: data/<word>)",
    )
    args = parser.parse_args()

    result_files = args.result_files or glob.glob(RESULTS_GLOB, recursive=True)
    dest_dir = Path(args.dest or f"data/{args.word.lower()}")

    matches = find_matching_images(args.word, result_files)

    dest_dir.mkdir(parents=True, exist_ok=True)
    copied, missing = 0, []
    for img in sorted(matches):
        if os.path.exists(img):
            shutil.copy2(img, dest_dir / Path(img).name)
            copied += 1
        else:
            missing.append(img)

    print(f"Searched {len(result_files)} result file(s) for '{args.word}'")
    print(f"Matched {len(matches)} unique image(s)")
    print(f"Copied {copied} to {dest_dir}")
    if missing:
        print(f"Missing (not copied): {missing}")


if __name__ == "__main__":
    main()
