import argparse
import random
from itertools import zip_longest
from pathlib import Path


def shuffle_parallel_data(
    src_path,
    tgt_path,
    output_src_path,
    output_tgt_path,
    seed,
):
    src_path = Path(src_path)
    tgt_path = Path(tgt_path)
    output_src_path = Path(output_src_path)
    output_tgt_path = Path(output_tgt_path)

    if output_src_path.exists() or output_tgt_path.exists():
        raise FileExistsError(
            "Output file already exists. Remove it explicitly before rerunning."
        )

    sentence_pairs = []
    missing = object()

    with (
        src_path.open("r", encoding="utf-8", newline="") as src_file,
        tgt_path.open("r", encoding="utf-8", newline="") as tgt_file,
    ):
        for line_number, (src_line, tgt_line) in enumerate(
            zip_longest(src_file, tgt_file, fillvalue=missing),
            start=1,
        ):
            if src_line is missing or tgt_line is missing:
                raise ValueError(
                    f"Source and target files have different lengths "
                    f"near line {line_number}."
                )

            sentence_pairs.append((src_line, tgt_line))

    print(f"loaded {len(sentence_pairs)} aligned sentence pairs")

    random_generator = random.Random(seed)
    random_generator.shuffle(sentence_pairs)

    with (
        output_src_path.open("x", encoding="utf-8", newline="") as output_src,
        output_tgt_path.open("x", encoding="utf-8", newline="") as output_tgt,
    ):
        for index, (src_line, tgt_line) in enumerate(sentence_pairs, start=1):
            output_src.write(src_line)
            output_tgt.write(tgt_line)

            if index % 500_000 == 0:
                print(f"written {index} sentence pairs")

    print(f"saved shuffled source: {output_src_path.resolve()}")
    print(f"saved shuffled target: {output_tgt_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Globally shuffle an aligned parallel corpus."
    )
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--output-src", required=True)
    parser.add_argument("--output-tgt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shuffle_parallel_data(
        src_path=args.src,
        tgt_path=args.tgt,
        output_src_path=args.output_src,
        output_tgt_path=args.output_tgt,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()