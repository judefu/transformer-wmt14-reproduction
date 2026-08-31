import re
from html import unescape
from itertools import zip_longest
from pathlib import Path


dataset_dir = Path.home() / "datasets/wmt14"
output_dir = dataset_dir / "processed/plain"

segment_pattern = re.compile(
    r'<seg id="([^"]+)">\s*(.*?)\s*</seg>'
)


def normalize_record(text):
    return text.replace("\r", " ").strip()


def prepare_plain_pair(
    en_input,
    de_input,
    en_output,
    de_output,
    expected_count,
):
    count = 0

    with en_input.open(
        encoding="utf-8", newline="\n"
    ) as en_file, de_input.open(
        encoding="utf-8", newline="\n"
    ) as de_file, en_output.open(
        "w", encoding="utf-8", newline="\n"
    ) as en_out, de_output.open(
        "w", encoding="utf-8", newline="\n"
    ) as de_out:

        for line_number, (en_line, de_line) in enumerate(
            zip_longest(en_file, de_file),
            start=1,
        ):
            if en_line is None or de_line is None:
                raise ValueError(
                    f"Line count mismatch at {line_number}"
                )

            en_text = normalize_record(en_line)
            de_text = normalize_record(de_line)

            if not en_text or not de_text:
                raise ValueError(
                    f"Empty evaluation sentence at {line_number}"
                )

            en_out.write(en_text + "\n")
            de_out.write(de_text + "\n")
            count += 1

    if count != expected_count:
        raise ValueError(
            f"Expected {expected_count} pairs, got {count}"
        )

    return count


def read_sgm(path):
    segments = []

    with path.open(encoding="utf-8", newline="\n") as file:
        for line_number, line in enumerate(file, start=1):
            match = segment_pattern.search(line)

            if match is None:
                continue

            segment_id = match.group(1)
            text = normalize_record(unescape(match.group(2)))

            if not text:
                raise ValueError(
                    f"{path}: empty segment at source line {line_number}"
                )

            segments.append((segment_id, text))

    return segments


def prepare_sgm_pair(
    en_input,
    de_input,
    en_output,
    de_output,
    expected_count,
):
    en_segments = read_sgm(en_input)
    de_segments = read_sgm(de_input)

    if len(en_segments) != len(de_segments):
        raise ValueError("SGM source/reference counts differ")

    with en_output.open(
        "w", encoding="utf-8", newline="\n"
    ) as en_out, de_output.open(
        "w", encoding="utf-8", newline="\n"
    ) as de_out:

        for position, (en_segment, de_segment) in enumerate(
            zip(en_segments, de_segments),
            start=1,
        ):
            en_id, en_text = en_segment
            de_id, de_text = de_segment

            if en_id != de_id:
                raise ValueError(
                    f"Segment ID mismatch at position {position}: "
                    f"{en_id} != {de_id}"
                )

            en_out.write(en_text + "\n")
            de_out.write(de_text + "\n")

    count = len(en_segments)

    if count != expected_count:
        raise ValueError(
            f"Expected {expected_count} segments, got {count}"
        )

    return count


def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_dir = dataset_dir / "extracted/dev/dev"

    valid_count = prepare_plain_pair(
        dev_dir / "newstest2013.en",
        dev_dir / "newstest2013.de",
        output_dir / "valid.en",
        output_dir / "valid.de",
        expected_count=3000,
    )

    filtered_dir = dataset_dir / "extracted/test_filtered/test"

    filtered_count = prepare_sgm_pair(
        filtered_dir / "newstest2014-deen-src.en.sgm",
        filtered_dir / "newstest2014-deen-ref.de.sgm",
        output_dir / "test_filtered.en",
        output_dir / "test_filtered.de",
        expected_count=2737,
    )

    full_dir = dataset_dir / "extracted/test_full/test-full"

    full_count = prepare_sgm_pair(
        full_dir / "newstest2014-deen-src.en.sgm",
        full_dir / "newstest2014-deen-ref.de.sgm",
        output_dir / "test_full.en",
        output_dir / "test_full.de",
        expected_count=3003,
    )

    print(f"validation pairs:    {valid_count:,}")
    print(f"filtered test pairs: {filtered_count:,}")
    print(f"full test pairs:     {full_count:,}")
    print(f"output directory:    {output_dir}")


if __name__ == "__main__":
    main()