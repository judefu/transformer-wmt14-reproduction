import os
from pathlib import Path
import sentencepiece as spm


project_dir=Path(__file__).resolve().parents[1]
data_dir=Path.home() / "datasets/wmt14/processed/plain"
output_dir=project_dir / "tokenizer"

train_en_path=data_dir / "train.en"
train_de_path=data_dir / "train.de"

model_prefix=output_dir / "wmt14_en_de_bpe_37k"
model_path=model_prefix.with_suffix(".model")
vocab_path=model_prefix.with_suffix(".vocab")


def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_path.exists() or vocab_path.exists():
        raise FileExistsError(
            "Tokenizer files already exist. "
            "Refusing to overwrite them automatically."
        )
    num_threads = min(8, os.cpu_count() or 1)

    spm.SentencePieceTrainer.train(input=[str(train_en_path),str(train_de_path),],model_prefix=str(model_prefix),
                                   model_type="bpe",vocab_size=37_000,character_coverage=1.0,input_sentence_size=2_000_000,
                                   shuffle_input_sentence=True,max_sentence_length=4192,normalization_rule_name="nmt_nfkc",
                                   num_threads=num_threads,hard_vocab_limit=True,pad_id=0,bos_id=1,eos_id=2,unk_id=3,
                                   pad_piece="<pad>",bos_piece="<bos>",eos_piece="<eos>",unk_piece="<unk>")
    
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))

    assert tokenizer.vocab_size() == 37_000
    assert tokenizer.pad_id() == 0
    assert tokenizer.bos_id() == 1
    assert tokenizer.eos_id() == 2
    assert tokenizer.unk_id() == 3

if __name__ == "__main__":
    main()