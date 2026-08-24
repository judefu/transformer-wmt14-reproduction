from collections.abc import Sequence
from pathlib import Path
import sentencepiece as spm


class WMT14Tokenizer:
    def __init__(self, model_path:str):
        self.model_path=Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer model not found: {self.model_path}"
            )
        
        self.processor=spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.vocab_size=self.processor.vocab_size()
        self.pad_id=self.processor.pad_id()
        self.bos_id=self.processor.bos_id()
        self.eos_id=self.processor.eos_id()
        self.unk_id=self.processor.unk_id()
        
    def encode(self,text: str,*,add_bos: bool = False,add_eos: bool = False):
        if not isinstance(text, str):
            raise TypeError(
                f"text must be str, got {type(text).__name__}"
            )

        token_ids = self.processor.encode(text,out_type=int,add_bos=add_bos,
                                          add_eos=add_eos)
        return token_ids

    def decode(self, token_ids: Sequence[int]):
        return self.processor.decode(list(token_ids))