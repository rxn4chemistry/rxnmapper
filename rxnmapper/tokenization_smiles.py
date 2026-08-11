# coding=utf-8
# made to match the Hugginface transformer interface

import collections
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from .smiles_utils import SMI_REGEX_PATTERN

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


class SmilesTokenizer(PreTrainedTokenizerFast):
    r"""
    Constructs a SmilesTokenizer.

    A SMILES string is split into tokens with the regex from
    :obj:`SMI_REGEX_PATTERN`; every token is then looked up in the vocabulary
    as a whole (there is no subword splitting), and unknown tokens are mapped
    to :obj:`unk_token`.

    Args:
        vocab_file: Path to a SMILES character per line vocabulary file
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs,
    ) -> None:
        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocab file at path '{}'.".format(vocab_file))

        self.vocab_file = vocab_file
        vocab = load_vocab(vocab_file)
        self.highest_unused_index = max(
            [i for i, v in enumerate(vocab.keys()) if v.startswith("[unused")]
        )

        tokenizer = Tokenizer(WordLevel(vocab=dict(vocab), unk_token=unk_token))
        # Inverting the split makes everything the regex does not match a
        # delimiter, which is then removed: the tokens that come out are the
        # regex matches, exactly as with re.findall() on the same pattern.
        tokenizer.pre_tokenizer = pre_tokenizers.Split(
            Regex(SMI_REGEX_PATTERN), behavior="removed", invert=True
        )
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls_token}:0 $A:0 {sep_token}:0",
            pair=f"{cls_token}:0 $A:0 {sep_token}:0 $B:1 {sep_token}:1",
            special_tokens=[
                (cls_token, vocab[cls_token]),
                (sep_token, vocab[sep_token]),
            ],
        )
        tokenizer.decoder = decoders.Fuse()

        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        # Kept for backwards compatibility: the pre-`tokenizers` implementation
        # exposed the plain-Python SMILES splitter under this attribute.
        self.basic_tokenizer = BasicSmilesTokenizer()

    @property
    def vocab_list(self) -> List[str]:
        return list(self.vocab.keys())

    def add_special_tokens_ids_single_sequence(self, token_ids: List[int]) -> List[int]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_single_sequence(self, tokens: List[str]) -> List[str]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        return [self.cls_token] + tokens + [self.sep_token]

    def add_special_tokens_sequence_pair(
        self, token_0: List[str], token_1: List[str]
    ) -> List[str]:
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]
        """
        sep = [self.sep_token]
        cls = [self.cls_token]
        return cls + token_0 + sep + token_1 + sep

    def add_special_tokens_ids_sequence_pair(
        self, token_ids_0: List[int], token_ids_1: List[int]
    ) -> List[int]:
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def add_padding_tokens(
        self, token_ids: List[int], length: int, right: bool = True
    ) -> List[int]:
        """
        Adds padding tokens to return a sequence of length max_length.
        By  default padding tokens are added to the right of the sequence.
        """
        padding = [self.pad_token_id] * (length - len(token_ids))
        if right:
            return token_ids + padding
        else:
            return padding + token_ids

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """Save the tokenizer vocabulary to a file."""
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "")
                + VOCAB_FILES_NAMES["vocab_file"],
            )
        else:
            vocab_file = save_directory

        index = 0
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(
                            vocab_file
                        )
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class BasicSmilesTokenizer(object):
    """Run basic SMILES tokenization"""

    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN) -> None:
        """Constructs a BasicSMILESTokenizer.

        Args:
            **regex**: SMILES token regex
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text: str) -> List[str]:
        """Basic Tokenization of a SMILES."""
        tokens = [token for token in self.regex.findall(text)]
        return tokens


def load_vocab(vocab_file: str) -> Dict[str, int]:
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab
