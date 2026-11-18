"""Text tokenizers: turn a raw instruction string into integer token ids.

This is the text-side analog of the image transforms. The Trainer instantiates a
tokenizer and hands it to EpisodicDataset (just like img_transforms); the dataset
applies it per sample so tokenization runs in the dataloader workers.

A tokenizer is a single composable callable ``str -> LongTensor[max_len]`` built from
two interchangeable pieces, mirroring mainstream NLP pipelines:

    Tokenizer = splitter (text -> tokens) + mapper (token -> id) + pad/truncate

- Splitters decide *how text is decomposed*: whole words, characters, or raw UTF-8
  bytes.
- Mappers decide *how a token becomes an id*: a stable hash into a fixed-size space
  (no vocabulary needed), or a lookup in a built/loaded corpus vocabulary (the path
  used with real / pretrained embeddings).

Special ids are configurable. Project-built vocabularies use ``<pad>=0`` and
``<unk>=1`` by convention, but an external vocab_file can define different ids; the
trainer resolves them and passes the ids into ``build_tokenizer``.
"""

import zlib
from abc import ABC, abstractmethod
from typing import List, Union

import torch


PAD_ID = 0
UNK_ID = 1  # matches vocab.SPECIALS["<unk>"] for project-built vocabularies

Token = Union[str, int]


# --------------------------------------------------------------------------- #
# Splitters: text -> list of tokens                                           #
# --------------------------------------------------------------------------- #
class Splitter(ABC):
    @abstractmethod
    def __call__(self, text: str) -> List[Token]:
        raise NotImplementedError


class WhitespaceWordSplitter(Splitter):
    """Lowercase then split on whitespace into word tokens."""

    def __call__(self, text: str) -> List[str]:
        return str(text).lower().split()


class CharSplitter(Splitter):
    """Split into individual characters (lowercased)."""

    def __call__(self, text: str) -> List[str]:
        return list(str(text).lower())


class ByteSplitter(Splitter):
    """Split into raw UTF-8 byte values (0..255). Tokens are already integers, so a
    byte-level pipeline can map them directly (HashMapper still applies if combined)."""

    def __call__(self, text: str) -> List[int]:
        return list(str(text).encode("utf-8"))


# --------------------------------------------------------------------------- #
# Mappers: token -> int id                                                    #
# --------------------------------------------------------------------------- #
class Mapper(ABC):
    @abstractmethod
    def __call__(self, token: Token) -> int:
        raise NotImplementedError


class HashMapper(Mapper):
    """Map a token into a fixed hash space while reserving ``pad_id``.

    Hash mode has no unknown-token concept: every token maps to one of the non-padding
    buckets. For the default pad_id=0 this reproduces the original miniVLA behavior:
    CRC32 % (vocab_size - 1) + 1.
    """

    def __init__(self, vocab_size: int, pad_id: int = PAD_ID):
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2 (one id is reserved for padding).")
        if not (0 <= pad_id < vocab_size):
            raise ValueError(f"pad_id={pad_id} must be in [0, {vocab_size}).")
        self.vocab_size = vocab_size
        self.pad_id = pad_id

    def __call__(self, token: Token) -> int:
        key = token if isinstance(token, str) else str(token)
        raw = zlib.crc32(key.encode("utf-8")) % (self.vocab_size - 1)
        # Map into [0, vocab_size) excluding pad_id.
        return raw + 1 if raw >= self.pad_id else raw


class VocabMapper(Mapper):
    """Map a token via a corpus vocabulary ``{token: id}``; unknown tokens -> unk_id."""

    def __init__(self, vocab: dict, unk_id: int = UNK_ID):
        if not vocab:
            raise ValueError("VocabMapper requires a non-empty vocab.")
        # vocab keys are strings; byte/char tokens are stringified to match build_vocab.
        self.vocab = vocab
        self.unk_id = unk_id

    def __call__(self, token: Token) -> int:
        key = token if isinstance(token, str) else str(token)
        return self.vocab.get(key, self.unk_id)


# --------------------------------------------------------------------------- #
# Tokenizer: splitter + mapper + pad/truncate                                 #
# --------------------------------------------------------------------------- #
class Tokenizer:
    """Composable tokenizer: ``str -> LongTensor[max_len]``.

    Splits the text, maps each token to an id, then truncates / right-pads to
    ``max_len`` using the configured ``pad_id``. Behavior is fully determined by the
    splitter and mapper passed in; e.g. ``WhitespaceWordSplitter`` + ``HashMapper``
    gives the hashing tokenizer, while any splitter + ``VocabMapper`` gives a
    corpus/external-vocab tokenizer suitable for pretrained embeddings.
    """

    def __init__(self, splitter: Splitter, mapper: Mapper, max_len: int, pad_id: int = PAD_ID):
        if max_len < 1:
            raise ValueError("max_len must be >= 1.")
        self.splitter = splitter
        self.mapper = mapper
        self.max_len = max_len
        self.pad_id = pad_id

    def __call__(self, text: str) -> torch.Tensor:
        tokens = self.splitter(text)
        ids = [self.mapper(tok) for tok in tokens][: self.max_len]
        ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #
_SPLITTERS = {
    "word": WhitespaceWordSplitter,
    "char": CharSplitter,
    "byte": ByteSplitter,
}


def build_tokenizer(
    split_mode: str,
    map_mode: str,
    max_len: int,
    vocab: dict = None,
    vocab_size: int = None,
    pad_id: int = PAD_ID,
    unk_id: int = UNK_ID,
) -> Tokenizer:
    """Build a Tokenizer from string mode names.

    split_mode: "word" | "char" | "byte"
    map_mode:   "hash" (needs vocab_size) | "vocab" (needs vocab)
    pad_id:     id used for right-padding and excluded from hash buckets
    unk_id:     id returned for OOV tokens in vocab mode
    """
    if split_mode not in _SPLITTERS:
        raise ValueError(
            f"Unknown split_mode '{split_mode}'. Choose from {list(_SPLITTERS)}."
        )
    splitter = _SPLITTERS[split_mode]()

    if map_mode == "hash":
        if vocab_size is None:
            raise ValueError("map_mode='hash' requires vocab_size.")
        mapper = HashMapper(vocab_size, pad_id=pad_id)
    elif map_mode == "vocab":
        if vocab is None:
            raise ValueError("map_mode='vocab' requires a vocab dict.")
        mapper = VocabMapper(vocab, unk_id=unk_id)
    else:
        raise ValueError(f"Unknown map_mode '{map_mode}'. Choose 'hash' or 'vocab'.")

    return Tokenizer(splitter, mapper, max_len, pad_id=pad_id)
