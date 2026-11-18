"""Pretrained word-embedding alignment.

word2vec / GloVe / fastText embeddings are all ultimately an ``nn.Embedding``; the only
real differences are (a) the on-disk serialization and (b) the fact that the pretrained
file has its *own* token->row order while our corpus vocab (vocab.py) has its own
token->id order. The actual work is **alignment**: build a ``(vocab_size, d_word)``
matrix whose row ``i`` holds the pretrained vector for the token with our id ``i``,
random-initializing tokens the pretrained file does not cover.

Two trivial serializations are read:
  - ``.pt``  : torch.save of ``{token: vector}`` or ``{"matrix": (N,d), "vocab": {tok:row}}``
  - text     : GloVe-style ``token v1 v2 ... vd`` per line (whitespace separated)

The aligned matrix is loaded into the text encoder via ``nn.Embedding.from_pretrained``.
``align_and_save`` persists the aligned matrix as a ``.pt`` so training and eval load the
*same* artifact (the file kept in dataset_dir and copied into the log dir).
"""

import os

import torch


def _load_source_vectors(source_path: str) -> dict:
    """Read a pretrained embedding file into ``{token: 1-D FloatTensor}``."""
    ext = os.path.splitext(source_path)[1].lower()
    if ext in (".pt", ".pth", ".bin"):
        obj = torch.load(source_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "matrix" in obj and "vocab" in obj:
            matrix = torch.as_tensor(obj["matrix"], dtype=torch.float32)
            return {tok: matrix[row] for tok, row in obj["vocab"].items()}
        if isinstance(obj, dict):
            return {
                str(tok): torch.as_tensor(vec, dtype=torch.float32)
                for tok, vec in obj.items()
            }
        raise ValueError(
            f"Unsupported .pt embedding structure in {source_path}; expected "
            "{token: vector} or {'matrix','vocab'}."
        )

    # GloVe / word2vec text format: "token v1 v2 ... vd" per line.
    vectors = {}
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 2:
                continue
            token = parts[0]
            vectors[token] = torch.tensor([float(x) for x in parts[1:]])
    if not vectors:
        raise ValueError(f"No vectors parsed from text embedding file {source_path}.")
    return vectors


def load_source_tokens(source_path: str) -> list:
    """Return the pretrained file's tokens in their native order.

    Used to build the *full* corpus vocab (keep the pretrained vocabulary instead of
    trimming to the dataset), so the model can map words unseen in the training
    instructions to their pretrained vectors at eval time. Order is preserved so vocab
    ids are stable for a given embedding file.
    """
    ext = os.path.splitext(source_path)[1].lower()
    if ext in (".pt", ".pth", ".bin"):
        obj = torch.load(source_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "matrix" in obj and "vocab" in obj:
            # order tokens by their row index in the matrix
            return [tok for tok, _ in sorted(obj["vocab"].items(), key=lambda kv: kv[1])]
        if isinstance(obj, dict):
            return [str(tok) for tok in obj.keys()]
        raise ValueError(
            f"Unsupported .pt embedding structure in {source_path}; expected "
            "{token: vector} or {'matrix','vocab'}."
        )

    tokens = []
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 2:
                continue
            tokens.append(parts[0])
    if not tokens:
        raise ValueError(f"No tokens parsed from text embedding file {source_path}.")
    return tokens


def load_aligned_embedding(
    source_path: str, vocab: dict, d_word: int = None, seed: int = 319, pad_token: str = "<pad>"
) -> torch.Tensor:
    """Build a ``(len(vocab), d_word)`` matrix aligned to ``vocab`` ids.

    Rows for tokens present in the pretrained source get the pretrained vector; misses
    (incl. <pad>/<unk> and any OOV) are randomly initialized. The <pad> row is zeroed.

    ``d_word`` is OPTIONAL: the pretrained file already fixes the vector width, so by
    default it is inferred from the source vectors (``d_word=None``). The aligned matrix
    shape ``(vocab_size, d_word)`` then fully determines the downstream ``nn.Embedding``,
    independent of whether the vocab was pruned. Pass an explicit ``d_word`` only to
    assert the source width matches an expected value.
    """
    vectors = _load_source_vectors(source_path)
    src_dim = len(next(iter(vectors.values())))
    if d_word is None:
        d_word = src_dim
    elif src_dim != d_word:
        raise ValueError(
            f"Pretrained embedding dim {src_dim} != requested d_word {d_word}. "
            "Omit d_word to infer it from the pretrained vectors, or set it to match."
        )

    vocab_size = max(vocab.values()) + 1
    gen = torch.Generator().manual_seed(seed)
    matrix = torch.randn(vocab_size, d_word, generator=gen) * 0.02

    n_hit = 0
    for token, idx in vocab.items():
        if token in vectors:
            matrix[idx] = vectors[token]
            n_hit += 1
    # PAD row must be zero so padding_idx contributes nothing.
    if pad_token in vocab:
        matrix[vocab[pad_token]].zero_()
    return matrix


def align_and_save(
    source_path: str,
    vocab: dict,
    out_path: str,
    d_word: int = None,
    seed: int = 319,
    pad_token: str = "<pad>",
) -> str:
    """Align a pretrained embedding to ``vocab`` and save it as a self-describing ``.pt``
    (``{"matrix": (V,d), "vocab": vocab}``) so train and eval reuse the exact same
    aligned table. ``d_word`` is optional (inferred from the source by default).
    Returns ``out_path``."""
    matrix = load_aligned_embedding(
        source_path, vocab, d_word=d_word, seed=seed, pad_token=pad_token
    )
    torch.save({"matrix": matrix, "vocab": vocab}, out_path)
    return out_path


def load_saved_embedding(aligned_path: str) -> torch.Tensor:
    """Load the ``(V, d_word)`` matrix from an align_and_save artifact."""
    obj = torch.load(aligned_path, map_location="cpu", weights_only=False)
    return torch.as_tensor(obj["matrix"], dtype=torch.float32)
