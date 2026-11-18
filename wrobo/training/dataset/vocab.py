"""Corpus vocabulary: build / cache / load a ``{token: id}`` map from a dataset's
language, for the corpus-vocab tokenizer path (the alternative to hash mapping, and the
prerequisite for aligning pretrained word embeddings).

The vocabulary is built by enumerating the *full instruction space* of the training
cases — every sentence a trajectory could be assigned — so the ids cover all language
the model can ever see, not just the wording sampled during one epoch:

  - template tasks: render every template with that segment's parameters;
  - library tasks: include every sentence for every referenced task.

Tokens are produced by the same splitter the tokenizer uses, so ids line up. This
mirrors how dataset_statistic.get_norm_stats scans the training cases to precompute
normalization stats.
"""

import os
import h5py

from wrobo.utils.file_operations import open_json, save_json
from wrobo.training.dataset.instruction import (
    INSTRUCTION_ATTR,
    UnifiedInstructionProvider,
    parse_instruction_segments,
)


SPECIALS = {"<pad>": 0, "<unk>": 1}


def _iter_instruction_sentences(dataset_dir, case_list):
    """Yield every possible instruction sentence over the training cases.

    For the unified format with frame segments:
    - Parse each episode's instruction metadata (frame segments)
    - For each task reference (int or dict):
      * If int: yield all sentences from that library task
      * If dict with template: render all templates with given parameters
    - Emit each unique sentence only once across all episodes
    """
    provider = UnifiedInstructionProvider(os.path.join(dataset_dir, "instruction_library.json"))
    emitted_sentences = set()

    for case in case_list:
        # Episodes are in episodes/ subdirectory
        path = os.path.join(
            dataset_dir, "episodes", case if case.endswith(".hdf5") else case + ".hdf5"
        )

        if not os.path.exists(path):
            continue

        # Read instruction metadata from HDF5
        with h5py.File(path, "r") as root:
            if INSTRUCTION_ATTR not in root.attrs:
                continue

            instruction_str = root.attrs[INSTRUCTION_ATTR]
            if isinstance(instruction_str, bytes):
                instruction_str = instruction_str.decode('utf-8')

            # Parse segments using the same logic as UnifiedInstructionProvider
            segments = parse_instruction_segments(instruction_str)

            for start, end, task_ref in segments:
                # Check if it's an integer (library) or dict (template)
                if isinstance(task_ref, int):
                    # Library task: yield all sentences
                    task_id = task_ref
                    try:
                        for sentence in provider.get_all_sentences(task_id):
                            if sentence not in emitted_sentences:
                                emitted_sentences.add(sentence)
                                yield sentence
                    except (KeyError, ValueError):
                        continue
                elif isinstance(task_ref, dict):
                    # Template task: extract the canonical task object.
                    try:
                        task_id = task_ref["task_id"]
                        params = task_ref.get("params")

                        # Yield all templates rendered with these params
                        for sentence in provider.get_all_sentences(task_id, params):
                            if sentence not in emitted_sentences:
                                emitted_sentences.add(sentence)
                                yield sentence
                    except (KeyError, ValueError, TypeError):
                        continue


def build_vocab(dataset_dir, case_list, splitter) -> dict:
    """Build a ``{token: id}`` vocab from the training instruction space.

    splitter: a tokenizer.Splitter; its tokens (stringified) become vocab keys, so the
              VocabMapper at train/eval time produces matching ids.
    Special tokens (<pad>=0, <unk>=1) occupy the first ids; corpus tokens follow in
    first-seen order (deterministic given the case_list order).
    """
    vocab = dict(SPECIALS)
    next_id = len(vocab)
    for sentence in _iter_instruction_sentences(dataset_dir, case_list):
        for tok in splitter(sentence):
            key = tok if isinstance(tok, str) else str(tok)
            if key not in vocab:
                vocab[key] = next_id
                next_id += 1
    return vocab


def build_vocab_with_pretrained(
    dataset_dir, case_list, splitter, pretrained_tokens
) -> dict:
    """Build a vocab that KEEPS the full pretrained vocabulary (not trimmed to the
    dataset), so words unseen in the training instructions still map to their pretrained
    vectors at eval time.

    Layout: <pad>=0, <unk>=1, then every pretrained token (in the embedding file's
    order, for stable ids), then any dataset token absent from the pretrained file
    (these get random-initialized embedding rows). Pruning the corpus to the dataset
    after alignment would throw away exactly the coverage a pretrained embedding is for,
    so this is the default when a pretrained embedding is given.
    """
    vocab = dict(SPECIALS)
    next_id = len(vocab)
    for tok in pretrained_tokens:
        key = str(tok)
        if key not in vocab:
            vocab[key] = next_id
            next_id += 1
    # add dataset tokens the pretrained file does not cover (random-init at alignment)
    for sentence in _iter_instruction_sentences(dataset_dir, case_list):
        for tok in splitter(sentence):
            key = tok if isinstance(tok, str) else str(tok)
            if key not in vocab:
                vocab[key] = next_id
                next_id += 1
    return vocab


def save_vocab(vocab: dict, path: str) -> None:
    save_json(vocab, path)


def load_vocab(path: str) -> dict:
    """Load a vocab json. ids are ints (JSON keys stay strings, which is what the
    tokenizer's VocabMapper looks up)."""
    raw = open_json(path)
    return {str(k): int(v) for k, v in raw.items()}
