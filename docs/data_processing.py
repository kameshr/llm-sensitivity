"""
Data Processing Module for Gradient-Based Sensitivity Analysis

Implements Algorithm A.1: Tokenization and Batching from the paper.
"""

import torch
from typing import List, Iterator, Tuple, Generator
from datasets import load_dataset
from transformers import AutoTokenizer


def chunk_generator(x: List[int], L: int) -> Generator[Tuple[List[int], List[int]], None, None]:
    """
    Generate input-label pairs from token sequence using rolling buffer.

    Args:
        x: Tokenized sequence [x_0, x_1, ..., x_{T-1}]
        L: Sequence length (SEQ_LEN)

    Yields:
        (input, labels): Input sequence and corresponding labels
    """
    cache = []
    for token in x:
        cache.append(token)
        while len(cache) >= L + 1:
            win, cache = cache[:L+1], cache[L+1:]
            inp, labels = win[:-1], win[1:]
            yield inp, labels


def get_batches(x: List[int], L: int, B: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Generate batches of input-label tensors.

    Args:
        x: Tokenized sequence
        L: Sequence length
        B: Batch size

    Yields:
        (I, Y): Input tensor [B x L] and label tensor [B x L]
    """
    gen = chunk_generator(x, L)
    buf = []
    for inp, labels in gen:
        buf.append((inp, labels))
        if len(buf) == B:
            I = torch.tensor([p[0] for p in buf], dtype=torch.long)
            Y = torch.tensor([p[1] for p in buf], dtype=torch.long)
            yield I, Y
            buf = []


class DataLoader:
    """
    Data loader for streaming text corpus with configurable parameters.
    """

    def __init__(self, dataset_name: str = "wikitext", subset: str = "wikitext-103-raw-v1",
                 split: str = "train", seq_len: int = 1024, batch_size: int = 16):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokenizer = None
        self.dataset = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)

    def set_tokenizer(self, tokenizer: AutoTokenizer):
        """Set the tokenizer to use for processing text."""
        self.tokenizer = tokenizer

    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize a single text string."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before tokenizing text")
        return self.tokenizer.encode(text, add_special_tokens=False)

    def get_token_stream(self, max_tokens: int = None) -> List[int]:
        """
        Create a continuous token stream from the dataset.

        Args:
            max_tokens: Maximum number of tokens to process (None for all)

        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before processing dataset")

        tokens = []
        total_tokens = 0

        for example in self.dataset:
            text = example['text'].strip()
            if not text:
                continue

            text_tokens = self.tokenize_text(text)
            tokens.extend(text_tokens)
            total_tokens += len(text_tokens)

            if max_tokens and total_tokens >= max_tokens:
                tokens = tokens[:max_tokens]
                break

        return tokens

    def get_batch_iterator(self, max_tokens: int = None) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Get an iterator over batches of (input, label) tensors.

        Args:
            max_tokens: Maximum number of tokens to process

        Yields:
            (I, Y): Input and label tensors
        """
        token_stream = self.get_token_stream(max_tokens)
        return get_batches(token_stream, self.seq_len, self.batch_size)


def step_through_corpus(data_loader: DataLoader, max_steps: int = None, step_size: int = 1000) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Step through corpus with configurable step size for gradient scanning.

    Args:
        data_loader: DataLoader instance
        max_steps: Maximum number of steps to take
        step_size: Number of tokens to advance between steps

    Yields:
        (I, Y): Input and label tensors for each step
    """
    batch_gen = data_loader.get_batch_iterator()
    step_count = 0

    for batch in batch_gen:
        yield batch
        step_count += 1

        if max_steps and step_count >= max_steps:
            break

        # Skip ahead by step_size tokens worth of batches if needed
        if step_size > data_loader.batch_size * data_loader.seq_len:
            skip_batches = (step_size // (data_loader.batch_size * data_loader.seq_len)) - 1
            for _ in range(skip_batches):
                try:
                    next(batch_gen)
                except StopIteration:
                    return


if __name__ == "__main__":
    # Test the data processing pipeline
    from transformers import AutoTokenizer

    print("Testing data processing pipeline...")

    # Initialize tokenizer and data loader
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    data_loader = DataLoader(seq_len=128, batch_size=4)
    data_loader.set_tokenizer(tokenizer)

    # Test batch generation
    print("Generating test batches...")
    batch_count = 0
    for inputs, labels in data_loader.get_batch_iterator(max_tokens=1000):
        print(f"Batch {batch_count + 1}: Input shape {inputs.shape}, Label shape {labels.shape}")
        print(f"First input tokens: {inputs[0][:10].tolist()}")
        print(f"First label tokens: {labels[0][:10].tolist()}")
        batch_count += 1
        if batch_count >= 3:
            break

    print("Data processing test completed!")