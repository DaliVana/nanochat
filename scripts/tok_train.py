"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.

Improvements over vanilla BPE training:
  --multilingual-dir : Interleave multilingual text (e.g. finetranslations) to improve
                       CJK, Arabic, Hindi, etc. vocabulary coverage. Default 5% ratio.
  --ios-prune        : Post-training junk token pruning via Intersection-over-Self (IoS).
                       Removes intermediate BPE artifacts that waste vocabulary slots.
                       See: Picky BPE (arXiv 2409.04599).
"""
import os
import time
import argparse
from pathlib import Path
import torch
import pyarrow.parquet as pq
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=8_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=5_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
# Multilingual data injection (addresses CJK, Arabic, Hindi under-representation)
parser.add_argument('--multilingual-dir', type=str, default=None,
                    help='Directory with multilingual parquet files (e.g. dev/finetranslations_output). '
                         'If provided, interleaves multilingual text to improve non-English vocab coverage.')
parser.add_argument('--multilingual-ratio', type=float, default=0.05,
                    help='Ratio of multilingual chars to total chars (default: 0.05 = 5%%). '
                         'arXiv 2506.10766 shows even 5%% multilingual data gives massive plasticity gains.')
# IoS-based junk token pruning (addresses wasted vocab slots from BPE merge artifacts)
parser.add_argument('--ios-prune', type=float, default=None,
                    help='IoS threshold for post-training junk token pruning (e.g. 0.8). '
                         'Trains with 5%% inflated vocab, then prunes tokens with IoS >= threshold. '
                         'Typical: 0.8 removes ~4%% of vocab, 0.9 removes ~2%%.')
parser.add_argument('--ios-sample-chars', type=int, default=800_000_000,
                    help='Characters to sample for IoS analysis (default: 800M)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
if args.multilingual_dir:
    print(f"multilingual_dir: {args.multilingual_dir}")
    print(f"multilingual_ratio: {args.multilingual_ratio:.1%}")
if args.ios_prune is not None:
    print(f"ios_prune threshold: {args.ios_prune}")

# -----------------------------------------------------------------------------
# Multilingual data setup (optional)

multilingual_parquets = []
if args.multilingual_dir:
    ml_path = Path(args.multilingual_dir)
    if not ml_path.is_absolute():
        ml_path = Path(__file__).parent.parent / args.multilingual_dir
    if ml_path.exists():
        multilingual_parquets = sorted(ml_path.glob("*.parquet"))
        print(f"Found {len(multilingual_parquets)} multilingual parquet files in {ml_path}")
    else:
        print(f"WARNING: Multilingual directory {ml_path} does not exist. "
              f"Run dev/download_finetranslations.py first. Continuing without multilingual data.")

def multilingual_doc_iterator():
    """Yield documents from multilingual parquet files, cycling indefinitely."""
    while True:
        for pf_path in multilingual_parquets:
            pf = pq.ParquetFile(pf_path)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                for doc in texts:
                    if doc:
                        yield doc

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    4) Optionally interleave multilingual documents at the specified ratio
    """
    nchars = 0
    ml_chars = 0
    ml_target_ratio = args.multilingual_ratio if multilingual_parquets else 0.0
    ml_iter = multilingual_doc_iterator() if multilingual_parquets else None

    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text

            # Interleave multilingual documents to reach target ratio
            # After each English doc, check if we need more multilingual text
            if ml_iter is not None and nchars > 0:
                current_ratio = ml_chars / (nchars + ml_chars) if (nchars + ml_chars) > 0 else 0
                while current_ratio < ml_target_ratio:
                    try:
                        ml_doc = next(ml_iter)
                        if len(ml_doc) > args.doc_cap:
                            ml_doc = ml_doc[:args.doc_cap]
                        ml_chars += len(ml_doc)
                        yield ml_doc
                        current_ratio = ml_chars / (nchars + ml_chars)
                    except StopIteration:
                        ml_iter = None
                        break

            if nchars > args.max_chars:
                if ml_chars > 0:
                    total = nchars + ml_chars
                    print(f"Training data: {nchars:,} base + {ml_chars:,} multilingual = {total:,} chars "
                          f"({ml_chars/total:.1%} multilingual)")
                return

    if ml_chars > 0:
        total = nchars + ml_chars
        print(f"Training data: {nchars:,} base + {ml_chars:,} multilingual = {total:,} chars "
              f"({ml_chars/total:.1%} multilingual)")

text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
# If IoS pruning is enabled, train with inflated vocab size to compensate for pruning
train_vocab_size = args.vocab_size
if args.ios_prune is not None:
    # Inflate by ~8% to compensate for junk tokens that will be pruned (~4-5% at threshold 0.8)
    ios_overshoot = int(args.vocab_size * 0.08)
    train_vocab_size = args.vocab_size + ios_overshoot
    print(f"IoS pruning enabled: training with inflated vocab {train_vocab_size} "
          f"(+{ios_overshoot} overshoot, target: {args.vocab_size})")

t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, train_vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# IoS-based junk token pruning (optional)
pruned_count = 0
if args.ios_prune is not None:
    print(f"\n--- Post-training IoS junk token pruning (threshold={args.ios_prune}) ---")
    # Collect sample text for IoS analysis
    ios_chars = 0
    ios_texts = []
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            ios_texts.append(doc[:args.doc_cap])
            ios_chars += len(ios_texts[-1])
        if ios_chars >= args.ios_sample_chars:
            break
    ios_corpus = "\n".join(ios_texts)

    tokenizer, pruned_ids = tokenizer.prune_junk_tokens(ios_corpus, threshold=args.ios_prune)
    pruned_count = len(pruned_ids)
    print(f"Final vocab size after pruning: {tokenizer.get_vocab_size()}")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {"pruned_junk_tokens": pruned_count},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
