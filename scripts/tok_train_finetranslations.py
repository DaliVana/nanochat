"""
Train a tokenizer using finetranslations data.
Uses the multilingual data from dev/finetranslations_output/ parquet files.
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
from pathlib import Path
import torch
import pyarrow.parquet as pq
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer on finetranslations data')
parser.add_argument('--data-dir', type=str, default='dev/finetranslations_output', 
                    help='Directory containing parquet files (default: dev/finetranslations_output)')
parser.add_argument('--doc-cap', type=int, default=10_000, 
                    help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, 
                    help='Vocabulary size (default: 32768 = 2^15)')
parser.add_argument('--tokenizer-name', type=str, default='tokenizer_finetranslations',
                    help='Name for saved tokenizer directory (default: tokenizer_finetranslations)')
args = parser.parse_args()

print(f"data_dir: {args.data_dir}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
print(f"tokenizer_name: {args.tokenizer_name}")
print("Using ALL data from dataset (no character limit)")

# -----------------------------------------------------------------------------
# Find parquet files

data_path = Path(args.data_dir)
if not data_path.is_absolute():
    # Resolve relative path from project root
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / args.data_dir

if not data_path.exists():
    print(f"Error: Data directory {data_path} does not exist")
    print(f"Please run dev/download_finetranslations.py first to generate the data")
    exit(1)

parquet_files = sorted(data_path.glob("*.parquet"))
if not parquet_files:
    print(f"Error: No parquet files found in {data_path}")
    print(f"Please run dev/download_finetranslations.py first to generate the data")
    exit(1)

print(f"\nFound {len(parquet_files)} parquet files:")
for f in parquet_files[:5]:
    print(f"  {f.name}")
if len(parquet_files) > 5:
    print(f"  ... and {len(parquet_files) - 5} more")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Iterate through all parquet files
    2) Read each parquet file row group by row group
    3) Crop every document to args.doc_cap characters
    4) Process ALL documents in the dataset
    """
    nchars = 0
    ndocs = 0
    
    for parquet_file in parquet_files:
        print(f"Processing {parquet_file.name}...")
        pf = pq.ParquetFile(parquet_file)
        
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            
            for doc in texts:
                # Skip empty documents (padding from parquet writing)
                if not doc:
                    continue
                
                doc_text = doc
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[:args.doc_cap]
                
                nchars += len(doc_text)
                ndocs += 1
                yield doc_text
                
                if ndocs % 10000 == 0:
                    print(f"  Processed {ndocs:,} docs, {nchars:,} chars ({nchars/1e9:.2f}B)")
    
    print(f"\nProcessed all data: {nchars:,} chars from {ndocs:,} docs ({nchars/1e9:.2f}B)")

text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
print("\nTraining tokenizer...")
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, args.tokenizer_name)
tokenizer.save(tokenizer_dir)
print(f"Saved tokenizer to {tokenizer_dir}")

# -----------------------------------------------------------------------------
# Quick inline sanity check with multilingual test text
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍
Français: Bonjour le monde!
Español: ¡Hola mundo!
Deutsch: Hallo Welt!
Русский: Привет мир!
العربية: مرحبا بالعالم
हिन्दी: नमस्ते दुनिया"""

encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text
print("\n✓ Multilingual sanity check passed")

# -----------------------------------------------------------------------------
# Cache a mapping from token id to number of bytes of that token
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
get_report().log(section="Tokenizer training (finetranslations)", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])

print("\n✓ Complete!")
