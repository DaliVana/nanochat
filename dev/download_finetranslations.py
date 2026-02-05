#!/usr/bin/env python3
"""
Download and repackage HuggingFace FineTranslations-Edu dataset by language quota.

Filters dataset by languages in world_languages_iso639-3.json, collecting up to
(speakers_millions × 1000) characters per language. Prioritizes higher edu_score.
For English: uses translated_text, for others: uses og_full_text.

Writes mixed parquet shards to dev/finetranslations_output/
"""

import json
import heapq
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random
import multiprocessing as mp

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq


# Default script mapping for ISO 639-3 codes
# Maps 3-letter ISO codes to language-script format used in dataset
DEFAULT_SCRIPTS = {
    "eng": "eng_Latn", "cmn": "cmn_Hans", "hin": "hin_Deva", "spa": "spa_Latn",
    "arb": "arb_Arab", "fra": "fra_Latn", "ben": "ben_Beng", "por": "por_Latn",
    "rus": "rus_Cyrl", "urd": "urd_Arab", "ind": "ind_Latn", "deu": "deu_Latn",
    "jpn": "jpn_Jpan", "swh": "swh_Latn", "mar": "mar_Deva", "vie": "vie_Latn",
    "tel": "tel_Telu", "pnb": "pnb_Arab", "tur": "tur_Latn", "ita": "ita_Latn",
    "yue": "yue_Hant", "tam": "tam_Taml", "kor": "kor_Hang", "wuu": "wuu_Hans",
    "tha": "tha_Thai", "jav": "jav_Latn", "guj": "guj_Gujr", "pus": "pus_Arab",
    "kan": "kan_Knda", "fas": "fas_Arab", "bho": "bho_Deva", "pol": "pol_Latn",
    "nan": "nan_Hant", "hak": "hak_Hans", "cjy": "cjy_Hans", "fil": "fil_Latn",
    "ukr": "ukr_Cyrl", "hau": "hau_Latn", "mya": "mya_Mymr", "sun": "sun_Latn",
    "mal": "mal_Mlym", "ory": "ory_Orya", "amh": "amh_Ethi", "orm": "orm_Latn",
    "zsm": "zsm_Latn", "uzb": "uzb_Latn", "nld": "nld_Latn", "ron": "ron_Latn",
    "hun": "hun_Latn", "ell": "ell_Grek", "ces": "ces_Latn", "swe": "swe_Latn",
    "bul": "bul_Cyrl", "dan": "dan_Latn", "fin": "fin_Latn", "slk": "slk_Latn",
    "hrv": "hrv_Latn", "lit": "lit_Latn", "slv": "slv_Latn", "lav": "lav_Latn",
    "est": "est_Latn", "mlt": "mlt_Latn", "gle": "gle_Latn",
}


@dataclass
class Document:
    """Document with text and quality score."""
    text: str
    edu_score: int
    char_count: int
    
    def __lt__(self, other):
        # For min heap: lower edu_score comes first (will be evicted first)
        return self.edu_score < other.edu_score


class LanguageCollector:
    """Collects documents per language with memory and quota limits."""
    
    def __init__(self, quotas: Dict[str, int], max_memory_gb: float = 20.0):
        self.quotas = quotas  # lang -> max chars
        self.char_counts = {lang: 0 for lang in quotas}
        self.heaps = {lang: [] for lang in quotas}  # lang -> min heap of Documents
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.current_memory_bytes = 0
        self.docs_processed = 0
        self.docs_collected = 0
        
    def get_current_memory_mb(self) -> float:
        """Estimate current memory usage in MB."""
        return self.current_memory_bytes / (1024 * 1024)
    
    def can_accept(self, lang: str, char_count: int) -> bool:
        """Check if we can accept a document for this language."""
        return self.char_counts.get(lang, 0) < self.quotas.get(lang, 0)
    
    def add_document(self, lang: str, text: str, edu_score: int):
        """Add document to language heap, evicting lower quality if needed."""
        if lang not in self.quotas:
            return
        
        self.docs_processed += 1
        
        # Estimate memory: ~2x text length for overhead
        doc_memory = len(text) * 2
        char_count = len(text)
        
        # Skip if quota already met and this doc is lower quality than worst in heap
        if self.char_counts[lang] >= self.quotas[lang]:
            if not self.heaps[lang]:
                return
            worst_doc = self.heaps[lang][0]
            if edu_score <= worst_doc.edu_score:
                return
        
        # Add to heap
        doc = Document(text=text, edu_score=edu_score, char_count=char_count)
        heapq.heappush(self.heaps[lang], doc)
        self.char_counts[lang] += char_count
        self.current_memory_bytes += doc_memory
        self.docs_collected += 1
        
        # Evict lowest quality docs if over quota
        while self.char_counts[lang] > self.quotas[lang] and self.heaps[lang]:
            evicted = heapq.heappop(self.heaps[lang])
            self.char_counts[lang] -= evicted.char_count
            self.current_memory_bytes -= evicted.char_count * 2
            self.docs_collected -= 1
        
        # Evict lowest quality docs across all languages if over memory limit
        while self.current_memory_bytes > self.max_memory_bytes * 0.9:  # 90% threshold
            # Find language with lowest quality document
            min_lang = None
            min_score = float('inf')
            for lang, heap in self.heaps.items():
                if heap and heap[0].edu_score < min_score:
                    min_score = heap[0].edu_score
                    min_lang = lang
            
            if min_lang is None:
                break
                
            evicted = heapq.heappop(self.heaps[min_lang])
            self.char_counts[min_lang] -= evicted.char_count
            self.current_memory_bytes -= evicted.char_count * 2
            self.docs_collected -= 1
    
    def get_all_documents(self) -> List[str]:
        """Get all collected documents, sorted by quality and shuffled within tiers."""
        all_docs = []
        for lang, heap in self.heaps.items():
            all_docs.extend(heap)
        
        # Sort by edu_score descending
        all_docs.sort(key=lambda d: d.edu_score, reverse=True)
        
        # Shuffle within edu_score tiers for better distribution
        result = []
        i = 0
        while i < len(all_docs):
            score = all_docs[i].edu_score
            tier = []
            while i < len(all_docs) and all_docs[i].edu_score == score:
                tier.append(all_docs[i].text)
                i += 1
            random.shuffle(tier)
            result.extend(tier)
        
        return result
    
    def print_stats(self):
        """Print collection statistics."""
        print("\n=== Collection Statistics ===")
        print(f"Documents processed: {self.docs_processed:,}")
        print(f"Documents collected: {self.docs_collected:,}")
        print(f"Memory usage: {self.get_current_memory_mb():.1f} MB")
        print("\nPer-language quotas:")
        
        for lang in sorted(self.quotas.keys()):
            quota = self.quotas[lang]
            collected = self.char_counts[lang]
            pct = (collected / quota * 100) if quota > 0 else 0
            heap_size = len(self.heaps[lang])
            print(f"  {lang:12s}: {collected:12,} / {quota:12,} chars ({pct:5.1f}%) - {heap_size:6,} docs")


def load_language_quotas(json_path: Path, multiplier: int = 1000) -> Dict[str, int]:
    """Load language quotas from world_languages JSON file."""
    with open(json_path) as f:
        languages = json.load(f)
    
    quotas = {}
    for entry in languages:
        iso_code = entry["iso639_3"]
        speakers_millions = entry["speakers_millions"]
        char_limit = int(speakers_millions * multiplier)  # speakers_millions × multiplier
        
        # Map to language-script format
        lang_script = DEFAULT_SCRIPTS.get(iso_code)
        if lang_script:
            quotas[lang_script] = char_limit
        else:
            print(f"Warning: No script mapping for {iso_code}, skipping")
    
    return quotas


def write_parquet_shards(documents: List[str], output_dir: Path, shard_size_chars: int = 250_000_000):
    """Write documents to parquet shards following repackage_data_reference.py pattern."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shard_idx = 0
    shard_docs = []
    shard_char_count = 0
    row_group_size = 1024
    
    print(f"\nWriting {len(documents):,} documents to parquet shards...")
    
    for doc in documents:
        shard_docs.append(doc)
        shard_char_count += len(doc)
        
        # Write shard when we hit target size AND have clean row group boundary
        if shard_char_count >= shard_size_chars and len(shard_docs) % row_group_size == 0:
            shard_path = output_dir / f"shard_{shard_idx:05d}.parquet"
            
            shard_table = pa.Table.from_pydict({"text": shard_docs})
            pq.write_table(
                shard_table,
                shard_path,
                row_group_size=row_group_size,
                compression="zstd",
                compression_level=3,
                use_dictionary=False,
                write_statistics=False,
            )
            
            print(f"  Wrote {shard_path.name}: {len(shard_docs):,} docs, {shard_char_count:,} chars")
            
            shard_idx += 1
            shard_docs = []
            shard_char_count = 0
    
    # Write final partial shard if any docs remain
    if shard_docs:
        # Pad to row group boundary if needed
        while len(shard_docs) % row_group_size != 0:
            shard_docs.append("")  # Add empty documents for padding
        
        shard_path = output_dir / f"shard_{shard_idx:05d}.parquet"
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            shard_table,
            shard_path,
            row_group_size=row_group_size,
            compression="zstd",
            compression_level=3,
            use_dictionary=False,
            write_statistics=False,
        )
        
        print(f"  Wrote {shard_path.name}: {len(shard_docs):,} docs, {shard_char_count:,} chars (final)")
    
    print(f"\nWrote {shard_idx + 1} shards to {output_dir}")


def main():
    """Main download and repackaging pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Download and repackage FineTranslations-Edu dataset')
    parser.add_argument('--num-workers', type=int, default=24, 
                        help='Number of parallel workers for dataset loading (default: 24)')
    parser.add_argument('--multiplier', type=int, default=1000,
                        help='Multiplier for character quota: speakers_millions × multiplier (default: 1000)')
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    languages_file = script_dir / "world_languages_iso639-3.json"
    output_dir = script_dir / "finetranslations_output"
    
    if not languages_file.exists():
        print(f"Error: {languages_file} not found")
        sys.exit(1)
    
    # Load quotas
    print("Loading language quotas...")
    print(f"Using multiplier: {args.multiplier} (speakers_millions × {args.multiplier})")
    quotas = load_language_quotas(languages_file, multiplier=args.multiplier)
    print(f"Loaded {len(quotas)} languages")
    total_chars = sum(quotas.values())
    print(f"Total target characters: {total_chars:,} ({total_chars / 1e9:.2f}B)")
    
    # Initialize collector
    collector = LanguageCollector(quotas, max_memory_gb=20.0)
    target_langs = set(quotas.keys())
    
    # Load and stream dataset
    print("\nLoading dataset from HuggingFace (streaming)...")
    print("Dataset: HuggingFaceFW/finetranslations-edu")
    print(f"Note: Streaming datasets automatically prefetch data in background")
    
    try:
        # Streaming mode automatically prefetches data in background
        ds = load_dataset(
            "HuggingFaceFW/finetranslations-edu", 
            split="train", 
            streaming=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have the datasets library installed: pip install datasets")
        sys.exit(1)
    
    # Process dataset
    print("\nProcessing documents...")
    report_interval = 10000
    lang_samples = {}  # Track sample language codes we see
    
    for idx, row in enumerate(ds):
        og_lang = row.get("og_language", "")
        
        # Collect sample language codes for debugging
        if len(lang_samples) < 50 and og_lang and og_lang not in lang_samples:
            lang_samples[og_lang] = True
            if len(lang_samples) <= 20:
                print(f"  Found language code: {og_lang}")
        
        # Skip if not a target language
        if og_lang not in target_langs:
            continue
        
        # Get edu_score (0-4 scale)
        edu_score = row.get("edu_score", 0)
        if edu_score is None:
            edu_score = 0
        
        # Get text: translated_text for English, og_full_text for others
        if og_lang == "eng_Latn":
            text = row.get("translated_text", "")
        else:
            text = row.get("og_full_text", "")
        
        if not text:
            continue
        
        # Add to collector
        collector.add_document(og_lang, text, edu_score)
        
        # Progress report
        if (idx + 1) % report_interval == 0:
            print(f"\rProcessed {idx + 1:,} rows | Collected {collector.docs_collected:,} docs | "
                  f"Memory: {collector.get_current_memory_mb():.1f} MB", end="")
        
        # Check if all quotas filled and memory stabilized
        if idx > 100000 and idx % 50000 == 0:
            all_filled = all(
                collector.char_counts[lang] >= quota * 0.95  # 95% of quota
                for lang, quota in quotas.items()
            )
            if all_filled:
                print(f"\n\nAll language quotas filled (>95%), stopping at row {idx + 1:,}")
                break
    
    print()  # New line after progress
    print(f"\nUnique language codes seen in dataset: {len(lang_samples)}")
    if lang_samples:
        print("Sample codes:", sorted(list(lang_samples.keys()))[:30])
    collector.print_stats()
    
    # Get all documents
    print("\nRetrieving and sorting documents by quality...")
    documents = collector.get_all_documents()
    print(f"Total documents to write: {len(documents):,}")
    
    # Write parquet shards
    write_parquet_shards(documents, output_dir)
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()
