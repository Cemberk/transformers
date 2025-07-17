#!/usr/bin/env python3
"""
AMD frameworks-CI utility - Test Bucketing and Analysis
======================================================
Bucket HuggingFace Transformers tests into P-1/P0/P1/P2/P3/CPU classes and analyze results.

Bucketing Strategy:
• P-1: Important model tests - critical models (auto, bert, clip, t5, etc.) (NON-CPU ONLY)
• P0: Critical GPU tests - heavily used GPU kernels (NON-CPU ONLY)
• P1: Framework basics - torch/tf/flax without device specificity (NON-CPU ONLY)
• P2: CPU-only/Flaky tests - not_device_test, pipelines, flaky tests (NON-CPU ONLY)
• P3: Everything else not captured above (NON-CPU ONLY)
• CPU: All CPU-only tests (explicit CPU tests, not_device_test, etc.)

Key CLI modes:
• Discovery only: python bucket_tests.py [--bucket P-1|P0|P1|P2|P3|CPU] [--yaml buckets.yaml] [--print]
• Export to Excel: python bucket_tests.py --export-excel results.xlsx
• Analyze CI results: python bucket_tests.py --analyze-ci results.txt --export-excel full_report.xlsx
"""

from __future__ import annotations
import argparse, ast, json, os, pathlib, re, sys, textwrap, xml.etree.ElementTree as ET
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Tuple, Set
import subprocess
import tempfile

# Path constants - adjusted for utils folder
SCRIPT_PATH = pathlib.Path(__file__).resolve()
UTILS_DIR = SCRIPT_PATH.parent
REPO_ROOT = UTILS_DIR.parent

# Find the actual repo root by looking for .git directory  
while not (REPO_ROOT / ".git").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent

TEST_ROOT = REPO_ROOT / "tests"
TEST_GLOB_ROOT = TEST_ROOT
PY_EXT = (".py")
DEBUG_MODE = False

# Important models for P-1 priority
IMPORTANT_MODELS = [
    "auto", "bert", "clip", "t5", "xlm-roberta", "gpt2", "bart", "mpnet", "gpt-j", 
    "wav2vec2", "deberta-v2", "layoutlm", "llama", "opt", "longformer", "vit", 
    "whisper", "tapas", "vilt", "clap", "detr", "owlvit", "dpt", "videomae"
]

IMPORTANT_MODELS_SET = set()
for model in IMPORTANT_MODELS:
    IMPORTANT_MODELS_SET.add(model.lower())
    IMPORTANT_MODELS_SET.add(model.replace("-", "_"))
    IMPORTANT_MODELS_SET.add(model.replace("_", "-"))

# CPU test identification decorators
CPU_DECOS = {
    "not_device_test", "is_flaky", "slow", "tooslow", "is_staging_test",
    "is_pt_tf_cross_test", "is_pt_flax_cross_test", "is_pipeline_test", "is_agent_test",
    "require_torch_xpu", "require_torch_npu", "require_torch_multi_npu",
    "require_torch_xla", "require_torch_neuroncore", "require_torch_multi_xpu",
    "require_intel_extension_for_pytorch", "require_non_xpu",
    "require_bs4", "require_cv2", "require_levenshtein", "require_nltk",
    "require_g2p_en", "require_rjieba", "require_jieba", "require_jinja",
    "require_pytesseract", "require_ftfy", "require_spacy", "require_datasets",
    "require_cython", "require_soundfile", "require_av", "require_librosa", 
    "require_phonemizer", "require_pyctcdecode", "require_essentia", "require_pretty_midi",
    "require_sudachi", "require_sudachi_projection", "require_jumanpp",
    "require_wandb", "require_clearml", "require_optuna", "require_ray", "require_sigopt",
    "require_galore_torch", "require_lomo", "require_grokadamw", "require_schedulefree",
    "require_peft", "require_read_token"
}

CPU_PATH_PATTERNS = [
    "cpu", "not_device", "flaky", "slow", "pipeline", "agent",
    "cross_test", "audio", "text_processing", "data_processing"
]

# P0: Critical GPU tests
P0_DECOS = {
    "require_torch_gpu", "require_torch_accelerator", "require_torch_fp16", "require_torch_bf16", 
    "require_torch_tf32", "require_fp8", "require_torch_bf16_gpu", "require_torch_bf16_cpu",
    "require_torch_large_gpu", "require_torch_large_accelerator", "require_torch_multi_gpu", 
    "require_torch_multi_accelerator", "require_torch_up_to_2_gpus", "require_torch_up_to_2_accelerators",
    "require_torch_non_multi_gpu", "require_torch_non_multi_accelerator",
    "require_torch_gpu_if_bnb_not_multi_backend_enabled", "require_accelerate", "require_fsdp", 
    "require_deepspeed", "require_flash_attn", "require_torch_sdpa", "require_apex",
    "require_torchdynamo", "require_deterministic_for_xpu", "require_bitsandbytes", 
    "require_pytorch_quantization", "require_torchao", "require_optimum", "require_optimum_quanto", 
    "require_gptq", "require_hqq", "require_vptq", "require_eetq", "require_aqlm", "require_auto_awq",
    "require_auto_round", "require_compressed_tensors", "require_fbgemm_gpu", "require_liger_kernel",
    "require_timm", "require_detectron2", "require_natten", "require_torch_tensorrt_fx",
    "require_faiss", "require_flute_hadamard", "run_first"
}

# P1: Framework basics
FRAMEWORK_DECOS = {
    "require_torch", "require_tf", "require_flax", "require_jax", "require_torch_or_tf",
    "require_tokenizers", "require_sentencepiece", "require_sacremoses", "require_tiktoken", 
    "require_seqio", "require_safetensors", "require_pandas", "require_scipy",
    "require_vision", "require_torchvision", "require_torchaudio", "require_tensorflow_text", 
    "require_keras_nlp", "require_tensorflow_probability", "require_onnx", "require_tf2onnx", 
    "require_gguf", "require_tensorboard"
}

P2_DECOS = set()
SLOW_DECOS = {"slow", "tooslow"}
PIPELINE_DECOS = {"is_pipeline_test", "is_agent_test"}
EXCLUDE_DIRS = {"examples", "templates"}

def is_excluded(path: pathlib.Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return True
    if any(pattern in str(path) for pattern in ["__pycache__", ".git", ".pytest_cache"]):
        return True
    return False

def is_cpu_test(decos: set[str], nodeid: str, file_path: pathlib.Path = None) -> bool:
    if decos & CPU_DECOS:
        return True
    nodeid_lower = nodeid.lower()
    if any(pattern in nodeid_lower for pattern in CPU_PATH_PATTERNS):
        return True
    if "not_device_test" in decos:
        return True
    if decos & PIPELINE_DECOS:
        return True
    if decos & SLOW_DECOS:
        return True
    return False

def is_important_model_test(nodeid: str, file_path: pathlib.Path = None) -> bool:
    nodeid_lower = nodeid.lower()
    models_match = re.search(r'tests/models/([^/]+)/', nodeid_lower)
    if models_match:
        model_name = models_match.group(1)
        model_clean = model_name.replace("_", "-").replace("-", "_")
        if (model_name in IMPORTANT_MODELS_SET or
            model_clean in IMPORTANT_MODELS_SET or
            any(important in model_name for important in IMPORTANT_MODELS_SET if len(important) > 3)):
            return True

    for important_model in IMPORTANT_MODELS_SET:
        if len(important_model) > 3:
            patterns = [f"test_{important_model}", f"test_modeling_{important_model}",
                       f"{important_model}_test", f"modeling_{important_model}"]
            for pattern in patterns:
                if pattern in nodeid_lower:
                    return True

    if "auto" in IMPORTANT_MODELS_SET:
        auto_patterns = ["test_auto_", "/auto/", "auto_test", "autoprocessor", 
                        "automodel", "autotokenizer"]
        for pattern in auto_patterns:
            if pattern in nodeid_lower:
                return True
    return False

def discover_tests_with_pytest() -> list[str]:
    print("Using pytest to discover all tests...", file=sys.stderr)
    cmd = [sys.executable, "-m", "pytest", str(TEST_GLOB_ROOT), "--collect-only", "--quiet",
           "--continue-on-collection-errors", "-p", "no:cacheprovider",
           "--ignore-glob=**/test_torch_compile.py", "--ignore-glob=**/test_doctests.py",
           "--ignore-glob=**/test_torchao.py", "--ignore-glob=**/test_trainer.py"]

    env = os.environ.copy()
    env.update({"RUN_SLOW": "0", "CUDA_VISIBLE_DEVICES": "", "TRANSFORMERS_VERBOSITY": "error",
                "TRANSFORMERS_TEST_DEVICE": "cpu"})

    try:
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env, cwd=REPO_ROOT)
        
        if result.returncode != 0:
            print(f"Pytest collection stderr: {result.stderr}", file=sys.stderr)

        nodeids = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if (line.startswith("tests/") and "::" in line and
                not line.startswith("<") and not line.startswith("=")):
                nodeid = line.strip()
                if nodeid.endswith(">"):
                    continue
                if " " in nodeid:
                    continue
                nodeids.append(nodeid)

        print(f"Pytest discovered {len(nodeids)} tests", file=sys.stderr)
        
        if len(nodeids) < 1000:
            print("Low test count from direct nodeids, complementing with file-based discovery...", file=sys.stderr)
            file_based_nodeids = discover_tests_from_files()
            all_nodeids = list(set(nodeids + file_based_nodeids))
            print(f"Combined discovery found {len(all_nodeids)} tests", file=sys.stderr)
            return all_nodeids

        return nodeids

    except subprocess.TimeoutExpired:
        print("Pytest collection timed out, falling back to file-based discovery", file=sys.stderr)
        return discover_tests_from_files()
    except Exception as e:
        print(f"Error during pytest collection: {e}", file=sys.stderr)
        return discover_tests_from_files()

def discover_tests_from_files() -> list[str]:
    print("Using file-based discovery...", file=sys.stderr)
    nodeids = []
    files_processed = 0

    if not TEST_GLOB_ROOT.exists():
        print(f"Test directory {TEST_GLOB_ROOT} does not exist", file=sys.stderr)
        return []

    for py_file in TEST_GLOB_ROOT.rglob("test_*.py"):
        if is_excluded(py_file):
            continue

        files_processed += 1
        try:
            content = py_file.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}", file=sys.stderr)
            continue

        try:
            rel_path = py_file.relative_to(REPO_ROOT)
        except ValueError:
            rel_path = py_file

        file_str = str(rel_path).replace('\\', '/')
        has_parameterized = bool(re.search(r'@parameterized\.expand', content))

        if has_parameterized:
            nodeids.append(file_str)
        else:
            test_functions = re.findall(r'def\s+(test_\w+)\s*\(', content)
            test_classes = re.findall(r'class\s+(\w*Test\w*)\s*[:\(]', content)

            if test_classes:
                for test_class in test_classes:
                    class_methods = re.findall(
                        rf'class\s+{re.escape(test_class)}.*?(?=class|\Z)', content, re.DOTALL)
                    if class_methods:
                        class_content = class_methods[0]
                        methods = re.findall(r'def\s+(test_\w+)\s*\(', class_content)
                        for method in methods:
                            nodeids.append(f"{file_str}::{test_class}::{method}")

            for test_func in test_functions:
                func_pos = content.find(f"def {test_func}")
                if func_pos > 0:
                    before_func = content[:func_pos]
                    last_class = before_func.rfind("class ")
                    last_def = before_func.rfind("def ")
                    if last_class > last_def:
                        continue
                nodeids.append(f"{file_str}::{test_func}")

    print(f"File-based discovery: processed {files_processed} files, found {len(nodeids)} tests", file=sys.stderr)
    return nodeids

class TestMetadataExtractor(ast.NodeVisitor):
    def __init__(self, file_path: pathlib.Path):
        super().__init__()
        self.file_path = file_path
        self.test_metadata: dict[str, set[str]] = {}
        self.class_decorators: dict[str, set[str]] = {}
        self.current_class: list[str] = []

    def _extract_decorator_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Call):
            node = node.func
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _extract_pytest_markers(self, node: ast.AST) -> set[str]:
        markers = set()
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Attribute) and
                isinstance(node.func.value.value, ast.Name) and
                node.func.value.value.id == "pytest" and
                node.func.value.attr == "mark"):
                markers.add(node.func.attr)
        elif isinstance(node, ast.Attribute):
            if (isinstance(node.value, ast.Attribute) and
                isinstance(node.value.value, ast.Name) and
                node.value.value.id == "pytest" and
                node.value.attr == "mark"):
                markers.add(node.attr)

        if isinstance(node, ast.Name):
            name = node.id
            relevant_decos = (P0_DECOS | CPU_DECOS | FRAMEWORK_DECOS |
                             {"not_device_test", "slow", "tooslow", "is_flaky", "is_staging_test",
                              "is_pt_tf_cross_test", "is_pt_flax_cross_test", "run_first"})
            if name in relevant_decos:
                markers.add(name)
        return markers

    def visit_ClassDef(self, node: ast.ClassDef):
        self.current_class.append(node.name)
        class_decos = set()
        for deco in node.decorator_list:
            name = self._extract_decorator_name(deco)
            if name:
                class_decos.add(name)
            class_decos.update(self._extract_pytest_markers(deco))
        self.class_decorators[node.name] = class_decos
        for child in node.body:
            self.visit(child)
        self.current_class.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        func_decos = set()
        for deco in node.decorator_list:
            name = self._extract_decorator_name(deco)
            if name:
                func_decos.add(name)
            func_decos.update(self._extract_pytest_markers(deco))

        for cls in reversed(self.current_class):
            func_decos.update(self.class_decorators.get(cls, set()))

        if (node.name.startswith("test") or "test" in node.name.lower() or func_decos):
            self.test_metadata[node.name] = func_decos

def extract_test_metadata(file_path: pathlib.Path) -> tuple[dict[str, set[str]], bool]:
    try:
        content = file_path.read_text(encoding='utf-8', errors='replace')
        regex_markers = set()

        decorator_patterns = [
            (r'@not_device_test\b', 'not_device_test'),
            (r'@pytest\.mark\.not_device_test\b', 'not_device_test'),
            (r'@slow\b', 'slow'), (r'@tooslow\b', 'tooslow'), (r'@is_flaky\b', 'is_flaky'),
            (r'@is_pipeline_test\b', 'is_pipeline_test'), (r'@is_agent_test\b', 'is_agent_test'),
            (r'@require_torch\b', 'require_torch'), (r'@require_tf\b', 'require_tf'),
            (r'@require_torch_gpu\b', 'require_torch_gpu'), (r'@require_deepspeed\b', 'require_deepspeed'),
            (r'@require_accelerate\b', 'require_accelerate'), (r'@require_flash_attn\b', 'require_flash_attn'),
            (r'@require_bitsandbytes\b', 'require_bitsandbytes'), (r'@require_vision\b', 'require_vision')
        ]

        for pattern, marker_name in decorator_patterns:
            if re.search(pattern, content):
                regex_markers.add(marker_name)

        tree = ast.parse(content)
        extractor = TestMetadataExtractor(file_path)
        extractor.visit(tree)

        for test_name, ast_markers in extractor.test_metadata.items():
            extractor.test_metadata[test_name] = ast_markers | regex_markers

        if regex_markers and not extractor.test_metadata:
            extractor.test_metadata['__file_markers__'] = regex_markers

        rocm_hint = bool(re.search(r'\b(IS_ROCM_SYSTEM|torch\.version\.hip|rocm|hip_version)\b', content, re.IGNORECASE))
        return extractor.test_metadata, rocm_hint

    except Exception as e:
        if DEBUG_MODE:
            print(f"Warning: Could not extract metadata from {file_path}: {e}", file=sys.stderr)
        return {}, False

def bucket_of(decos: set[str], had_rocm_hint: bool, nodeid: str) -> str:
    if is_cpu_test(decos, nodeid):
        return "CPU"
    if is_important_model_test(nodeid):
        return "P-1"
    if had_rocm_hint:
        return "P0"
    if decos & P0_DECOS:
        return "P0"
    if decos & FRAMEWORK_DECOS:
        return "P1"
    
    nodeid_lower = nodeid.lower()
    if any(pattern in nodeid_lower for pattern in [
        "deepspeed", "fsdp", "accelerate", "flash_attn", "gpu", "cuda", "multi_gpu", "bitsandbytes"
    ]):
        return "P0"
    return "P2"

def scan_tests() -> tuple[dict[str, list[str]], dict[str, int], dict[str, dict[str, set[str]]]]:
    buckets: dict[str, list[str]] = {"P-1": [], "P0": [], "P1": [], "P2": [], "P3": [], "CPU": []}
    test_metadata_map: dict[str, dict[str, set[str]]] = {}

    all_nodeids = discover_tests_with_pytest()
    if not all_nodeids:
        print("No tests discovered!", file=sys.stderr)
        return buckets, {}, test_metadata_map

    print(f"Processing {len(all_nodeids)} discovered tests for bucketing with CPU separation...", file=sys.stderr)

    tests_by_file = defaultdict(list)
    for nodeid in all_nodeids:
        if "::" in nodeid:
            file_part = nodeid.split("::")[0]
        else:
            file_part = nodeid
        file_path = REPO_ROOT / file_part
        tests_by_file[file_path].append(nodeid)

    print(f"Found tests in {len(tests_by_file)} files", file=sys.stderr)

    decorator_stats = defaultdict(int)
    slow_test_buckets = {"P-1": 0, "P0": 0, "P1": 0, "P2": 0, "P3": 0, "CPU": 0}
    files_with_metadata = 0
    important_model_count = 0
    cpu_test_count = 0

    for file_path, nodeids_in_file in tests_by_file.items():
        test_metadata, rocm_hint = extract_test_metadata(file_path)
        if test_metadata:
            files_with_metadata += 1

        for nodeid in nodeids_in_file:
            if "::" in nodeid:
                parts = nodeid.split("::")
                test_name = parts[-1]
            else:
                test_name = "__file_level__"

            decos = test_metadata.get(test_name, set())
            if not decos:
                for meta_test_name, meta_decos in test_metadata.items():
                    if test_name in meta_test_name or meta_test_name in test_name:
                        decos = meta_decos
                        break
                if not decos and '__file_markers__' in test_metadata:
                    decos = test_metadata['__file_markers__']

            test_metadata_map[nodeid] = {
                'decorators': decos, 'rocm_hint': rocm_hint,
                'file_path': str(file_path), 'test_name': test_name
            }

            if is_important_model_test(nodeid):
                important_model_count += 1
            if is_cpu_test(decos, nodeid):
                cpu_test_count += 1

            bucket = bucket_of(decos, rocm_hint, nodeid)
            buckets[bucket].append(nodeid)

            for deco in decos:
                decorator_stats[deco] += 1
            if decos & SLOW_DECOS:
                slow_test_buckets[bucket] += 1

    total_tests = sum(len(tests) for tests in buckets.values())
    print(f"Bucketed {total_tests} tests from {files_with_metadata} files with metadata", file=sys.stderr)
    print(f"Found {important_model_count} important model tests for P-1", file=sys.stderr)
    print(f"Found {cpu_test_count} CPU tests for CPU bucket", file=sys.stderr)

    for lst in buckets.values():
        lst.sort()

    return buckets, dict(decorator_stats), test_metadata_map

def analyze_bucket_distribution(buckets: dict[str, list[str]], decorator_stats: dict[str, int] = None, test_metadata_map=None) -> None:
    total_tests = sum(len(tests) for tests in buckets.values())

    print("\n=== Bucket Distribution Analysis (with CPU separation) ===", file=sys.stderr)
    for bucket in ["P-1", "P0", "P1", "P2", "P3", "CPU"]:
        count = len(buckets[bucket])
        percentage = (count / total_tests * 100) if total_tests > 0 else 0

        bucket_desc = {
            "P-1": "Important Models (auto, bert, clip, t5, etc.) - NON-CPU ONLY",
            "P0": "Critical GPU (DeepSpeed, Flash Attention, Quantization, Multi-GPU) - NON-CPU ONLY",
            "P1": "Framework Basics (torch/tf/flax, tokenizers, vision, data handling) - NON-CPU ONLY",
            "P2": "Remaining Non-CPU Tests - NON-CPU ONLY",
            "P3": "Uncategorized Non-CPU Tests - NON-CPU ONLY",
            "CPU": "ALL CPU-only tests (not_device_test, flaky, pipelines, audio, etc.)"
        }

        print(f"{bucket}: {count:4d} tests ({percentage:5.1f}%) - {bucket_desc[bucket]}", file=sys.stderr)

    print(f"Total: {total_tests} tests", file=sys.stderr)

    if buckets["CPU"]:
        print(f"\nSample CPU tests:", file=sys.stderr)
        for i, nodeid in enumerate(buckets["CPU"][:5]):
            print(f"   {i+1}. {nodeid}", file=sys.stderr)
        if len(buckets["CPU"]) > 5:
            print(f"   ... and {len(buckets['CPU']) - 5} more", file=sys.stderr)

    if buckets["P-1"]:
        print(f"\nSample P-1 (Important Model) tests:", file=sys.stderr)
        for i, nodeid in enumerate(buckets["P-1"][:5]):
            print(f"   {i+1}. {nodeid}", file=sys.stderr)
        if len(buckets["P-1"]) > 5:
            print(f"   ... and {len(buckets['P-1']) - 5} more", file=sys.stderr)

    if decorator_stats:
        cpu_count = len(buckets["CPU"])
        non_cpu_total = sum(len(buckets[b]) for b in ["P-1", "P0", "P1", "P2", "P3"])
        
        print(f"\nBucket Analysis with CPU Separation:", file=sys.stderr)
        print(f"   - CPU (All CPU-only): {cpu_count} tests ({cpu_count/total_tests*100:.1f}%)", file=sys.stderr)
        print(f"   - Non-CPU Total: {non_cpu_total} tests ({non_cpu_total/total_tests*100:.1f}%)", file=sys.stderr)

def parse_ci_results(ci_file: pathlib.Path) -> dict[str, dict[str, int]]:
    """Parse CI results file and extract test outcomes by model/bucket."""
    print(f"Parsing CI results from {ci_file}", file=sys.stderr)
    
    try:
        content = ci_file.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"Error reading CI results file: {e}", file=sys.stderr)
        return {}

    results = {}
    current_model = None
    model_results = {"PASSED": 0, "FAILED": 0, "SKIPPED": 0}

    for line in content.splitlines():
        line = line.strip()
        
        # Check for summary line like {'PASSED': 27453, 'FAILED': 445, 'SKIPPED': 21588, 'TOTAL': 49486}
        if line.startswith("{") and "PASSED" in line:
            try:
                summary = eval(line)
                if isinstance(summary, dict) and "PASSED" in summary:
                    results["OVERALL"] = summary
                    continue
            except:
                pass

        # Check for model name (single word lines that aren't test results)
        if (line and not line.startswith("PASSED") and not line.startswith("FAILED") and 
            not line.startswith("SKIPPED") and "::" not in line and 
            line.replace("_", "").replace("-", "").isalnum()):
            if current_model and any(model_results.values()):
                results[current_model] = model_results.copy()
            current_model = line
            model_results = {"PASSED": 0, "FAILED": 0, "SKIPPED": 0}
            continue

        # Check for test result lines
        if line.startswith(("PASSED ", "FAILED ", "SKIPPED ")):
            status = line.split()[0]
            if current_model and status in model_results:
                model_results[status] += 1

    # Add the last model
    if current_model and any(model_results.values()):
        results[current_model] = model_results.copy()

    print(f"Parsed results for {len(results)} models/sections", file=sys.stderr)
    return results

def create_excel_export(buckets: dict[str, list[str]], test_metadata_map: dict[str, dict[str, set[str]]], 
                       output_path: str, ci_results: dict = None) -> None:
    try:
        import pandas as pd
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils.dataframe import dataframe_to_rows
    except ImportError:
        print("Error: pandas and openpyxl are required for Excel export.", file=sys.stderr)
        print("Install with: pip install pandas openpyxl", file=sys.stderr)
        return

    print(f"Creating Excel export with CPU separation: {output_path}", file=sys.stderr)

    wb = openpyxl.Workbook()
    if 'Sheet' in wb.sheetnames:
        wb.remove(wb['Sheet'])

    _create_summary_sheet(wb, buckets, ci_results or {})

    for bucket in ["P-1", "P0", "P1", "P2", "P3", "CPU"]:
        if not buckets[bucket]:
            continue
        print(f"Creating sheet for bucket {bucket} ({len(buckets[bucket])} tests)", file=sys.stderr)
        _create_bucket_sheet(wb, bucket, buckets[bucket], test_metadata_map, ci_results or {})

    _create_decorator_analysis_sheet(wb, test_metadata_map)
    _create_cpu_analysis_sheet(wb, buckets, test_metadata_map)

    if ci_results:
        _create_ci_results_sheet(wb, ci_results)

    try:
        wb.save(output_path)
        print(f"✅ Excel file saved: {output_path}", file=sys.stderr)
        print(f"📊 Created {len(wb.sheetnames)} sheets with test data and analysis", file=sys.stderr)
    except Exception as e:
        print(f"Error saving Excel file: {e}", file=sys.stderr)

def _create_summary_sheet(wb, buckets: dict[str, list[str]], ci_results: dict) -> None:
    import pandas as pd
    from openpyxl.styles import Font, PatternFill

    ws = wb.create_sheet("Summary", 0)
    summary_data = []
    total_tests = sum(len(tests) for tests in buckets.values())

    for bucket in ["P-1", "P0", "P1", "P2", "P3", "CPU"]:
        count = len(buckets[bucket])
        percentage = (count / total_tests * 100) if total_tests > 0 else 0

        bucket_desc = {
            "P-1": "Important Models (auto, bert, clip, t5, etc.) - NON-CPU",
            "P0": "Critical GPU (DeepSpeed, Flash Attention, Quantization) - NON-CPU",
            "P1": "Framework Basics (torch/tf/flax, tokenizers, vision) - NON-CPU",
            "P2": "Remaining Non-CPU Tests - NON-CPU", 
            "P3": "Uncategorized Non-CPU Tests - NON-CPU",
            "CPU": "ALL CPU-only tests (not_device_test, flaky, pipelines, etc.)"
        }

        summary_data.append({
            'Bucket': bucket, 'Description': bucket_desc[bucket], 'Test Count': count,
            'Percentage': f"{percentage:.1f}%", 'CI Status': 'Available' if ci_results else 'Not Available'
        })

    df = pd.DataFrame(summary_data)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for column in ws.columns:
        max_length = max(len(str(cell.value)) for cell in column)
        ws.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)

def _create_bucket_sheet(wb, bucket: str, test_nodeids: list[str], test_metadata_map: dict, ci_results: dict) -> None:
    import pandas as pd
    ws = wb.create_sheet(f"Bucket_{bucket}")

    test_data = []
    for nodeid in test_nodeids:
        metadata = test_metadata_map.get(nodeid, {})
        decorators = metadata.get('decorators', set())

        parts = nodeid.split("::")
        file_path = parts[0] if len(parts) > 0 else ""
        test_class = parts[1] if len(parts) > 1 else ""
        test_method = parts[2] if len(parts) > 2 else parts[-1]

        gpu_decorators = [d for d in decorators if d in P0_DECOS]
        framework_decorators = [d for d in decorators if d in FRAMEWORK_DECOS]
        cpu_decorators = [d for d in decorators if d in CPU_DECOS]
        other_decorators = [d for d in decorators if d not in (P0_DECOS | FRAMEWORK_DECOS | CPU_DECOS)]

        test_data.append({
            'Test NodeID': nodeid, 'File Path': file_path, 'Test Class': test_class,
            'Test Method': test_method, 'Bucket': bucket,
            'Is CPU Test': 'Yes' if is_cpu_test(decorators, nodeid) else 'No',
            'Important Model': 'Yes' if is_important_model_test(nodeid) else 'No',
            'ROCm Hint': 'Yes' if metadata.get('rocm_hint', False) else 'No',
            'GPU Decorators': ', '.join(sorted(gpu_decorators)),
            'Framework Decorators': ', '.join(sorted(framework_decorators)),
            'CPU Decorators': ', '.join(sorted(cpu_decorators)),
            'Other Decorators': ', '.join(sorted(other_decorators)),
            'All Decorators': ', '.join(sorted(decorators)),
            'Decorator Count': len(decorators)
        })

    df = pd.DataFrame(test_data)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)

def _create_decorator_analysis_sheet(wb, test_metadata_map: dict) -> None:
    import pandas as pd
    ws = wb.create_sheet("Decorator_Analysis")

    decorator_counts = defaultdict(int)
    decorator_buckets = defaultdict(lambda: defaultdict(int))

    for nodeid, metadata in test_metadata_map.items():
        decorators = metadata.get('decorators', set())
        bucket = bucket_of(decorators, metadata.get('rocm_hint', False), nodeid)

        for decorator in decorators:
            decorator_counts[decorator] += 1
            decorator_buckets[decorator][bucket] += 1

    decorator_data = []
    for decorator, total_count in sorted(decorator_counts.items(), key=lambda x: x[1], reverse=True):
        bucket_usage = decorator_buckets[decorator]

        category = "Other"
        if decorator in P0_DECOS:
            category = "GPU/Critical"
        elif decorator in FRAMEWORK_DECOS:
            category = "Framework"
        elif decorator in CPU_DECOS:
            category = "CPU"

        decorator_data.append({
            'Decorator': decorator, 'Category': category, 'Total Usage': total_count,
            'P-1 Usage': bucket_usage.get('P-1', 0), 'P0 Usage': bucket_usage.get('P0', 0),
            'P1 Usage': bucket_usage.get('P1', 0), 'P2 Usage': bucket_usage.get('P2', 0),
            'P3 Usage': bucket_usage.get('P3', 0), 'CPU Usage': bucket_usage.get('CPU', 0),
            'Primary Bucket': max(bucket_usage.items(), key=lambda x: x[1])[0] if bucket_usage else 'None'
        })

    df = pd.DataFrame(decorator_data)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)

def _create_cpu_analysis_sheet(wb, buckets: dict[str, list[str]], test_metadata_map: dict) -> None:
    import pandas as pd
    ws = wb.create_sheet("CPU_Analysis")

    cpu_tests = buckets["CPU"]
    non_cpu_buckets = ["P-1", "P0", "P1", "P2", "P3"]
    non_cpu_tests = []
    for bucket in non_cpu_buckets:
        non_cpu_tests.extend(buckets[bucket])

    total_tests = len(cpu_tests) + len(non_cpu_tests)

    cpu_analysis_data = [
        {'Category': 'CPU Tests', 'Count': len(cpu_tests), 
         'Percentage': f"{len(cpu_tests)/total_tests*100:.1f}%",
         'Description': 'Tests marked as CPU-only (not_device_test, flaky, pipelines, etc.)'},
        {'Category': 'Non-CPU Tests', 'Count': len(non_cpu_tests),
         'Percentage': f"{len(non_cpu_tests)/total_tests*100:.1f}%",
         'Description': 'Tests that can run on GPU/accelerators'},
        {'Category': 'Total Tests', 'Count': total_tests, 'Percentage': '100.0%',
         'Description': 'All tests combined'}
    ]

    for bucket in non_cpu_buckets:
        count = len(buckets[bucket])
        cpu_analysis_data.append({
            'Category': f'  └─ {bucket}', 'Count': count,
            'Percentage': f"{count/total_tests*100:.1f}%",
            'Description': f'Non-CPU tests in {bucket} bucket'
        })

    df = pd.DataFrame(cpu_analysis_data)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)

def _create_ci_results_sheet(wb, ci_results: dict) -> None:
    import pandas as pd
    ws = wb.create_sheet("CI_Results")

    results_data = []
    for model_or_section, stats in ci_results.items():
        if isinstance(stats, dict):
            total = stats.get('TOTAL', sum(stats.values()) if 'TOTAL' not in stats else stats['TOTAL'])
            passed = stats.get('PASSED', 0)
            failed = stats.get('FAILED', 0)
            skipped = stats.get('SKIPPED', 0)
            
            success_rate = (passed / total * 100) if total > 0 else 0
            
            results_data.append({
                'Model/Section': model_or_section, 'Total': total, 'Passed': passed,
                'Failed': failed, 'Skipped': skipped, 'Success Rate (%)': f"{success_rate:.1f}"
            })

    df = pd.DataFrame(results_data)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)

def safe_print(text: str) -> bool:
    try:
        print(text)
        return True
    except BrokenPipeError:
        return False

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_argument
    
    # Discovery options
    g("--bucket", choices=["P-1", "P0", "P1", "P2", "P3", "CPU"], help="Emit only that bucket's nodeids to stdout")
    g("--yaml", default="buckets.yaml", help="Where to save YAML when not printing")
    g("--print", action="store_true", help="Pretty-print bucket contents")
    g("--export-excel", metavar="FILE.xlsx", help="Export test data to Excel file")
    
    # CI results analysis
    g("--analyze-ci", metavar="CI_FILE", help="Analyze CI results file and integrate with buckets")
    
    g("--debug", action="store_true", help="Show detailed debug information")
    args = ap.parse_args()

    global DEBUG_MODE
    DEBUG_MODE = args.debug

    print("=== Scanning and analyzing test distribution with CPU separation ===", file=sys.stderr)
    buckets, decorator_stats, test_metadata_map = scan_tests()

    ci_results = {}
    if args.analyze_ci:
        ci_file = pathlib.Path(args.analyze_ci)
        if ci_file.exists():
            ci_results = parse_ci_results(ci_file)
        else:
            print(f"CI results file not found: {ci_file}", file=sys.stderr)

    if args.print or not (args.export_excel or args.analyze_ci):
        analyze_bucket_distribution(buckets, decorator_stats, test_metadata_map)

    # Discovery mode
    if not args.export_excel and not args.analyze_ci:
        if args.bucket:
            for n in buckets[args.bucket]:
                if not safe_print(n):
                    sys.exit(0)
            return
        if args.print:
            for b in ("P-1", "P0", "P1", "P2", "P3", "CPU"):
                if not safe_print(f"\n=== {b} ({len(buckets[b])} tests) ==="):
                    sys.exit(0)
                for n in buckets[b]:
                    if not safe_print(n):
                        sys.exit(0)
        else:
            try:
                with open(args.yaml, "w") as f:
                    json.dump(buckets, f, indent=2)
                print(f"Wrote {args.yaml}")
            except Exception as e:
                print(f"Error writing {args.yaml}: {e}", file=sys.stderr)
                sys.exit(1)
        return

    # Excel export
    if args.export_excel:
        create_excel_export(buckets, test_metadata_map, args.export_excel, ci_results)

if __name__ == "__main__":
    main()
