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
• Multi-GPU analysis: python bucket_tests.py --analyze-ci file1.txt file2.txt file3.txt --export-excel multi_gpu_report.xlsx

Multi-GPU CI File Format:
Each CI results file should start with a JSON metadata line:
{"gpu_name": "h100", "commit_hash": "4d57c39", "total_status_count": {"passed": 24879, "failed": 1114, "skipped": 25752}}

Followed by test results grouped by model:
model_name
PASSED test_path::TestClass::test_method
FAILED test_path::TestClass::test_method
...

Key fixes in this version:
- Correctly parses CI files into *per-test* (nodeid) status maps (individual_tests), so coverage is non-zero.
- Normalizes CI status keys to UPPERCASE and computes TOTAL per section.
- Passes individual test statuses into Excel bucket sheets.
- Restores and cleans up `analyze_bucket_distribution` (separate from coverage).
- Fixes Excel export imports (Alignment, dataframe_to_rows) inside helper functions to avoid NameError.

"""

from __future__ import annotations
import argparse, ast, json, os, pathlib, re, sys, xml.etree.ElementTree as ET
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Tuple, Set
import subprocess
import tempfile

from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment

# Import important models and CPU test identifiers from external modules
#from utils.test_fetcher import IMPORTANT_MODELS
#from conftest import NOT_DEVICE_TESTS

IMPORTANT_MODELS = [
    "auto",
    # Most downloaded models
    "bert",
    "clip",
    "t5",
    "xlm-roberta",
    "gpt2",
    "bart",
    "mpnet",
    "gpt-j",
    "wav2vec2",
    "deberta-v2",
    "layoutlm",
    "llama",
    "opt",
    "longformer",
    "vit",
    "whisper",
    # Pipeline-specific model (to be sure each pipeline has one model in this list)
    "tapas",
    "vilt",
    "clap",
    "detr",
    "owlvit",
    "dpt",
    "videomae",
]

NOT_DEVICE_TESTS = {
    "test_tokenization",
    "test_tokenization_mistral_common",
    "test_processor",
    "test_processing",
    "test_beam_constraints",
    "test_configuration_utils",
    "test_data_collator",
    "test_trainer_callback",
    "test_trainer_utils",
    "test_feature_extraction",
    "test_image_processing",
    "test_image_processor",
    "test_image_transforms",
    "test_optimization",
    "test_retrieval",
    "test_config",
    "test_from_pretrained_no_checkpoint",
    "test_keep_in_fp32_modules",
    "test_gradient_checkpointing_backward_compatibility",
    "test_gradient_checkpointing_enable_disable",
    "test_torch_save_load",
    "test_initialization",
    "test_forward_signature",
    "test_model_get_set_embeddings",
    "test_model_main_input_name",
    "test_correct_missing_keys",
    "test_tie_model_weights",
    "test_can_use_safetensors",
    "test_load_save_without_tied_weights",
    "test_tied_weights_keys",
    "test_model_weights_reload_no_missing_tied_weights",
    "test_mismatched_shapes_have_properly_initialized_weights",
    "test_matched_shapes_have_loaded_weights_when_some_mismatched_shapes_exist",
    "test_model_is_small",
    "test_tf_from_pt_safetensors",
    "test_flax_from_pt_safetensors",
    "ModelTest::test_pipeline_",  # None of the pipeline tests from PipelineTesterMixin (of which XxxModelTest inherits from) are running on device
    "ModelTester::test_pipeline_",
    "/repo_utils/",
    "/utils/",
}

#decorator_stats = defaultdict(int) above scan_tests()


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

# Create normalized set of important models for efficient lookup
IMPORTANT_MODELS_SET = set()
for model in IMPORTANT_MODELS:
    IMPORTANT_MODELS_SET.add(model.lower())
    IMPORTANT_MODELS_SET.add(model.replace("-", "_"))
    IMPORTANT_MODELS_SET.add(model.replace("_", "-"))

# CPU test identification decorators - these mark tests that should run on CPU only
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

# CPU test path patterns - additional patterns to identify CPU-only tests
CPU_PATH_PATTERNS = [
    "cpu", "not_device", "flaky", "slow", "pipeline", "agent",
    "cross_test", "audio", "text_processing", "data_processing"
]

# P0: Critical GPU tests - these require GPU/accelerator resources and are high priority
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

# P1: Framework basics - core framework dependencies without device specificity
FRAMEWORK_DECOS = {
    "require_torch", "require_tf", "require_flax", "require_jax", "require_torch_or_tf",
    "require_tokenizers", "require_sentencepiece", "require_sacremoses", "require_tiktoken",
    "require_seqio", "require_safetensors", "require_pandas", "require_scipy",
    "require_vision", "require_torchvision", "require_torchaudio", "require_tensorflow_text",
    "require_keras_nlp", "require_tensorflow_probability", "require_onnx", "require_tf2onnx",
    "require_gguf", "require_tensorboard"
}

# P2: Additional decorators (currently empty)
P2_DECOS = set()

# Decorator sets for slow tests and pipeline tests
SLOW_DECOS = {"slow", "tooslow"}
PIPELINE_DECOS = {"is_pipeline_test", "is_agent_test"}

# Directories to exclude from test discovery
EXCLUDE_DIRS = {"examples", "templates"}

def is_excluded(path: pathlib.Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return True
    if any(pattern in str(path) for pattern in ["__pycache__", ".git", ".pytest_cache"]):
        return True
    return False


def is_not_device_test(nodeid: str) -> bool:
    return any(test_name in nodeid for test_name in NOT_DEVICE_TESTS)


def is_cpu_test(decos: set[str], nodeid: str, file_path: pathlib.Path | None = None) -> bool:
    if decos & CPU_DECOS:
        return True
    if is_not_device_test(nodeid):
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


def is_important_model_test(nodeid: str, file_path: pathlib.Path | None = None) -> bool:
    nodeid_lower = nodeid.lower()
    models_match = re.search(r'tests/models/([^/]+)/', nodeid_lower)
    if models_match:
        model_name = models_match.group(1)
        model_clean = model_name.replace("_", "-").replace("-", "_")
        if (
            model_name in IMPORTANT_MODELS_SET
            or model_clean in IMPORTANT_MODELS_SET
            or any(important in model_name for important in IMPORTANT_MODELS_SET if len(important) > 3)
        ):
            return True
    for important_model in IMPORTANT_MODELS_SET:
        if len(important_model) > 3:
            patterns = [
                f"test_{important_model}",
                f"test_modeling_{important_model}",
                f"{important_model}_test",
                f"modeling_{important_model}",
            ]
            if any(p in nodeid_lower for p in patterns):
                return True
    if "auto" in IMPORTANT_MODELS_SET:
        for pattern in [
            "test_auto_",
            "/auto/",
            "auto_test",
            "autoprocessor",
            "automodel",
            "autotokenizer",
        ]:
            if pattern in nodeid_lower:
                return True
    return False


def discover_tests_with_pytest() -> list[str]:
    print("Using pytest to discover all tests...", file=sys.stderr)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(TEST_GLOB_ROOT),
        "--collect-only",
        "--quiet",
        "--continue-on-collection-errors",
        "-p",
        "no:cacheprovider",
        "--ignore-glob=**/test_torch_compile.py",
        "--ignore-glob=**/test_doctests.py",
        "--ignore-glob=**/test_torchao.py",
        "--ignore-glob=**/test_trainer.py",
    ]

    env = os.environ.copy()
    env.update(
        {
            "RUN_SLOW": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "TRANSFORMERS_VERBOSITY": "error",
            "TRANSFORMERS_TEST_DEVICE": "cpu",
        }
    )

    try:
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env, cwd=REPO_ROOT
        )

        if result.returncode != 0:
            print(f"Pytest collection stderr: {result.stderr}", file=sys.stderr)

        nodeids: list[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("tests/") and "::" in line and not line.startswith("<") and not line.startswith("="):
                nodeid = line.strip()
                if nodeid.endswith(">"):
                    continue
                if " " in nodeid:
                    continue
                nodeids.append(nodeid)

        print(f"Pytest discovered {len(nodeids)} tests", file=sys.stderr)

        if len(nodeids) < 1000:
            print(
                "Low test count from direct nodeids, complementing with file-based discovery...",
                file=sys.stderr,
            )
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
    nodeids: list[str] = []
    files_processed = 0

    if not TEST_GLOB_ROOT.exists():
        print(f"Test directory {TEST_GLOB_ROOT} does not exist", file=sys.stderr)
        return []

    for py_file in TEST_GLOB_ROOT.rglob("test_*.py"):
        if is_excluded(py_file):
            continue

        files_processed += 1
        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}", file=sys.stderr)
            continue

        try:
            rel_path = py_file.relative_to(REPO_ROOT)
        except ValueError:
            rel_path = py_file

        file_str = str(rel_path).replace("\\", "/")
        has_parameterized = bool(re.search(r"@parameterized\.expand", content))

        if has_parameterized:
            nodeids.append(file_str)
        else:
            test_functions = re.findall(r"def\s+(test_\w+)\s*\(", content)
            test_classes = re.findall(r"class\s+(\w*Test\w*)\s*[:\(]", content)

            if test_classes:
                for test_class in test_classes:
                    class_methods = re.findall(
                        rf"class\s+{re.escape(test_class)}.*?(?=class|\Z)", content, re.DOTALL
                    )
                    if class_methods:
                        class_content = class_methods[0]
                        methods = re.findall(r"def\s+(test_\w+)\s*\(", class_content)
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

    print(
        f"File-based discovery: processed {files_processed} files, found {len(nodeids)} tests",
        file=sys.stderr,
    )
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
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "pytest"
                and node.func.value.attr == "mark"
            ):
                markers.add(node.func.attr)
        elif isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "pytest"
                and node.value.attr == "mark"
            ):
                markers.add(node.attr)
        if isinstance(node, ast.Name):
            name = node.id
            relevant_decos = (
                P0_DECOS
                | CPU_DECOS
                | FRAMEWORK_DECOS
                | {
                    "not_device_test",
                    "slow",
                    "tooslow",
                    "is_flaky",
                    "is_staging_test",
                    "is_pt_tf_cross_test",
                    "is_pt_flax_cross_test",
                    "run_first",
                }
            )
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
        if node.name.startswith("test") or "test" in node.name.lower() or func_decos:
            self.test_metadata[node.name] = func_decos


def extract_test_metadata(file_path: pathlib.Path) -> tuple[dict[str, set[str]], bool]:
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        regex_markers = set()
        decorator_patterns = [
            (r"@not_device_test\b", "not_device_test"),
            (r"@pytest\.mark\.not_device_test\b", "not_device_test"),
            (r"@slow\b", "slow"),
            (r"@tooslow\b", "tooslow"),
            (r"@is_flaky\b", "is_flaky"),
            (r"@is_pipeline_test\b", "is_pipeline_test"),
            (r"@is_agent_test\b", "is_agent_test"),
            (r"@require_torch\b", "require_torch"),
            (r"@require_tf\b", "require_tf"),
            (r"@require_torch_gpu\b", "require_torch_gpu"),
            (r"@require_deepspeed\b", "require_deepspeed"),
            (r"@require_accelerate\b", "require_accelerate"),
            (r"@require_flash_attn\b", "require_flash_attn"),
            (r"@require_bitsandbytes\b", "require_bitsandbytes"),
            (r"@require_vision\b", "require_vision"),
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
            extractor.test_metadata["__file_markers__"] = regex_markers
        rocm_hint = bool(
            re.search(r"\b(IS_ROCM_SYSTEM|torch\.version\.hip|rocm|hip_version)\b", content, re.IGNORECASE)
        )
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
    if any(
        pattern in nodeid_lower
        for pattern in ["deepspeed", "fsdp", "accelerate", "flash_attn", "gpu", "cuda", "multi_gpu", "bitsandbytes"]
    ):
        return "P0"
    return "P2"

decorator_stats = defaultdict(int)
def scan_tests() -> tuple[dict[str, list[str]], dict[str, int], dict[str, dict[str, set[str]]]]:
    buckets: dict[str, list[str]] = {"P-1": [], "P0": [], "P1": [], "P2": [], "P3": [], "CPU": []}
    test_metadata_map: dict[str, dict[str, set[str]]] = {}

    all_nodeids = discover_tests_with_pytest()
    if not all_nodeids:
        print("No tests discovered!", file=sys.stderr)
        return buckets, {}, test_metadata_map

    print(
        f"Processing {len(all_nodeids)} discovered tests for bucketing with CPU separation...",
        file=sys.stderr,
    )

    tests_by_file = defaultdict(list)
    for nodeid in all_nodeids:
        file_part = nodeid.split("::")[0] if "::" in nodeid else nodeid
        file_path = REPO_ROOT / file_part
        tests_by_file[file_path].append(nodeid)

    print(f"Found tests in {len(tests_by_file)} files", file=sys.stderr)

    #decorator_stats = defaultdict(int)
    files_with_metadata = 0
    important_model_count = 0
    cpu_test_count = 0

    for file_path, nodeids_in_file in tests_by_file.items():
        test_metadata, rocm_hint = extract_test_metadata(file_path)
        if test_metadata:
            files_with_metadata += 1
        for nodeid in nodeids_in_file:
            test_name = nodeid.split("::")[-1] if "::" in nodeid else "__file_level__"
            decos = test_metadata.get(test_name, set())
            if not decos:
                for meta_test_name, meta_decos in test_metadata.items():
                    if test_name in meta_test_name or meta_test_name in test_name:
                        decos = meta_decos
                        break
                if not decos and "__file_markers__" in test_metadata:
                    decos = test_metadata["__file_markers__"]
            test_metadata_map[nodeid] = {
                "decorators": decos,
                "rocm_hint": rocm_hint,
                "file_path": str(file_path),
                "test_name": test_name,
            }
            if is_important_model_test(nodeid):
                important_model_count += 1
            if is_cpu_test(decos, nodeid):
                cpu_test_count += 1
            bucket = bucket_of(decos, rocm_hint, nodeid)
            buckets[bucket].append(nodeid)
            for deco in decos:
                decorator_stats[deco] += 1

    total_tests = sum(len(tests) for tests in buckets.values())
    print(
        f"Bucketed {total_tests} tests from {files_with_metadata} files with metadata",
        file=sys.stderr,
    )
    print(f"Found {important_model_count} important model tests for P-1", file=sys.stderr)
    print(f"Found {cpu_test_count} CPU tests for CPU bucket", file=sys.stderr)

    for lst in buckets.values():
        lst.sort()

    return buckets, dict(decorator_stats), test_metadata_map


def analyze_test_coverage(buckets: dict[str, list[str]], ci_results: dict) -> None:
    if not ci_results:
        return
    all_discovered_tests = set()
    for bucket_tests in buckets.values():
        all_discovered_tests.update(bucket_tests)

    print(f"\n=== Test Coverage Analysis ===", file=sys.stderr)
    print(f"Total discovered tests: {len(all_discovered_tests)}", file=sys.stderr)

    for gpu_name, gpu_data in ci_results.items():
        individual_tests: dict[str, str] = gpu_data.get("individual_tests", {})
        ci_test_nodeids = set(individual_tests.keys())

        print(f"\nGPU: {gpu_name}", file=sys.stderr)
        print(f"  CI test count: {len(ci_test_nodeids)}", file=sys.stderr)
        overlap = all_discovered_tests & ci_test_nodeids
        only_in_discovery = all_discovered_tests - ci_test_nodeids
        only_in_ci = ci_test_nodeids - all_discovered_tests
        print(f"  Overlap: {len(overlap)} tests", file=sys.stderr)
        print(f"  Only in discovery: {len(only_in_discovery)} tests", file=sys.stderr)
        print(f"  Only in CI: {len(only_in_ci)} tests", file=sys.stderr)
        if DEBUG_MODE and overlap:
            print(f"  Sample overlapping tests:", file=sys.stderr)
            for i, test in enumerate(list(overlap)[:3]):
                status = individual_tests[test]
                print(f"    {i+1}. {status} {test}", file=sys.stderr)
        if DEBUG_MODE and only_in_discovery:
            print(f"  Sample discovery-only tests:", file=sys.stderr)
            for i, test in enumerate(list(only_in_discovery)[:3]):
                print(f"    {i+1}. {test}", file=sys.stderr)
        if DEBUG_MODE and only_in_ci:
            print(f"  Sample CI-only tests:", file=sys.stderr)
            for i, test in enumerate(list(only_in_ci)[:3]):
                status = individual_tests[test]
                print(f"    {i+1}. {status} {test}", file=sys.stderr)


def analyze_bucket_distribution(
    buckets: dict[str, list[str]], decorator_stats: dict[str, int], test_metadata_map: dict[str, dict]
) -> None:
    total_tests = sum(len(tests) for tests in buckets.values())
    print("\n=== Bucket Distribution Analysis (with CPU separation) ===", file=sys.stderr)
    bucket_descriptions = {
        "P-1": "Important Models (auto, bert, clip, t5, etc.) - NON-CPU ONLY",
        "P0": "Critical GPU (DeepSpeed, Flash Attention, Quantization, Multi-GPU) - NON-CPU ONLY",
        "P1": "Framework Basics (torch/tf/flax, tokenizers, vision, data handling) - NON-CPU ONLY",
        "P2": "Remaining Non-CPU Tests - NON-CPU ONLY",
        "P3": "Uncategorized Non-CPU Tests - NON-CPU ONLY",
        "CPU": "ALL CPU-only tests (not_device_test, flaky, pipelines, audio, etc.)",
    }
    for bucket in ["P-1", "P0", "P1", "P2", "P3", "CPU"]:
        count = len(buckets[bucket])
        percentage = (count / total_tests * 100) if total_tests > 0 else 0
        print(f"{bucket}: {count:4d} tests ({percentage:5.1f}%) - {bucket_descriptions[bucket]}", file=sys.stderr)
    print(f"Total: {total_tests} tests", file=sys.stderr)


# ------------------------------
# CI parsing (fixed)
# ------------------------------

def _normalize_status_key(k: str) -> str:
    return k.strip().upper()


def parse_ci_results(ci_file: pathlib.Path) -> tuple[dict[str, dict[str, int]], dict[str, str], dict[str, str]]:
    """
    Returns (results_by_model, metadata, individual_tests)
    - results_by_model includes an "OVERALL" section with UPPERCASE keys and TOTAL.
    - individual_tests maps exact nodeid -> status (UPPERCASE).
    """
    print(f"Parsing CI results from {ci_file}", file=sys.stderr)
    try:
        content = ci_file.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"Error reading CI results file: {e}", file=sys.stderr)
        return {}, {}, {}

    results: dict[str, dict[str, int]] = {}
    metadata: dict[str, str] = {"gpu_name": "unknown", "commit_hash": "unknown", "file_name": ci_file.name}
    individual_tests: dict[str, str] = {}

    current_model: str | None = None
    model_results = {"PASSED": 0, "FAILED": 0, "SKIPPED": 0, "ERROR": 0}
    lines = content.splitlines()

    # First line metadata (JSON)
    if lines and lines[0].lstrip().startswith("{"):
        try:
            first_line_data = json.loads(lines[0])
            if isinstance(first_line_data, dict):
                metadata.update(
                    {
                        "gpu_name": first_line_data.get("gpu_name", "unknown"),
                        "commit_hash": first_line_data.get("commit_hash", "unknown"),
                        "total_status_count": first_line_data.get("total_status_count", {}),
                    }
                )
                # Normalize to uppercase keys for OVERALL
                overall = {}
                raw_overall = first_line_data.get("total_status_count", {}) or {}
                for k, v in raw_overall.items():
                    k_up = _normalize_status_key(k)
                    if k_up in {"PASSED", "FAILED", "SKIPPED", "ERROR"}:
                        overall[k_up] = int(v)
                overall["TOTAL"] = sum(overall.values())
                results["OVERALL"] = overall
        except json.JSONDecodeError:
            print(f"Warning: Could not parse metadata from first line of {ci_file}", file=sys.stderr)

    # Remaining lines
    for raw in lines[1:]:
        line = raw.strip()
        if not line:
            continue
        # Heuristic for section/model header: single token (letters/underscores/dashes), no '::'
        if (
            not line.startswith(("PASSED ", "FAILED ", "SKIPPED ", "ERROR "))
            and "::" not in line
            and line.replace("_", "").replace("-", "").isalnum()
        ):
            if current_model and any(model_results.values()):
                model_results["TOTAL"] = sum(model_results.values())
                results[current_model] = model_results.copy()
            current_model = line
            model_results = {"PASSED": 0, "FAILED": 0, "SKIPPED": 0, "ERROR": 0}
            continue
        # Test line
        if line.startswith(("PASSED ", "FAILED ", "SKIPPED ", "ERROR ")):
            try:
                status, nodeid = line.split(" ", 1)
            except ValueError:
                continue
            status = _normalize_status_key(status)
            nodeid = nodeid.strip()
            # Record individual test outcome
            individual_tests[nodeid] = status
            # Tally under current model if any
            if current_model and status in model_results:
                model_results[status] += 1

    if current_model and any(model_results.values()):
        model_results["TOTAL"] = sum(model_results.values())
        results[current_model] = model_results.copy()

    print(
        f"Parsed results for {len(results)} models/sections from GPU: {metadata['gpu_name']}",
        file=sys.stderr,
    )
    return results, metadata, individual_tests


def parse_multiple_ci_results(ci_files: list[pathlib.Path]) -> dict[str, dict[str, dict[str, int] | dict | str]]:
    all_gpu_results: dict[str, dict[str, dict | str]] = {}
    for ci_file in ci_files:
        if not ci_file.exists():
            print(f"CI results file not found: {ci_file}", file=sys.stderr)
            continue
        results, metadata, individual_tests = parse_ci_results(ci_file)
        gpu_name = metadata.get("gpu_name", "unknown")
        all_gpu_results[gpu_name] = {
            "results": results,
            "metadata": metadata,
            "individual_tests": individual_tests,
        }
    print(
        f"Parsed CI results from {len(all_gpu_results)} GPU types: {list(all_gpu_results.keys())}",
        file=sys.stderr,
    )
    return all_gpu_results


# ------------------------------
# Excel export helpers (fixed imports per function)
# ------------------------------

def create_excel_export(
    buckets: dict[str, list[str]],
    test_metadata_map: dict[str, dict[str, set[str]]],
    output_path: str,
    ci_results: dict | None = None,
) -> None:
    try:
        import pandas as pd  # noqa: F401
        import openpyxl  # noqa: F401
        from openpyxl.styles import Font, PatternFill, Alignment  # noqa: F401
        from openpyxl.utils.dataframe import dataframe_to_rows  # noqa: F401
    except ImportError:
        print("Error: pandas and openpyxl are required for Excel export.", file=sys.stderr)
        print("Install with: pip install pandas openpyxl", file=sys.stderr)
        return

    print(f"Creating Excel export with CPU separation: {output_path}", file=sys.stderr)
    if ci_results:
        gpu_names = list(ci_results.keys())
        print(f"Including CI results from GPUs: {gpu_names}", file=sys.stderr)

    import openpyxl
    wb = openpyxl.Workbook()
    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])

    _create_summary_sheet(wb, buckets, ci_results or {})

    for bucket in ["P-1", "P0", "P1", "P2", "P3", "CPU"]:
        if not buckets[bucket]:
            continue
        print(
            f"Creating sheet for bucket {bucket} ({len(buckets[bucket])} tests)",
            file=sys.stderr,
        )
        _create_bucket_sheet(wb, bucket, buckets[bucket], test_metadata_map, ci_results or {})

    _create_decorator_analysis_sheet(wb, test_metadata_map)
    _create_cpu_analysis_sheet(wb, buckets, test_metadata_map)

    if ci_results:
        _create_multi_gpu_ci_results_sheet(wb, ci_results)
        _create_gpu_comparison_sheet(wb, ci_results)

    try:
        wb.save(output_path)
        print(f"✅ Excel file saved: {output_path}", file=sys.stderr)
        print(f"📊 Created {len(wb.sheetnames)} sheets with test data and analysis", file=sys.stderr)
    except Exception as e:
        print(f"Error saving Excel file: {e}", file=sys.stderr)


def _create_summary_sheet(wb, buckets: dict[str, list[str]], ci_results: dict) -> None:
    import pandas as pd
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows

    ws = wb.create_sheet("Summary", 0)
    summary_data = []
    total_tests = sum(len(tests) for tests in buckets.values())

    bucket_desc = {
        "P-1": "Important Models (auto, bert, clip, t5, etc.) - NON-CPU",
        "P0": "Critical GPU (DeepSpeed, Flash Attention, Quantization) - NON-CPU",
        "P1": "Framework Basics (torch/tf/flax, tokenizers, vision) - NON-CPU",
        "P2": "Remaining Non-CPU Tests - NON-CPU",
        "P3": "Uncategorized Non-CPU Tests - NON-CPU",
        "CPU": "ALL CPU-only tests (not_device_test, flaky, pipelines, etc.)",
    }

    ci_status = "Not Available"
    if ci_results:
        gpu_names = list(ci_results.keys())
        ci_status = f"Available ({len(gpu_names)} GPUs: {', '.join(gpu_names)})"

    for bucket in ["P-1", "P0", "P1", "P2", "P3", "CPU"]:
        count = len(buckets[bucket])
        percentage = (count / total_tests * 100) if total_tests > 0 else 0
        summary_data.append(
            {
                "Bucket": bucket,
                "Description": bucket_desc[bucket],
                "Test Count": count,
                "Percentage": f"{percentage:.1f}%",
                "CI Status": ci_status,
            }
        )

    df = pd.DataFrame(summary_data)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)

    if ci_results:
        ws.append([])
        ws.append(["GPU Comparison Summary"])
        ws.append(["GPU Name", "Commit Hash", "Total Tests", "Passed", "Failed", "Skipped", "Error", "Success Rate"])
        for gpu_name, gpu_data in ci_results.items():
            metadata = gpu_data.get("metadata", {})
            overall_stats = gpu_data.get("results", {}).get("OVERALL", {})
            commit_hash = metadata.get("commit_hash", "unknown")
            total = overall_stats.get("TOTAL", sum(overall_stats.values()) if overall_stats else 0)
            passed = overall_stats.get("PASSED", 0)
            failed = overall_stats.get("FAILED", 0)
            skipped = overall_stats.get("SKIPPED", 0)
            error = overall_stats.get("ERROR", 0)
            success_rate = (passed / total * 100) if total > 0 else 0
            ws.append([gpu_name, commit_hash, total, passed, failed, skipped, error, f"{success_rate:.1f}%"])

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for column in ws.columns:
        max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
        ws.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)


def _create_bucket_sheet(wb, bucket: str, test_nodeids: list[str], test_metadata_map: dict, ci_results: dict) -> None:
    import pandas as pd
    from openpyxl.utils.dataframe import dataframe_to_rows

    ws = wb.create_sheet(f"Bucket_{bucket}")

    test_data = []
    for nodeid in test_nodeids:
        metadata = test_metadata_map.get(nodeid, {})
        decorators = metadata.get("decorators", set())
        parts = nodeid.split("::")
        file_path = parts[0] if len(parts) > 0 else ""
        test_class = parts[1] if len(parts) > 1 else ""
        test_method = parts[2] if len(parts) > 2 else parts[-1]

        gpu_decorators = [d for d in decorators if d in P0_DECOS]
        framework_decorators = [d for d in decorators if d in FRAMEWORK_DECOS]
        cpu_decorators = [d for d in decorators if d in CPU_DECOS]
        other_decorators = [d for d in decorators if d not in (P0_DECOS | FRAMEWORK_DECOS | CPU_DECOS)]

        test_row = {
            "Test NodeID": nodeid,
            "File Path": file_path,
            "Test Class": test_class,
            "Test Method": test_method,
            "Bucket": bucket,
            "Is CPU Test": "Yes" if is_cpu_test(decorators, nodeid) else "No",
            "Important Model": "Yes" if is_important_model_test(nodeid) else "No",
            "ROCm Hint": "Yes" if metadata.get("rocm_hint", False) else "No",
            "GPU Decorators": ", ".join(sorted(gpu_decorators)),
            "Framework Decorators": ", ".join(sorted(framework_decorators)),
            "CPU Decorators": ", ".join(sorted(cpu_decorators)),
            "Other Decorators": ", ".join(sorted(other_decorators)),
            "All Decorators": ", ".join(sorted(decorators)),
            "Decorator Count": len(decorators),
        }

        if ci_results:
            for gpu_name, gpu_data in ci_results.items():
                individual_tests = gpu_data.get("individual_tests", {})
                test_status = individual_tests.get(nodeid, "Not Found")
                test_row[f"{gpu_name}_Status"] = test_status

        test_data.append(test_row)

    df = pd.DataFrame(test_data)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)


def _create_decorator_analysis_sheet(wb, test_metadata_map: dict) -> None:
    import pandas as pd
    from openpyxl.utils.dataframe import dataframe_to_rows

    ws = wb.create_sheet("Decorator_Analysis")

    decorator_counts = defaultdict(int)
    decorator_buckets = defaultdict(lambda: defaultdict(int))

    for nodeid, metadata in test_metadata_map.items():
        decorators = metadata.get("decorators", set())
        bucket = bucket_of(decorators, metadata.get("rocm_hint", False), nodeid)
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
        decorator_data.append(
            {
                "Decorator": decorator,
                "Category": category,
                "Total Usage": total_count,
                "P-1 Usage": bucket_usage.get("P-1", 0),
                "P0 Usage": bucket_usage.get("P0", 0),
                "P1 Usage": bucket_usage.get("P1", 0),
                "P2 Usage": bucket_usage.get("P2", 0),
                "P3 Usage": bucket_usage.get("P3", 0),
                "CPU Usage": bucket_usage.get("CPU", 0),
                "Primary Bucket": max(bucket_usage.items(), key=lambda x: x[1])[0] if bucket_usage else "None",
            }
        )

    df = pd.DataFrame(decorator_data)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)


def _create_cpu_analysis_sheet(wb, buckets: dict[str, list[str]], test_metadata_map: dict) -> None:
    import pandas as pd
    from openpyxl.utils.dataframe import dataframe_to_rows

    ws = wb.create_sheet("CPU_Analysis")

    cpu_tests = buckets["CPU"]
    non_cpu_buckets = ["P-1", "P0", "P1", "P2", "P3"]
    non_cpu_tests: list[str] = []
    for bucket in non_cpu_buckets:
        non_cpu_tests.extend(buckets[bucket])

    total_tests = len(cpu_tests) + len(non_cpu_tests)

    cpu_analysis_data = [
        {
            "Category": "CPU Tests",
            "Count": len(cpu_tests),
            "Percentage": f"{len(cpu_tests)/total_tests*100:.1f}%" if total_tests else "0.0%",
            "Description": "Tests marked as CPU-only (not_device_test, flaky, pipelines, etc.)",
        },
        {
            "Category": "Non-CPU Tests",
            "Count": len(non_cpu_tests),
            "Percentage": f"{len(non_cpu_tests)/total_tests*100:.1f}%" if total_tests else "0.0%",
            "Description": "Tests that can run on GPU/accelerators",
        },
        {"Category": "Total Tests", "Count": total_tests, "Percentage": "100.0%", "Description": "All tests combined"},
    ]

    for bucket in non_cpu_buckets:
        count = len(buckets[bucket])
        cpu_analysis_data.append(
            {
                "Category": f"  └─ {bucket}",
                "Count": count,
                "Percentage": f"{count/total_tests*100:.1f}%" if total_tests else "0.0%",
                "Description": f"Non-CPU tests in {bucket} bucket",
            }
        )

    df = pd.DataFrame(cpu_analysis_data)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)


def _create_multi_gpu_ci_results_sheet(wb, ci_results: dict) -> None:
    import pandas as pd
    from openpyxl.utils.dataframe import dataframe_to_rows

    ws = wb.create_sheet("Multi_GPU_CI_Results")

    results_data = []
    for gpu_name, gpu_data in ci_results.items():
        gpu_results = gpu_data.get("results", {})
        metadata = gpu_data.get("metadata", {})
        commit_hash = metadata.get("commit_hash", "unknown")
        for model_or_section, stats in gpu_results.items():
            if isinstance(stats, dict) and model_or_section != "OVERALL":
                total = stats.get("TOTAL", sum(v for k, v in stats.items() if k != "TOTAL"))
                passed = stats.get("PASSED", 0)
                failed = stats.get("FAILED", 0)
                skipped = stats.get("SKIPPED", 0)
                error = stats.get("ERROR", 0)
                success_rate = (passed / total * 100) if total > 0 else 0
                results_data.append(
                    {
                        "GPU Type": gpu_name,
                        "Commit Hash": commit_hash,
                        "Model/Section": model_or_section,
                        "Total Tests": total,
                        "Passed": passed,
                        "Failed": failed,
                        "Skipped": skipped,
                        "Error": error,
                        "Success Rate (%)": f"{success_rate:.1f}",
                        "Failure Rate (%)": f"{(failed/total*100) if total > 0 else 0:.1f}",
                    }
                )

    df = pd.DataFrame(results_data)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)


def _create_gpu_comparison_sheet(wb, ci_results: dict) -> None:
    import pandas as pd
    from openpyxl.utils.dataframe import dataframe_to_rows

    ws = wb.create_sheet("GPU_Comparison")

    all_models = set()
    for gpu_data in ci_results.values():
        gpu_results = gpu_data.get("results", {})
        all_models.update(model for model in gpu_results.keys() if model != "OVERALL")

    comparison_data = []
    for model in sorted(all_models):
        row_data: dict[str, Any] = {"Model": model}
        for gpu_name, gpu_data in ci_results.items():
            gpu_results = gpu_data.get("results", {})
            model_stats = gpu_results.get(model, {})
            if model_stats:
                total = model_stats.get("TOTAL", sum(v for k, v in model_stats.items() if k != "TOTAL"))
                passed = model_stats.get("PASSED", 0)
                failed = model_stats.get("FAILED", 0)
                success_rate = (passed / total * 100) if total > 0 else 0
                row_data[f"{gpu_name}_Total"] = total
                row_data[f"{gpu_name}_Passed"] = passed
                row_data[f"{gpu_name}_Failed"] = failed
                row_data[f"{gpu_name}_Success_Rate"] = f"{success_rate:.1f}%"
            else:
                row_data[f"{gpu_name}_Total"] = 0
                row_data[f"{gpu_name}_Passed"] = 0
                row_data[f"{gpu_name}_Failed"] = 0
                row_data[f"{gpu_name}_Success_Rate"] = "N/A"
        comparison_data.append(row_data)

    df = pd.DataFrame(comparison_data)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)


# ------------------------------
# Entrypoint
# ------------------------------

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
    g(
        "--analyze-ci",
        metavar="CI_FILE",
        nargs="+",
        help=(
            "Analyze one or more CI results files for multi-GPU comparison. "
            + "Each file should start with JSON metadata: {\"gpu_name\": \"h100\", \"commit_hash\": \"abc123\", ...}"
        ),
    )

    g("--debug", action="store_true", help="Show detailed debug information including test matching analysis")
    args = ap.parse_args()

    global DEBUG_MODE
    DEBUG_MODE = args.debug

    print("=== Scanning and analyzing test distribution with CPU separation ===", file=sys.stderr)
    buckets, decorator_stats, test_metadata_map = scan_tests()

    ci_results = {}
    if args.analyze_ci:
        ci_file_paths = [pathlib.Path(f) for f in args.analyze_ci]
        print(f"Multi-GPU CI analysis mode: processing {len(ci_file_paths)} files", file=sys.stderr)
        for i, path in enumerate(ci_file_paths):
            print(f"  {i+1}. {path}", file=sys.stderr)
        ci_results = parse_multiple_ci_results(ci_file_paths)

    if args.print or not (args.export_excel or args.analyze_ci):
        analyze_bucket_distribution(buckets, decorator_stats, test_metadata_map)

    if ci_results:
        analyze_test_coverage(buckets, ci_results)

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

    if args.export_excel:
        create_excel_export(buckets, test_metadata_map, args.export_excel, ci_results)


if __name__ == "__main__":
    main()
