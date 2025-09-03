#!/usr/bin/env python3
"""
AMD frameworks-CI utility - Complete Test Bucketing with Status Mapping
========================================================================
Properly handles test discovery, bucketing, and CI status mapping.
Achieves high match rates and correctly maps pass/fail/skip/error status.
Now supports pulling data from upstream HuggingFace datasets by date.
Added inspection capabilities for debugging data structure issues.

python3 scn.py --analyze-ci collated_reports_MI325_4d57c39.json collated_reports_h100_4d57c39.json collated_reports_MI355_4d57c39.json --export-excel multi_gpu_report.xlsx --debug --dates 2025-09-03

"""

from __future__ import annotations
import argparse, json, os, pathlib, re, sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Set
import subprocess
from datetime import datetime

# Excel imports
try:
    import pandas as pd
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("Warning: pandas/openpyxl not available, Excel export disabled", file=sys.stderr)

# HuggingFace imports
try:
    from huggingface_hub import HfFileSystem
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available, date-based fetching disabled", file=sys.stderr)

# Test classification constants
IMPORTANT_MODELS = [
    "auto", "bert", "clip", "t5", "xlm-roberta", "gpt2", "bart", "mpnet",
    "gpt-j", "wav2vec2", "deberta-v2", "layoutlm", "llama", "opt",
    "longformer", "vit", "whisper", "tapas", "vilt", "clap", "detr",
    "owlvit", "dpt", "videomae",
]

NOT_DEVICE_TESTS = {
    "test_tokenization", "test_processor", "test_processing",
    "test_configuration_utils", "test_data_collator", "test_trainer_callback",
    "test_trainer_utils", "test_feature_extraction", "test_image_processing",
    "test_optimization", "test_retrieval", "test_config",
    "/repo_utils/", "/utils/",
}

# Data source configurations
DATA_SOURCES = {
    "amd": {
        "repo": "optimum-amd/transformers_daily_ci",
        "pattern": "{date}/runs/**/ci_results_run_models_gpu/collated_reports*.json",
        "multiple_runs": True
    },
    "nvidia": {
        "repo": "hf-internal-testing/transformers_daily_ci",
        "pattern": "{date}/ci_results_run_models_gpu/collated_reports*.json",
        "multiple_runs": False
    }
}

# Paths
SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
while not (REPO_ROOT / ".git").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
TEST_ROOT = REPO_ROOT / "tests"

DEBUG_MODE = False


class CITestStatusMapper:
    """Maps CI test results to discovered tests with comprehensive matching."""

    def __init__(self, source_name: str = "unknown"):
        self.source_name = source_name
        self.status_map = {}  # nodeid -> status
        self.base_map = {}    # base_name -> status
        self.stats = defaultdict(int)
        self.all_discovered_tests = set()

    def set_discovered_tests(self, tests: list[str]):
        """Set the list of all discovered tests for better matching."""
        self.all_discovered_tests = set(tests)

    def add_ci_entry(self, nodeid: str, status: str, count: int = 1):
        """Add a CI test entry with simplified handling."""
        if not nodeid:
            return

        # Normalize status
        status = status.upper()
        if status not in {"PASSED", "FAILED", "SKIPPED", "ERROR"}:
            status = "SKIPPED"

        # Store original
        self.status_map[nodeid] = status
        self.stats[status] += count if count > 1 else 1

        # Store base name for matching
        base = self.extract_base_name(nodeid)
        if base not in self.base_map:
            self.base_map[base] = status

        # Handle count expansion only for SKIPPED (as per original logic)
        if count > 1 and status == "SKIPPED":
            base_nodeid = self.extract_base_name(nodeid)
            for i in range(count):
                self.status_map[f"{base_nodeid}[{i}]"] = status
                self.status_map[f"{base_nodeid}_{i}"] = status

    def extract_base_name(self, nodeid: str) -> str:
        """Extract base test name without parameters."""
        if "::" not in nodeid:
            return nodeid

        parts = nodeid.split("::")
        method = parts[-1]

        # Remove bracketed params
        if "[" in method:
            method = method.split("[")[0]

        # Remove numeric suffixes
        method = re.sub(r'_\d+$', '', method)

        # Remove type/device suffixes
        for suffix in ["_bf16", "_fp16", "_fp32", "_cpu", "_cuda", "_gpu", "_eager", "_compile"]:
            if method.endswith(suffix):
                method = method[:-len(suffix)]
                break

        return "::".join(parts[:-1] + [method])

    def find_status(self, test_nodeid: str) -> str:
        """Find status for a test using simplified matching."""
        # Direct match
        if test_nodeid in self.status_map:
            return self.status_map[test_nodeid]

        # Base name match
        base = self.extract_base_name(test_nodeid)
        if base in self.base_map:
            return self.base_map[base]

        # Without parameters
        if "[" in test_nodeid:
            without_params = test_nodeid.split("[")[0]
            if without_params in self.status_map:
                return self.status_map[without_params]
            base_without_params = self.extract_base_name(without_params)
            if base_without_params in self.base_map:
                return self.base_map[base_without_params]

        # Default to SKIPPED
        return "SKIPPED"


def discover_all_tests() -> list[str]:
    """Discover tests using only pytest (as requested)."""
    print("Starting pytest test discovery...", file=sys.stderr)

    cmd = [
        sys.executable, "-m", "pytest",
        str(TEST_ROOT),
        "--collect-only", "--quiet",
        "--continue-on-collection-errors",
        "-p", "no:cacheprovider",
        "--tb=no",
    ]

    env = os.environ.copy()
    env["TRANSFORMERS_VERBOSITY"] = "error"

    nodeids = set()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            env=env, cwd=REPO_ROOT
        )

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("tests/") and "::" in line:
                nodeid = line.split()[0]
                nodeids.add(nodeid)

    except Exception as e:
        print(f"Pytest error: {e}", file=sys.stderr)

    normalized = []
    for test in sorted(nodeids):
        test = test.strip().replace("\\", "/")
        if test and (test.startswith("tests/") or "::" in test):
            normalized.append(test)

    print(f"Discovered: {len(normalized)} tests", file=sys.stderr)
    return normalized


def bucket_tests(tests: list[str]) -> dict[str, list[str]]:
    """Apply bucketing logic - each test goes into exactly one bucket (like water)."""
    buckets = {"P-1": [], "P0": [], "P1": [], "P2": [], "CPU": []}

    for nodeid in tests:
        nodeid_lower = nodeid.lower()

        # Each test goes into the FIRST matching bucket only (water bucket principle)
        # Important models (highest priority)
        if any(model in nodeid_lower for model in IMPORTANT_MODELS):
            buckets["P-1"].append(nodeid)
        # CPU only tests
        elif any(pattern in nodeid_lower for pattern in NOT_DEVICE_TESTS):
            buckets["CPU"].append(nodeid)
        # GPU critical
        elif any(p in nodeid_lower for p in ["gpu", "cuda", "deepspeed", "accelerate"]):
            buckets["P0"].append(nodeid)
        # Framework tests
        elif any(p in nodeid_lower for p in ["torch", "tensorflow", "flax"]):
            buckets["P1"].append(nodeid)
        else:
            buckets["P2"].append(nodeid)

    return buckets


def fetch_latest_run_for_date(source: str, date: str) -> tuple[dict, str]:
    """
    Fetch the latest run for a given date from specified data source.

    Args:
        source: 'amd' or 'nvidia'
        date: Date string in YYYY-MM-DD format

    Returns:
        (metadata, file_path): Tuple of parsed data and source path
    """
    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub required for date-based fetching")

    if source not in DATA_SOURCES:
        raise ValueError(f"Unknown source: {source}. Available: {list(DATA_SOURCES.keys())}")

    config = DATA_SOURCES[source]
    fs = HfFileSystem()

    # Build search pattern
    if config["multiple_runs"]:
        # AMD pattern: date/runs/**/ci_results_run_models_gpu/collated_reports*.json
        search_pattern = f"datasets/{config['repo']}/{date}/runs/**/ci_results_run_models_gpu/collated_reports*.json"
    else:
        # NVIDIA pattern: date/ci_results_run_models_gpu/collated_reports*.json
        search_pattern = f"datasets/{config['repo']}/{date}/ci_results_run_models_gpu/collated_reports*.json"

    print(f"Searching {source} for {date}: {search_pattern}", file=sys.stderr)

    try:
        # Find all matching files
        matching_files = fs.glob(search_pattern)

        if not matching_files:
            print(f"No files found for {source} on {date}", file=sys.stderr)
            return {}, ""

        # Get the latest run (sort by full path, reverse=True for latest)
        latest_file = sorted(matching_files, reverse=True)[0]
        print(f"Loading latest {source} report from: {latest_file}", file=sys.stderr)

        # Read and parse the JSON
        with fs.open(latest_file, 'r') as f:
            data = json.load(f)

        return data, latest_file

    except Exception as e:
        print(f"Error fetching {source} data for {date}: {e}", file=sys.stderr)
        return {}, ""


def inspect_data_structure(data: dict, source_name: str = "unknown"):
    """Inspect and print the structure of CI data for debugging."""
    print(f"\n=== INSPECTING DATA STRUCTURE for {source_name} ===", file=sys.stderr)
    print(f"Top-level keys: {list(data.keys())}", file=sys.stderr)

    # Check for results
    if "results" in data:
        results = data["results"]
        print(f"Results type: {type(results)}, length: {len(results) if isinstance(results, (list, dict)) else 'N/A'}", file=sys.stderr)

        if isinstance(results, list) and results:
            print(f"First result keys: {list(results[0].keys()) if isinstance(results[0], dict) else 'Not a dict'}", file=sys.stderr)

            # Check if first result has its own results
            first_result = results[0]
            if isinstance(first_result, dict) and "results" in first_result:
                inner_results = first_result["results"]
                print(f"Inner results type: {type(inner_results)}, length: {len(inner_results) if isinstance(inner_results, (list, dict)) else 'N/A'}", file=sys.stderr)

                if isinstance(inner_results, list) and inner_results:
                    first_inner = inner_results[0]
                    print(f"First inner result: {first_inner}", file=sys.stderr)
                    if isinstance(first_inner, dict):
                        print(f"First inner result keys: {list(first_inner.keys())}", file=sys.stderr)
            else:
                print(f"First result sample: {str(first_result)[:200]}...", file=sys.stderr)
    else:
        print("No 'results' key found in data", file=sys.stderr)

    # Show a few more top-level details
    for key, value in list(data.items())[:3]:
        if key != "results":
            print(f"{key}: {type(value)} - {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}", file=sys.stderr)

    print("=== END INSPECTION ===\n", file=sys.stderr)


def parse_ci_results(ci_source, source_name: str = None, inspect: bool = False) -> tuple[dict, CITestStatusMapper]:
    """
    Parse CI results from either file path or fetched data.

    Args:
        ci_source: Either a pathlib.Path to local file or dict from fetch_latest_run_for_date
        source_name: Name to identify this source in reports
        inspect: If True, inspect the data structure before parsing
    """
    if isinstance(ci_source, (str, pathlib.Path)):
        # Local file
        ci_file = pathlib.Path(ci_source)
        print(f"Parsing local file: {ci_file}", file=sys.stderr)
        source_name = source_name or ci_file.stem

        try:
            data = json.loads(ci_file.read_text())
        except Exception as e:
            print(f"Error reading {ci_file}: {e}", file=sys.stderr)
            return {}, CITestStatusMapper(source_name)

    elif isinstance(ci_source, dict):
        # Already parsed data from HuggingFace
        data = ci_source
        print(f"Processing fetched data: {source_name}", file=sys.stderr)

    else:
        print(f"Invalid CI source type: {type(ci_source)}", file=sys.stderr)
        return {}, CITestStatusMapper(source_name)

    # Inspect data structure if requested
    if inspect:
        inspect_data_structure(data, source_name)

    metadata = {
        "gpu_name": data.get("gpu_name", source_name or "unknown"),
        "commit_hash": data.get("commit_hash", "unknown"),
        "source": source_name,
    }

    mapper = CITestStatusMapper(source_name)

    # Parse test results with enhanced debugging
    total_entries = 0
    results = data.get("results", [])

    if DEBUG_MODE:
        print(f"Found {len(results)} top-level result entries", file=sys.stderr)

    for idx, model_data in enumerate(results):
        if not isinstance(model_data, dict):
            if DEBUG_MODE:
                print(f"Skipping non-dict model data at index {idx}: {type(model_data)}", file=sys.stderr)
            continue

        model_results = model_data.get("results", [])

        if DEBUG_MODE:
            print(f"Model {idx}: {len(model_results)} test results", file=sys.stderr)

        for test_result in model_results:
            if not isinstance(test_result, dict):
                continue

            # Try multiple field names for the test nodeid
            line = ""
            for field_name in ["line", "nodeid", "test_name", "name", "test", "test_id", "id"]:
                if field_name in test_result and test_result[field_name]:
                    line = test_result[field_name]
                    break

            status = test_result.get("status", "").upper()
            count = test_result.get("count", 1)

            if DEBUG_MODE and total_entries < 5:  # Show first few entries
                print(f"Processing: line='{line}', status='{status}', count={count}", file=sys.stderr)
                if not line:
                    print(f"  Available fields: {list(test_result.keys())}", file=sys.stderr)

            nodeid = extract_nodeid_from_line(line)
            if nodeid:
                mapper.add_ci_entry(nodeid, status, count)
                total_entries += 1

                if DEBUG_MODE and total_entries <= 3:
                    print(f"CI Entry: {nodeid} -> {status} (count: {count})", file=sys.stderr)
            elif DEBUG_MODE and total_entries < 5:
                print(f"No nodeid extracted from line: '{line}'", file=sys.stderr)
                print(f"  Full test_result: {test_result}", file=sys.stderr)

    print(f"Parsed {total_entries} CI entries from {source_name}", file=sys.stderr)
    print(f"Status distribution: {dict(mapper.stats)}", file=sys.stderr)

    return metadata, mapper


def extract_nodeid_from_line(line: str) -> str:
    """Extract test nodeid from CI line with enhanced debugging."""
    if not line:
        return None

    original_line = line
    line = line.strip()

    # Remove [N] prefix
    if line.startswith("["):
        match = re.match(r'\[\d+\]\s+(.*)', line)
        if match:
            line = match.group(1)

    # Remove status prefixes
    for status in ["PASSED", "FAILED", "SKIPPED", "ERROR"]:
        if line.startswith(status):
            line = line[len(status):].strip()

    # Handle different test path patterns
    # 1. Clean nodeids (already in correct format with ::)
    if "::" in line and not line.startswith("../"):
        # Extract the nodeid part (before any " - " error messages)
        nodeid = line.split(" - ")[0].split()[0]
        # Accept any valid test path pattern, not just tests/
        if "/" in nodeid and (
            nodeid.startswith("tests/") or
            nodeid.startswith("examples/") or
            nodeid.startswith("src/") or
            "test_" in nodeid
        ):
            return nodeid

    # 2. Legacy format: starts with tests/ or examples/ (space-separated)
    if line.startswith(("tests/", "examples/")):
        return line.split()[0]

    # 3. Search for test paths in the line
    for pattern in [
        r'(tests/[^\s]+)',           # tests/ paths
        r'(examples/[^\s]*test[^\s]*)', # examples/ with test in name
        r'(src/[^\s]*test[^\s]*)',   # src/ with test in name
    ]:
        match = re.search(pattern, line)
        if match:
            potential_nodeid = match.group(1)
            # Ensure it looks like a valid test nodeid
            if "::" in potential_nodeid or "test_" in potential_nodeid:
                return potential_nodeid

    # 4. For file paths with line numbers (like src/transformers/testing_utils.py:646:)
    file_line_match = re.search(r'([^/\s]+/[^\s:]+\.py):(\d+):', line)
    if file_line_match:
        # These aren't really test nodeids, but diagnostic info - skip them
        return None

    if DEBUG_MODE:
        print(f"Could not extract nodeid from: '{original_line}' -> '{line}'", file=sys.stderr)

    return None


def parse_date_string(date_str: str) -> str:
    """Parse date string and return in YYYY-MM-DD format."""
    try:
        # Try parsing various formats
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"]:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # If no format matches, assume it's already correct
        return date_str
    except Exception:
        return date_str


def fetch_latest_run_for_date(source: str, date: str) -> tuple[dict, str]:
    """
    Fetch the latest run for a given date from specified data source.

    Args:
        source: 'amd' or 'nvidia'
        date: Date string in YYYY-MM-DD format

    Returns:
        (metadata, file_path): Tuple of parsed data and source path
    """
    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub required for date-based fetching")

    if source not in DATA_SOURCES:
        raise ValueError(f"Unknown source: {source}. Available: {list(DATA_SOURCES.keys())}")

    config = DATA_SOURCES[source]
    fs = HfFileSystem()

    # Build search pattern (fixed: removed hf:// prefix)
    if config["multiple_runs"]:
        # AMD pattern: date/runs/**/ci_results_run_models_gpu/collated_reports*.json
        search_pattern = f"datasets/{config['repo']}/{date}/runs/**/ci_results_run_models_gpu/collated_reports*.json"
    else:
        # NVIDIA pattern: date/ci_results_run_models_gpu/collated_reports*.json
        search_pattern = f"datasets/{config['repo']}/{date}/ci_results_run_models_gpu/collated_reports*.json"

    print(f"Searching {source} for {date}: {search_pattern}", file=sys.stderr)

    try:
        # Find all matching files
        matching_files = fs.glob(search_pattern)

        if not matching_files:
            print(f"No files found for {source} on {date}", file=sys.stderr)
            return {}, ""

        # Get the latest run (sort by full path, reverse=True for latest)
        latest_file = sorted(matching_files, reverse=True)[0]
        print(f"Loading latest {source} report from: {latest_file}", file=sys.stderr)

        # Read and parse the JSON (fixed: specify text mode)
        with fs.open(latest_file, 'r') as f:
            data = json.load(f)

        return data, latest_file

    except Exception as e:
        print(f"Error fetching {source} data for {date}: {e}", file=sys.stderr)
        return {}, ""


def fetch_all_data_sources(dates: list[str], sources: list[str] = None, inspect: bool = False) -> dict:
    """
    Fetch data from multiple sources and dates.

    Args:
        dates: List of date strings
        sources: List of source names ('amd', 'nvidia'), defaults to both
        inspect: If True, inspect data structure for each fetch

    Returns:
        Dict mapping source_name -> (metadata, mapper)
    """
    if not HF_AVAILABLE:
        print("Error: huggingface_hub required for date-based fetching", file=sys.stderr)
        return {}

    if sources is None:
        sources = list(DATA_SOURCES.keys())

    all_mappers = {}

    for date_str in dates:
        date_formatted = parse_date_string(date_str)

        for source in sources:
            try:
                data, file_path = fetch_latest_run_for_date(source, date_formatted)

                if data:
                    source_key = f"{source}_{date_formatted}"
                    metadata, mapper = parse_ci_results(data, source_key, inspect=inspect)
                    metadata["date"] = date_formatted
                    metadata["file_path"] = file_path
                    all_mappers[source_key] = (metadata, mapper)

            except Exception as e:
                print(f"Failed to fetch {source} data for {date_formatted}: {e}", file=sys.stderr)

    return all_mappers


def inspect_command(dates: list[str], sources: list[str] = None):
    """Standalone inspection command to debug data structure."""
    print("=== DATA INSPECTION MODE ===", file=sys.stderr)

    if sources is None:
        sources = list(DATA_SOURCES.keys())

    for date_str in dates:
        date_formatted = parse_date_string(date_str)
        print(f"\nInspecting data for {date_formatted}:", file=sys.stderr)

        for source in sources:
            print(f"\n--- {source.upper()} ---", file=sys.stderr)
            try:
                data, file_path = fetch_latest_run_for_date(source, date_formatted)

                if data:
                    print(f"✅ Successfully fetched: {file_path}", file=sys.stderr)
                    inspect_data_structure(data, f"{source}_{date_formatted}")

                    # Quick sample of what would be parsed
                    print("Sample parsing attempt:", file=sys.stderr)
                    temp_metadata, temp_mapper = parse_ci_results(data, f"{source}_{date_formatted}", inspect=False)
                    print(f"Would parse {sum(temp_mapper.stats.values())} entries", file=sys.stderr)
                    print(f"Status breakdown: {dict(temp_mapper.stats)}", file=sys.stderr)
                else:
                    print(f"❌ No data found for {source} on {date_formatted}", file=sys.stderr)

            except Exception as e:
                print(f"❌ Error inspecting {source}: {e}", file=sys.stderr)


def create_excel_report(buckets: dict, ci_mappers: dict, output_path: str):
    """Create Excel report with optimized processing."""
    if not EXCEL_AVAILABLE:
        print("Excel libraries not available", file=sys.stderr)
        return

    print(f"Creating Excel report: {output_path}", file=sys.stderr)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"

    # Quick summary
    total_tests = sum(len(b) for b in buckets.values())

    ws.append(["Test Bucketing Summary"])
    ws.append([])
    ws.append(["Bucket", "Count", "Percentage"])
    for name, tests in buckets.items():
        pct = len(tests) / total_tests * 100 if total_tests else 0
        ws.append([name, len(tests), f"{pct:.1f}%"])

    ws.append([])
    ws.append(["Data Source Information"])
    ws.append(["Source", "GPU/Type", "Total Entries", "Passed", "Failed", "Skipped", "Error"])

    for source_name, (metadata, mapper) in ci_mappers.items():
        gpu_name = metadata.get("gpu_name", "unknown")
        date = metadata.get("date", "")
        file_path = metadata.get("file_path", "")

        display_name = f"{source_name}"
        if date:
            display_name += f" ({date})"

        ws.append([
            display_name,
            gpu_name,
            sum(mapper.stats.values()),
            mapper.stats.get("PASSED", 0),
            mapper.stats.get("FAILED", 0),
            mapper.stats.get("SKIPPED", 0),
            mapper.stats.get("ERROR", 0)
        ])

        if file_path:
            ws.append(["", f"Source: {file_path}", "", "", "", "", ""])

    # Create bucket sheets with simplified processing
    for bucket_name, test_list in buckets.items():
        if not test_list:
            continue

        print(f"Processing bucket {bucket_name} ({len(test_list)} tests)...", file=sys.stderr)

        ws = wb.create_sheet(f"Bucket_{bucket_name}")

        # Pre-calculate status for efficiency
        test_statuses = {}
        for test in test_list:
            test_statuses[test] = {}
            for source_name, (metadata, mapper) in ci_mappers.items():
                test_statuses[test][source_name] = mapper.find_status(test)

        # Build header
        headers = ["Test", "File", "Method"]
        for source_name in ci_mappers.keys():
            headers.append(f"{source_name}_Status")

        ws.append(headers)

        # Add data rows
        for test in sorted(test_list):
            parts = test.split("::")
            row = [
                test,
                parts[0] if parts else test,
                parts[-1] if len(parts) > 1 else "",
            ]

            # Add status columns
            for source_name in ci_mappers.keys():
                row.append(test_statuses[test][source_name])

            ws.append(row)

        # Apply basic formatting
        header_row = ws[1]
        for cell in header_row:
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor="DDDDDD")

        # Color status cells
        status_colors = {
            "PASSED": "90EE90", "FAILED": "FFB6C1",
            "SKIPPED": "FFFFE0", "ERROR": "FFA07A"
        }

        for row_idx in range(2, ws.max_row + 1):
            for col_idx, header in enumerate(headers, 1):
                if "_Status" in header:
                    cell = ws.cell(row=row_idx, column=col_idx)
                    if cell.value in status_colors:
                        cell.fill = PatternFill("solid", fgColor=status_colors[cell.value])

    wb.save(output_path)
    print(f"✅ Report saved with {len(wb.sheetnames)} sheets", file=sys.stderr)


def validate_date_format(date_str: str) -> bool:
    """Validate that date string can be parsed."""
    try:
        parse_date_string(date_str)
        return True
    except:
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CI Test Analysis with upstream data source support")

    # Original arguments
    parser.add_argument("--analyze-ci", nargs="*", help="CI result files (local JSON files)")
    parser.add_argument("--export-excel", help="Output Excel file")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--bucket", choices=["P-1", "P0", "P1", "P2", "CPU"])
    parser.add_argument("--yaml", default="buckets.yaml")

    # Date-based arguments
    parser.add_argument("--dates", nargs="*", help="Dates to fetch (YYYY-MM-DD format)")
    parser.add_argument("--sources", nargs="*", choices=["amd", "nvidia"],
                       help="Data sources to fetch from (default: both)")
    parser.add_argument("--dates-only", action="store_true",
                       help="Only analyze date-based sources, ignore local files")
    parser.add_argument("--files-only", action="store_true",
                       help="Only analyze local files, ignore dates")

    # New inspection arguments
    parser.add_argument("--inspect", action="store_true",
                       help="Inspect data structure before parsing (for debugging)")
    parser.add_argument("--inspect-only", action="store_true",
                       help="Only inspect data structure, don't generate reports")

    args = parser.parse_args()

    # Validate arguments
    if args.dates_only and args.files_only:
        print("Error: Cannot use both --dates-only and --files-only", file=sys.stderr)
        sys.exit(1)

    if args.dates and not HF_AVAILABLE:
        print("Error: huggingface_hub required for date-based fetching", file=sys.stderr)
        sys.exit(1)

    # Validate date formats
    if args.dates:
        for date in args.dates:
            if not validate_date_format(date):
                print(f"Error: Invalid date format: {date}. Use YYYY-MM-DD", file=sys.stderr)
                sys.exit(1)

    global DEBUG_MODE
    DEBUG_MODE = args.debug

    # Handle inspect-only mode
    if args.inspect_only:
        if not args.dates:
            print("Error: --inspect-only requires --dates", file=sys.stderr)
            sys.exit(1)
        inspect_command(args.dates, args.sources)
        return

    # Discover tests (pytest only as requested)
    tests = discover_all_tests()

    # Bucket tests (water bucket principle - each test goes into exactly one bucket)
    buckets = bucket_tests(tests)

    # Verify water bucket principle
    total_bucketed = sum(len(b) for b in buckets.values())
    if total_bucketed != len(tests):
        print(f"WARNING: Bucket count mismatch! {total_bucketed} bucketed vs {len(tests)} discovered", file=sys.stderr)

    # Show distribution
    print(f"\nBucket Distribution ({len(tests)} tests):", file=sys.stderr)
    for name, test_list in buckets.items():
        pct = len(test_list) / len(tests) * 100 if tests else 0
        print(f"  {name}: {len(test_list):6d} ({pct:5.1f}%)", file=sys.stderr)

    # Initialize CI mappers
    ci_mappers = {}

    # Process local files (unless --dates-only)
    if not args.dates_only and args.analyze_ci:
        for ci_file in args.analyze_ci:
            file_path = pathlib.Path(ci_file)
            if file_path.exists():
                metadata, mapper = parse_ci_results(file_path, inspect=args.inspect)
                mapper.set_discovered_tests(tests)
                source_name = f"file_{file_path.stem}"
                ci_mappers[source_name] = (metadata, mapper)
            else:
                print(f"Warning: File not found: {ci_file}", file=sys.stderr)

    # Process date-based sources (unless --files-only)
    if not args.files_only and args.dates:
        date_mappers = fetch_all_data_sources(args.dates, args.sources, inspect=args.inspect)

        for source_name, (metadata, mapper) in date_mappers.items():
            mapper.set_discovered_tests(tests)
            ci_mappers[source_name] = (metadata, mapper)

    # Fallback: if no specific data sources specified, show help
    if not ci_mappers and not args.bucket:
        print("\nNo data sources specified. Use either:", file=sys.stderr)
        print("  --analyze-ci file1.json file2.json  (for local files)", file=sys.stderr)
        print("  --dates 2025-09-03 2025-09-02       (for upstream data)", file=sys.stderr)
        print("  --dates 2025-09-03 --sources amd    (specific source)", file=sys.stderr)
        print("  --inspect-only --dates 2025-09-03   (debug data structure)", file=sys.stderr)
        print("\nExample: python script.py --dates 2025-09-03 --export-excel report.xlsx", file=sys.stderr)
        print("Example: python script.py --inspect-only --dates 2025-09-03 --sources amd", file=sys.stderr)

    # Export results
    if args.export_excel and ci_mappers:
        create_excel_report(buckets, ci_mappers, args.export_excel)
    elif args.bucket:
        # Output specific bucket tests
        for test in buckets[args.bucket]:
            print(test)
    elif ci_mappers:
        # Create summary
        with open(args.yaml, "w") as f:
            summary = {
                "buckets": buckets,
                "sources": {
                    name: {
                        "metadata": metadata,
                        "stats": dict(mapper.stats)
                    }
                    for name, (metadata, mapper) in ci_mappers.items()
                }
            }
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {args.yaml}", file=sys.stderr)
    else:
        # Just save buckets as before
        with open(args.yaml, "w") as f:
            json.dump(buckets, f, indent=2)
        print(f"Saved buckets to {args.yaml}", file=sys.stderr)


if __name__ == "__main__":
    main()
