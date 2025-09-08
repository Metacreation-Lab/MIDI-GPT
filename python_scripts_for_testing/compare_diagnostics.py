#!/usr/bin/env python3
"""
MIDI-GPT Diagnostic Comparator
Compares diagnostic files from original and refactored versions
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

class DiagnosticComparator:
    def __init__(self):
        self.mismatches = []
        self.matches = []
        self.warnings = []
    
    def load_diagnostic(self, filepath: str) -> Dict[str, Any]:
        """Load diagnostic JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load {filepath}: {e}")
    
    def compare_basic_values(self, orig_val: Any, new_val: Any, path: str) -> bool:
        """Compare basic values and record differences"""
        if orig_val == new_val:
            self.matches.append(f"✅ {path}: MATCH")
            return True
        else:
            self.mismatches.append(f"❌ {path}: {orig_val} != {new_val}")
            return False
    
    def compare_success_status(self, orig: Dict, new: Dict, path: str) -> bool:
        """Compare success status of operations"""
        orig_success = orig.get("success", False)
        new_success = new.get("success", False)
        
        if orig_success == new_success:
            if orig_success:
                self.matches.append(f"✅ {path}: Both succeeded")
                return True
            else:
                self.warnings.append(f"⚠️  {path}: Both failed")
                # Show the errors for comparison
                orig_error = orig.get("error", "Unknown error")
                new_error = new.get("error", "Unknown error")
                if orig_error == new_error:
                    self.matches.append(f"✅ {path} error: Same error message")
                else:
                    self.mismatches.append(f"❌ {path} error: '{orig_error}' != '{new_error}'")
                return False
        else:
            self.mismatches.append(f"❌ {path}: Success status differs (orig: {orig_success}, new: {new_success})")
            return False
    
    def compare_hashes(self, orig: Dict, new: Dict, path: str) -> bool:
        """Compare data hashes for identical output"""
        orig_hash = orig.get("data_hash")
        new_hash = new.get("data_hash")
        
        if orig_hash and new_hash:
            if orig_hash == new_hash:
                self.matches.append(f"✅ {path}: Data hashes match ({orig_hash[:8]}...)")
                return True
            else:
                self.mismatches.append(f"❌ {path}: Data hashes differ (orig: {orig_hash[:8]}..., new: {new_hash[:8]}...)")
                return False
        else:
            self.warnings.append(f"⚠️  {path}: No hash data available for comparison")
            return False
    
    def compare_data_lengths(self, orig: Dict, new: Dict, path: str) -> bool:
        """Compare data lengths"""
        orig_len = orig.get("data_length")
        new_len = new.get("data_length")
        
        if orig_len is not None and new_len is not None:
            if orig_len == new_len:
                self.matches.append(f"✅ {path}: Data lengths match ({orig_len})")
                return True
            else:
                diff_pct = abs(orig_len - new_len) / max(orig_len, new_len) * 100
                self.mismatches.append(f"❌ {path}: Data lengths differ (orig: {orig_len}, new: {new_len}, {diff_pct:.1f}% diff)")
                return False
        else:
            self.warnings.append(f"⚠️  {path}: No length data available")
            return False
    
    def compare_token_samples(self, orig: Dict, new: Dict, path: str) -> bool:
        """Compare token sample data"""
        orig_sample = orig.get("data_sample", {})
        new_sample = new.get("data_sample", {})
        
        if not orig_sample or not new_sample:
            self.warnings.append(f"⚠️  {path}: No token sample data")
            return False
        
        matches = 0
        total = 0
        
        for key in ["first_10", "last_10", "min_token", "max_token"]:
            if key in orig_sample and key in new_sample:
                total += 1
                if orig_sample[key] == new_sample[key]:
                    matches += 1
                    self.matches.append(f"✅ {path}.{key}: Match")
                else:
                    self.mismatches.append(f"❌ {path}.{key}: {orig_sample[key]} != {new_sample[key]}")
        
        return matches == total
    
    def compare_file_info(self, orig: Dict, new: Dict, path: str) -> bool:
        """Compare file information"""
        orig_info = orig.get("file_info", {})
        new_info = new.get("file_info", {})
        
        if not orig_info or not new_info:
            self.warnings.append(f"⚠️  {path}: No file info available")
            return False
        
        # File hashes should match (same input file)
        orig_hash = orig_info.get("md5_hash")
        new_hash = new_info.get("md5_hash")
        
        if orig_hash and new_hash:
            if orig_hash == new_hash:
                self.matches.append(f"✅ {path}: Input file hashes match")
                return True
            else:
                self.mismatches.append(f"❌ {path}: Input file hashes differ (different files used?)")
                return False
        else:
            self.warnings.append(f"⚠️  {path}: No input file hash data")
            return False
    
    def compare_api_attributes(self, orig: Dict, new: Dict) -> bool:
        """Compare available API attributes"""
        orig_attrs = orig.get("api_tests", {}).get("available_attributes", {})
        new_attrs = new.get("api_tests", {}).get("available_attributes", {})
        
        if not orig_attrs or not new_attrs:
            self.warnings.append("⚠️  API attributes: No data available")
            return False
        
        all_match = True
        for attr_name in set(orig_attrs.keys()) | set(new_attrs.keys()):
            orig_has = orig_attrs.get(attr_name, False)
            new_has = new_attrs.get(attr_name, False)
            
            if orig_has == new_has:
                status = "available" if orig_has else "missing"
                self.matches.append(f"✅ API.{attr_name}: Both {status}")
            else:
                self.mismatches.append(f"❌ API.{attr_name}: orig={orig_has}, new={new_has}")
                all_match = False
        
        return all_match
    
    def compare_encoder_methods(self, orig: Dict, new: Dict) -> bool:
        """Compare encoder methods"""
        orig_methods = orig.get("encoder_tests", {}).get("available_methods", {})
        new_methods = new.get("encoder_tests", {}).get("available_methods", {})
        
        if not orig_methods or not new_methods:
            self.warnings.append("⚠️  Encoder methods: No data available")
            return False
        
        all_match = True
        for method_name in set(orig_methods.keys()) | set(new_methods.keys()):
            orig_has = orig_methods.get(method_name, False)
            new_has = new_methods.get(method_name, False)
            
            if orig_has == new_has:
                status = "available" if orig_has else "missing"
                self.matches.append(f"✅ Encoder.{method_name}: Both {status}")
            else:
                self.mismatches.append(f"❌ Encoder.{method_name}: orig={orig_has}, new={new_has}")
                all_match = False
        
        return all_match
    
    def compare_midi_file_tests(self, orig: Dict, new: Dict) -> bool:
        """Compare MIDI file processing results"""
        orig_tests = orig.get("midi_file_tests", {})
        new_tests = new.get("midi_file_tests", {})
        
        if not orig_tests or not new_tests:
            self.warnings.append("⚠️  MIDI file tests: No data available")
            return False
        
        # Find common files (by name, not exact key)
        orig_files = set(key.split('_', 2)[-1] for key in orig_tests.keys())
        new_files = set(key.split('_', 2)[-1] for key in new_tests.keys())
        common_files = orig_files & new_files
        
        if not common_files:
            self.warnings.append("⚠️  MIDI file tests: No common files found")
            return False
        
        all_match = True
        
        for filename in common_files:
            # Find the full keys for this filename
            orig_key = next(k for k in orig_tests.keys() if k.endswith(filename))
            new_key = next(k for k in new_tests.keys() if k.endswith(filename))
            
            orig_file_data = orig_tests[orig_key]
            new_file_data = new_tests[new_key]
            
            print(f"\n--- Comparing {filename} ---")
            
            # Compare file info
            self.compare_file_info(orig_file_data, new_file_data, f"File({filename})")
            
            # Compare midi_to_json
            if "midi_to_json" in orig_file_data and "midi_to_json" in new_file_data:
                success_match = self.compare_success_status(
                    orig_file_data["midi_to_json"], 
                    new_file_data["midi_to_json"], 
                    f"File({filename}).midi_to_json"
                )
                if success_match and orig_file_data["midi_to_json"].get("success"):
                    hash_match = self.compare_hashes(
                        orig_file_data["midi_to_json"], 
                        new_file_data["midi_to_json"], 
                        f"File({filename}).midi_to_json"
                    )
                    self.compare_data_lengths(
                        orig_file_data["midi_to_json"], 
                        new_file_data["midi_to_json"], 
                        f"File({filename}).midi_to_json"
                    )
                    if not hash_match:
                        all_match = False
            
            # Compare midi_to_tokens
            if "midi_to_tokens" in orig_file_data and "midi_to_tokens" in new_file_data:
                success_match = self.compare_success_status(
                    orig_file_data["midi_to_tokens"], 
                    new_file_data["midi_to_tokens"], 
                    f"File({filename}).midi_to_tokens"
                )
                if success_match and orig_file_data["midi_to_tokens"].get("success"):
                    hash_match = self.compare_hashes(
                        orig_file_data["midi_to_tokens"], 
                        new_file_data["midi_to_tokens"], 
                        f"File({filename}).midi_to_tokens"
                    )
                    token_match = self.compare_token_samples(
                        orig_file_data["midi_to_tokens"], 
                        new_file_data["midi_to_tokens"], 
                        f"File({filename}).midi_to_tokens"
                    )
                    if not hash_match or not token_match:
                        all_match = False
        
        return all_match
    
    def compare_sampling_tests(self, orig: Dict, new: Dict) -> bool:
        """Compare sampling test results"""
        orig_sampling = orig.get("sampling_tests", {})
        new_sampling = new.get("sampling_tests", {})
        
        if not orig_sampling or not new_sampling:
            self.warnings.append("⚠️  Sampling tests: No data available")
            return False
        
        if "test_run" in orig_sampling and "test_run" in new_sampling:
            success_match = self.compare_success_status(
                orig_sampling["test_run"], 
                new_sampling["test_run"], 
                "Sampling.test_run"
            )
            
            if success_match and orig_sampling["test_run"].get("success"):
                hash_match = self.compare_hashes(
                    orig_sampling["test_run"], 
                    new_sampling["test_run"], 
                    "Sampling.test_run"
                )
                return hash_match
            
            return success_match
        else:
            self.warnings.append("⚠️  Sampling tests: No test_run data")
            return False
    
    def compare_diagnostics(self, orig_file: str, new_file: str) -> Dict[str, bool]:
        """Compare two diagnostic files and return results"""
        print(f"Comparing diagnostics:")
        print(f"  Original: {orig_file}")
        print(f"  Refactored: {new_file}")
        print()
        
        # Load files
        orig_data = self.load_diagnostic(orig_file)
        new_data = self.load_diagnostic(new_file)
        
        results = {}
        
        # Compare metadata
        print("=== Metadata Comparison ===")
        orig_py = orig_data.get("metadata", {}).get("python_version", "unknown")
        new_py = new_data.get("metadata", {}).get("python_version", "unknown")
        print(f"Original Python: {orig_py}")
        print(f"Refactored Python: {new_py}")
        
        # Compare library import
        print("\n=== Library Import ===")
        orig_import = orig_data.get("library_info", {}).get("import_success", False)
        new_import = new_data.get("library_info", {}).get("import_success", False)
        results["library_import"] = self.compare_basic_values(orig_import, new_import, "Library import")
        
        # Compare version
        print("\n=== Version Check ===")
        if "version" in orig_data.get("library_info", {}) and "version" in new_data.get("library_info", {}):
            results["version"] = self.compare_success_status(
                orig_data["library_info"]["version"],
                new_data["library_info"]["version"],
                "Version check"
            )
        else:
            self.warnings.append("⚠️  Version: No data available")
            results["version"] = False
        
        # Compare API surface
        print("\n=== API Surface ===")
        results["api_attributes"] = self.compare_api_attributes(orig_data, new_data)
        
        # Compare encoder
        print("\n=== Encoder Tests ===")
        if "creation" in orig_data.get("encoder_tests", {}) and "creation" in new_data.get("encoder_tests", {}):
            results["encoder_creation"] = self.compare_success_status(
                orig_data["encoder_tests"]["creation"],
                new_data["encoder_tests"]["creation"],
                "Encoder creation"
            )
        else:
            results["encoder_creation"] = False
        
        results["encoder_methods"] = self.compare_encoder_methods(orig_data, new_data)
        
        # Compare MIDI file processing
        print("\n=== MIDI File Processing ===")
        results["midi_file_tests"] = self.compare_midi_file_tests(orig_data, new_data)
        
        # Compare sampling
        print("\n=== Sampling Tests ===")
        results["sampling_tests"] = self.compare_sampling_tests(orig_data, new_data)
        
        return results
    
    def generate_report(self, results: Dict[str, bool]) -> None:
        """Generate final comparison report"""
        print("\n" + "=" * 60)
        print("COMPATIBILITY COMPARISON REPORT")
        print("=" * 60)
        
        # Summary by category
        categories = {
            "Core Library": ["library_import", "version"],
            "API Surface": ["api_attributes", "encoder_creation", "encoder_methods"],
            "Data Processing": ["midi_file_tests"],
            "Model Inference": ["sampling_tests"]
        }
        
        overall_pass = True
        
        for category, tests in categories.items():
            print(f"\n{category}:")
            category_pass = True
            for test in tests:
                if test in results:
                    status = "✅ PASS" if results[test] else "❌ FAIL"
                    print(f"  {test:<20} {status}")
                    if not results[test]:
                        category_pass = False
                        overall_pass = False
                else:
                    print(f"  {test:<20} ⚠️  NOT TESTED")
                    category_pass = False
            
            category_status = "✅ PASS" if category_pass else "❌ FAIL"
            print(f"  {category} Overall: {category_status}")
        
        # Detailed results
        print(f"\n" + "-" * 60)
        print("DETAILED RESULTS")
        print("-" * 60)
        
        if self.matches:
            print(f"\n✅ MATCHES ({len(self.matches)}):")
            for match in self.matches:
                print(f"  {match}")
        
        if self.mismatches:
            print(f"\n❌ MISMATCHES ({len(self.mismatches)}):")
            for mismatch in self.mismatches:
                print(f"  {mismatch}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        # Final verdict
        print(f"\n" + "=" * 60)
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        print(f"OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        if overall_pass and not self.mismatches:
            print("🎉 COMPATIBILITY CONFIRMED - Refactor maintains 1:1 functionality!")
        elif passed_tests >= total_tests * 0.8:  # 80% pass rate
            print("⚠️  MOSTLY COMPATIBLE - Minor differences detected")
        else:
            print("❌ COMPATIBILITY ISSUES - Significant differences found")
        
        print("=" * 60)
    
    def save_detailed_report(self, results: Dict[str, bool], output_file: str) -> None:
        """Save detailed comparison results to JSON"""
        report = {
            "summary": {
                "total_tests": len(results),
                "passed_tests": sum(1 for result in results.values() if result),
                "overall_compatible": all(results.values()) and not self.mismatches
            },
            "test_results": results,
            "matches": self.matches,
            "mismatches": self.mismatches,
            "warnings": self.warnings
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nDetailed report saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save detailed report: {e}")


def main():
    parser = argparse.ArgumentParser(description="Compare MIDI-GPT diagnostic files")
    parser.add_argument("--original", required=True, 
                       help="Diagnostic file from original version")
    parser.add_argument("--refactored", required=True,
                       help="Diagnostic file from refactored version")
    parser.add_argument("--report", default="comparison_report.json",
                       help="Output file for detailed report")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.original).exists():
        print(f"Error: Original diagnostic file not found: {args.original}")
        return 1
    
    if not Path(args.refactored).exists():
        print(f"Error: Refactored diagnostic file not found: {args.refactored}")
        return 1
    
    # Run comparison
    try:
        comparator = DiagnosticComparator()
        results = comparator.compare_diagnostics(args.original, args.refactored)
        comparator.generate_report(results)
        comparator.save_detailed_report(results, args.report)
        
        # Return exit code based on results
        if all(results.values()) and not comparator.mismatches:
            return 0  # All tests passed
        else:
            return 1  # Some tests failed
            
    except Exception as e:
        print(f"Error during comparison: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())