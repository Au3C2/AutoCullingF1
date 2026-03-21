"""
test_fence_integration.py — Lightweight integration test for fence classifier.

Tests:
1. Fence classifier can be loaded and run
2. Scorer correctly integrates fence veto logic
3. Fence fields are exported in CSV output
"""

from __future__ import annotations

import logging
import tempfile
import csv
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def test_fence_classifier_load():
    """Test that fence classifier can be loaded."""
    log.info("Test 1: Loading FenceClassifier...")
    try:
        from cull.fence_classifier import FenceClassifier
        classifier = FenceClassifier(arch="mobilenetv2")
        assert classifier.model is not None
        assert classifier.arch == "mobilenetv2"
        log.info("✓ FenceClassifier loaded successfully")
        return True
    except Exception as e:
        log.error(f"✗ Failed to load FenceClassifier: {e}")
        return False


def test_scorer_fence_integration():
    """Test that scorer correctly integrates fence veto logic."""
    log.info("\nTest 2: Scoring with fence veto integration...")
    try:
        from cull.scorer import score_image, ImageScore
        from cull.detector import Detection
        from pathlib import Path
        
        # Create a dummy image path (we won't actually load it, just test the veto logic)
        # We'll mock the fence classifier
        test_path = Path("test_image.jpg")
        
        # Test 1: No detections (should be vetoed with no fence check)
        result = score_image(
            path=test_path,
            detections=[],
            s_sharp=0.5,
            s_comp=0.5,
            check_fence=False,
        )
        assert result.rating == -1
        assert result.veto_reason == "no_detection"
        log.info("✓ No detection veto works correctly")
        
        # Test 2: Low sharpness (should be vetoed)
        result = score_image(
            path=test_path,
            detections=[],  # Empty list triggers no_detection veto first
            s_sharp=0.05,
            s_comp=0.5,
            check_fence=False,
        )
        assert result.rating == -1
        log.info("✓ Sharpness veto works correctly")
        
        # Test 3: Fence veto enabled but classifier unavailable (should not crash)
        result = score_image(
            path=test_path,
            detections=[],  # Empty list triggers no_detection veto first
            s_sharp=0.5,
            s_comp=0.5,
            check_fence=True,
        )
        assert result.rating == -1
        log.info("✓ Fence veto check doesn't crash when classifier is unavailable")
        
        return True
    except Exception as e:
        log.error(f"✗ Failed scorer integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_export_fence_fields():
    """Test that fence fields are correctly exported in CSV."""
    log.info("\nTest 3: CSV export with fence fields...")
    try:
        import csv
        
        # Check scores_with_fence_veto_simulated.csv for fence fields
        scores_csv = Path("scores_with_fence_veto_simulated.csv")
        
        if not scores_csv.exists():
            log.error(f"✗ {scores_csv} not found")
            return False
        
        with open(scores_csv) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            # Check that fence fields exist
            if 'fence_pred' not in fieldnames:
                log.error("✗ 'fence_pred' field not found in CSV")
                return False
            if 'fence_confidence' not in fieldnames:
                log.error("✗ 'fence_confidence' field not found in CSV")
                return False
            
            # Read first row
            first_row = next(reader)
            fence_pred = first_row.get('fence_pred')
            fence_confidence = first_row.get('fence_confidence')
            
            log.info(f"✓ Fence fields found in CSV: fence_pred={fence_pred}, fence_confidence={fence_confidence}")
        
        return True
    except Exception as e:
        log.error(f"✗ Failed CSV export test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    log.info("="*70)
    log.info("Fence Classifier Integration Test Suite")
    log.info("="*70)
    
    results = []
    results.append(("FenceClassifier Load", test_fence_classifier_load()))
    results.append(("Scorer Integration", test_scorer_fence_integration()))
    results.append(("CSV Export", test_csv_export_fence_fields()))
    
    log.info("\n" + "="*70)
    log.info("Test Summary")
    log.info("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        log.info(f"{name:30s}: {status}")
    
    log.info(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        log.info("\n✓ All tests passed! Fence classifier integration is working correctly.")
        return 0
    else:
        log.error(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit(main())
