#!/usr/bin/env bash
# Runs the sanity checks of every phase; exits non-zero if any fails.
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
failed=0
for t in test_phase1.py test_phase2.py test_phase3.py test_phase4.py test_phase5.py; do
    echo "=== $t"
    if ! $PY "$t"; then
        failed=1
    fi
done
if [ -d data/fw_uav/val_meta ] && [ -f data/fw_uav/val.zip ]; then   # Phase 7a needs the FW-UAV6DPose data
    echo "=== test_phase7a.py"
    $PY test_phase7a.py || failed=1
fi
if [ -f data/fw_uav/yolo_fw_uav.zip ]; then   # Phase 7b part A needs the prepared YOLO dataset
    echo "=== test_phase7b.py"
    $PY test_phase7b.py || failed=1
fi
if [ $failed -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $failed
