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
if [ $failed -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $failed
