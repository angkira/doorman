#!/bin/bash
# Quick test script to verify bounding box size fix

echo "================================"
echo "Doorman BBox Test"
echo "================================"
echo ""
echo "This will start the daemon with preview enabled."
echo "Then you can run 'doorman preview --debug' in another terminal."
echo ""
echo "What to check:"
echo "  ✓ Green/red box should cover just your face (not whole frame)"
echo "  ✓ Box should be ~150-250 pixels wide in logs"
echo "  ✓ Box should track face movement smoothly"
echo ""
echo "Press Ctrl+C to stop daemon when done testing."
echo ""
echo "Starting in 3 seconds..."
sleep 3

cd "$(dirname "$0")"
exec ./target/release/doormand --user --preview
