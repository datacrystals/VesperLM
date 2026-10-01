#!/bin/bash
# chat.sh — launch the VesperLM agent chat REPL in your terminal.
# Usage: bash /home/tliao/VesperLM/Agent/chat.sh
# Type messages; Ctrl-D or 'exit' to quit. Sandbox mode optional.

cd "$(dirname "$0")"
exec /home/tliao/venvs/vesper/bin/python agent_harness.py "$@"
