#!/data/data/com.termux/files/usr/bin/bash
# heart.c — runtime/boot/heart-daemon.sh
# Symlink target for ~/.termux/boot/02-heart.sh per ARCHITECTURE.md §11.
# Watchdog respawn + wake-lock re-acquire pattern (matches 01-mesh-agent.sh).

LOG=~/.heart/heart.log
mkdir -p ~/.heart
while true; do
    termux-wake-lock 2>/dev/null
    echo "[$(date -Iseconds)] heart-daemon: starting" >> "$LOG"
    ~/.local/bin/heart serve >> "$LOG" 2>&1
    echo "[$(date -Iseconds)] heart-daemon: exited (code $?), restarting in 5s" >> "$LOG"
    sleep 5
done
