#!/bin/bash
# pretrain_watchdog.sh — silent-stall detector for the VesperLM pretrain.
#
# Fires when pretrain_470m.log is stale (>2h) AND no pretrain process is alive,
# so a finished or hung run can never again masquerade as a healthy one
# (see lab/FAILURES.md 2026-10-16 pretrain-470m-silent-stall).
# Alarms go to the training log (where the heartbeat health-check already
# looks) and to pretrain_watchdog.log. The staleness clock ignores this
# script's own alarm writes (either alert format).
#
# Install (survives reboots — keep the script outside /tmp):
#   install -m 755 pretrain_watchdog.sh /home/tliao/pretrain_watchdog.sh
#   crontab:  */30 * * * * /home/tliao/pretrain_watchdog.sh
#
# Tunables: PRETRAIN_LOG, PRETRAIN_WATCHDOG_LOG, PRETRAIN_STALE_S

LOG="${PRETRAIN_LOG:-/home/tliao/pretrain_470m.log}"
WLOG="${PRETRAIN_WATCHDOG_LOG:-/home/tliao/pretrain_watchdog.log}"
STALE_S="${PRETRAIN_STALE_S:-7200}"
STATE="${PRETRAIN_WATCHDOG_STATE:-/home/tliao/.pretrain_watchdog_state}"
ALARM_EVERY_S=3600
PROC_PAT='0[12]_pretrain(_linear)?\.py'

ts() { date -u +%FT%TZ; }

# 1) A live pretrain process means everything is fine.
if pgrep -f "$PROC_PAT" >/dev/null 2>&1; then
    rm -f "$STATE"
    exit 0
fi

now=$(date +%s)

# 2) Missing log is an immediate (rate-limited) alarm.
if [ ! -f "$LOG" ]; then
    last_alarm=$(awk '{print $1}' "$STATE" 2>/dev/null)
    case "$last_alarm" in ''|*[!0-9]*) last_alarm=0 ;; esac
    if [ $(( now - last_alarm )) -ge "$ALARM_EVERY_S" ]; then
        echo "$now 0" >"$STATE"
        echo "[$(ts)] [WATCHDOG-ALERT] pretrain log $LOG missing and no pretrain process running" | tee -a "$WLOG"
    fi
    exit 0
fi

# 3) Staleness is measured against the last real (non-watchdog) write.
file_mtime=$(stat -c %Y "$LOG")
real_mtime="$file_mtime"
if tail -n 1 "$LOG" | grep -qE '\[WATCHDOG-ALERT\]|ALERT: pretrain log'; then
    stored=$(awk '{print $2}' "$STATE" 2>/dev/null)
    case "$stored" in ''|*[!0-9]*) ;; *) real_mtime="$stored" ;; esac
else
    echo "0 $file_mtime" >"$STATE"
fi

age=$(( now - real_mtime ))
if [ "$age" -le "$STALE_S" ]; then
    exit 0
fi

# 4) Rate-limit to one alarm per ALARM_EVERY_S.
last_alarm=$(awk '{print $1}' "$STATE" 2>/dev/null)
case "$last_alarm" in ''|*[!0-9]*) last_alarm=0 ;; esac
if [ $(( now - last_alarm )) -lt "$ALARM_EVERY_S" ]; then
    exit 0
fi

last=$(tail -n 1 "$LOG" | tr -d '\r')
line="[$(ts)] [WATCHDOG-ALERT] pretrain log stale ${age}s (> ${STALE_S}s) and no pretrain process running — silent stall? log=$LOG last: ${last}"
echo "$line" >>"$WLOG"
echo "$line" >>"$LOG"
echo "$now $real_mtime" >"$STATE"
