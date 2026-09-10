#!/usr/bin/env bash
# tqdm-style live progress bar for a YARN/Spark job.
#
#   bash scripts/watch_yarn.sh                    # newest running tl_* app, one snapshot
#   bash scripts/watch_yarn.sh <app_id>           # a specific application
#   WATCH=1 bash scripts/watch_yarn.sh <app_id>   # redraw in place until the app ends
#   INTERVAL=5 WATCH=1 bash scripts/watch_yarn.sh <app_id>
#
# Why not `yarn application -status`: it reports a flat 10% for every Spark job
# and never moves. The real task counters come from the Spark REST API, reached
# through the ResourceManager proxy. That proxy base is read from
# `yarn applicationattempt -list`, so no cluster host is hardcoded here.
set -uo pipefail

INTERVAL="${INTERVAL:-10}"
HTTP_TIMEOUT="${HTTP_TIMEOUT:-10}"
WATCH="${WATCH:-0}"

APP="${1:-${APP:-}}"
if [[ -z "$APP" ]]; then
  APP="$(yarn application -list -appStates RUNNING,ACCEPTED,SUBMITTED 2>/dev/null \
         | awk 'NR>1 && $2 ~ /^tl_/ {print $1}' | head -1)"
fi
if [[ -z "$APP" ]]; then
  echo "no running tl_* application; pass one explicitly:" >&2
  echo "  bash $0 application_1782979140233_47610667" >&2
  exit 1
fi

first=1
while :; do
  if [[ "$WATCH" == "1" && $first -eq 0 ]]; then printf '\033[H\033[J'; fi
  first=0

  STATUS="$(yarn application -status "$APP" 2>/dev/null)"
  if [[ -z "$STATUS" ]]; then
    echo "cannot read $APP (yarn application -status returned nothing)" >&2
    exit 1
  fi
  STATE="$(awk -F': ' '/^[[:space:]]*State /{print $2; exit}' <<<"$STATUS")"
  START_MS="$(awk -F': ' '/Start-Time/{print $2; exit}' <<<"$STATUS")"
  NAME="$(awk -F': ' '/Application-Name/{print $2; exit}' <<<"$STATUS")"
  DIAG="$(awk -F': ' '/Diagnostics/{print $2; exit}' <<<"$STATUS")"

  TRACK="$(yarn applicationattempt -list "$APP" 2>/dev/null \
           | awk '/appattempt_/{u=$NF} END{print u}')"
  BASE="${TRACK%/}"
  JOBS_JSON=""
  if [[ -n "$BASE" ]]; then
    JOBS_JSON="$(curl -s -m "$HTTP_TIMEOUT" \
                 "$BASE/api/v1/applications/$APP/jobs" 2>/dev/null)"
    [[ "$JOBS_JSON" == "<html"* ]] && JOBS_JSON=""
  fi

  # -u: flush every line, so piping/tee still shows live progress
  JOBS_JSON="$JOBS_JSON" python3 -u - "$APP" "$NAME" "$STATE" "${START_MS:-0}" <<'PYEOF'
import json, os, sys, time
from datetime import datetime, timezone

app, name, state, start_ms = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4] or 0)
raw = os.environ.get("JOBS_JSON", "")
now = int(time.time() * 1000)

def hms(ms):
    s = max(0, int(ms)) // 1000
    return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)

def bar(frac, width=24):
    if frac != frac:
        frac = 0.0
    k = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * k + "░" * (width - k)

def epoch_ms(s):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fGMT", "%Y-%m-%dT%H:%M:%S.GMT"):
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except Exception:
            pass
    return 0

try:
    jobs = json.loads(raw) if raw.strip() else []
except Exception:
    jobs = []

done = sum(1 for j in jobs if j.get("status") == "SUCCEEDED")
running = [j for j in jobs if j.get("status") == "RUNNING"]
elapsed = hms(now - start_ms) if start_ms > 0 else "--:--:--"
print("%s  %s  %s  elapsed %s  jobs %d done" % (name, app, state, elapsed, done))

if not raw.strip():
    print("AM 未就绪:Spark REST 还没起来或代理不可达,稍后再看")
elif running:
    j = running[-1]
    tot = int(j.get("numTasks", 0))
    comp = int(j.get("numCompletedTasks", 0))
    act = int(j.get("numActiveTasks", 0))
    bad = int(j.get("numFailedTasks", 0)) + int(j.get("numKilledTasks", 0))
    frac = comp / tot if tot else 0.0
    eta = "--:--:--"
    if comp and tot > comp:
        t0 = epoch_ms(j.get("firstTaskLaunchedTime") or j.get("submissionTime") or "")
        if t0 and now > t0:
            rate = comp / float(now - t0)
            eta = hms((tot - comp) / rate)
    jname = (j.get("name") or "")[:34]
    print("%s %3d%%  %d/%d tasks  %d active  %d bad  ETA %s"
          % (bar(frac), round(frac * 100), comp, tot, act, bad, eta))
    print("job %s  %s" % (j.get("jobId"), jname))
else:
    print("no RUNNING spark job (between jobs, or all done)")
PYEOF

  case "$STATE" in
    FINISHED|FAILED|KILLED)
      [[ -n "${DIAG:-}" ]] && echo "diagnostics: $DIAG"
      echo "---"
      echo "logs:  yarn logs -applicationId $APP | tail -60"
      break
      ;;
  esac
  [[ "$WATCH" == "1" ]] || break
  sleep "$INTERVAL"
done
