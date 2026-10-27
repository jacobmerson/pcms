#!/bin/bash
# Two-process streaming demo over the PCMS coupler (ADIOS2 SST by default).
#
#   ./run_pcms_demo.sh NPRODUCER [options...]
#
# Starts the consumer (1 rank) and the producer (NPRODUCER ranks) as separate
# MPI jobs; options are passed to both, e.g.
#   ./run_pcms_demo.sh 8 --source exodus --exo flow.exo --plot demo.png
# Consumer-only options (--plot, --record, --model, --verify, ...) are
# dropped for the producer, and --rate and --partition for the consumer. MPIEXEC overrides the launcher (default: mpiexec).
set -euo pipefail
nprod=${1:?usage: $0 NPRODUCER [options...]}
shift
here=$(cd "$(dirname "$0")" && pwd)
mpiexec=${MPIEXEC:-mpiexec}

producer_args=()
skip=0
for arg in "$@"; do
  if ((skip)); then skip=0; continue; fi
  case $arg in
    --plot|--record|--model|--nx|--ny|--mode|--anchor-every|--hnode-dir)
      skip=1 ;;
    --plot=*|--record=*|--model=*|--nx=*|--ny=*|--mode=*|--anchor-every=*|\
    --hnode-dir=*) ;;
    --verify) ;;
    *) producer_args+=("$arg") ;;
  esac
done
consumer_args=()
skip=0
for arg in "$@"; do
  if ((skip)); then skip=0; continue; fi
  case $arg in
    --rate|--partition) skip=1 ;;
    --rate=*|--partition=*) ;;
    *) consumer_args+=("$arg") ;;
  esac
done

$mpiexec -n 1 python "$here/run_consumer.py" "${consumer_args[@]}" &
consumer=$!
$mpiexec -n "$nprod" python "$here/run_producer.py" "${producer_args[@]}" &
producer=$!
status=0
if ! wait $producer; then
  status=1
  # The consumer would otherwise wait for the stream until its timeout.
  kill $consumer 2>/dev/null || true
fi
wait $consumer || status=1
exit $status
