# MITL streaming example

Streaming ML ingestion for the hnode plasma-gun (inner MITL) surrogate
(arXiv:2604.21101, Fig. 3). A producer sends native unstructured-mesh fields
step by step; the consumer ingests them conservatively with PCMS into
hnode-style node frames and runs a surrogate in the loop against the
streamed truth.

Design and rationale: `docs/mitl_pcms_data_loader_plan.md` (section 8 for
streaming).

## Status

| Piece | State |
|---|---|
| Synthetic EMPIRE stand-in (`synthetic_source.py`) | Working |
| EMPIRE Exodus replay (`exodus_source.py`) | Working on pcms_testcases `exodus/flow/flow.exo` |
| Conservative / point rasterizer (`rasterizer.py`) | Working (exact electron count) |
| Loopback transport, SST-like block/discard queue (`transport.py`) | Working |
| PCMS coupler transport over ADIOS2 SST or BP4 (`PcmsTransport`) | Working, two programs, Block policy only (see Notes) |
| Partitioned producer, N ranks -> 1 consumer rank (`partition.py`) | Working; a distributed Omega_h mesh balanced by Omega_h (default), or serial pieces of EMPIRE's own `PROC_ID` decomposition (`--partition source`) |
| Cheap models: persistence, linear, pooled torch (`models.py`) | Working |
| hnode surrogate adapter (`hnode_model.py`) | Working (smoke test on hnode's bundled frames, CPU) |
| Consumer with re-anchoring and gap handling (`consumer.py`) | Working |
| Online training on the stream, point vs conservative (`online_model.py`, `run_trainer.py`) | Working; video via `make_video.py` |
| hnode fine-tuning on the stream (`HNODEStreamTrainer`) | Working on the synthetic source (code path only), CPU |

## Files

| File | Role |
|---|---|
| `geometry.py` | L-shaped domain, `Grid` (600 x 100 default), mesh helpers |
| `synthetic_source.py` | Analytic, deterministic run on a ~55k-triangle jittered mesh |
| `exodus_source.py` | `ExodusRun`: EMPIRE Exodus replay (netCDF4, no SEACAS) |
| `sources.py` | Source and frame-grid selection shared by producer and consumer |
| `rasterizer.py` | `GridRasterizer` (conservative P1 on covered pixels, or point interpolation) and `make_frame` |
| `transport.py` | `LoopbackTransport` (one process) and `PcmsTransport` (coupler) |
| `partition.py` | Per-rank meshes: Omega_h-balanced distributed mesh, or serial pieces with global IDs and vertex ownership |
| `producer.py` | `run_producer(source, transport, rate)` |
| `consumer.py` | `Consumer`: rasterize, count, forecast, compare |
| `models.py` | `rollout(frames[3, C, H, W], horizon)` implementations |
| `hnode_model.py` | The pretrained hnode checkpoint behind `rollout`; `train_step` and `HNODEStreamTrainer` fine-tune it on streamed frames |
| `online_model.py` | `OnlineTrainer`: small next-frame CNN trained on frames as they arrive, scored before it trains on each |
| `report.py` | Console summary and figure |
| `runtime.py` | One `OmegaHLibrary` per process, released at exit |
| `run_loopback_demo.py` | One-process end-to-end demo |
| `run_producer.py`, `run_consumer.py`, `run_pcms_demo.sh` | Two-program demo over the PCMS coupler |
| `run_trainer.py`, `make_video.py` | Consumer that trains online (CNN per rasterization mode, or hnode fine-tuning), and its video |
| `hnode_smoke.py` | hnode adapter check on hnode's bundled frames |
| `tests/` | `unittest` suites (no pytest needed) |

## Running

`pcms` (built with PETSc, MeshFields and Python), `PyOmega_h` and `numpy`
must be importable; `netCDF4` for Exodus input; `torch` and `matplotlib` are
optional. Run from this directory:

```bash
PCMS_TEST_DATA_DIR=/path/to/pcms_testcases python -m unittest discover -s tests -t .
python run_loopback_demo.py --steps 120                 # 600 x 100, conservative
python run_loopback_demo.py --steps 120 --mode point    # hnode-style sampling
python run_loopback_demo.py --nx 120 --ny 20 --policy discard --queue-limit 1 \
    --model pooled-torch --plot demo.png
```

The demo reports received/dropped steps, per-step cost, the ingested
electron count against the exact source integral, and forecast error.

### Over the PCMS coupler (two programs)

`run_pcms_demo.sh N` starts the consumer (one rank) and the producer (N
ranks) as two MPI jobs that rendezvous over ADIOS2 (SST by default, BP4 with
`--transport BP4`); run it in an empty directory, which receives the ADIOS2
contact files, and the producer's temporary mesh file, so it must be visible
to every producer rank. Options go to both programs.

```bash
# EMPIRE test run from 8 producer ranks (distributed mesh balanced by Omega_h;
# add --partition source for EMPIRE's own decomposition):
./run_pcms_demo.sh 8 --source exodus --exo $PCMS_TEST_DATA_DIR/exodus/flow/flow.exo \
    --verify --plot flow.png --record flow.npz
# Synthetic source with the hnode checkpoint in the loop:
./run_pcms_demo.sh 2 --source synthetic --steps 12 --anchor-every 5 \
    --model hnode --hnode-dir /path/to/hnode --plot synthetic_hnode.png
```

### Training on the stream

`run_trainer.py` is a consumer that trains instead of only forecasting. By
default it trains two identical next-frame CNNs (same initial weights and
sample order) on the same stream, one on point-interpolated frames and one
on frames from the conservative transfer. Every received frame is first
predicted by the model (scored against persistence), then added to a replay
buffer, then trained on (`--train-steps` optimizer steps on random patches).
`make_video.py` renders the record: per mode the ingested density, the
prediction made before arrival and the forecast from `--horizon` steps
earlier, plus the training loss, error relative to persistence and the
electron count against the exact source integral.

```bash
CONSUMER=run_trainer.py PRODUCER_NICE=19 ./run_pcms_demo.sh 8 --source exodus \
    --exo $PCMS_TEST_DATA_DIR/exodus/flow/flow.exo --record train.npz
python make_video.py train.npz train.mp4           # needs imageio-ffmpeg
# Fine-tune the pretrained hnode checkpoint on the synthetic 6-channel stream:
CONSUMER=run_trainer.py PRODUCER_NICE=19 ./run_pcms_demo.sh 8 --source synthetic \
    --steps 12 --model hnode --hnode-dir /path/to/hnode --modes point \
    --lr 1e-6 --record hnode.npz
python make_video.py hnode.npz hnode.mp4
```

`PRODUCER_NICE` matters when producer and trainer share a node: producer
ranks blocked by a slow consumer busy-wait in MPI and would take the
trainer's cores. Under `mpiexec` torch defaults to one thread, so the trainer
sets its thread count to the cores it may use (`--threads` overrides).

`--verify` checks every received field bit for bit against the source read
locally by the consumer. The two programs can also be started by hand:
`mpiexec -n 1 python run_consumer.py ...` and
`mpiexec -n N python run_producer.py ...` with the same source and
transport options.

`python hnode_smoke.py /path/to/hnode` checks the hnode adapter on hnode's
bundled `example_traj` frames (needs `einops` and `scikit-fem`).

## Notes

- The synthetic step lies on grid lines of the frame grid, so the
  conservative target mesh conforms to the domain and integrals are exact.
- The EMPIRE mesh (`flow.exo`) is the same L shape in metres with the step
  at y = 5 mm, which is on a grid line only for ny a multiple of 33; the
  Exodus demo uses 600 x 99 (square 1/12 mm pixels). hnode's 600 x 100 grid
  needs per-pixel coverage (plan section 3.5).
- `flow.exo` has electron moments only, so Exodus frames have 3 channels
  (vx, vy, n_e) and the hnode checkpoint cannot run on them; the in-loop
  model is a cheap stand-in there.
- SST `QueueFullPolicy=Discard` does not work with redev: with
  `QueueLimit=1` the channel setup fails, and with a larger queue a dropped
  end-of-stream step leaves the consumer at EndOfStream, where redev aborts.
  Use Block (the default).
- `flow.exo` is 51 noisy particle-in-cell frames 0.2 ns apart: the
  change from one frame to the next is mostly noise, so an online model
  trained on it does not beat persistence on unseen frames (even spatial
  smoothing only reaches 0.90-0.95 of persistence's error). Its training
  loss falls; forecast skill needs more and denser output.
- hnode training differentiates through the solver's implicit backward,
  which needs a double backward of attention; on the CPU this uses the math
  attention kernel. One step takes about 20 s and 10 GB on 4-8 cores.
  Adam at lr 1e-5 already raises the loss of the pretrained model after one
  step; use about 1e-6.
- Frames stay float64 inside the pipeline; hnode stores float32, so cast at
  the disk or model boundary.
- Density uses a lumped mass matrix (non-negative). Consistent projection
  undershoots below zero at sharp density fronts.
