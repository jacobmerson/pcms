"""Source and frame-grid choices shared by the producer and consumer."""

from exodus_source import ExodusRun
from geometry import EMPIRE_STEP_Y, Grid
from synthetic_source import SyntheticMITLSource

SOURCES = ("synthetic", "exodus")


def add_source_args(parser):
    parser.add_argument("--source", choices=SOURCES, default="synthetic")
    parser.add_argument("--exo", default=None,
                        help="Exodus file for --source exodus "
                             "(pcms_testcases exodus/flow/flow.exo)")
    parser.add_argument("--steps", type=int, default=None,
                        help="steps to replay (default: synthetic 120, "
                             "exodus all)")


def make_source(args, comm):
    """comm None: no mesh (an exodus producer piece builds its own)."""
    if args.source == "exodus":
        if args.exo is None:
            raise SystemExit("--source exodus needs --exo PATH")
        return ExodusRun(args.exo, comm, args.steps)
    return SyntheticMITLSource(comm, num_steps=args.steps or 120)


def default_grid(source_name, nx=None, ny=None):
    """Frame grid whose step lies on grid lines. The EMPIRE step at y = 5 mm
    is on a grid line for ny a multiple of 33; 600 x 99 gives square pixels
    of 1/12 mm."""
    if source_name == "exodus":
        return Grid(nx=nx or 600, ny=ny or 99, step_y=EMPIRE_STEP_Y)
    return Grid(nx=nx or 600, ny=ny or 100)
