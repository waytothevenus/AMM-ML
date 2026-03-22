#!/usr/bin/env python3
"""
Single entry point for the Vulnerability Risk Assessment System.

Usage:
    python run.py serve                    # Start the web dashboard
    python run.py batch                    # Run the daily batch pipeline
    python run.py import <path>            # Import a file or directory
    python run.py assess <vuln> <asset>    # On-demand single-pair assessment
    python run.py backtest                 # Run backtesting validation
    python run.py freshness                # Show data-freshness report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import get_config, resolve_path

logger = logging.getLogger("run")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directories(cfg) -> None:
    """Create required data directories if they don't exist."""
    for attr in ("imports_dir", "db_dir", "graph_dir", "reports_dir"):
        path = resolve_path(getattr(cfg.data, attr))
        path.mkdir(parents=True, exist_ok=True)
    for attr in ("elp_dir", "isa_dir", "acc_dir", "embeddings_dir"):
        path = resolve_path(getattr(cfg.models, attr))
        path.mkdir(parents=True, exist_ok=True)


# ── CLI Commands ─────────────────────────────────────────────


def cmd_serve(args) -> None:
    """Start the local web server (Layer 5)."""
    import uvicorn
    from layer5_presentation.app import create_app

    cfg = get_config()
    app = create_app()
    uvicorn.run(app, host=cfg.layer5.host, port=cfg.layer5.port)


def cmd_batch(args) -> None:
    """Run the full daily batch pipeline (L0 → L1 → L2 ↔ L3 → L4)."""
    from pipeline.daily_batch import DailyBatchPipeline

    pipeline = DailyBatchPipeline()
    result = pipeline.run(cycle_id=args.cycle_id if hasattr(args, "cycle_id") else None)
    logger.info("Batch complete: %s", json.dumps(result, indent=2, default=str))


def cmd_import(args) -> None:
    """Import a data file or directory into the knowledge graph."""
    from layer0_knowledge_graph.graph_store import GraphStore
    from layer0_knowledge_graph.import_manager import ImportManager

    gs = GraphStore()
    mgr = ImportManager(gs)
    path = Path(args.path)

    if path.is_dir():
        report = mgr.import_directory(path)
        logger.info(report.summarize())
    elif path.is_file():
        result = mgr.import_file(path)
        logger.info(
            "Imported %s via %s – %d nodes, %d edges",
            result.file.name, result.connector, result.nodes_added, result.edges_added,
        )
    else:
        logger.error("Path does not exist: %s", path)


def cmd_assess(args) -> None:
    """Run on-demand assessment for a single vuln–asset pair."""
    from pipeline.on_demand import OnDemandPipeline

    pipeline = OnDemandPipeline()
    result = pipeline.assess_pair(args.vuln_id, args.asset_id)
    print(json.dumps(result, indent=2, default=str))


def cmd_backtest(args) -> None:
    """Run backtesting validation (requires historical ground-truth data)."""
    from backtesting.backtesting_engine import BacktestingEngine

    engine = BacktestingEngine()
    # Load ground-truth data from reports directory
    cfg = get_config()
    gt_path = resolve_path(cfg.data.reports_dir) / "ground_truth.json"
    pred_path = resolve_path(cfg.data.reports_dir) / "predictions.json"

    if not gt_path.exists() or not pred_path.exists():
        logger.error(
            "Backtesting requires %s and %s. "
            "Run a batch pipeline first and export predictions.",
            gt_path, pred_path,
        )
        return

    with open(pred_path) as f:
        predictions = json.load(f)
    with open(gt_path) as f:
        actuals = json.load(f)

    result = engine.run_backtest(predictions, actuals, period=args.period or "")
    print(json.dumps(result.__dict__, indent=2, default=str))


def cmd_freshness(args) -> None:
    """Print data freshness report."""
    from layer0_knowledge_graph.graph_store import GraphStore
    from layer0_knowledge_graph.data_freshness_monitor import DataFreshnessMonitor

    gs = GraphStore()
    gs.load()
    monitor = DataFreshnessMonitor(gs)
    reports = monitor.check_all()
    for r in reports:
        print(f"  {r.source:<25} score={r.freshness_score:.2f}  age={r.age_days:.1f}d  stale={r.is_stale}")
    overall = monitor.get_overall_freshness()
    print(f"\n  Overall freshness: {overall:.2f}")


# ── Main ─────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid AMM+ML Vulnerability Risk Assessment System"
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    sub.add_parser("serve", help="Start the web dashboard")

    # batch
    p_batch = sub.add_parser("batch", help="Run the daily batch pipeline")
    p_batch.add_argument("--cycle-id", default=None, help="Override cycle identifier")

    # import
    p_import = sub.add_parser("import", help="Import a data file or directory")
    p_import.add_argument("path", type=str, help="Path to file or directory")

    # assess
    p_assess = sub.add_parser("assess", help="On-demand single-pair assessment")
    p_assess.add_argument("vuln_id", type=str, help="CVE identifier")
    p_assess.add_argument("asset_id", type=str, help="Asset identifier")

    # backtest
    p_bt = sub.add_parser("backtest", help="Run backtesting validation")
    p_bt.add_argument("--period", default=None, help="Period label for run")

    # freshness
    sub.add_parser("freshness", help="Show data-freshness report")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cfg = get_config()
    setup_logging(cfg.system.log_level)
    ensure_directories(cfg)

    commands = {
        "serve": cmd_serve,
        "batch": cmd_batch,
        "import": cmd_import,
        "assess": cmd_assess,
        "backtest": cmd_backtest,
        "freshness": cmd_freshness,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
