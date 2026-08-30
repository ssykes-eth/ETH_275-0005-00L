"""Run every test suite with one command and print a combined summary.

    uv run python -m tests

Each suite is executed as its own subprocess, its report is streamed through, and a
final table shows which suites passed. Exits 0 only if every suite passes (CI-friendly).
"""

from __future__ import annotations

import subprocess
import sys

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Order follows the pipeline: input -> context -> retrieval -> verify -> fix -> display.
SUITES = [
    "tests.test_action_validation",
    "tests.test_context_builder",
    "tests.test_policy_tool",
    "tests.test_verifier_agent",
    "tests.test_solution_agent",
    "tests.test_display_agent",
]


def main() -> int:
    results: list[tuple[str, bool]] = []
    for module in SUITES:
        console.rule(f"[bold cyan]{module}")
        completed = subprocess.run([sys.executable, "-m", module])
        results.append((module, completed.returncode == 0))

    table = Table(box=box.SIMPLE_HEAVY, title="All Suites", title_style="bold")
    table.add_column("Suite", style="bold")
    table.add_column("Status", justify="center")
    for module, ok in results:
        table.add_row(module, "[green]✓ PASS[/]" if ok else "[red]✗ FAIL[/]")

    console.print()
    console.print(table)

    failed = [module for module, ok in results if not ok]
    if failed:
        body = "Suites needing attention:\n" + "\n".join(f"  • {m}" for m in failed)
        console.print(Panel(body, style="yellow", box=box.ROUNDED, expand=False))
        return 1

    console.print(
        Panel(f"All {len(results)} suites passed — nice work! 🎉", style="bold green", box=box.ROUNDED, expand=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
