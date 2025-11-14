import argparse, torch
from rich import print
from .io import load_config
from .smoke import run_smoke

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--smoke", action="store_true", help="run 60s integrated smoke")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[bold]TFAN Prototype[/bold] :: loading {args.config}")

    if args.smoke:
        ok, path, rep = run_smoke()
        print(f"[cyan]Smoke report:[/cyan] {rep}")
        if not ok:
            raise SystemExit(2)
        print(f"[green]PASS[/green] :: artifact -> {path}")
        return

    print("[yellow]No mode specified. Use --smoke for the integrated test.[/yellow]")

if __name__ == "__main__":
    main()
