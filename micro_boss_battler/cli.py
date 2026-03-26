from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .engine import CARD_ORDER, CARD_POOL, DeckValidationError, STARTING_DECK, simulate_game
from .evaluation import evaluate_deck
from .optimizer import DEFAULT_SEEDS, OpenAIDeckOptimizer, load_env_file, run_optimization_loop
from .player import OpenAIPlayerPolicy


def _load_deck(deck_file: str | None, deck_json: str | None) -> dict[str, int]:
    if deck_file:
        with Path(deck_file).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if deck_json:
        return json.loads(deck_json)
    return STARTING_DECK.copy()


def _dump_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _format_deck(deck: dict[str, int]) -> str:
    return ", ".join(
        f"{name}={deck[name]}"
        for name in CARD_ORDER
        if deck.get(name, 0) > 0
    )


def _print_optimization_summary(run_payload: dict[str, Any], output_file: str | None) -> None:
    rounds = run_payload["rounds"]
    best_iteration = int(run_payload["best_iteration"])
    best_round = rounds[best_iteration]
    final_round = rounds[-1]
    best_metrics = _format_progress_metrics(best_round["evaluation"]["aggregate"])
    final_metrics = _format_progress_metrics(final_round["evaluation"]["aggregate"])

    summary = {
        "optimizer_model": run_payload["optimizer_model"],
        "best_iteration": best_iteration,
        "best_deck": best_round["deck"],
        "best_record": best_metrics["record"],
        "best_win_rate": best_metrics["win_rate_pct"],
        "final_iteration": final_round["iteration"],
        "final_deck": final_round["deck"],
        "final_record": final_metrics["record"],
        "final_win_rate": final_metrics["win_rate_pct"],
    }
    if output_file:
        summary["output_file"] = output_file
    _dump_json(summary)


def _require_tqdm():
    try:
        from tqdm import tqdm
    except ImportError as error:
        raise RuntimeError(
            "tqdm is required for progress bars. Install dependencies with `python3 -m pip install -r requirements.txt`."
        ) from error
    return tqdm


def _add_player_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--player-policy",
        choices=("search", "openai"),
        default="search",
        help="Turn policy to use during combat, defaults to deterministic search",
    )
    parser.add_argument(
        "--player-model",
        help="Override the OpenAI model used for player turns when --player-policy openai",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the environment file containing OPENAI_API_KEY, defaults to .env",
    )


def _add_worker_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of seeds to evaluate in parallel, defaults to 1",
    )


def _build_player_policy(args: argparse.Namespace):
    if getattr(args, "player_policy", "search") == "openai":
        return OpenAIPlayerPolicy.from_env(model=getattr(args, "player_model", None))
    return None


def _format_progress_metrics(aggregate: dict[str, Any]) -> dict[str, str | int]:
    wins = int(aggregate["wins"])
    games_played = int(aggregate["games_played"])
    win_rate = float(aggregate["win_rate"]) * 100.0
    return {
        "wins": wins,
        "games_played": games_played,
        "record": f"{wins}/{games_played}",
        "win_rate_pct": f"{win_rate:.2f}%",
    }


def _build_round_progress_callback(total_rounds: int):
    tqdm = _require_tqdm()
    progress = tqdm(total=total_rounds, desc="Optimization rounds", unit="round")

    def update(round_result) -> None:
        metrics = _format_progress_metrics(round_result.evaluation.aggregate)
        deck_text = _format_deck(round_result.deck)
        progress.set_postfix(
            iteration=round_result.iteration,
            wins=metrics["record"],
            win_rate=metrics["win_rate_pct"],
            refresh=False,
        )
        progress.update(1)
        progress.write(
            f"Iteration {round_result.iteration}: deck [{deck_text}] "
            f"record={metrics['record']} win_rate={metrics['win_rate_pct']}"
        )

    return progress, update


def _build_game_progress_callback(
    total_games: int,
    *,
    desc: str,
    position: int = 0,
    leave: bool = True,
):
    tqdm = _require_tqdm()
    progress = tqdm(
        total=total_games,
        desc=desc,
        unit="game",
        position=position,
        leave=leave,
    )
    wins = 0

    def update(completed_games: int, total: int, game) -> None:
        del completed_games
        del total
        nonlocal wins
        if game.result == "WIN":
            wins += 1
        progress.set_postfix(
            seed=game.seed,
            result=game.result,
            wins=f"{wins}/{progress.n + 1}",
            refresh=False,
        )
        progress.update(1)

    return progress, update


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Micro Boss Battler simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser("simulate", help="Run a single seeded combat")
    simulate_parser.add_argument("--deck-file", help="Path to a JSON deck file")
    simulate_parser.add_argument("--deck-json", help="Deck JSON string")
    simulate_parser.add_argument("--seed", type=int, required=True, help="Shuffle seed")
    _add_player_args(simulate_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run a deck across multiple seeds")
    evaluate_parser.add_argument("--deck-file", help="Path to a JSON deck file")
    evaluate_parser.add_argument("--deck-json", help="Deck JSON string")
    evaluate_parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(DEFAULT_SEEDS),
        help="Seed list to evaluate, defaults to 0..99",
    )
    _add_player_args(evaluate_parser)
    _add_worker_arg(evaluate_parser)

    optimize_parser = subparsers.add_parser("optimize", help="Run the OpenAI deck optimization loop")
    optimize_parser.add_argument("--deck-file", help="Path to a JSON deck file")
    optimize_parser.add_argument("--deck-json", help="Deck JSON string")
    optimize_parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of optimizer revisions to run, defaults to 3",
    )
    optimize_parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(DEFAULT_SEEDS),
        help="Seed list to evaluate on each iteration, defaults to 0..99",
    )
    optimize_parser.add_argument("--model", help="Override the OpenAI model from .env")
    _add_player_args(optimize_parser)
    _add_worker_arg(optimize_parser)
    optimize_parser.add_argument(
        "--output-file",
        help="Optional path to write the full optimization run JSON",
    )
    optimize_parser.add_argument(
        "--include-iteration-history",
        action="store_true",
        help="Include compact prior deck/rationale/win-rate history in each optimizer prompt",
    )

    subparsers.add_parser("card-pool", help="Print the card pool and starting deck")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "card-pool":
            _dump_json(
                {
                    "starting_deck": STARTING_DECK,
                    "card_pool": {name: card.__dict__ for name, card in CARD_POOL.items()},
                }
            )
            return 0

        if (
            getattr(args, "player_policy", "search") == "openai"
            or (args.command == "optimize" and args.iterations > 0)
        ):
            load_env_file(args.env_file)

        deck = _load_deck(getattr(args, "deck_file", None), getattr(args, "deck_json", None))
        if args.command == "simulate":
            _dump_json(
                simulate_game(
                    deck,
                    args.seed,
                    player_policy=_build_player_policy(args),
                ).to_dict()
            )
            return 0

        if args.command == "evaluate":
            progress, progress_callback = _build_game_progress_callback(
                len(args.seeds),
                desc="Evaluation games",
            )
            try:
                result = evaluate_deck(
                    deck,
                    args.seeds,
                    player_policy=_build_player_policy(args),
                    progress_callback=progress_callback,
                    workers=args.workers,
                )
            finally:
                progress.close()
            _dump_json(
                result.to_dict()
            )
            return 0

        if args.command == "optimize":
            player_policy = _build_player_policy(args)
            optimizer = None
            if args.iterations > 0:
                optimizer = OpenAIDeckOptimizer.from_env(
                    model=args.model,
                    include_iteration_history=args.include_iteration_history,
                )
            total_rounds = args.iterations + 1
            progress, progress_callback = _build_round_progress_callback(total_rounds)

            def evaluation_progress_factory(iteration: int, total_games: int):
                game_progress, game_update = _build_game_progress_callback(
                    total_games,
                    desc=f"Games (round {iteration})",
                    position=1,
                    leave=False,
                )
                return game_update, game_progress.close

            try:
                run = run_optimization_loop(
                    initial_deck=deck,
                    optimizer=optimizer,
                    iterations=args.iterations,
                    seeds=args.seeds,
                    player_policy=player_policy,
                    workers=args.workers,
                    progress_callback=progress_callback,
                    evaluation_progress_factory=evaluation_progress_factory,
                )
            finally:
                progress.close()
            payload = run.to_dict()
            if args.output_file:
                Path(args.output_file).write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            _print_optimization_summary(payload, args.output_file)
            return 0
    except DeckValidationError as error:
        parser.exit(status=2, message=f"{error}\n")
    except ValueError as error:
        parser.exit(status=2, message=f"{error}\n")
    except RuntimeError as error:
        parser.exit(status=1, message=f"{error}\n")

    parser.exit(status=2, message="Unknown command.\n")
    return 2
