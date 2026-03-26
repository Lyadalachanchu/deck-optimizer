from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Callable

from .engine import SimulationResult, TurnPlayerPolicy, normalize_deck, simulate_game


GameProgressCallback = Callable[[int, int, SimulationResult], None]


@dataclass
class EvaluationResult:
    deck: dict[str, int]
    seeds: list[int]
    aggregate: dict[str, float | int | None]
    representative_win: SimulationResult | None
    representative_loss: SimulationResult | None
    games: list[SimulationResult]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return payload


def evaluation_rank(result: EvaluationResult) -> tuple[float, float, float]:
    average_turns_to_win = result.aggregate["average_turns_to_win"]
    turns_component = (
        float("inf") if average_turns_to_win is None else float(average_turns_to_win)
    )
    return (
        float(result.aggregate["win_rate"]),
        -float(result.aggregate["average_boss_hp_remaining_on_losses"]),
        -turns_component,
    )


def evaluate_deck(
    deck: dict[str, int],
    seeds: list[int] | range | tuple[int, ...],
    player_policy: TurnPlayerPolicy | None = None,
    progress_callback: GameProgressCallback | None = None,
    workers: int = 1,
) -> EvaluationResult:
    normalized_deck = normalize_deck(deck)
    seed_list = list(seeds)
    total_games = len(seed_list)
    if workers < 1:
        raise ValueError("workers must be at least 1.")

    if workers == 1 or total_games <= 1:
        games: list[SimulationResult] = []
        for completed_games, seed in enumerate(seed_list, start=1):
            game = simulate_game(normalized_deck, seed, player_policy=player_policy)
            games.append(game)
            if progress_callback is not None:
                progress_callback(completed_games, total_games, game)
    else:
        games_by_index: list[SimulationResult | None] = [None] * total_games
        completed_games = 0
        max_workers = min(workers, total_games)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(simulate_game, normalized_deck, seed, player_policy): (index, seed)
                for index, seed in enumerate(seed_list)
            }
            for future in as_completed(futures):
                index, _seed = futures[future]
                game = future.result()
                games_by_index[index] = game
                completed_games += 1
                if progress_callback is not None:
                    progress_callback(completed_games, total_games, game)
        games = [game for game in games_by_index if game is not None]

    wins = [game for game in games if game.result == "WIN"]
    losses = [game for game in games if game.result == "LOSS"]

    aggregate: dict[str, float | int | None] = {
        "games_played": len(games),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(games) if games else 0.0,
        "average_boss_hp_remaining": mean(game.boss_hp_remaining for game in games) if games else 0.0,
        "average_player_hp_remaining": mean(game.player_hp_remaining for game in games) if games else 0.0,
        "average_boss_hp_remaining_on_losses": mean(game.boss_hp_remaining for game in losses) if losses else 0.0,
        "average_turns_to_win": mean(game.turn_reached for game in wins) if wins else None,
    }

    representative_win = None
    if wins:
        representative_win = min(
            wins,
            key=lambda game: (game.turn_reached, -game.player_hp_remaining, game.seed),
        )

    representative_loss = None
    if losses:
        representative_loss = min(
            losses,
            key=lambda game: (game.boss_hp_remaining, -game.turn_reached, game.seed),
        )

    return EvaluationResult(
        deck=normalized_deck,
        seeds=seed_list,
        aggregate=aggregate,
        representative_win=representative_win,
        representative_loss=representative_loss,
        games=games,
    )
