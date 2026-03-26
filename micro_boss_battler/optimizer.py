from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Callable, Protocol

from .engine import CARD_ORDER, CARD_POOL, DECK_SIZE, STARTING_DECK, TurnPlayerPolicy, normalize_deck
from .evaluation import EvaluationResult, GameProgressCallback, evaluate_deck, evaluation_rank


DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_OPENAI_REASONING_EFFORT = None
DEFAULT_SEEDS = tuple(range(100))

GAME_RULES = """Micro Boss Battler rules:
- Player starts at 36 HP.
- Boss starts at 72 HP.
- The player gets 3 energy each turn.
- The player draws 3 cards on turn 1, then 2 cards at the start of each later turn.
- Decks must contain exactly 12 cards.
- Shuffle randomness is the only source of variance.
- Boss attack pattern repeats every 3 turns: 8, 8, 16.
- Combat ends when the boss dies, the player dies, or after turn 10.
- Unplayed cards stay in hand between turns.
- Played cards go to the discard pile after resolving.

Allowed cards:
- Strike: cost 1, deal 5 damage
- Block: cost 1, gain 5 block
- Smash: cost 2, deal 11 damage
- Wall: cost 2, gain 11 block
- Scout: cost 1, draw 2 cards
- Forge: cost 1, your attacks deal +2 damage for the rest of combat

Optimization objective:
1. Maximize win rate over the evaluated seeds.
2. Break ties by lowering average boss HP remaining on losses.
3. Break remaining ties by lowering average turns to win.
"""

OPTIMIZER_SYSTEM_PROMPT = """You are optimizing decks for Micro Boss Battler.

You will receive the game rules, the current deck, aggregate results, and only the loss
turn logs from the current deck's evaluation. Winning game logs are omitted.
Use that evidence to improve the next deck.
If a compact iteration history summary is included, use it to avoid repeating failed
changes and to preserve changes that improved win rate.

Rules for your response:
- Only use allowed cards.
- Return exactly 12 total cards.
- Prefer meaningful changes supported by the logs.
- If the deck is already strong, small changes are acceptable, but still explain why.
- Output must match the provided JSON schema exactly.
"""

OPTIMIZER_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "analysis": {"type": "string"},
        "changes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "deck": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                name: {"type": "integer", "minimum": 0} for name in CARD_ORDER
            },
            "required": list(CARD_ORDER),
        },
    },
    "required": ["analysis", "changes", "deck"],
}


class DeckSuggestionSource(Protocol):
    def propose_next_deck(
        self,
        *,
        iteration: int,
        current_deck: dict[str, int],
        evaluation: EvaluationResult,
        history: list["OptimizationRound"],
    ) -> "OptimizerSuggestion":
        ...


@dataclass
class OptimizerSuggestion:
    analysis: str
    changes: list[str]
    deck: dict[str, int]
    model: str
    response_id: str | None = None


@dataclass
class OptimizationRound:
    iteration: int
    deck: dict[str, int]
    evaluation: EvaluationResult
    suggestion: OptimizerSuggestion | None = None


@dataclass
class OptimizationRun:
    seeds: list[int]
    optimizer_model: str
    rounds: list[OptimizationRound]
    best_iteration: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ProgressCallback = Callable[[OptimizationRound], None]
EvaluationProgressFactory = Callable[
    [int, int],
    tuple[GameProgressCallback | None, Callable[[], None] | None],
]


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _turn_for_prompt(turn: dict[str, object]) -> dict[str, object]:
    return {
        "turn": turn["turn"],
        "boss_intent": turn["boss_intent"],
        "player_hp_start": turn["player_hp_start"],
        "boss_hp_start": turn["boss_hp_start"],
        "player_attack_bonus_start": turn["player_attack_bonus_start"],
        "drawn": turn["drawn"],
        "actions": [
            {
                "card": action["card"],
                "damage_dealt": action["damage_dealt"],
                "block_gained": action["block_gained"],
                "drawn": action["drawn"],
                "attack_bonus_gained": action["attack_bonus_gained"],
                "player_attack_bonus_after": action["player_attack_bonus_after"],
            }
            for action in turn["actions"]
        ],
        "energy_spent": turn["energy_spent"],
        "damage_dealt": turn["damage_dealt"],
        "block_gained": turn["block_gained"],
        "damage_taken": turn["damage_taken"],
        "player_hp_end": turn["player_hp_end"],
        "boss_hp_end": turn["boss_hp_end"],
        "player_attack_bonus_end": turn["player_attack_bonus_end"],
    }


def _game_for_prompt(game: dict[str, object]) -> dict[str, object]:
    return {
        "seed": game["seed"],
        "result": game["result"],
        "loss_reason": game["loss_reason"],
        "turn_reached": game["turn_reached"],
        "boss_hp_remaining": game["boss_hp_remaining"],
        "player_hp_remaining": game["player_hp_remaining"],
        "turns": [_turn_for_prompt(turn) for turn in game["turns"]],
    }


def _compact_aggregate(aggregate: dict[str, object]) -> dict[str, object]:
    return {
        "games_played": aggregate["games_played"],
        "wins": aggregate["wins"],
        "losses": aggregate["losses"],
        "win_rate": aggregate["win_rate"],
        "average_boss_hp_remaining_on_losses": aggregate["average_boss_hp_remaining_on_losses"],
        "average_turns_to_win": aggregate["average_turns_to_win"],
    }


def _build_iteration_history(history: list["OptimizationRound"]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for index, round_result in enumerate(history):
        if round_result.suggestion is None:
            continue

        next_round = history[index + 1] if index + 1 < len(history) else None
        entries.append(
            {
                "iteration": round_result.iteration,
                "deck": round_result.deck,
                "aggregate_results": _compact_aggregate(round_result.evaluation.aggregate),
                "rationale": round_result.suggestion.analysis,
                "changes": round_result.suggestion.changes,
                "next_deck": round_result.suggestion.deck,
                "next_deck_win_rate": (
                    next_round.evaluation.aggregate["win_rate"] if next_round is not None else None
                ),
            }
        )
    return entries


def build_optimizer_payload(
    *,
    iteration: int,
    current_deck: dict[str, int],
    evaluation: EvaluationResult,
    history: list[OptimizationRound],
    include_iteration_history: bool = False,
) -> dict[str, object]:
    current_evaluation = evaluation.to_dict()
    loss_games = [
        _game_for_prompt(game)
        for game in current_evaluation["games"]
        if game["result"] == "LOSS"
    ]
    payload = {
        "iteration": iteration,
        "game_rules": GAME_RULES,
        "allowed_cards": {
            name: {
                "cost": definition.cost,
                "damage": definition.damage,
                "block": definition.block,
                "draw": definition.draw,
                "attack_bonus": definition.attack_bonus,
            }
            for name, definition in CARD_POOL.items()
        },
        "current_deck": current_deck,
        "objective": {
            "primary": "maximize win rate",
            "tie_breaker_1": "lower average boss HP remaining on losses",
            "tie_breaker_2": "lower average turns to win",
        },
        "aggregate_results": current_evaluation["aggregate"],
        "loss_games": loss_games,
    }
    if include_iteration_history:
        payload["iteration_history"] = _build_iteration_history(history)
    return payload


def _supports_temperature(model: str) -> bool:
    normalized = model.strip().lower()
    return not normalized.startswith("gpt-5")


def _supports_reasoning(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("gpt-5") or normalized.startswith("o")


def _truncate_for_error(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated>"


def _deck_total_guidance(deck: object) -> str | None:
    if not isinstance(deck, dict):
        return None

    values = list(deck.values())
    if any(not isinstance(value, int) for value in values):
        return None

    total = sum(values)
    if total == DECK_SIZE:
        return None

    delta = DECK_SIZE - total
    action = "add" if delta > 0 else "remove"
    count = abs(delta)
    noun = "card" if count == 1 else "cards"
    return f"Your last deck totaled {total}, you must {action} {count} {noun}."


def _build_openai_request(
    *,
    model: str,
    instructions: str,
    payload: dict[str, object],
    reasoning_effort: str | None,
    temperature: float,
    max_output_tokens: int | None,
) -> dict[str, object]:
    request: dict[str, object] = {
        "model": model,
        "instructions": instructions,
        "input": json.dumps(payload, indent=2, sort_keys=True),
        "store": False,
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "deck_optimizer_response",
                "strict": True,
                "schema": OPTIMIZER_RESPONSE_SCHEMA,
            },
        },
    }
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens
    if _supports_reasoning(model) and reasoning_effort is not None:
        request["reasoning"] = {"effort": reasoning_effort}
    if _supports_temperature(model):
        request["temperature"] = temperature
    return request


class OpenAIDeckOptimizer:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
        include_iteration_history: bool = False,
        client: object | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")

        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)

        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.include_iteration_history = include_iteration_history

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
        include_iteration_history: bool = False,
    ) -> "OpenAIDeckOptimizer":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        chosen_model = (model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL).strip()
        return cls(
            api_key=api_key,
            model=chosen_model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            include_iteration_history=include_iteration_history,
        )

    def propose_next_deck(
        self,
        *,
        iteration: int,
        current_deck: dict[str, int],
        evaluation: EvaluationResult,
        history: list[OptimizationRound],
    ) -> OptimizerSuggestion:
        payload = build_optimizer_payload(
            iteration=iteration,
            current_deck=current_deck,
            evaluation=evaluation,
            history=history,
            include_iteration_history=self.include_iteration_history,
        )
        correction_note: str | None = None
        last_failure_detail = "The response was empty."

        max_attempts = 4
        for _attempt in range(max_attempts):
            instructions = OPTIMIZER_SYSTEM_PROMPT
            if correction_note:
                instructions = (
                    f"{instructions}\n\nYour previous proposal was invalid: {correction_note}\n"
                    "Return a corrected deck that follows all game constraints."
                )

            response = self.client.responses.create(
                **_build_openai_request(
                    model=self.model,
                    instructions=instructions,
                    payload=payload,
                    reasoning_effort=self.reasoning_effort,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
            )
            raw_output = (response.output_text or "").strip()
            if not raw_output:
                correction_note = "The response was empty."
                last_failure_detail = correction_note
                continue

            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError as error:
                correction_note = f"The response was not valid JSON: {error}"
                last_failure_detail = (
                    f"{correction_note} Raw response: {_truncate_for_error(raw_output)}"
                )
                continue
            try:
                next_deck = normalize_deck(parsed["deck"])
            except KeyError:
                correction_note = "The response did not include a 'deck' field."
                last_failure_detail = (
                    f"{correction_note} Raw response: {_truncate_for_error(raw_output)}"
                )
            except Exception as error:
                correction_note = str(error)
                invalid_deck = parsed.get("deck")
                total_guidance = _deck_total_guidance(invalid_deck)
                if total_guidance:
                    correction_note = f"{correction_note} {total_guidance}"
                last_failure_detail = correction_note
                if invalid_deck is not None:
                    last_failure_detail = (
                        f"{last_failure_detail} Returned deck: "
                        f"{json.dumps(invalid_deck, sort_keys=True)}"
                    )
                continue

            return OptimizerSuggestion(
                analysis=parsed["analysis"],
                changes=parsed["changes"],
                deck=next_deck,
                model=self.model,
                response_id=getattr(response, "id", None),
            )

        raise RuntimeError(
            f"OpenAI did not return a valid deck proposal after {max_attempts} attempts. "
            f"Last failure: {last_failure_detail}"
        )


def run_optimization_loop(
    *,
    initial_deck: dict[str, int] | None = None,
    optimizer: DeckSuggestionSource | None = None,
    iterations: int,
    seeds: list[int] | range | tuple[int, ...] = DEFAULT_SEEDS,
    player_policy: TurnPlayerPolicy | None = None,
    workers: int = 1,
    progress_callback: ProgressCallback | None = None,
    evaluation_progress_factory: EvaluationProgressFactory | None = None,
) -> OptimizationRun:
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if iterations > 0 and optimizer is None:
        raise ValueError("optimizer is required when iterations is greater than 0.")
    if workers < 1:
        raise ValueError("workers must be at least 1.")

    current_deck = normalize_deck(initial_deck or STARTING_DECK)
    seed_list = list(seeds)
    rounds: list[OptimizationRound] = []

    for iteration in range(iterations + 1):
        game_progress_callback = None
        close_game_progress = None
        if evaluation_progress_factory is not None:
            game_progress_callback, close_game_progress = evaluation_progress_factory(
                iteration,
                len(seed_list),
            )

        try:
            evaluation = evaluate_deck(
                current_deck,
                seed_list,
                player_policy=player_policy,
                progress_callback=game_progress_callback,
                workers=workers,
            )
        finally:
            if close_game_progress is not None:
                close_game_progress()

        round_result = OptimizationRound(
            iteration=iteration,
            deck=current_deck,
            evaluation=evaluation,
        )
        rounds.append(round_result)
        if progress_callback is not None:
            progress_callback(round_result)

        if iteration == iterations:
            break

        suggestion = optimizer.propose_next_deck(
            iteration=iteration,
            current_deck=current_deck,
            evaluation=evaluation,
            history=rounds,
        )
        round_result.suggestion = suggestion
        current_deck = suggestion.deck

    best_iteration = max(
        range(len(rounds)),
        key=lambda index: (evaluation_rank(rounds[index].evaluation), -rounds[index].iteration),
    )
    return OptimizationRun(
        seeds=seed_list,
        optimizer_model=(
            getattr(optimizer, "model", DEFAULT_OPENAI_MODEL)
            if optimizer is not None
            else DEFAULT_OPENAI_MODEL
        ),
        rounds=rounds,
        best_iteration=best_iteration,
    )
