from __future__ import annotations

from collections import Counter
import json
import os

from .engine import (
    BOSS_DAMAGE_PATTERN,
    CARD_ORDER,
    CARD_POOL,
    MAX_TURNS,
    TurnState,
    _play_card,
    boss_intent_for_turn,
    legal_actions_in_hand,
)


DEFAULT_OPENAI_PLAYER_MODEL = "gpt-5.4"
DEFAULT_OPENAI_PLAYER_REASONING_EFFORT = "medium"

PLAYER_SYSTEM_PROMPT = """You are playing Micro Boss Battler one action at a time.

Your goal is to maximize the chance to win the full fight, not just this turn.
Balance immediate survival, preserving HP, efficient damage, and setting up future turns.
Choose exactly one next action from the provided legal actions, or choose END_TURN.

Rules:
- Player starts at 36 HP.
- Boss starts at 72 HP.
- The player gets 3 energy each turn.
- The player draws 3 cards on turn 1, then 2 cards at the start of each later turn.
- Decks contain exactly 12 cards.
- Boss attack pattern repeats every 3 turns: 8, 8, 16.
- Combat ends when the boss dies, the player dies, or after turn 10.
- Unplayed cards stay in hand between turns.
- Played cards go to the discard pile after resolving.
- Strike: cost 1, deal 5 damage.
- Block: cost 1, gain 5 block.
- Smash: cost 2, deal 11 damage.
- Wall: cost 2, gain 11 block.
- Scout: cost 1, draw 2 cards.
- Forge: cost 1, your attacks deal +2 damage for the rest of combat. Multiple Forges stack.

Only use the visible information in the payload. Do not assume the order of unseen cards.
Prefer legal actions that improve survival or damage output.
Choose END_TURN only when there are no legal actions or every legal action is strictly worse than passing.
"""

PLAYER_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "analysis": {"type": "string"},
        "action": {
            "type": "string",
            "enum": [*CARD_ORDER, "END_TURN"],
        },
    },
    "required": ["analysis", "action"],
}


def _supports_temperature(model: str) -> bool:
    normalized = model.strip().lower()
    return not normalized.startswith("gpt-5")


def _supports_reasoning(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("gpt-5") or normalized.startswith("o")


def _build_openai_request(
    *,
    model: str,
    instructions: str,
    payload: dict[str, object],
    reasoning_effort: str,
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
                "name": "player_turn_response",
                "strict": True,
                "schema": PLAYER_RESPONSE_SCHEMA,
            },
        },
    }
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens
    if _supports_reasoning(model):
        request["reasoning"] = {"effort": reasoning_effort}
    if _supports_temperature(model):
        request["temperature"] = temperature
    return request


def _truncate_for_error(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated>"


class PlayerResponseError(RuntimeError):
    """Raised when the OpenAI player fails to return a valid action."""


def _response_failure_detail(response: object, base_message: str) -> str:
    parts = [base_message]
    status = getattr(response, "status", None)
    if status:
        parts.append(f"status={status}")

    incomplete_details = getattr(response, "incomplete_details", None)
    incomplete_reason = getattr(incomplete_details, "reason", None)
    if incomplete_reason:
        parts.append(f"incomplete_reason={incomplete_reason}")

    error = getattr(response, "error", None)
    error_code = getattr(error, "code", None)
    error_message = getattr(error, "message", None)
    if error_code or error_message:
        formatted_error = ": ".join(part for part in [error_code, error_message] if part)
        parts.append(f"error={formatted_error}")

    return ". ".join(parts)


def _remaining_unknown_counts(deck: dict[str, int], state: TurnState) -> dict[str, int]:
    visible_counts = Counter(state.hand)
    visible_counts.update(state.discard_pile)
    remaining_counts: dict[str, int] = {}
    for card_name in CARD_ORDER:
        remaining = deck.get(card_name, 0) - visible_counts[card_name]
        if remaining > 0:
            remaining_counts[card_name] = remaining
    return remaining_counts


def _action_for_prompt(action: object) -> dict[str, object]:
    return {
        "card": getattr(action, "card"),
        "damage_dealt": getattr(action, "damage_dealt"),
        "block_gained": getattr(action, "block_gained"),
        "drawn": getattr(action, "drawn"),
        "attack_bonus_gained": getattr(action, "attack_bonus_gained"),
        "boss_hp_after": getattr(action, "boss_hp_after"),
        "player_block_after": getattr(action, "player_block_after"),
        "player_attack_bonus_after": getattr(action, "player_attack_bonus_after"),
    }


def _build_turn_payload(
    *,
    deck: dict[str, int],
    state: TurnState,
    boss_intent: int,
    drawn: list[str],
    actions_taken: list[object],
    legal_actions: list[str],
) -> dict[str, object]:
    return {
        "objective": "Choose the single best next action for this turn.",
        "deck": deck,
        "turn_state": {
            "turn": state.turn,
            "turns_remaining_including_this_one": MAX_TURNS - state.turn + 1,
            "player_hp": state.player_hp,
            "player_block": state.player_block,
            "player_attack_bonus": state.attack_bonus,
            "boss_hp": state.boss_hp,
            "boss_intent": boss_intent,
            "next_boss_intent": (
                boss_intent_for_turn(state.turn + 1) if state.turn < MAX_TURNS else None
            ),
            "energy": state.energy,
            "hand": state.hand,
            "drawn_this_turn": drawn,
            "actions_taken_this_turn": [_action_for_prompt(action) for action in actions_taken],
            "draw_pile_count": len(state.draw_pile),
            "discard_pile_count": len(state.discard_pile),
            "discard_pile": state.discard_pile,
            "remaining_unknown_counts": _remaining_unknown_counts(deck, state),
            "boss_pattern": list(BOSS_DAMAGE_PATTERN),
        },
        "legal_actions": [*legal_actions, "END_TURN"],
    }


class OpenAIPlayerPolicy:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_OPENAI_PLAYER_MODEL,
        reasoning_effort: str = DEFAULT_OPENAI_PLAYER_REASONING_EFFORT,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
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

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        reasoning_effort: str = DEFAULT_OPENAI_PLAYER_REASONING_EFFORT,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> "OpenAIPlayerPolicy":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        chosen_model = (
            model or os.getenv("OPENAI_PLAYER_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_PLAYER_MODEL
        ).strip()
        return cls(
            api_key=api_key,
            model=chosen_model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def _choose_next_action(
        self,
        *,
        deck: dict[str, int],
        state: TurnState,
        boss_intent: int,
        drawn: list[str],
        actions_taken: list[object],
    ) -> str:
        correction_note: str | None = None
        last_failure_detail = "The response was empty."

        for _attempt in range(2):
            legal_actions = legal_actions_in_hand(state.hand, state.energy)
            payload = _build_turn_payload(
                deck=deck,
                state=state,
                boss_intent=boss_intent,
                drawn=drawn,
                actions_taken=actions_taken,
                legal_actions=legal_actions,
            )
            instructions = PLAYER_SYSTEM_PROMPT
            if correction_note:
                instructions = (
                    f"{instructions}\n\nYour previous action was invalid: {correction_note}\n"
                    "Choose only from the listed legal actions or END_TURN."
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
                correction_note = _response_failure_detail(response, "The response was empty")
                last_failure_detail = correction_note
                continue

            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError as error:
                correction_note = f"The response was not valid JSON: {error}"
                last_failure_detail = (
                    f"{correction_note}. Raw response: {_truncate_for_error(raw_output)}"
                )
                continue

            action = parsed.get("action")
            if action == "END_TURN":
                return action
            if action in legal_actions:
                return action
            correction_note = f"{action!r} was not in legal_actions."
            last_failure_detail = (
                f"{correction_note} Legal actions: {legal_actions!r}. "
                f"Raw response: {_truncate_for_error(raw_output)}"
            )

        raise PlayerResponseError(
            "OpenAI player did not return a valid action after 2 attempts. "
            f"Last failure: {last_failure_detail}"
        )

    def choose_turn(
        self,
        *,
        deck: dict[str, int],
        turn_state: TurnState,
        boss_intent: int,
        drawn: list[str],
    ) -> list[str]:
        working_state = turn_state.clone()
        requested_actions: list[str] = []
        actions_taken: list[object] = []

        while working_state.boss_hp > 0:
            legal_actions = legal_actions_in_hand(working_state.hand, working_state.energy)
            if not legal_actions:
                break

            action = self._choose_next_action(
                deck=deck,
                state=working_state,
                boss_intent=boss_intent,
                drawn=drawn,
                actions_taken=actions_taken,
            )
            if action == "END_TURN":
                break
            if action not in legal_actions:
                break

            resolved = _play_card(working_state, action)
            actions_taken.append(resolved)
            requested_actions.append(action)

        return requested_actions
