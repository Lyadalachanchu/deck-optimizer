from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Literal, Protocol


PLAYER_MAX_HP = 36
BOSS_MAX_HP = 72
ENERGY_PER_TURN = 3
CARDS_DRAWN_PER_TURN = 2
OPENING_HAND_SIZE = 3
DECK_SIZE = 12
MAX_TURNS = 10
BOSS_DAMAGE_PATTERN = (8, 8, 16)
CARD_ORDER = ("Strike", "Block", "Smash", "Wall", "Scout", "Forge")
CARD_INDEX = {name: index for index, name in enumerate(CARD_ORDER)}


class DeckValidationError(ValueError):
    """Raised when a submitted deck does not match the game rules."""


@dataclass(frozen=True)
class CardDefinition:
    name: str
    cost: int
    damage: int = 0
    block: int = 0
    draw: int = 0
    attack_bonus: int = 0


CARD_POOL: dict[str, CardDefinition] = {
    "Strike": CardDefinition(name="Strike", cost=1, damage=5),
    "Block": CardDefinition(name="Block", cost=1, block=5),
    "Smash": CardDefinition(name="Smash", cost=2, damage=11),
    "Wall": CardDefinition(name="Wall", cost=2, block=11),
    "Scout": CardDefinition(name="Scout", cost=1, draw=2),
    "Forge": CardDefinition(name="Forge", cost=1, attack_bonus=2),
}

STARTING_DECK: dict[str, int] = {
    "Strike": 6,
    "Block": 4,
    "Scout": 2,
}


@dataclass
class ActionSummary:
    card: str
    cost: int
    damage_dealt: int
    block_gained: int
    drawn: list[str]
    attack_bonus_gained: int
    boss_hp_after: int
    player_block_after: int
    player_attack_bonus_after: int


@dataclass
class TurnSummary:
    turn: int
    boss_intent: int
    player_hp_start: int
    boss_hp_start: int
    player_attack_bonus_start: int
    drawn: list[str]
    actions: list[ActionSummary]
    energy_spent: int
    damage_dealt: int
    block_gained: int
    damage_taken: int
    player_hp_end: int
    boss_hp_end: int
    player_attack_bonus_end: int


@dataclass
class SimulationResult:
    deck: dict[str, int]
    seed: int
    result: Literal["WIN", "LOSS"]
    loss_reason: Literal["PLAYER_DEATH", "TIMEOUT"] | None
    turn_reached: int
    boss_hp_remaining: int
    player_hp_remaining: int
    turns: list[TurnSummary]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TurnState:
    turn: int
    player_hp: int
    player_block: int
    boss_hp: int
    attack_bonus: int
    energy: int
    hand: list[str]
    draw_pile: list[str]
    discard_pile: list[str]
    rng: random.Random

    def clone(self) -> TurnState:
        clone_rng = random.Random()
        clone_rng.setstate(self.rng.getstate())
        return TurnState(
            turn=self.turn,
            player_hp=self.player_hp,
            player_block=self.player_block,
            boss_hp=self.boss_hp,
            attack_bonus=self.attack_bonus,
            energy=self.energy,
            hand=self.hand.copy(),
            draw_pile=self.draw_pile.copy(),
            discard_pile=self.discard_pile.copy(),
            rng=clone_rng,
        )


@dataclass
class TurnPlan:
    score: tuple[int, ...]
    sequence_key: tuple[int, ...]
    state: TurnState
    actions: list[ActionSummary]


class TurnPlayerPolicy(Protocol):
    name: str

    def choose_turn(
        self,
        *,
        deck: dict[str, int],
        turn_state: TurnState,
        boss_intent: int,
        drawn: list[str],
    ) -> list[str]:
        ...


def boss_intent_for_turn(turn: int) -> int:
    return BOSS_DAMAGE_PATTERN[(turn - 1) % len(BOSS_DAMAGE_PATTERN)]


def normalize_deck(deck: dict[str, int]) -> dict[str, int]:
    if not isinstance(deck, dict):
        raise DeckValidationError("Deck must be a mapping of card names to counts.")

    unknown_cards = sorted(name for name in deck if name not in CARD_POOL)
    if unknown_cards:
        cards = ", ".join(unknown_cards)
        raise DeckValidationError(f"Unknown cards in deck: {cards}.")

    normalized: dict[str, int] = {}
    total_cards = 0
    for name in CARD_ORDER:
        count = deck.get(name, 0)
        if not isinstance(count, int):
            raise DeckValidationError(f"Count for {name} must be an integer.")
        if count < 0:
            raise DeckValidationError(f"Count for {name} must be non-negative.")
        if count > 0:
            normalized[name] = count
            total_cards += count

    if total_cards != DECK_SIZE:
        raise DeckValidationError(
            f"Deck must contain exactly {DECK_SIZE} cards, got {total_cards}."
        )

    return normalized


def expand_deck(deck: dict[str, int]) -> list[str]:
    normalized = normalize_deck(deck)
    cards: list[str] = []
    for name in CARD_ORDER:
        cards.extend([name] * normalized.get(name, 0))
    return cards


def _draw_cards(
    draw_pile: list[str],
    discard_pile: list[str],
    hand: list[str],
    count: int,
    rng: random.Random,
) -> list[str]:
    drawn: list[str] = []
    for _ in range(count):
        if not draw_pile:
            if not discard_pile:
                break
            rng.shuffle(discard_pile)
            draw_pile.extend(discard_pile)
            discard_pile.clear()
        card = draw_pile.pop()
        hand.append(card)
        drawn.append(card)
    return drawn


def _play_card(state: TurnState, card_name: str) -> ActionSummary:
    card = CARD_POOL[card_name]
    if card_name not in state.hand:
        raise ValueError(f"{card_name} is not in hand.")
    if card.cost > state.energy:
        raise ValueError(f"Not enough energy to play {card_name}.")

    state.hand.remove(card_name)
    state.energy -= card.cost

    attack_damage = card.damage + (state.attack_bonus if card.damage > 0 else 0)
    damage_dealt = min(attack_damage, state.boss_hp)
    state.boss_hp -= damage_dealt
    state.player_block += card.block
    state.attack_bonus += card.attack_bonus
    drawn = _draw_cards(
        draw_pile=state.draw_pile,
        discard_pile=state.discard_pile,
        hand=state.hand,
        count=card.draw,
        rng=state.rng,
    )
    state.discard_pile.append(card_name)

    return ActionSummary(
        card=card_name,
        cost=card.cost,
        damage_dealt=damage_dealt,
        block_gained=card.block,
        drawn=drawn,
        attack_bonus_gained=card.attack_bonus,
        boss_hp_after=state.boss_hp,
        player_block_after=state.player_block,
        player_attack_bonus_after=state.attack_bonus,
    )


def legal_actions_in_hand(hand: list[str], energy: int) -> list[str]:
    legal_actions: list[str] = []
    seen_cards: set[str] = set()
    for card_name in hand:
        if card_name in seen_cards:
            continue
        seen_cards.add(card_name)
        if CARD_POOL[card_name].cost <= energy:
            legal_actions.append(card_name)
    return legal_actions


def _attack_cards_remaining(state: TurnState) -> int:
    zones = (state.hand, state.draw_pile, state.discard_pile)
    return sum(
        1
        for zone in zones
        for card_name in zone
        if CARD_POOL[card_name].damage > 0
    )


def _score_turn_end(state: TurnState, boss_intent: int, sequence_length: int) -> tuple[int, ...]:
    if state.boss_hp <= 0:
        return (
            3,
            -state.turn,
            state.player_hp,
            -state.energy,
            -sequence_length,
        )

    damage_taken = max(0, boss_intent - state.player_block)
    player_hp_after = state.player_hp - damage_taken
    survives = 2 if player_hp_after > 0 else 0
    remaining_turns = max(1, MAX_TURNS - state.turn)
    required_damage_rate = math.ceil(state.boss_hp / remaining_turns)
    future_bonus_damage = state.attack_bonus * _attack_cards_remaining(state)

    return (
        survives,
        -state.boss_hp,
        future_bonus_damage,
        player_hp_after,
        -required_damage_rate,
        -damage_taken,
        -state.energy,
        -sequence_length,
    )


def _choose_turn_plan(state: TurnState, boss_intent: int) -> TurnPlan:
    def search(current_state: TurnState, actions: list[ActionSummary]) -> TurnPlan:
        best_plan = TurnPlan(
            score=_score_turn_end(current_state, boss_intent, len(actions)),
            sequence_key=tuple(CARD_INDEX[action.card] for action in actions),
            state=current_state.clone(),
            actions=list(actions),
        )

        seen_cards: set[str] = set()
        for card_name in current_state.hand:
            if card_name in seen_cards:
                continue
            seen_cards.add(card_name)
            if CARD_POOL[card_name].cost > current_state.energy:
                continue

            next_state = current_state.clone()
            action = _play_card(next_state, card_name)
            candidate = search(next_state, [*actions, action])

            if candidate.score > best_plan.score:
                best_plan = candidate
                continue
            if candidate.score == best_plan.score and candidate.sequence_key < best_plan.sequence_key:
                best_plan = candidate

        return best_plan

    return search(state, [])


def _resolve_requested_actions(state: TurnState, requested_actions: list[str]) -> list[ActionSummary]:
    actions: list[ActionSummary] = []
    for card_name in requested_actions:
        if card_name not in CARD_POOL:
            break
        if card_name not in state.hand:
            break
        if CARD_POOL[card_name].cost > state.energy:
            break
        actions.append(_play_card(state, card_name))
        if state.boss_hp <= 0:
            break
    return actions


class SearchPlayerPolicy:
    name = "search"

    def choose_turn(
        self,
        *,
        deck: dict[str, int],
        turn_state: TurnState,
        boss_intent: int,
        drawn: list[str],
    ) -> list[str]:
        del deck
        del drawn
        plan = _choose_turn_plan(turn_state, boss_intent)
        return [action.card for action in plan.actions]


DEFAULT_PLAYER_POLICY = SearchPlayerPolicy()


def simulate_game(
    deck: dict[str, int],
    seed: int,
    player_policy: TurnPlayerPolicy | None = None,
) -> SimulationResult:
    normalized_deck = normalize_deck(deck)
    chosen_policy = player_policy or DEFAULT_PLAYER_POLICY
    rng = random.Random(seed)
    draw_pile = expand_deck(normalized_deck)
    rng.shuffle(draw_pile)
    discard_pile: list[str] = []
    hand: list[str] = []

    player_hp = PLAYER_MAX_HP
    boss_hp = BOSS_MAX_HP
    attack_bonus = 0
    turns: list[TurnSummary] = []

    for turn in range(1, MAX_TURNS + 1):
        player_hp_start = player_hp
        boss_hp_start = boss_hp
        player_attack_bonus_start = attack_bonus
        boss_intent = boss_intent_for_turn(turn)
        player_block = 0
        cards_to_draw = OPENING_HAND_SIZE if turn == 1 else CARDS_DRAWN_PER_TURN
        drawn = _draw_cards(
            draw_pile=draw_pile,
            discard_pile=discard_pile,
            hand=hand,
            count=cards_to_draw,
            rng=rng,
        )

        turn_state = TurnState(
            turn=turn,
            player_hp=player_hp,
            player_block=player_block,
            boss_hp=boss_hp,
            attack_bonus=attack_bonus,
            energy=ENERGY_PER_TURN,
            hand=hand.copy(),
            draw_pile=draw_pile.copy(),
            discard_pile=discard_pile.copy(),
            rng=random.Random(),
        )
        turn_state.rng.setstate(rng.getstate())

        requested_actions = chosen_policy.choose_turn(
            deck=normalized_deck,
            turn_state=turn_state.clone(),
            boss_intent=boss_intent,
            drawn=drawn.copy(),
        )
        actions = _resolve_requested_actions(turn_state, requested_actions)

        hand = turn_state.hand
        draw_pile = turn_state.draw_pile
        discard_pile = turn_state.discard_pile
        boss_hp = turn_state.boss_hp
        player_block = turn_state.player_block
        attack_bonus = turn_state.attack_bonus
        rng = turn_state.rng

        damage_taken = 0
        if boss_hp > 0:
            damage_taken = max(0, boss_intent - player_block)
            player_hp = max(0, player_hp - damage_taken)
            player_block = max(0, player_block - boss_intent)

        turns.append(
            TurnSummary(
                turn=turn,
                boss_intent=boss_intent,
                player_hp_start=player_hp_start,
                boss_hp_start=boss_hp_start,
                player_attack_bonus_start=player_attack_bonus_start,
                drawn=drawn,
                actions=actions,
                energy_spent=ENERGY_PER_TURN - turn_state.energy,
                damage_dealt=sum(action.damage_dealt for action in actions),
                block_gained=sum(action.block_gained for action in actions),
                damage_taken=damage_taken,
                player_hp_end=player_hp,
                boss_hp_end=boss_hp,
                player_attack_bonus_end=attack_bonus,
            )
        )

        if boss_hp <= 0:
            return SimulationResult(
                deck=normalized_deck,
                seed=seed,
                result="WIN",
                loss_reason=None,
                turn_reached=turn,
                boss_hp_remaining=0,
                player_hp_remaining=player_hp,
                turns=turns,
            )

        if player_hp <= 0:
            return SimulationResult(
                deck=normalized_deck,
                seed=seed,
                result="LOSS",
                loss_reason="PLAYER_DEATH",
                turn_reached=turn,
                boss_hp_remaining=boss_hp,
                player_hp_remaining=max(0, player_hp),
                turns=turns,
            )

    return SimulationResult(
        deck=normalized_deck,
        seed=seed,
        result="LOSS",
        loss_reason="TIMEOUT",
        turn_reached=MAX_TURNS,
        boss_hp_remaining=boss_hp,
        player_hp_remaining=player_hp,
        turns=turns,
    )
