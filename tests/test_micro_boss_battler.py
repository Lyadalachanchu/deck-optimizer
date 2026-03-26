from __future__ import annotations

import random
import unittest

from micro_boss_battler import (
    DEFAULT_OPENAI_MODEL,
    DeckValidationError,
    OpenAIDeckOptimizer,
    OpenAIPlayerPolicy,
    OptimizerSuggestion,
    PlayerResponseError,
    STARTING_DECK,
    build_optimizer_payload,
    evaluate_deck,
    evaluation_rank,
    run_optimization_loop,
    simulate_game,
)
from micro_boss_battler.cli import _format_progress_metrics, build_parser
from micro_boss_battler.engine import TurnState, _play_card
from micro_boss_battler.optimizer import _build_openai_request
from micro_boss_battler.player import _build_openai_request as _build_player_openai_request


class DeckValidationTests(unittest.TestCase):
    def test_rejects_unknown_cards(self) -> None:
        with self.assertRaises(DeckValidationError):
            simulate_game({"Strike": 11, "Zap": 1}, seed=0)

    def test_rejects_wrong_total(self) -> None:
        with self.assertRaises(DeckValidationError):
            simulate_game({"Strike": 11}, seed=0)


class SimulationTests(unittest.TestCase):
    class NoOpPolicy:
        name = "noop"

        def choose_turn(self, **_: object) -> list[str]:
            return []

    def test_same_seed_is_deterministic(self) -> None:
        first = simulate_game(STARTING_DECK, seed=5).to_dict()
        second = simulate_game(STARTING_DECK, seed=5).to_dict()
        self.assertEqual(first, second)

    def test_all_strike_deck_loses(self) -> None:
        evaluation = evaluate_deck({"Strike": 12}, seeds=range(8))
        self.assertEqual(evaluation.aggregate["wins"], 0)
        self.assertGreater(evaluation.aggregate["average_boss_hp_remaining_on_losses"], 0)

    def test_all_block_deck_loses_without_dealing_damage(self) -> None:
        result = simulate_game({"Block": 12}, seed=0)
        self.assertEqual(result.result, "LOSS")
        self.assertEqual(result.boss_hp_remaining, 72)

    def test_mixed_deck_outperforms_starting_deck(self) -> None:
        starting = evaluate_deck(STARTING_DECK, seeds=range(8))
        mixed = evaluate_deck(
            {"Strike": 2, "Block": 2, "Smash": 4, "Wall": 2, "Scout": 2},
            seeds=range(8),
        )
        self.assertGreater(evaluation_rank(mixed), evaluation_rank(starting))

    def test_draws_two_cards_per_turn_and_keeps_hand_between_turns(self) -> None:
        result = simulate_game({"Scout": 10, "Smash": 2}, seed=1)
        self.assertEqual(len(result.turns[0].drawn), 3)
        self.assertEqual(len(result.turns[1].drawn), 2)
        self.assertEqual(len(result.turns[2].drawn), 2)
        self.assertEqual(result.turns[2].actions[0].card, "Smash")

    def test_known_good_mixed_deck_has_non_zero_win_rate(self) -> None:
        evaluation = evaluate_deck(
            {"Strike": 4, "Block": 3, "Smash": 3, "Wall": 2},
            seeds=range(10),
        )
        self.assertGreater(evaluation.aggregate["wins"], 0)

    def test_forge_buffs_future_attacks(self) -> None:
        state = TurnState(
            turn=1,
            player_hp=36,
            player_block=0,
            boss_hp=72,
            attack_bonus=0,
            energy=3,
            hand=["Forge", "Strike"],
            draw_pile=[],
            discard_pile=[],
            rng=random.Random(0),
        )
        forge = _play_card(state, "Forge")
        strike = _play_card(state, "Strike")

        self.assertEqual(forge.attack_bonus_gained, 2)
        self.assertEqual(forge.player_attack_bonus_after, 2)
        self.assertEqual(strike.damage_dealt, 7)
        self.assertEqual(strike.player_attack_bonus_after, 2)
        self.assertEqual(state.boss_hp, 65)

    def test_custom_player_policy_is_used_by_simulation_and_evaluation(self) -> None:
        result = simulate_game({"Strike": 12}, seed=0, player_policy=self.NoOpPolicy())
        self.assertEqual(result.turns[0].actions, [])
        self.assertEqual(result.boss_hp_remaining, 72)

        evaluation = evaluate_deck({"Strike": 12}, seeds=[0, 1], player_policy=self.NoOpPolicy())
        self.assertEqual(evaluation.aggregate["wins"], 0)
        self.assertEqual(evaluation.aggregate["average_boss_hp_remaining_on_losses"], 72)

    def test_evaluate_deck_reports_progress_for_each_game(self) -> None:
        updates: list[tuple[int, int, int]] = []

        def on_progress(completed_games: int, total_games: int, game) -> None:
            updates.append((completed_games, total_games, game.seed))

        evaluate_deck({"Strike": 12}, seeds=[3, 5, 8], progress_callback=on_progress)
        self.assertEqual(updates, [(1, 3, 3), (2, 3, 5), (3, 3, 8)])

    def test_parallel_evaluation_preserves_seed_order_in_results(self) -> None:
        evaluation = evaluate_deck({"Strike": 12}, seeds=[5, 2, 9], workers=3)
        self.assertEqual([game.seed for game in evaluation.games], [5, 2, 9])


class OptimizerLoopTests(unittest.TestCase):
    def test_progress_metrics_show_percent_and_record(self) -> None:
        metrics = _format_progress_metrics(
            {
                "wins": 25,
                "games_played": 100,
                "win_rate": 0.25,
            }
        )
        self.assertEqual(metrics["record"], "25/100")
        self.assertEqual(metrics["win_rate_pct"], "25.00%")

    def test_default_model_and_seed_count(self) -> None:
        from micro_boss_battler.optimizer import DEFAULT_OPENAI_REASONING_EFFORT, DEFAULT_SEEDS

        self.assertEqual(DEFAULT_OPENAI_MODEL, "gpt-5.4")
        self.assertIsNone(DEFAULT_OPENAI_REASONING_EFFORT)
        self.assertEqual(len(DEFAULT_SEEDS), 100)
        self.assertEqual(DEFAULT_SEEDS[0], 0)
        self.assertEqual(DEFAULT_SEEDS[-1], 99)

    def test_gpt5_requests_omit_temperature(self) -> None:
        request = _build_openai_request(
            model="gpt-5-mini",
            instructions="optimize",
            payload={"deck": STARTING_DECK},
            reasoning_effort="high",
            temperature=0.2,
            max_output_tokens=500,
        )
        self.assertNotIn("temperature", request)
        self.assertEqual(request["reasoning"], {"effort": "high"})

    def test_gpt5_requests_omit_reasoning_when_default(self) -> None:
        request = _build_openai_request(
            model="gpt-5.4",
            instructions="optimize",
            payload={"deck": STARTING_DECK},
            reasoning_effort=None,
            temperature=0.2,
            max_output_tokens=None,
        )
        self.assertNotIn("reasoning", request)
        self.assertNotIn("temperature", request)

    def test_optimizer_request_omits_max_output_tokens_when_uncapped(self) -> None:
        request = _build_openai_request(
            model="gpt-5.4",
            instructions="optimize",
            payload={"deck": STARTING_DECK},
            reasoning_effort="xhigh",
            temperature=0.2,
            max_output_tokens=None,
        )
        self.assertNotIn("max_output_tokens", request)

    def test_non_gpt5_requests_keep_temperature(self) -> None:
        request = _build_openai_request(
            model="gpt-4.1-mini",
            instructions="optimize",
            payload={"deck": STARTING_DECK},
            reasoning_effort="high",
            temperature=0.2,
            max_output_tokens=500,
        )
        self.assertEqual(request["temperature"], 0.2)
        self.assertNotIn("reasoning", request)

    def test_player_gpt5_requests_default_to_medium_reasoning(self) -> None:
        request = _build_player_openai_request(
            model="gpt-5.4-nano",
            instructions="play",
            payload={"turn_state": {}},
            reasoning_effort="medium",
            temperature=0.2,
            max_output_tokens=None,
        )
        self.assertEqual(request["reasoning"], {"effort": "medium"})
        self.assertNotIn("temperature", request)
        self.assertNotIn("max_output_tokens", request)

    def test_player_request_omits_max_output_tokens_when_uncapped(self) -> None:
        request = _build_player_openai_request(
            model="gpt-5.4-mini",
            instructions="play",
            payload={"turn_state": {}},
            reasoning_effort="medium",
            temperature=0.2,
            max_output_tokens=None,
        )
        self.assertNotIn("max_output_tokens", request)

    def test_openai_player_raises_on_empty_response(self) -> None:
        class FakeIncompleteDetails:
            reason = "max_output_tokens"

        class FakeResponse:
            def __init__(self) -> None:
                self.output_text = ""
                self.status = "incomplete"
                self.incomplete_details = FakeIncompleteDetails()
                self.error = None

        class FakeResponses:
            def create(self, **_: object) -> FakeResponse:
                return FakeResponse()

        class FakeClient:
            responses = FakeResponses()

        policy = OpenAIPlayerPolicy(api_key="test-key", client=FakeClient())
        with self.assertRaises(PlayerResponseError) as ctx:
            policy.choose_turn(
                deck={"Strike": 12},
                turn_state=TurnState(
                    turn=1,
                    player_hp=36,
                    player_block=0,
                    boss_hp=72,
                    attack_bonus=0,
                    energy=3,
                    hand=["Strike", "Strike", "Strike"],
                    draw_pile=["Strike"] * 9,
                    discard_pile=[],
                    rng=random.Random(0),
                ),
                boss_intent=8,
                drawn=["Strike", "Strike", "Strike"],
            )
        message = str(ctx.exception)
        self.assertIn("OpenAI player did not return a valid action after 2 attempts.", message)
        self.assertIn("The response was empty", message)
        self.assertIn("incomplete_reason=max_output_tokens", message)

    def test_prompt_payload_contains_only_loss_turn_logs(self) -> None:
        evaluation = evaluate_deck(STARTING_DECK, seeds=[0, 1])
        payload = build_optimizer_payload(
            iteration=0,
            current_deck=STARTING_DECK,
            evaluation=evaluation,
            history=[],
        )
        self.assertIn("Micro Boss Battler rules", payload["game_rules"])
        self.assertEqual(payload["current_deck"], STARTING_DECK)
        self.assertNotIn("iteration_history", payload)
        self.assertEqual(
            len(payload["loss_games"]),
            payload["aggregate_results"]["losses"],
        )
        self.assertTrue(payload["loss_games"])
        self.assertTrue(all(game["result"] == "LOSS" for game in payload["loss_games"]))
        self.assertTrue(payload["loss_games"][0]["turns"])

    def test_prompt_payload_can_include_iteration_history(self) -> None:
        round_zero_evaluation = evaluate_deck(STARTING_DECK, seeds=[0, 1])
        round_one_deck = {"Strike": 4, "Block": 2, "Smash": 3, "Wall": 1, "Scout": 2}
        round_one_evaluation = evaluate_deck(round_one_deck, seeds=[0, 1])
        history = [
            type("RoundStub", (), {
                "iteration": 0,
                "deck": STARTING_DECK,
                "evaluation": round_zero_evaluation,
                "suggestion": OptimizerSuggestion(
                    analysis="Increase damage density without cutting all draw.",
                    changes=["Cut 2 Strike", "Add 3 Smash", "Add 1 Wall", "Cut 2 Block"],
                    deck=round_one_deck,
                    model="fake-optimizer",
                ),
            })(),
            type("RoundStub", (), {
                "iteration": 1,
                "deck": round_one_deck,
                "evaluation": round_one_evaluation,
                "suggestion": None,
            })(),
        ]
        payload = build_optimizer_payload(
            iteration=1,
            current_deck=round_one_deck,
            evaluation=round_one_evaluation,
            history=history,
            include_iteration_history=True,
        )
        self.assertIn("iteration_history", payload)
        self.assertEqual(len(payload["iteration_history"]), 1)
        entry = payload["iteration_history"][0]
        self.assertEqual(entry["iteration"], 0)
        self.assertEqual(entry["deck"], STARTING_DECK)
        self.assertEqual(entry["rationale"], "Increase damage density without cutting all draw.")
        self.assertEqual(entry["changes"], ["Cut 2 Strike", "Add 3 Smash", "Add 1 Wall", "Cut 2 Block"])
        self.assertEqual(entry["next_deck"], round_one_deck)
        self.assertEqual(
            entry["next_deck_win_rate"],
            round_one_evaluation.aggregate["win_rate"],
        )

    def test_prompt_payload_omits_logs_when_no_games_are_evaluated(self) -> None:
        evaluation = evaluate_deck(STARTING_DECK, seeds=[])
        payload = build_optimizer_payload(
            iteration=0,
            current_deck=STARTING_DECK,
            evaluation=evaluation,
            history=[],
        )
        self.assertEqual(payload["aggregate_results"]["losses"], 0)
        self.assertEqual(payload["loss_games"], [])

    def test_optimizer_error_includes_last_invalid_deck(self) -> None:
        class FakeResponse:
            def __init__(self, output_text: str) -> None:
                self.output_text = output_text
                self.id = "fake-response"

        class FakeResponses:
            def __init__(self, outputs: list[str]) -> None:
                self.outputs = outputs

            def create(self, **_: object) -> FakeResponse:
                return FakeResponse(self.outputs.pop(0))

        class FakeClient:
            def __init__(self, outputs: list[str]) -> None:
                self.responses = FakeResponses(outputs)

        optimizer = OpenAIDeckOptimizer(
            api_key="test-key",
            client=FakeClient(
                [
                    '{"analysis":"bad","changes":[],"deck":{"Strike":13,"Block":0,"Smash":0,"Wall":0,"Scout":0}}',
                    '{"analysis":"bad again","changes":[],"deck":{"Strike":14,"Block":0,"Smash":0,"Wall":0,"Scout":0}}',
                    '{"analysis":"still bad","changes":[],"deck":{"Strike":10,"Block":0,"Smash":0,"Wall":0,"Scout":0}}',
                    '{"analysis":"still bad","changes":[],"deck":{"Strike":11,"Block":0,"Smash":0,"Wall":0,"Scout":0}}',
                ]
            ),
        )
        with self.assertRaises(RuntimeError) as ctx:
            optimizer.propose_next_deck(
                iteration=0,
                current_deck=STARTING_DECK,
                evaluation=evaluate_deck(STARTING_DECK, seeds=[0]),
                history=[],
            )
        message = str(ctx.exception)
        self.assertIn("OpenAI did not return a valid deck proposal after 4 attempts.", message)
        self.assertIn("Deck must contain exactly 12 cards, got 11.", message)
        self.assertIn("Your last deck totaled 11, you must add 1 card.", message)
        self.assertIn('"Strike": 11', message)

    def test_optimizer_retry_note_includes_total_adjustment(self) -> None:
        class FakeResponse:
            def __init__(self, output_text: str) -> None:
                self.output_text = output_text
                self.id = "fake-response"

        class FakeResponses:
            def __init__(self, outputs: list[str]) -> None:
                self.outputs = outputs
                self.requests: list[dict[str, object]] = []

            def create(self, **kwargs: object) -> FakeResponse:
                self.requests.append(kwargs)
                return FakeResponse(self.outputs.pop(0))

        class FakeClient:
            def __init__(self, outputs: list[str]) -> None:
                self.responses = FakeResponses(outputs)

        client = FakeClient(
            [
                '{"analysis":"bad","changes":[],"deck":{"Strike":4,"Block":3,"Smash":4,"Wall":2,"Forge":1}}',
                '{"analysis":"fixed","changes":["-2 Smash"],"deck":{"Strike":4,"Block":3,"Smash":2,"Wall":2,"Forge":1}}',
            ]
        )
        optimizer = OpenAIDeckOptimizer(
            api_key="test-key",
            client=client,
        )

        suggestion = optimizer.propose_next_deck(
            iteration=0,
            current_deck=STARTING_DECK,
            evaluation=evaluate_deck(STARTING_DECK, seeds=[0]),
            history=[],
        )

        self.assertEqual(suggestion.deck, {"Strike": 4, "Block": 3, "Smash": 2, "Wall": 2, "Forge": 1})
        self.assertEqual(len(client.responses.requests), 2)
        self.assertIn(
            "Your last deck totaled 14, you must remove 2 cards.",
            str(client.responses.requests[1]["instructions"]),
        )

    def test_progress_callback_receives_each_round(self) -> None:
        class FakeOptimizer:
            model = "fake-optimizer"

            def propose_next_deck(self, **_: object) -> OptimizerSuggestion:
                return OptimizerSuggestion(
                    analysis="Shift into heavier cards.",
                    changes=["Cut 2 Strike", "Add 1 Smash", "Add 1 Wall"],
                    deck={"Strike": 4, "Block": 4, "Smash": 1, "Wall": 1, "Scout": 2},
                    model=self.model,
                )

        seen_iterations: list[int] = []

        run_optimization_loop(
            initial_deck=STARTING_DECK,
            optimizer=FakeOptimizer(),
            iterations=1,
            seeds=[0],
            progress_callback=lambda round_result: seen_iterations.append(round_result.iteration),
        )
        self.assertEqual(seen_iterations, [0, 1])

    def test_optimization_loop_applies_suggested_decks(self) -> None:
        class FakeOptimizer:
            model = "fake-optimizer"

            def __init__(self) -> None:
                self.calls = 0

            def propose_next_deck(self, **_: object) -> OptimizerSuggestion:
                decks = [
                    {"Strike": 4, "Block": 2, "Smash": 3, "Wall": 1, "Scout": 2},
                    {"Strike": 2, "Block": 2, "Smash": 4, "Wall": 2, "Scout": 2},
                ]
                analyses = [
                    "Add more damage density while keeping some draw.",
                    "Lean further into Smash and Wall after the first improvement.",
                ]
                changes = [
                    ["Cut 2 Strike", "Add 3 Smash", "Add 1 Wall", "Cut 2 Block"],
                    ["Cut 2 Strike", "Add 1 Smash", "Add 1 Wall"],
                ]
                index = self.calls
                self.calls += 1
                return OptimizerSuggestion(
                    analysis=analyses[index],
                    changes=changes[index],
                    deck=decks[index],
                    model=self.model,
                )

        run = run_optimization_loop(
            initial_deck=STARTING_DECK,
            optimizer=FakeOptimizer(),
            iterations=2,
            seeds=[0, 1],
        )
        self.assertEqual(run.optimizer_model, "fake-optimizer")
        self.assertEqual(len(run.rounds), 3)
        self.assertEqual(run.rounds[1].deck, {"Strike": 4, "Block": 2, "Smash": 3, "Wall": 1, "Scout": 2})
        self.assertEqual(run.rounds[2].deck, {"Strike": 2, "Block": 2, "Smash": 4, "Wall": 2, "Scout": 2})
        self.assertEqual(run.rounds[0].suggestion.model, "fake-optimizer")

    def test_optimization_loop_passes_player_policy_into_evaluation(self) -> None:
        class NoOpPolicy:
            name = "noop"

            def choose_turn(self, **_: object) -> list[str]:
                return []

        run = run_optimization_loop(
            initial_deck={"Strike": 12},
            iterations=0,
            player_policy=NoOpPolicy(),
            seeds=[0],
        )
        self.assertEqual(run.rounds[0].evaluation.aggregate["average_boss_hp_remaining_on_losses"], 72)

    def test_optimization_loop_calls_evaluation_progress_factory_each_round(self) -> None:
        seen_calls: list[tuple[int, int]] = []
        closed_rounds: list[int] = []

        def evaluation_progress_factory(iteration: int, total_games: int):
            seen_calls.append((iteration, total_games))

            def on_progress(completed_games: int, total: int, game) -> None:
                del completed_games
                del total
                del game

            def close() -> None:
                closed_rounds.append(iteration)

            return on_progress, close

        run_optimization_loop(
            initial_deck=STARTING_DECK,
            iterations=1,
            optimizer=type(
                "FakeOptimizer",
                (),
                {
                    "model": "fake",
                    "propose_next_deck": lambda self, **_: OptimizerSuggestion(
                        analysis="noop",
                        changes=[],
                        deck=STARTING_DECK,
                        model="fake",
                    ),
                },
            )(),
            seeds=[0, 1, 2],
            evaluation_progress_factory=evaluation_progress_factory,
        )
        self.assertEqual(seen_calls, [(0, 3), (1, 3)])
        self.assertEqual(closed_rounds, [0, 1])

    def test_cli_parser_accepts_openai_player_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "simulate",
                "--seed",
                "3",
                "--player-policy",
                "openai",
                "--player-model",
                "gpt-5.4",
            ]
        )
        self.assertEqual(args.command, "simulate")
        self.assertEqual(args.player_policy, "openai")
        self.assertEqual(args.player_model, "gpt-5.4")

    def test_cli_parser_accepts_workers_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["evaluate", "--workers", "4"])
        self.assertEqual(args.command, "evaluate")
        self.assertEqual(args.workers, 4)


if __name__ == "__main__":
    unittest.main()
