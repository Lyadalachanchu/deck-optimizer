# Micro Boss Battler

Small deterministic combat simulator for iterative deck optimization with an LLM.

## Rules

- Player starts at `36 HP`, with `3 energy` each turn.
- Player draws `3` cards on turn `1`, then `2` cards at the start of each later turn.
- Boss starts at `72 HP` and attacks for `8`, `8`, `16`, then repeats.
- Combat ends on boss death, player death, or after turn `10`.
- Decks must contain exactly `12` cards.
- Unplayed cards stay in hand between turns.

## Card Pool

- `Strike`: cost `1`, deal `5`.
- `Block`: cost `1`, gain `5 block`.
- `Smash`: cost `2`, deal `11`.
- `Wall`: cost `2`, gain `11 block`.
- `Scout`: cost `1`, draw `2`.
- `Forge`: cost `1`, your attacks deal `+2` damage for the rest of combat.

## Play Policies

By default, the simulator uses a deterministic turn search. For each turn, it explores all
legal action sequences under the current hand and energy budget, then picks the best line
using a fixed ranking:

- kill the boss this turn if possible
- otherwise survive the incoming hit
- otherwise push boss HP down while preserving useful future scaling such as `Forge`

This keeps the optimization target on deck composition instead of tactical variance.

You can also switch to an OpenAI-backed player policy. In that mode, the model is given the
rules, current turn state, current hand, current block/energy, visible discard pile, remaining
unknown deck counts, and the legal actions for the next play. It then chooses one action at a
time until it ends the turn. This is much slower and more expensive than the default search
policy because it can make multiple API calls per turn.

## Usage

Show the card pool and starting deck:

```bash
python3 -m micro_boss_battler card-pool
```

Run a single simulation:

```bash
python3 -m micro_boss_battler simulate --deck-file examples/starting_deck.json --seed 3
```

Run a single simulation with the OpenAI player:

```bash
python3 -m micro_boss_battler simulate \
  --deck-file examples/starting_deck.json \
  --seed 3 \
  --player-policy openai \
  --player-model gpt-5.4
```

Evaluate a deck across the default `100` seeds:

```bash
python3 -m micro_boss_battler evaluate --deck-file examples/starting_deck.json
```

The `evaluate` command shows a `tqdm` bar for the per-seed game progress.

To evaluate seeds in parallel, add `--workers`:

```bash
python3 -m micro_boss_battler evaluate \
  --deck-file examples/starting_deck.json \
  --seeds 0 1 2 3 \
  --workers 4
```

Evaluate using the OpenAI player policy:

```bash
python3 -m micro_boss_battler evaluate \
  --deck-file examples/starting_deck.json \
  --player-policy openai \
  --player-model gpt-5.4
```

## OpenAI Optimization Loop

Install the SDK dependency:

```bash
python3 -m pip install -r requirements.txt
```

Fill in `.env` with your API key:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.4
```

Run the iterative optimizer:

```bash
python3 -m micro_boss_battler optimize \
  --deck-file examples/starting_deck.json \
  --iterations 3 \
  --output-file optimization-run.json
```

Example longer run:

```bash
python3 -m micro_boss_battler optimize \
  --deck-file examples/starting_deck.json \
  --iterations 30 \
  --output-file optimization-run.json
```

Run the optimizer while also using the OpenAI player policy for the game simulations:

```bash
python3 -m micro_boss_battler optimize \
  --deck-file examples/starting_deck.json \
  --iterations 3 \
  --player-policy openai \
  --player-model gpt-5.4 \
  --output-file optimization-run.json
```

The `optimize` command shows nested `tqdm` bars: one for optimization rounds and one for the
per-seed games inside each round.

You can also parallelize the seed evaluations inside each round:

```bash
python3 -m micro_boss_battler optimize \
  --deck-file examples/starting_deck.json \
  --iterations 3 \
  --seeds 0 1 2 3 \
  --workers 4
```

Each optimizer step sends the model:

- the game rules
- the current deck
- aggregate evaluation metrics
- only the turn logs from seeds where the deck lost
- a compact history summary of earlier iterations

The optimizer must return strict JSON with:

- a short analysis
- a list of deck changes
- the next 12-card deck

To include a compact history of prior decks, win rates, and change rationales in each
optimizer turn, add:

```bash
python3 -m micro_boss_battler optimize \
  --deck-file examples/starting_deck.json \
  --iterations 3 \
  --include-iteration-history
```
