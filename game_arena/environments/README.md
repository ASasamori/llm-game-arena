# Standalone game environments

These rules engines are ordinary Python. They import neither OpenSpiel nor model
SDKs and need no credentials, database, Redis, or network. The existing runner
uses a small compatibility adapter; the original OpenSpiel entry points remain
available. All random deals and replacement cards are explicit chance actions.

## Validate rules before using models

```bash
uv run python -m unittest discover -s tests -p 'test_*environment.py'
uv run python -m game_arena environments list
uv run python -m game_arena environments inspect --game 'avalon(players=7)'
uv run python -m game_arena environments simulate --game 'coup(players=4,max_turns=200)' --seed 123 --output analytics/coup-baseline.json
uv run python -m game_arena environments replay analytics/coup-baseline.json
```

`simulate` uses random legal actions by default, or `--policy first`; it does not
construct a model. Saved traces include the exact configuration, implementation
fingerprint, actions, player observations, state hashes, terminal status, and
return vector. Replay rejects changed evidence or a changed implementation.
Trace files contain private evaluator evidence, are created with mode `0600`, and
are never overwritten. A transition limit produces `TRUNCATED`, not a win.

Python callers use the same rules directly:

```python
from game_arena.environments.registry import load_game

game = load_game('avalon(players=5,discussion_rounds=0)')
state = game.new_initial_state()
# Resolve explicit chance nodes with chance_outcomes(); then only the current
# player's legal_actions() may be applied. Policy/randomness belongs to the caller.
while state.is_chance_node():
    state.apply_action(state.chance_outcomes()[0][0])
player = state.current_player()
print(state.observation_string(player))
print([(a, state.action_to_string(player, a)) for a in state.legal_actions(player)])
```

Invalid parameters and illegal actions raise `ValueError`. Actions are checked
before state mutation. `clone()` isolates a copy. `serialize()` is a full private
audit representation; only `observation_string(player)` is safe for a player.
`public_state()` provides spectator facts for the board viewer. It excludes
secret roles, hands, unrevealed ballots, deck contents, and private trade drafts.

## Games and parameters

`one_night_werewolf(players=3,discussion_rounds=1,lone_wolf=true)` supports 3–10
players. `werewolf` and `one_night_ultimate_werewolf` are aliases. Set `roles` to a
quoted pipe-separated deck with exactly `players + 3` cards; Python also accepts
a list. Supported roles are werewolf, minion, mason, seer, robber, troublemaker,
drunk, insomniac, villager, hunter, and tanner. Physical base-deck card limits are
enforced, masons must occur as a pair, and insomniac requires a role-swapping
robber or troublemaker. Night actions use initial roles; victory uses final
roles. Secret voting, ties, hunter kills, tanner, minion, and no-wolf outcomes are
resolved. Doppelganger and expansion roles are rejected. The implementation
follows the [original English Bezier rules](https://cdn.shopify.com/s/files/1/0740/4855/files/ONUW_rules-updated_for_BGG.pdf),
including its no-wolf minion outcome; translated editions can differ.

`avalon(players=5,discussion_rounds=1)` supports 5–10 players. An optional quoted
pipe-separated `roles` deck must contain one Merlin, one Assassin, the correct
good/evil counts, and at most one each of Percival, Morgana, Mordred, and Oberon.
Generic servant and minion roles fill remaining seats. This is the
[original 2012 ruleset](https://avalon.fun/pdfs/rules.pdf): Merlin sees Oberon but
not Mordred; other evil players do not know Oberon. Includes team proposals,
secret approval votes, five-rejection loss, private quest cards, the two-fail
fourth quest at 7+ players, and assassination after three successful quests.
Big Box character changes and Lady of the Lake are unsupported.

For Werewolf and Avalon, `discussion_rounds` is an integer from 0–3. Discussion
uses bounded public claims, accusations, and passes, which may be deceptive.
It is a structured arena protocol, not unrestricted free-text conversation.
Private votes are submitted in deterministic seat order and revealed only at
the rule-defined resolution. Opponents' pending votes never enter player views.

`coup(players=3,starting_player=0,max_turns=200)` supports 2–6 players. The starting
seat must exist; `max_turns` is an integer from 1–10000. Implements the standard
15-card deck and every base action, mandatory coup at ten coins, bluffing,
action/block challenges, proof and hidden replacement, exchange, chosen influence
loss, and elimination. Two-player starting coins follow the
[publisher-authored base rulebook](https://officialgamerules.org/wp-content/uploads/2025/02/Coup-Rulebook.pdf).
Clockwise reaction priority resolves simultaneous physical-table challenges.
The declared tournament turn cap gives everyone zero utility if nobody has won;
it never declares the remaining players winners. Reformation and the alternate
two-player deal are unsupported.

`catan(players=4,starting_player=-1,max_turns=500,max_actions_per_turn=200)`
supports the base game for 3–4 players, following the publisher's
[2020 rules and combined trade/build variant](https://www.catan.com/sites/default/files/2021-07/catan-25th-rules_eng-200313.pdf).
It uses 19 hexes, 54 intersections, 72 road locations, variable terrain,
standard spiral number tokens, nine fixed ports, and reverse-order second
placement. Production, bank shortages, seven/discards, robber theft, all five
development card types, piece supplies, Longest Road, Largest Army, and the
10-point victory condition are implemented. A development card bought this turn
cannot be played until a later turn; victory-point cards count immediately.
Expansion content, Helpers, and 2/5/6-player variants are unsupported.

`starting_player=-1` selects the first seat through a recorded chance action;
otherwise it must identify a seat. `max_turns` accepts 1–10000 and
`max_actions_per_turn` accepts 20–10000. These are declared arena safety caps,
including negotiation actions; reaching either without a winner records a draw.
`num_steps` remains the independent harness transition cap and records truncation.

Domestic trades support arbitrary resource bundles, consent, rejection, and
counteroffers. During the active player's main phase, choose `Propose domestic trade`,
select a partner, edit the give/receive bundles, then send the offer. Only the
recipient can accept, reject, or counter; acceptance checks both holdings before
moving cards atomically. Drafts are private to their editor. No gifts, loans,
same-resource exchanges, or development-card trading are permitted. Every trade
involves the active player. Maritime actions apply 4:1, general-port 3:1, or
resource-port 2:1 rates and check bank supply. The report displays offered bundles
and completed trades while keeping each player's resource mix private.

```bash
uv run python -m game_arena environments simulate --game 'catan(players=3,max_turns=20)' --seed 0 --output analytics/catan-baseline.json
uv run python -m game_arena environments replay analytics/catan-baseline.json
uv run python -m game_arena plan --games 'catan(players=3,max_turns=20)' --models smoke --episodes 2 --seed 0 --max-steps 5000 --output analytics/catan-smoke --execute-mock --workers 2
```

## Use the existing arena after validation

The game catalog and `/v1/games/inspect` expose these configurations alongside
OpenSpiel games. `env` takes the same specification, and PVP still requires one
model per player. The queue and `runner.py` remain responsible for execution.
Built-in game instructions, private observations, and legal action labels enter
each model prompt. Full hidden state and other seats' private messages do not.
Team and individual utility vectors are preserved; these games are not converted
into two-player Elo rankings.

```bash
uv run python -m game_arena plan --games 'one_night_werewolf(players=3)' 'avalon(players=5)' 'coup(players=3)' --models smoke --episodes 2 --seed 123 --max-steps 10000 --output analytics/social-smoke --execute-mock --workers 2
uv run python -m game_arena export --dataset analytics/social-smoke --output analytics/social-export
```

This second check uses only offline mock competitors and exercises the arena
integration after the model-independent rules checks. Mock players always select
the first legal action; their results measure infrastructure, not strategic skill.
