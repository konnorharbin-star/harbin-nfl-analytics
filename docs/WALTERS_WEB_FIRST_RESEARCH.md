# Web-first Walters football price research (no odds API key)

**Primary strategy**: collect publicly visible source evidence by normal web
search in a ChatGPT research session, preserve original independent forecasts in
Git, and run deterministic, exact-line win/push/loss math offline.

The connected GitHub repositories are the statistical engine, **not** a
paid odds-feed dependency. No API key or signup is needed for this process.
When a webpage is read in a ChatGPT session, the publicly observed prices
must be transcribed, with evidence URL, into a pregame Git research record.

## Live source hierarchy

- Primary: **named sportsbook publisher** with BOTH sides and both American
  prices at exactly the same line, e.g. FanDuel Research public odds page.
  Explicitly identify page type, the game, home/away, kickoff, and date/time
  the researcher actually read it. Official publisher is still **not** proof
  that a particular price is executable in a sportsbook account.
- Cross-check: a second independent publisher where possible, e.g. FOX
  Sports quoting bet365, NFL.com publisher-labeled reference odds, CBS Sports
  odds. These are additional *observations*, not invented same-book prices.
  Do **not** combine two different bookmakers into one two-sided quote.
- Context: NFL/NCAA/ESPN official statistics and roster/injury reports, team
  disclosures, public weather forecasts, stadium/time-zone data. Each claim
  should cite its own URL and publication time. Unknown player or weather
  information is **unknown**, not assumed fresh.
- Close: on later web research, read actual late/closing prices at the same
  named book and preserve a separate source observation if obtainable.
  Without reliable quotes and times, CLV is **not measured**.
- Outcomes: independent final scores are graded by the existing immutable
  forward evaluator, after kickoff. This record is never an order to wager.

## Exact research record

A document is stored under
`history/walters_key_forward_v1/web_observations/<observation_id>.json`.

Requirements for accepted **research-only** price comparisons:

1. Schema `walters_web_research_observation_v1`, stable file identifier,
   public HTTPS source, named verified publisher and corresponding book.
2. `observed_at` is when the researcher **actually read** the odds online.
   It is NOT the article's publication time, a game's UTC kickoff widget,
   or a sportsbook's own price update time.
3. Quote includes exact frozen `game_id`, `home_team`, `away_team`,
   `kickoff`, home/away opposing handicap and corresponding American odds
   from **the same book**. Named source and on-page quote context are required.
4. The original model file was Git-published before the web observation;
   the web evidence file was first Git-published at or after reading the page
   and before the game, all within seven days of kickoff.
5. No invented pair price, malformed 1/2-point line, other game, source-domain
   mismatch, duplicated record, or post-kickoff price gets admitted.
6. Article's price timestamp is recorded `null` if unverified. This blocks
   upgrades to executable or "fresh" status even if the article displays
   a kickoff timestamp. Set
   `verified_executable_sportsbook_quote: false` always.
7. `compare` reuses frozen independent margin and historical exact-score
   distribution: baseline Gaussian vs 3/7 challenger, proper exact-line
   WIN/PUSH/LOSS, independent matched-book no-vig conditional comparison,
   and **unshrunk raw EV hypothesis**. Neither is validated economic profit.

The output `reports/walters_web_research.json` is labelled
`RESEARCH_ONLY_NO_VERIFIED_ECONOMIC_EDGE`, with
`betting_authorized=false` and zero wagers. This workflow **does not**
mutate the betting engine's probabilities, release gates or bankroll.

## Actual October 10 research

For NFL, `nfl_20261010_2220_fanduel_week5.json` freezes 14 publicly
displayed FanDuel Research Week 5 two-sided spread pairs read Oct 10 2026
at 22:20:02 UTC (17:20:02 Central). Source:
https://fanduel.com/research/nfl-week-5-schedule-odds-for-every-game

The page explicitly warns that published odds may change; its displayed
`Oct 11 ... UTC` labels are event kickoff, **not** bookmaker quote update
timestamps. No price provenance beyond the page read is claimed.

The CFB repo can accept a separate observation only when it is read while
the game is still future and matched to one of its own frozen forecasts.
We do not retroactively fabricate CFB pregame quotes.

## Running in GitHub

Both canonical football model writers now run:

```bash
python -m scripts.walters_web_research
```

The read-only research CI tests web quote validation and uploads audit
evidence. No secret and no third-party odds API are needed.

**Scope of automation:** ChatGPT web searches are initiated in the research
conversation, **not** invisibly executed from GitHub Actions. GitHub Actions
can automatically *audit previously frozen web observations*, but does not
gain access to ChatGPT's interactive web search. Each new slate requires
another public webpage read and Git publication, or an explicitly designed
and tested public-site collector that respects site restrictions.

**Evidence thresholds for bet recommendations remain strict:** valid
original price/time and market reference, independently verified personnel,
prospectively tested predictive improvement vs market, adequate sample sizes,
bettable quote in an eligible account, and closing-line/settlement audit.
A search result with apparent +30% raw model EV alone is a **NO BET**.
