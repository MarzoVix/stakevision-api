from paddleocr import PaddleOCR
import re
import json

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

def extract_lines(img_path: str) -> list[dict]:
    result = ocr.ocr(img_path, cls=True)
    if not result or not result[0]:
        return []
    lines = []
    for item in result[0]:
        bbox = item[0]
        text = item[1][0].strip()
        conf = item[1][1]
        x_center = (bbox[0][0] + bbox[2][0]) / 2
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        if text:
            lines.append({'text': text, 'x': x_center, 'y': y_center, 'conf': conf})
    return sorted(lines, key=lambda l: l['y'])

def group_lines(lines: list[dict], threshold: int = 20) -> list[list[dict]]:
    if not lines: return []
    groups = []
    current = [lines[0]]
    for line in lines[1:]:
        if abs(line['y'] - current[-1]['y']) <= threshold:
            current.append(line)
        else:
            groups.append(sorted(current, key=lambda l: l['x']))
            current = [line]
    groups.append(sorted(current, key=lambda l: l['x']))
    return groups

def row_text(group: list[dict]) -> str:
    return ' '.join(item['text'] for item in group)

def clean_odds(text: str) -> str:
    text = text.strip().replace('O', '0').replace('l', '1')
    match = re.search(r'[+-]\d+', text)
    return match.group(0) if match else text

def clean_money(text: str) -> str:
    match = re.search(r'\$?([\d,]+\.?\d*)', text)
    return match.group(1).replace(',', '') if match else text

def detect_sportsbook(lines: list[dict]) -> str:
    all_text = ' '.join(l['text'].lower() for l in lines)
    # Normalize OCR concatenation for matching
    no_space = all_text.replace(' ', '')

    # ── Brand name matches (highest confidence) ──
    if 'draftkings' in all_text or 'raftkings' in all_text or 'draftkngs' in all_text:
        return 'DraftKings'
    if re.search(r'bet\s*id[:\s]*dk\d', all_text):
        return 'DraftKings'
    if 'fanduel' in all_text:
        return 'FanDuel'
    if 'prizepicks' in all_text:
        return 'PrizePicks'
    if 'betmgm' in all_text:
        return 'BetMGM'
    if 'fanatics' in all_text or 'fancash' in all_text:
        return 'Fanatics'
    if 'onyx' in all_text:
        return 'Onyx'
    if 'hard rock' in all_text or 'hardrock' in no_space:
        return 'Hard Rock'
    if 'sgpmax' in no_space or re.search(r'\d+-?\s*Bet\s+Parlay', all_text, re.I):
        return 'Hard Rock'
    # Hard Rock has "PLAYER - MARKET" format with "Hide selections" and "EDT/CDT" times
    if 'hide selections' in all_text and re.search(r'\b(edt|cdt|est|cst)\b', all_text):
        return 'Hard Rock'
    if 'kalshi' in all_text:
        return 'Kalshi'

    # ── Underdog: "higher"/"lower" + "reuse picks" ──
    if ('higher' in all_text or 'lower' in all_text) and 'reusepicks' in no_space:
        return 'Underdog'

    # ── Layout/phrasing pattern matches ──

    # DraftKings: "pick parlay", "pick sgp", "sgpx", "to pay:", "parlay boost"
    if 'sgpx' in no_space:
        return 'DraftKings'
    if (re.search(r'pick\s*parlay|pick\s*sgp', all_text) or 'pickparlay' in no_space) and 'total alternate' not in all_text:
        return 'DraftKings'
    if ('to pay:' in all_text or 'topay:' in no_space or 'topay$' in no_space) and 'total alternate' not in all_text and all_text.count('moneyline') < 3:
        return 'DraftKings'
    if 'parlayboost' in no_space or 'parlay boost' in all_text:
        return 'DraftKings'

    # FanDuel: "leg parlay", "profitboost", "cash out", "betid:", "total wager"
    if re.search(r'leg\s*parlay|\bleg parlay\b', all_text) or 'legparlay' in no_space:
        return 'FanDuel'
    if 'profitboost' in no_space or 'profit boost' in all_text:
        return 'FanDuel'
    if 'betid:' in no_space or 'betid :' in all_text:
        return 'FanDuel'
    if 'total wager' in all_text or 'totalwager' in no_space:
        return 'FanDuel'
    if 'total alternate' in all_text:
        return 'FanDuel'
    if 'cash out' in all_text and ('parlay' in all_text or 'moneyline' in all_text):
        return 'FanDuel'
    if '1st inning' in all_text or 'same game parlay+' in all_text:
        return 'FanDuel'

    # BetMGM: "hide legs", "stake:", "edit bet", "result-" prefix
    if 'hide legs' in all_text or 'hidelegs' in no_space:
        return 'BetMGM'
    if 'stake:' in all_text or 'stake:$' in no_space:
        return 'BetMGM'
    if 'editbet' in no_space or 'edit bet' in all_text:
        return 'BetMGM'

    # PrizePicks: "starts in", "power play", "flex play"
    if 'starts in' in all_text:
        return 'PrizePicks'
    if re.search(r'(power|flex|standard)\s*play', all_text):
        return 'PrizePicks'

    # Onyx: "pick placed", "pick receipt", "bet#", "hide selections"
    if 'pickplaced' in no_space or 'pick placed' in all_text:
        return 'Onyx'
    if 'pickreceipt' in no_space or 'pick receipt' in all_text:
        return 'Onyx'
    if 'bet#' in all_text or 'bet #' in all_text:
        return 'Onyx'
    if re.search(r'hide\s*selections?', all_text):
        return 'Onyx'

    # Hard Rock: "to win" + "hide selections" combo
    if 'to win' in all_text and 'hide selections' in all_text:
        return 'Hard Rock'

    # Underdog fallback: "higher"/"lower" without other brand signals
    if ('higher' in all_text or 'lower' in all_text):
        return 'Underdog'

    # ── Fallback: detect by content patterns (cropped screenshots) ──

    # FanDuel: "Straight Bet", "Same Game Parlay", "ANY TIME GOAL SCORER",
    # "PLAYER TO RECORD", "ALTERNATE PUCK LINE", "60 MIN MONEYLINE"
    if 'straight bet' in all_text:
        return 'FanDuel'
    if 'same game parlay' in all_text:
        return 'FanDuel'
    if 'any time goal scorer' in all_text or 'anytime goal scorer' in all_text:
        return 'FanDuel'
    if 'player to record' in all_text:
        return 'FanDuel'
    if 'alternate puck line' in all_text or 'alt total goals' in all_text:
        return 'FanDuel'
    if '60 min moneyline' in all_text:
        return 'FanDuel'
    if 'first goal scorer' in all_text:
        return 'FanDuel'

    # DraftKings: "Anytime Goalscorer" (DK-style formatting), stat combo lines
    if 'anytime goalscorer' in all_text:
        return 'DraftKings'
    if re.search(r'hits \+ runs \+ (rbis?|stolen bases|walks)', all_text):
        return 'DraftKings'
    if re.search(r'hits \+ walks \+ stolen bases', all_text):
        return 'DraftKings'
    if 'both teams to score' in all_text and 'shots on target' in all_text:
        return 'DraftKings'
    # DK cropped: "Player Points" lines with team abbreviation matchups
    if re.search(r'\w+ points\b', all_text) and re.search(r'\b[a-z]{2,4} today \d+:\d+', all_text):
        return 'DraftKings'

    return 'Unknown'

# ── Shared Utilities ──────────────────────────────────────────────────────────

def clean_event(text: str) -> str:
    """Strip trailing time stamps and ellipsis from event/matchup strings."""
    text = re.sub(r'\s*\d{1,2}:\d{2}\s*(AM|PM|am|pm)\s*(ET|CT|MT|PT)?\s*$', '', text)
    text = re.sub(r'\s*\.{2,}\s*$', '', text)
    return text.strip()

NOISE_WORDS = {'TO', 'No', 'Yes', 'Out', 'Win', 'Bet', 'The', 'For', 'All',
               'Or', 'At', 'In', 'On', 'If', 'An', 'Up', 'My', 'So', 'Do'}

def is_valid_player_name(name: str) -> bool:
    if not name or len(name) < 4:
        return False
    if not re.search(r'[A-Za-z]{2,}', name):
        return False
    if name.strip() in NOISE_WORDS:
        return False
    return True

def is_fake_leg(sel: dict) -> bool:
    combined = f"{sel.get('player','') or ''} {sel.get('stat','') or ''} {sel.get('market','') or ''}"
    return bool(re.search(r'\d+\s*(Pick|Leg)\s*(Parlay|SGP)', combined, re.I))

SPORT_TEAMS = {
    'MLB': ['Cardinals', 'Yankees', 'Red Sox', 'Cubs', 'Dodgers', 'Mets', 'Braves',
            'Athletics', 'Tigers', 'Nationals', 'Blue Jays', 'Padres', 'Giants',
            'Phillies', 'Brewers', 'Royals', 'Reds', 'Rangers', 'Orioles', 'Pirates',
            'Astros', 'Angels', 'Rays', 'Twins', 'Mariners', 'Guardians', 'Rockies',
            'Marlins', 'Diamondbacks', 'White Sox'],
    'NBA': ['Lakers', 'Celtics', 'Warriors', 'Heat', 'Bulls', 'Knicks', 'Nets',
            'Bucks', 'Suns', 'Clippers', 'Nuggets', 'Mavericks', 'Thunder', 'Spurs',
            '76ers', 'Raptors', 'Grizzlies', 'Pelicans', 'Timberwolves', 'Hawks',
            'Cavaliers', 'Pacers', 'Magic', 'Pistons', 'Wizards', 'Hornets', 'Jazz',
            'Trail Blazers', 'Kings', 'Rockets'],
    'NHL': ['Maple Leafs', 'Canadiens', 'Bruins', 'Penguins', 'Blackhawks', 'Rangers',
            'Ducks', 'Kings', 'Sharks', 'Senators', 'Flames', 'Oilers', 'Canucks',
            'Lightning', 'Panthers', 'Hurricanes', 'Devils', 'Islanders', 'Flyers',
            'Capitals', 'Blue Jackets', 'Predators', 'Stars', 'Wild', 'Jets',
            'Avalanche', 'Kraken', 'Sabres', 'Red Wings', 'Coyotes',
            'Golden Knights', 'Mammoth'],
    'NFL': ['Chiefs', 'Eagles', 'Cowboys', 'Patriots', 'Packers', 'Bears', '49ers',
            'Broncos', 'Ravens', 'Steelers', 'Seahawks', 'Bills', 'Dolphins',
            'Bengals', 'Chargers', 'Raiders', 'Colts', 'Texans', 'Titans',
            'Jaguars', 'Browns', 'Saints', 'Buccaneers', 'Falcons', 'Panthers',
            'Cardinals', 'Rams', 'Giants', 'Commanders', 'Lions', 'Vikings'],
}

def detect_sport_from_text(text: str) -> str:
    text_lower = text.lower()
    for sport, teams in SPORT_TEAMS.items():
        for team in teams:
            if team.lower() in text_lower:
                return sport
    # Keyword-based fallback for sports without team lists
    if re.search(r'\b(NCAAB|NCAAF|March Madness)\b', text, re.I):
        return 'NCAAB'
    if re.search(r'\b(soccer|La Liga|Serie A|Bundesliga|Premier League|MLS|FC Barcelona|Girona)\b', text, re.I):
        return 'Soccer'
    if re.search(r'\b(UFC|MMA)\b', text, re.I):
        return 'MMA'
    if re.search(r'\b(ATP|WTA|Tennis)\b', text, re.I):
        return 'Tennis'
    return ''

# Stat keywords used to identify player prop lines (not team/date lines)
DK_STAT_WORDS = [
    # MLB
    'Hits', 'Runs', 'RBI', 'RBl', 'RBIs', 'Runs + RBI', 'HR', 'Home Run', 'Home Runs',
    'Singles', 'Doubles', 'Triples', 'Stolen Bases', 'Total Bases', 'Walks',
    'Strikeouts', 'Strikeouts Thrown', 'Earned Runs', 'Innings Pitched',
    'Hits Allowed', 'Earned Runs Allowed',
    'Bases', 'Outs', 'Pitching Outs',
    # NBA
    'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers',
    'Three Pointers Made', 'Three Pointers', '3-Pointers Made',
    '3-Pointers', '3 Pointers Made', '3 Pointers',
    'Double Double', 'Triple Double',
    'Points + Rebounds + Assists', 'Points + Rebounds', 'Points + Assists',
    'Rebounds + Assists',
    # NFL
    'Touchdowns', 'Yards', 'Passing Yards', 'Rushing Yards', 'Receiving Yards',
    'Receptions', 'Tackles', 'Sacks', 'Interceptions',
    # NHL
    'Goals', 'Saves', 'Shots on Goal', 'Blocked Shots', 'Faceoffs Won',
    # Soccer
    'Anytime Goalscorer', 'Shots on Target', 'Both Teams to Score',
    # O/U variants (DK appends O/U to stat names)
    'Points O/U', 'Assists O/U', 'Rebounds O/U', 'Hits O/U',
    'Runs O/U', 'Strikeouts O/U', 'Singles O/U', 'Doubles O/U',
    'RBIs O/U', 'Strikeouts Thrown O/U', 'Home Runs O/U',
    'Stolen Bases O/U', 'Total Bases O/U', 'Walks O/U',
    'Three Pointers Made O/U', 'Goals O/U', 'Saves O/U',
    'Hits Allowed O/U', 'Earned Runs Allowed O/U',
    'Shots on Goal O/U', 'Blocked Shots O/U',
]

# Game total / market keywords that appear as standalone lines (no player name)
DK_GAME_MARKETS = [
    r'Runs\s*[-–]\s*1st\s*Inning', r'Runs\s*[-–]\s*\d+\w*\s*Inning',
    r'Total\s+Runs', r'Total\s+Points', r'Total\s+Goals',
    r'1st\s*Half', r'2nd\s*Half', r'1st\s*Quarter',
    r'Alternate\s+Total', r'Alternate\s+Run\s+Line',
    r'Team\s+Total',
    r'1st\s+Home\s+Run\s+Type',
    r'Both\s+Teams\s+to\s+Score',
    r'1st\s+Half\s+Total\s+Goals',
    r'Total\s+Goals',
]

def _normalize_text(text: str) -> str:
    """Normalize common OCR errors and split concatenated OCR text."""
    text = text.replace('RBls', 'RBIs').replace('RBl', 'RBI')
    # Split concatenated player+stat: "BallThree" → "Ball Three", "ReavesThree" → "Reaves Three"
    text = re.sub(r'([a-z])(Three|Points|Rebounds|Assists|Hits|Runs|Strikeouts|Blocks|Steals|Goals|Saves|Shots|Home|Stolen|Total|Earned|Double|Triple|Walks|Singles|Doubles|Triples|Faceoffs|Blocked|Innings|Anytime|Both)', r'\1 \2', text)
    # Fix "PointersMade" → "Pointers Made"
    text = re.sub(r'(Pointers)(Made)', r'\1 \2', text)
    return text

def _is_stat_line(text: str) -> bool:
    """True only if text contains an actual stat keyword — not team names or dates."""
    # Reject lines that look like team/date/matchup
    if re.search(r'(BRAVES|DIAMONDBACKS|METS|GIANTS|YANKEES|DODGERS|CUBS|SOX)', text, re.I):
        return False
    if re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d', text, re.I):
        return False
    norm = _normalize_text(text)
    return any(stat in norm for stat in DK_STAT_WORDS)

def _extract_player_stat(text: str) -> tuple[str, str]:
    """Split 'Gabriel Moreno Hits + Runs + RBIs' into (player, stat)."""
    norm = _normalize_text(text)
    # Find where the stat portion starts
    earliest = len(norm)
    for stat in DK_STAT_WORDS:
        idx = norm.find(stat)
        if idx != -1 and idx < earliest:
            earliest = idx
    if earliest > 0 and earliest < len(norm):
        player = norm[:earliest].strip()
        stat = norm[earliest:].strip()
        return player, stat
    return '', norm

# ── DraftKings Parser ─────────────────────────────────────────────────────────
def parse_draftkings(lines: list[dict]) -> dict:
    groups = group_lines(lines, threshold=25)
    texts = [row_text(g) for g in groups]

    result = {'sportsbook': 'DraftKings', 'bet_type': '', 'total_odds': '',
              'wager': '', 'payout': '', 'status': '', 'selections': []}

    current_line = ''
    current_event = ''
    for i, text in enumerate(texts):
        # ── Matchup with @: "LA Dodgers @ TOR Blue Jays • Today 2:07PM"
        # Set current_event for @ patterns (these appear inline with selections)
        if re.search(r'\s@\s', text):
            matchup_m = re.search(r'([A-Z].*?@[^•\u00ab\u00bb]+)', text)
            current_event = matchup_m.group(1).strip() if matchup_m else text.strip()
            current_event = re.sub(r'[\u00ab\u00bb\s]+$', '', current_event).strip()
            if result['selections'] and not result['selections'][-1].get('event'):
                result['selections'][-1]['event'] = current_event
            continue

        # Skip team name lines and dates — handled in post-processing
        if re.match(r'^[A-Z]{2,4}\s+[A-Z]{2,}', text.strip()):
            continue
        if re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d', text, re.I):
            continue

        # Status (standalone or at end of line like "Won")
        if re.match(r'^(Open|Won|Lost|Push|Settled|Void)$', text, re.I):
            result['status'] = text.strip()
            continue

        # Bet type + odds — only set once
        if not result['bet_type']:
            m = re.search(r'(\d+)\s*Pick\s*(Parlay|SGP)', text, re.I)
            if m:
                result['bet_type'] = f"{m.group(1)} Pick {m.group(2)}"
                all_odds = re.findall(r'[+-]\d{2,}', text)
                if all_odds:
                    result['total_odds'] = all_odds[-1]
                continue

        # Wager + payout
        wager_m = re.search(r'Wager[:\s]+\$?([\d,\.]+)', text, re.I)
        pay_m = re.search(r'(?:To Pay|Payout|Paid)[:\s]+\$?([\d,\.]+)', text, re.I)
        if wager_m: result['wager'] = wager_m.group(1).replace(',', '')
        if pay_m: result['payout'] = pay_m.group(1).replace(',', '')

        # Sub-leg odds: "Pick SGP | +400"
        if re.match(r'\d*\s*Pick\s*SGP\s*[|I]\s*[+-]\d+', text, re.I):
            continue

        # Line value: standalone "3+" or "2.5"
        line_m = re.match(r'^(\d+\.?\d*)\+?$', text.strip())
        if line_m:
            current_line = line_m.group(1) + '+'
            continue

        # ── Game-level market: "No Home Run" with "1st Home Run Type" below
        if re.match(r'^(No Home Run|Home Run|No Run)', text.strip(), re.I):
            market = ''
            if i + 1 < len(texts) and re.search(r'Home Run Type|Inning', texts[i+1], re.I):
                market = texts[i+1].strip()
            result['selections'].append({
                'pick': text.strip(), 'market': market or '1st Home Run Type',
                'pick_type': 'GAME_PROP', 'event': current_event
            })
            current_line = ''
            continue

        # ── Player + odds / market below: "Sidney Crosby +215" / "Anytime Goalscorer"
        player_odds_m = re.match(r'^([A-Z][a-zA-Z\'\-\.]+(?:\s[A-Z][a-zA-Z\'\-\.]+)*(?:\s*\([A-Z]+\))?)\s+([+-]\d+)$', text.strip())
        if player_odds_m and i + 1 < len(texts):
            next_lower = texts[i+1].lower()
            if any(w.lower() in next_lower for w in ['Goalscorer', 'Goal Scorer', 'Shots on Target',
                                                       'Both Teams', 'Moneyline']):
                result['selections'].append({
                    'player': player_odds_m.group(1), 'odds': player_odds_m.group(2),
                    'market': texts[i+1].strip(), 'pick_type': 'PROP',
                    'event': current_event
                })
                continue

        # ── Player name / market on next line: "Macklin Celebrini" / "Anytime Goalscorer"
        if (re.match(r'^[A-Z][a-zA-Z\'\-\.]+(?:\s[A-Z][a-zA-Z\'\-\.]+)+(?:\s*\([A-Z]+\))?$', text.strip())
                and i + 1 < len(texts)
                and any(w.lower() in texts[i+1].lower() for w in ['Goalscorer', 'Goal Scorer',
                                                                    'Shots on Target', 'Both Teams'])):
            result['selections'].append({
                'player': text.strip(), 'market': texts[i+1].strip(),
                'pick_type': 'PROP', 'event': current_event
            })
            continue

        # Normalize text for stat matching
        text_norm = _normalize_text(text)

        # Player + stat line (H+R+RBI type props)
        if _is_stat_line(text_norm):
            player, stat = _extract_player_stat(text_norm)
            if is_valid_player_name(player):
                result['selections'].append({
                    'player': player, 'stat': stat, 'pick': 'Over',
                    'line': current_line, 'pick_type': 'PROP',
                    'event': current_event
                })
            current_line = ''
            continue

        # ── Live bet: "Strike/Foul 1 -120 Won"
        live_m = re.match(r'^(.+?)\s+(\d+)\s+([+-]\d+)\s+(Won|Lost|Push)', text, re.I)
        if live_m:
            result['selections'].append({
                'market': live_m.group(1), 'line': live_m.group(2),
                'odds': live_m.group(3), 'pick_type': 'LIVE',
                'event': current_event
            })
            result['status'] = live_m.group(4)
            if not result['total_odds']: result['total_odds'] = live_m.group(3)
            if not result['bet_type']: result['bet_type'] = 'Live'
            continue

        # ── Live bet context: "Live Yoan Moncada - 5th Plate Appearance..."
        if text.startswith('Live ') and result['selections']:
            current_event = text[5:].strip()
            result['selections'][-1]['event'] = current_event
            continue

        # ── Moneyline leg
        ml_m = re.match(r'^(?:\d+[A-Z]?\s+)?([A-Z]{2,4})\s+([A-Za-z\s]+?)\s+(-?\d+)$', text.strip())
        if ml_m and i + 1 < len(texts) and re.match(r'^(Moneyline|Run Line|Spread)', texts[i+1], re.I):
            result['selections'].append({
                'team': f"{ml_m.group(1)} {ml_m.group(2).strip()}",
                'odds': ml_m.group(3), 'pick_type': 'MONEYLINE',
                'event': current_event
            })
            if not result['bet_type']: result['bet_type'] = 'Parlay'
            continue

        # ── Spread/Run Line leg — also handle OCR merging "A Run Line LA Dodgers -1.5 -118"
        inline_spread = re.match(r'^[A-Za-z]?\s*(?:Run Line|Spread)\s+(.+?)\s+([+-][\d\.]+)\s+([+-]?\d+)$', text.strip(), re.I)
        if inline_spread:
            result['selections'].append({
                'team': inline_spread.group(1).strip(), 'line': inline_spread.group(2),
                'odds': inline_spread.group(3), 'pick_type': 'SPREAD',
                'event': current_event
            })
            if not result['bet_type']: result['bet_type'] = 'Parlay'
            continue

        spread_m = re.match(r'^(?:\d+[A-Z]?\s+)?([A-Z][A-Za-z\s]+?)\s+([+-][\d\.]+)\s+(-?\d+)$', text.strip())
        if spread_m and i + 1 < len(texts) and re.match(r'^(Run Line|Spread)', texts[i+1], re.I):
            result['selections'].append({
                'team': spread_m.group(1).strip(), 'line': spread_m.group(2),
                'odds': spread_m.group(3), 'pick_type': 'SPREAD',
                'event': current_event
            })
            if not result['bet_type']: result['bet_type'] = 'Parlay'
            continue

        # ── Single bet moneyline — "NY Yankees" then "Moneyline"
        if re.match(r'^Moneyline$', text.strip(), re.I):
            if not result['selections'] and i > 0:
                team = texts[i-1].strip()
                if is_valid_player_name(team):
                    result['selections'].append({
                        'team': team, 'market': 'Moneyline', 'pick_type': 'MONEYLINE',
                        'event': current_event
                    })
                    if not result['bet_type']: result['bet_type'] = 'Straight'
                    continue
            if result['selections']:
                result['selections'][-1]['market'] = text.strip()
            continue

        # Market type label (attach to last selection)
        if re.match(r'^(Run Line|Spread|Total)', text, re.I):
            if result['selections']:
                result['selections'][-1]['market'] = text.strip()
            continue

        # ── CASH OUT line with odds
        if 'cash out' in text.lower() and not result['total_odds']:
            odds_m = re.search(r'[+-]\d{2,}', text)
            if odds_m:
                result['total_odds'] = odds_m.group(0)
            continue

    # Filter out fake header legs
    result['selections'] = [s for s in result['selections'] if not is_fake_leg(s)]

    # Backfill empty events — scan texts for team pairs and build events
    # DK layout: selections appear BEFORE team names, so we need post-processing
    # Team lines: "ATL BRAVES" or "ARI DIAMONDBACKS Apr 2, 2026, 9:40 PM"
    team_events = []
    team_buf = []
    for text in texts:
        # Extract team name (strip any trailing date)
        tm = re.match(r'^([A-Z]{2,4}\s+[A-Z][A-Z\s]+?)(?:\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|\s*$)', text.strip())
        if tm:
            team_buf.append(tm.group(1).strip())
            if len(team_buf) == 2:
                team_events.append(f"{team_buf[0]} vs {team_buf[1]}")
                team_buf = []
        elif re.search(r'\s@\s', text):
            matchup_m = re.search(r'([A-Z].*?@[^•\u00ab\u00bb]+)', text)
            if matchup_m:
                team_events.append(re.sub(r'[\u00ab\u00bb\s]+$', '', matchup_m.group(1)).strip())
            team_buf = []

    # Assign events to selections — distribute team_events in order to empty selections
    if team_events:
        event_idx = 0
        sels_without_event = [s for s in result['selections'] if not s.get('event')]
        # If we have same number of events as SGP sub-legs, map them
        # Otherwise just fill sequentially
        for sel in result['selections']:
            if not sel.get('event') and event_idx < len(team_events):
                sel['event'] = team_events[event_idx]
            # Advance to next event after seeing all selections for this matchup
            # Heuristic: advance when we've filled a group (use leg structure)

        # Better: distribute events evenly across selections
        if len(team_events) >= 2 and sels_without_event:
            sels_per_event = max(1, len(sels_without_event) // len(team_events))
            idx = 0
            for ei, event in enumerate(team_events):
                for _ in range(sels_per_event):
                    if idx < len(sels_without_event):
                        sels_without_event[idx]['event'] = event
                        idx += 1
            # Assign remaining to last event
            while idx < len(sels_without_event):
                sels_without_event[idx]['event'] = team_events[-1]
                idx += 1
        elif len(team_events) == 1:
            for sel in result['selections']:
                if not sel.get('event'):
                    sel['event'] = team_events[0]

    # Extract leg count from bet_type
    leg_count_m = re.search(r'(\d+)\s*Pick', result.get('bet_type', ''), re.I)
    if leg_count_m:
        result['leg_count'] = int(leg_count_m.group(1))

    # Detect sport from all text
    all_text = ' '.join(l['text'] for l in lines)
    result['sport'] = detect_sport_from_text(all_text)

    return result

# ── FanDuel Parser ────────────────────────────────────────────────────────────
FD_MARKET_WORDS = ['STRIKEOUTS', 'HOME RUN', 'MONEYLINE', 'SPREAD', 'TOTAL',
                   'OVER/UNDER', 'TO RECORD', 'HITS', 'RUNS', 'RBI', 'POINTS',
                   'REBOUNDS', 'ASSISTS', 'GOALS', 'TOUCHDOWN', 'YARDS',
                   'GOALSCORER', 'GOAL SCORER', 'DOUBLE DOUBLE', 'WINNER',
                   'SHOTS ON GOAL', 'PUCKLINE', 'ALT TOTAL', 'ALT STRIKEOUTS',
                   'MADE THREES', 'MADE THREE', 'ALT', 'SAVES', '1ST HALF',
                   'TRIPLE DOUBLE', 'BLOCKED SHOTS',
                   'ANY TIME', 'FIRST GOAL', '60 MIN', 'PUCK LINE',
                   'ALTERNATE', 'PLAYER TO']

def parse_fanduel(lines: list[dict]) -> dict:
    groups = group_lines(lines, threshold=20)
    texts = [row_text(g) for g in groups]

    result = {'sportsbook': 'FanDuel', 'bet_type': '', 'total_odds': '',
              'wager': '', 'payout': '', 'status': '', 'selections': []}

    # Pre-normalize all OCR text for FanDuel concatenation issues
    for ti in range(len(texts)):
        t = texts[ti]
        t = re.sub(r'TORECORD', 'TO RECORD', t)
        t = re.sub(r'TOHIT', 'TO HIT', t)
        t = re.sub(r'TO\s*HITA\b', 'TO HIT A', t)
        t = re.sub(r'RECORDA', 'RECORD A', t)
        t = re.sub(r'\bAHIT\b', 'A HIT', t)
        t = re.sub(r'\bAHOME\b', 'A HOME', t)
        t = re.sub(r'PLAYER(\d)', r'PLAYER \1', t)
        t = re.sub(r'(\w)TOTAL\s*SAVES', r'\1 TOTAL SAVES', t)
        t = re.sub(r'Over(\d)', r'Over \1', t)
        texts[ti] = t

    current_event = ''
    i = 0
    while i < len(texts):
        text = texts[i]
        text_lower = text.lower()

        # ── Status ──
        if re.match(r'^(Won|Lost|Open|Settled|Void|Cashed Out)$', text.strip(), re.I):
            result['status'] = text.strip()
            i += 1; continue

        # ── Bet type + total odds: "6 Leg Parlay +2613" or "3 Leg Parlay. +2084"
        # or "SGP 7Leg Same Game Parlay+ +3435" or "7 Pick Parlay 1 3890+5446 Open"
        bt_m = re.match(r'^(?:SGP\s*[\+]?\s*)?(\d+)\s*Leg\s*(.*?Parlay\.?\+?|SGP[^+]*)\s*([+-]\d+)', text, re.I)
        if not bt_m:
            bt_m = re.match(r'^(\d+)\s*Pick\s*(Parlay)\.?\s*.*?([+-]\d+)', text, re.I)
        if bt_m and not result['bet_type']:
            result['bet_type'] = f"{bt_m.group(1)} Leg {bt_m.group(2).strip().rstrip('.')}"
            result['total_odds'] = bt_m.group(3)
            i += 1; continue

        # "Straight Bet +1100"
        straight_m = re.match(r'^Straight\s+Bet\s+([+-]\d+)', text, re.I)
        if straight_m and not result['bet_type']:
            result['bet_type'] = 'Straight'
            result['total_odds'] = straight_m.group(1)
            i += 1; continue

        # "SGP Same Game Parlay" header or "SGP Same Game Parlay +117" — skip/extract odds
        sgp_m = re.match(r'^(?:s|S)GP\s*\]?\s*Same Game', text, re.I)
        if sgp_m:
            odds_m = re.search(r'([+-]\d+)', text)
            if odds_m and not result['bet_type']:
                result['bet_type'] = 'SGP'
                result['total_odds'] = odds_m.group(1)
            i += 1; continue

        # ── Wager / payout ──
        # "TOTAL WAGER TOTAL PAYOUT" header + "$7.00 $38.50" amounts
        if 'total wager' in text_lower or 'total payout' in text_lower:
            # Amounts may be on same line or next line
            amounts = re.findall(r'\$?([\d,\.]+)', text)
            if not amounts and i + 1 < len(texts):
                amounts = re.findall(r'\$?([\d,\.]+)', texts[i + 1])
                i += 1
            if len(amounts) >= 2:
                result['wager'] = amounts[0].replace(',', '')
                result['payout'] = amounts[1].replace(',', '')
            i += 1; continue

        # "Wager Payout" or standalone wager/payout
        if re.match(r'^Wager\s+Payout', text, re.I):
            if i + 1 < len(texts):
                amounts = re.findall(r'\$?([\d,\.]+)', texts[i + 1])
                if len(amounts) >= 2:
                    result['wager'] = amounts[0].replace(',', '')
                    result['payout'] = amounts[1].replace(',', '')
            i += 1; continue

        # Inline wager/payout: "Wager:$5.00 ToPay:$277.30"
        wager_m = re.search(r'(?:wager|stake)[:\s]*\$?([\d,\.]+)', text, re.I)
        pay_m = re.search(r'(?:payout|to\s*pay|to win|paid)[:\s]*\$?([\d,\.]+)', text, re.I)
        if wager_m: result['wager'] = wager_m.group(1).replace(',', '')
        if pay_m: result['payout'] = pay_m.group(1).replace(',', '')

        # ── Total odds standalone ──
        if re.match(r'^[+-]\d{3,}$', text.strip()):
            if not result['total_odds']:
                result['total_odds'] = text.strip()
            i += 1; continue

        # ── Pattern 1: NRFI — "Under -122" / market / matchup ──
        pick_m = re.match(r'^(Over|Under|Yes|No)\s*([+-]\d+)?$', text.strip(), re.I)
        if pick_m:
            pick = pick_m.group(1)
            odds = pick_m.group(2) or ''
            next_idx = i + 1
            if not odds and next_idx < len(texts) and re.match(r'^[+-]\d+$', texts[next_idx].strip()):
                odds = texts[next_idx].strip()
                next_idx += 1
            market = texts[next_idx].strip() if next_idx < len(texts) else ''
            next_idx += 1
            event_parts = []
            while next_idx < len(texts):
                peek = texts[next_idx].strip()
                if re.match(r'^(Over|Under|Yes|No)\s*([+-]\d+)?$', peek, re.I): break
                if re.match(r'^[+-]\d{3,}$', peek): break
                event_parts.append(peek)
                next_idx += 1
                if len(event_parts) >= 2: break
            market_norm = re.sub(r'(?<!\w)O(?=\.?\d)', '0', market)
            line_m = re.search(r'(?:OVER/UNDER|O/U|TOTAL)\s+(\d+\.?\d*)', market_norm, re.I)
            if not line_m: line_m = re.search(r'(\d+\.\d+)', market_norm)
            result['selections'].append({
                'pick': pick, 'odds': clean_odds(odds) if odds else '',
                'market': market, 'event': ' '.join(event_parts),
                'line': line_m.group(1) if line_m else ''
            })
            i = next_idx; continue

        # ── Pattern 1b: "Player-Over/Under XX.X odds" — "Dustin Wolf-Over 24.5 +108"
        player_ou_m = re.match(r'^(.+?)[-–]\s*(Over|Under)\s*([\d\.]+)\s*([+-]\d+)?$', text.strip(), re.I)
        if player_ou_m and i + 1 < len(texts):
            next_upper = texts[i+1].upper()
            if any(w in next_upper for w in FD_MARKET_WORDS):
                result['selections'].append({
                    'player': player_ou_m.group(1).strip(),
                    'pick': player_ou_m.group(2),
                    'line': player_ou_m.group(3),
                    'odds': player_ou_m.group(4) or '',
                    'market': texts[i+1].strip()
                })
                i += 2; continue

        # ── Pattern 2: Player prop — "Salvador Perez +300 +450" / "TO HIT A HOME RUN"
        # or "Robbie Ray Under +5.5" / "ROBBIE RAY-STRIKEOUTS"
        # or "Player Name" / "TO RECORD A HIT"
        player_odds_m = re.match(r'^([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)+)\.?\s+(Over|Under)?\s*([+-][\d\.]+)\s*([+-]\d+)?', text)
        if not player_odds_m:
            # "Salvador Perez +300 +450" or "Andrei Syechnikoy. +200"
            player_odds_m = re.match(r'^([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)+)\.?\s+([+-]\d+)\s*([+-]\d+)?$', text)
        if player_odds_m:
            player = player_odds_m.group(1).rstrip('.')
            # Check next line is a market
            if i + 1 < len(texts) and any(w in texts[i + 1].upper() for w in FD_MARKET_WORDS):
                market = texts[i + 1].strip()
                event = ''
                if i + 2 < len(texts) and re.search(r'(at|vs\.?|@)', texts[i + 2], re.I):
                    event = texts[i + 2].strip()
                # Extract odds and line from the player line
                nums = re.findall(r'[+-][\d\.]+', text)
                odds = nums[-1] if nums else ''
                line = nums[0] if len(nums) >= 2 else ''
                pick_dir = ''
                if 'under' in text.lower(): pick_dir = 'Under'
                elif 'over' in text.lower(): pick_dir = 'Over'
                result['selections'].append({
                    'player': player, 'odds': odds, 'line': line,
                    'pick': pick_dir, 'market': market, 'event': event
                })
                i += 3 if event else i + 2; continue

        # ── Pattern 3: Standalone player name followed by market ──
        # "Marcus Semien" / "TO RECORD A HIT" or "Pete Crow-Armstrong" / "TO RECORD A HIT"
        if (re.match(r'^[A-Z][a-zA-Z\'\-\.]+(\s[A-Za-z][a-zA-Z\'\-\.]+)+$', text.strip())
                and i + 1 < len(texts)
                and any(w in texts[i + 1].upper() for w in FD_MARKET_WORDS)):
            player = text.strip()
            market = texts[i + 1].strip()
            # Extract line from market if present
            line_m = re.search(r'(\d+\.?\d*)\+?', market)
            result['selections'].append({
                'player': player, 'market': market,
                'line': line_m.group(1) if line_m else ''
            })
            i += 2; continue

        # ── Pattern 4: SGP/Includes header — skip ──
        if re.match(r'^Includes:', text, re.I):
            i += 1; continue

        # ── Pattern 5: "Player X+ Stat" — "Shohei Ohtani 6+ Strikeouts"
        prop_line_m = re.match(r'^(.+?)\s+(\d+)\+\s*(.+)$', text)
        if prop_line_m:
            player = prop_line_m.group(1).strip()
            line = prop_line_m.group(2) + '+'
            stat = prop_line_m.group(3).strip()
            sel = {'player': player, 'line': line, 'market': stat}
            # Check next line for detailed market name
            if i + 1 < len(texts) and any(w in texts[i+1].upper() for w in FD_MARKET_WORDS):
                sel['market'] = texts[i+1].strip()
                i += 1
            result['selections'].append(sel)
            i += 1; continue

        # ── Pattern 6: "Over/Under X.X" or "Over/Under X.X odds" → market on next line
        ou_m = re.match(r'^(Over|Under)\s+([\d\.]+)\s*([+-]\d+)?$', text.strip(), re.I)
        if not ou_m:
            # "O Over 139.5 -140"
            ou_m = re.match(r'^O?\s*(Over|Under)\s*(\d+\.?\d*)\s*([+-]\d+)?$', text.strip(), re.I)
        if ou_m:
            pick = ou_m.group(1)
            line = ou_m.group(2)
            odds = ou_m.group(3) or ''
            market = ''
            if i + 1 < len(texts) and any(w in texts[i+1].upper() for w in FD_MARKET_WORDS + ['ALTERNATE', 'PUCKLINE', 'ALT']):
                market = texts[i+1].strip()
                i += 1
            result['selections'].append({
                'pick': pick, 'line': line, 'odds': odds, 'market': market
            })
            i += 1; continue

        # ── Pattern 7: "Team" → "MONEYLINE" (standalone team name)
        if (re.match(r'^[A-Z][a-zA-Z\s\.]+$', text.strip())
                and i + 1 < len(texts)
                and re.match(r'^MONEYLINE', texts[i+1].strip(), re.I)):
            result['selections'].append({
                'team': text.strip(), 'market': texts[i+1].strip()
            })
            i += 2; continue

        # ── Pattern 8: "Team+/-spread" or "Team +/-spread odds" → "SPREAD/PUCKLINE"
        # Handles "USA (W)-2.5" / "ALTERNATE PUCK LINE (INCL.OT/SO"
        spread_m = re.match(r'^(.+?)\s*([+-][\d\.]+)\s*([+-]\d+)?$', text.strip())
        if spread_m and i + 1 < len(texts) and re.search(r'(Spread|PUCK\s*LINE|Run Line|ALT.*LINE)', texts[i+1], re.I):
            result['selections'].append({
                'team': spread_m.group(1).strip(), 'line': spread_m.group(2),
                'odds': spread_m.group(3) or '', 'market': texts[i+1].strip()
            })
            i += 2; continue

        # ── Pattern 9: "Under/Over X.X odds" with market on same line — "Under 165.5 188"
        ou_inline_m = re.match(r'^(Under|Over)\s*([\d\.]+)\s+(-?\d+)$', text.strip(), re.I)
        if ou_inline_m and i + 1 < len(texts):
            market = texts[i+1].strip() if any(w in texts[i+1].upper() for w in ['TOTAL', 'ALTERNATE', 'ALT']) else ''
            result['selections'].append({
                'pick': ou_inline_m.group(1), 'line': ou_inline_m.group(2),
                'odds': ou_inline_m.group(3), 'market': market
            })
            if market: i += 1
            i += 1; continue

        # ── Pattern 10: "Team odds" — "Kansas -258" / "Moneyline"
        team_odds_m = re.match(r'^([A-Z][a-zA-Z\s\.]+?)\s+([+-]\d+)$', text.strip())
        if team_odds_m and i + 1 < len(texts) and re.match(r'^(Moneyline|Spread)', texts[i+1], re.I):
            result['selections'].append({
                'team': team_odds_m.group(1).strip(), 'odds': team_odds_m.group(2),
                'market': texts[i+1].strip()
            })
            i += 2; continue

        # ── PROFIT BOOST / promo lines — skip ──
        if re.match(r'^(PROFIT BOOST|If the bet|Reuse|Share|Follow|Cash out|BETID)', text, re.I):
            i += 1; continue

        # ── Matchup line (set current_event + attach to last selection) ──
        if re.search(r'\s(at|vs\.?|@)\s', text, re.I):
            current_event = text.strip()
            if result['selections'] and not result['selections'][-1].get('event'):
                result['selections'][-1]['event'] = current_event
            i += 1; continue

        # ── Matchup line alt: "TEAM1 TEAM2 date" like "UMASS MIAMI OH Mar 12..."
        if re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d', text, re.I):
            if not current_event:
                current_event = text.strip()
            if result['selections'] and not result['selections'][-1].get('event'):
                result['selections'][-1]['event'] = text.strip()
            i += 1; continue

        # ── Standalone player name (no odds) — "PavelMintyukov" or "Caroline Harvey"
        clean = text.replace(' ', '')
        if (re.match(r'^[A-Z][a-zA-Z\.]+$', clean) and len(clean) > 5
                and i + 1 < len(texts)
                and any(w in texts[i+1].upper() for w in FD_MARKET_WORDS)):
            # Split concatenated name: "PavelMintyukov" → "Pavel Mintyukov"
            player = re.sub(r'([a-z])([A-Z])', r'\1 \2', text.strip())
            market = texts[i+1].strip()
            result['selections'].append({'player': player, 'market': market})
            i += 2; continue

        i += 1

    # Backfill event on all selections that don't have one
    if current_event:
        for sel in result['selections']:
            if not sel.get('event'):
                sel['event'] = current_event

    # Detect sport
    all_text = ' '.join(l['text'] for l in lines)
    result['sport'] = detect_sport_from_text(all_text)

    return result

# ── PrizePicks Parser ─────────────────────────────────────────────────────────
PP_STAT_KEYWORDS = ['Points', 'Rebounds', 'Assists', 'Strikeouts', 'Hits',
                    'RBIs', 'Goals', 'Saves', 'Pts+Rebs+Asts', '3-Pointers',
                    'Rebs+Asts', 'Pts+Rebs', 'Pts + Rebs', 'Pts+Asts',
                    'Pts + Asts', 'PRA', 'Tackles',
                    'Fantasy Score', 'Fantasy Points', 'Fantasy',
                    'Passing', 'Rushing', 'Receiving',
                    'Shots on Goal', 'Blocked Shots']

def parse_prizepicks(lines: list[dict]) -> dict:
    groups = group_lines(lines, threshold=20)
    texts = [row_text(g) for g in groups]

    result = {'sportsbook': 'PrizePicks', 'bet_type': '', 'entry_fee': '',
              'payout': '', 'status': '', 'picks': []}

    # ── Detect format: settled results vs active picks ──
    all_lower = ' '.join(t.lower() for t in texts)
    is_settled = ('entry' in all_lower and 'paid' in all_lower) or 'paid' in all_lower
    if not is_settled:
        is_settled = any(re.match(r'^\d+\s*picks?\s+[\d\.]+x', t, re.I) for t in texts)
    if not is_settled:
        is_settled = bool(re.search(r'(won|lost|win|loss)\b', all_lower)) and 'pick' in all_lower

    if is_settled:
        # ── Settled format ──
        # Format A: "3 picks 6.5x" / "Won" / player / "Higher 87.99 Fantasy Points" / score
        # Format B: "$5 paid $15 2-Pick Power Play Win" / "Paul George 15.5" / stat / score
        current_pick = {}
        for text in texts:
            # Bet type: "3 picks 6.5x" or "2picks 3.08x"
            bt_m = re.match(r'^(\d+)\s*picks?\s+([\d\.]+)x', text, re.I)
            if bt_m:
                result['bet_type'] = f"{bt_m.group(1)} picks {bt_m.group(2)}x"
                continue

            # Combined line: "$5 paid $15 2-Pick Power Play Win"
            combined_m = re.search(r'\$([\d,\.]+)\s+paid\s+\$([\d,\.]+)\s+(.+)', text, re.I)
            if combined_m:
                result['entry_fee'] = combined_m.group(1).replace(',', '')
                result['payout'] = combined_m.group(2).replace(',', '')
                bt = combined_m.group(3).strip()
                # Extract status from end: "2-Pick Power Play Win"
                status_m = re.search(r'\b(Win|Won|Lost|Loss|Push|Void)\s*$', bt, re.I)
                if status_m:
                    result['status'] = status_m.group(1)
                    bt = bt[:status_m.start()].strip()
                if bt: result['bet_type'] = bt
                continue

            # Status
            if re.match(r'^(Won|Lost|Push|Void|Win|Loss)$', text.strip(), re.I):
                result['status'] = text.strip()
                continue

            # Entry + Paid amounts: "Entry Paid" header or "$5 $32.50"
            if re.match(r'^Entry\s+Paid', text, re.I):
                continue
            amounts = re.findall(r'\$([\d,\.]+)', text)
            if len(amounts) == 2 and not result['entry_fee']:
                result['entry_fee'] = amounts[0].replace(',', '')
                result['payout'] = amounts[1].replace(',', '')
                continue

            # Pick detail: "Higher 87.99 Fantasy Points" or "Lower4.5Strikeouts"
            pick_text = re.sub(r'^(Higher|Lower)(\d)', r'\1 \2', text, flags=re.I)
            pick_text = re.sub(r'(\d+\.5)(\d)', r'\1 \2', pick_text)
            pick_text = re.sub(r'(\d+)\s*([A-Z])', r'\1 \2', pick_text)
            pick_m = re.match(r'^(Higher|Lower)\s+([\d\.]+)\s+(.+)$', pick_text, re.I)
            if pick_m:
                if current_pick.get('player'):
                    current_pick['pick'] = pick_m.group(1)
                    current_pick['line'] = pick_m.group(2)
                    current_pick['stat'] = pick_m.group(3)
                    result['picks'].append(current_pick)
                    current_pick = {}
                continue

            # Player name + line on same row: "Paul George 15.5" or "Mitchell Robinson 15.5"
            pl_m = re.match(r'^([A-Z][a-zA-Z\'-]+(?:\s[A-Z][a-zA-Z\'-]+)+)\s+([\d\.]+)$', text.strip())
            if pl_m:
                # Save previous pick if it has enough data
                if current_pick.get('player') and (current_pick.get('stat') or current_pick.get('line')):
                    result['picks'].append(current_pick)
                current_pick = {'player': pl_m.group(1), 'line': pl_m.group(2)}
                continue

            # Standalone player name (multi-word, not UI text)
            clean = text.rstrip('.')
            skip = {'Won', 'Lost', 'Push', 'Share', 'Entry', 'Paid', 'Copy', 'Win', 'Loss'}
            if re.match(r'^[A-Z][a-zA-Z\'-]+(\s[A-Z][a-zA-Z\'-]+)+$', clean):
                if not any(w in clean for w in skip):
                    if current_pick.get('player') and current_pick.get('stat'):
                        result['picks'].append(current_pick)
                    current_pick = {'player': clean}
                    continue

            # Stat line: "PHIF.#8 Points" or "NYKC-F#23 Pts+Rebs" — extract stat keyword
            for kw in PP_STAT_KEYWORDS:
                if kw.lower() in text.lower():
                    if current_pick.get('player'):
                        current_pick['stat'] = kw
                    break

            # Matchup
            if re.search(r'\bvs\b', text, re.I):
                if current_pick:
                    current_pick['event'] = text
                continue

            # Standalone score number (e.g. "23", "28") — skip
            if re.match(r'^\d+\.?\d*$', text.strip()):
                if current_pick.get('player') and current_pick.get('line'):
                    current_pick['result'] = text.strip()
                continue

        if current_pick.get('player') and (current_pick.get('stat') or current_pick.get('line')):
            result['picks'].append(current_pick)
    else:
        # ── Active picks format ──
        # Two sub-formats:
        # A) Y-gap cards: "Isaiah Hartenstein C-F 7.5" / "NBA OKC @ UTA ... Points"
        # B) Sequential: "$5 to pay $70" / "5-Pick Flex Play" / "NBA BKN vs NYK" /
        #    "Jalen Brunson 41" / "NYKG#11 Fantasy Score" / ...

        current_pick = {}
        for text in texts:
            # ── Entry/payout: "$5 to pay $70" or "$5 to pay$50"
            tp_m = re.search(r'\$([\d,\.]+)\s*to\s*pay\s*\$?([\d,\.]+)', text, re.I)
            if tp_m:
                result['entry_fee'] = tp_m.group(1).replace(',', '')
                result['payout'] = tp_m.group(2).replace(',', '')
                continue

            # Bet type: "5-Pick Flex Play"
            bt_m = re.search(r'(\d+)[- ]?Pick\s*(Power|Flex|Standard)\s*Play', text, re.I)
            if bt_m:
                result['bet_type'] = f"{bt_m.group(1)}-Pick {bt_m.group(2)} Play"
                continue

            # Matchup line: "NBA BKN vs NYK 7:40pm" — start of a new pick context
            sport_m = re.match(r'^(NBA|NFL|MLB|NHL|NCAAB|NCAAF|MMA|PGA|UFC)\s+(.+)$', text)
            if sport_m:
                # Save previous pick if complete
                if current_pick.get('player'):
                    result['picks'].append(current_pick)
                    current_pick = {}
                remainder = sport_m.group(2)
                matchup_m = re.search(r'([A-Z]{2,4})\s*(@|vs\.?)\s*([A-Z]{2,4})', remainder)
                if matchup_m:
                    current_pick['event'] = f"{matchup_m.group(1)} {matchup_m.group(2)} {matchup_m.group(3)}"
                current_pick['sport'] = sport_m.group(1)
                for kw in PP_STAT_KEYWORDS:
                    if kw.lower() in remainder.lower():
                        current_pick['stat'] = kw
                        break
                continue

            # Player+position+line merged: "Isaiah Hartenstein C-F 7.5"
            pp_m = re.match(r'^(.+?)\s+([A-Z]{1,3}-?[A-Z]?)\s+(?:[↑↓▲▼]\s*)?(\d+\.?\d*)$', text)
            if pp_m:
                if current_pick.get('player'):
                    result['picks'].append(current_pick)
                current_pick = {**current_pick, 'player': pp_m.group(1),
                               'position': pp_m.group(2), 'line': pp_m.group(3), 'pick': 'More'}
                continue

            # Player + line (no position): "Jalen Brunson 41" or "Paul George 15.5"
            pl_m = re.match(r'^([A-Z][a-zA-Z\'-]+(?:\s[A-Z][a-zA-Z\'-]+)+)\s+([\d\.]+)$', text.strip())
            if pl_m:
                if current_pick.get('player') and current_pick.get('player') != pl_m.group(1):
                    result['picks'].append(current_pick)
                    event = current_pick.get('event', '')
                    sport = current_pick.get('sport', '')
                    current_pick = {'event': event, 'sport': sport} if event else {}
                current_pick['player'] = pl_m.group(1)
                current_pick['line'] = pl_m.group(2)
                continue

            # Stat line: "NYKG#11 Fantasy Score" or "Points" or "Rebs+Asts"
            for kw in PP_STAT_KEYWORDS:
                if kw.lower() in text.lower():
                    current_pick['stat'] = kw
                    break

            # Arrow + line
            if re.search(r'[↑↓▲▼]', text):
                line_m = re.search(r'(\d+\.?\d+)', text)
                if line_m:
                    current_pick['line'] = line_m.group(1)
                    current_pick['pick'] = 'More' if '↑' in text else 'Less'
                continue

            # "Starts in Xh Ym" — skip
            if 'starts in' in text.lower():
                continue

        if current_pick.get('player'):
            if 'pick' not in current_pick: current_pick['pick'] = 'More'
            result['picks'].append(current_pick)

    return result

# ── Underdog Parser ───────────────────────────────────────────────────────────
def parse_underdog(lines: list[dict]) -> dict:
    groups = group_lines(lines, threshold=15)
    texts = [row_text(g) for g in groups]

    result = {'sportsbook': 'Underdog', 'bet_type': '', 'multiplier': '',
              'entry_fee': '', 'payout': '', 'picks': []}

    # Collect player names and pick details separately, then zip them
    player_names = []
    pick_details = []

    for text in texts:
        # Bet type + multiplier: "Champions: 3 Picks 6.3x"
        bt_m = re.match(r'^(Champions|Rivals|Legends|Flex)\s*:?\s*(\d+)\s*Picks?\s*([\d\.]+)x?', text, re.I)
        if bt_m:
            result['bet_type'] = bt_m.group(1)
            result['multiplier'] = bt_m.group(3) + 'x'
            continue

        # Entry / payout
        if re.search(r'\$\d', text):
            amounts = re.findall(r'\$?([\d,\.]+)', text)
            if amounts and not result['entry_fee']:
                result['entry_fee'] = amounts[0]
            continue

        # Pick detail: "Higher 2.5 3-Pointers Made" or "Higher2.53-Pointers Made" (v2.9.1)
        # Normalize: insert space after Higher/Lower and between line number and stat
        pick_text = re.sub(r'^(Higher|Lower)(\d)', r'\1 \2', text, flags=re.I)
        # Split "2.53-Pointers" → "2.5 3-Pointers" (line ends at .5 or whole number)
        pick_text = re.sub(r'(\d+\.5)(\d)', r'\1 \2', pick_text)
        pick_text = re.sub(r'(\d+)\s*([A-Z])', r'\1 \2', pick_text)
        pick_m = re.match(r'^(Higher|Lower)\s+([\d\.]+)\s+(.+)$', pick_text, re.I)
        if pick_m:
            pick_details.append({
                'pick': pick_m.group(1),
                'line': pick_m.group(2),
                'stat': pick_m.group(3)
            })
            continue

        # Player name: multi-word name (handles McName, O'Name, hyphenated, lowercase particles)
        skip_words = {'Reuse', 'Submit', 'Entry', 'Review', 'Share', 'Cancel',
                      'Higher', 'Lower', 'Picks'}
        clean = text.rstrip('.')
        # Match "James van Riemsdyk", "Olivier-Maxence Prosper", "TaijuanWalker"
        if re.match(r'^[A-Z][a-zA-Z\'\-]+(\s+[a-zA-Z][a-zA-Z\'\-]+)+$', clean):
            if not any(w in clean for w in skip_words):
                # Split concatenated names: "TaijuanWalker" → "Taijuan Walker"
                clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
                player_names.append(clean)
                continue

    # Pair players with picks in order
    for i, detail in enumerate(pick_details):
        pick = dict(detail)
        if i < len(player_names):
            pick['player'] = player_names[i]
        result['picks'].append(pick)

    return result

# ── BetMGM Parser ─────────────────────────────────────────────────────────────
def parse_betmgm(lines: list[dict]) -> dict:
    groups = group_lines(lines, threshold=20)
    texts = [row_text(g) for g in groups]

    result = {'sportsbook': 'BetMGM', 'bet_type': '', 'total_odds': '',
              'wager': '', 'payout': '', 'status': '', 'selections': []}

    i = 0
    while i < len(texts):
        text = texts[i]

        # Status + branding: "WON BETMGM" or "LOST BETMGM"
        status_m = re.match(r'^(WON|LOST|OPEN|SETTLED|CASHED OUT|VOID)\b', text, re.I)
        if status_m:
            result['status'] = status_m.group(1)
            i += 1
            continue

        # Bet type + legs + odds: "SGP 2 Legs +140" or "Parlay 3 Legs +450"
        bt_m = re.match(r'^(SGP|Parlay|Straight|Single)\s+(\d+)\s*Legs?\s*([+-]\d+)', text, re.I)
        if bt_m:
            result['bet_type'] = f"{bt_m.group(1)} {bt_m.group(2)} Legs"
            result['total_odds'] = bt_m.group(3)
            i += 1
            continue

        # Single bet odds: just "+140" or "-110" as standalone
        if not result['total_odds'] and re.match(r'^[+-]\d+$', text.strip()):
            result['total_odds'] = text.strip()
            i += 1
            continue

        # Stake + payout: "Stake: $50.00 Payout: $120.00" or "Stake: $50.00"
        stake_m = re.search(r'Stake[:\s]+\$?([\d,\.]+)', text, re.I)
        pay_m = re.search(r'Payout[:\s]+\$?([\d,\.]+)', text, re.I)
        if stake_m:
            result['wager'] = stake_m.group(1).replace(',', '')
        if pay_m:
            result['payout'] = pay_m.group(1).replace(',', '')
        if stake_m or pay_m:
            i += 1
            continue

        # Skip summary line (before "Hide legs") — it's a condensed version of the legs
        # OCR may read "|" as ")" or merge text, so also skip if next line is "Hide legs"
        if i + 1 < len(texts) and re.match(r'^Hide\s*legs', texts[i+1], re.I):
            i += 1; continue
        # Skip "Hide legs" itself
        if re.match(r'^Hide\s*legs', text, re.I):
            i += 1; continue

        # Skip combined summary line with "|" separator
        if '|' in text and re.search(r'(Result|Total|Spread|Anytime)', text, re.I):
            i += 1; continue

        # ── "Anytime point - Player (Match)" — extract player from market
        anytime_m = re.match(r'^(Anytime\s+\w+)\s*[-–]\s*(.+?)(?:\s*\((?:Match|Game)\))?$', text.strip(), re.I)
        if anytime_m:
            result['selections'].append({
                'player': anytime_m.group(2).strip(),
                'market': anytime_m.group(1).strip(),
            })
            i += 1; continue

        # ── "Result - Team (Game)" — extract team from market
        result_m = re.match(r'^(Result|Spread)\s*[-–]\s*(.+?)(?:\s+[+-][\d\.]+)?\s*\((?:Game|Match)\)?$', text.strip(), re.I)
        if result_m:
            market_type = result_m.group(1).strip()
            team_or_detail = result_m.group(2).strip()
            sel = {'market': market_type}
            # Extract spread line if present
            spread_m = re.search(r'(.+?)\s+([+-][\d\.]+)', team_or_detail)
            if spread_m:
                sel['team'] = spread_m.group(1).strip()
                sel['line'] = spread_m.group(2)
            else:
                sel['team'] = team_or_detail
            result['selections'].append(sel)
            i += 1; continue

        # ── "Total goals - Under 7.5 (Game)" — extract pick + line
        total_m = re.match(r'^(Total\s+\w+)\s*[-–]\s*(Over|Under)\s+([\d\.]+)\s*\((?:Game|Match)\)?$', text.strip(), re.I)
        if total_m:
            result['selections'].append({
                'market': total_m.group(1).strip(),
                'pick': total_m.group(2),
                'line': total_m.group(3),
            })
            i += 1; continue

        # ── Other market descriptions (fallback)
        if re.search(r'(Result|Total|Over|Under|Moneyline|Spread|First|Last|Anytime)', text, re.I):
            if 'hide' not in text.lower():
                result['selections'].append({'market': text})
                i += 1; continue

        # Matchup: "Panthers at Devils" or "Lakers vs Celtics"
        if re.search(r'\s(at|vs\.?|@)\s', text, re.I) and not re.search(r'(Stake|Payout)', text, re.I):
            if result['selections']:
                result['selections'][-1]['event'] = text
            i += 1; continue

        # Date/time: "3/3/26 • 7:05 PM" or "Today 7:08 PM"
        if re.search(r'(\d+/\d+/\d+|Today|Tomorrow|Starting)', text, re.I):
            if result['selections']:
                result['selections'][-1]['date'] = text
            i += 1; continue

        i += 1

    # Detect sport
    all_text = ' '.join(l['text'] for l in lines)
    result['sport'] = detect_sport_from_text(all_text)

    return result

# ── Fanatics Parser ───────────────────────────────────────────────────────────
FAN_MARKET_WORDS = ['Home Run', 'Moneyline', 'Spread', 'Total', 'Alt', 'Anytime',
                    'Goalscorer', 'Points', 'Rebounds', 'Assists', 'Strikeouts',
                    'Touchdown', 'Winner', 'Double Double', 'First',
                    'Blocked Shots', 'Shots on Goal', 'Race To', 'Neither',
                    'Triple Double', 'Saves', 'Goals']

def parse_fanatics(lines: list[dict]) -> dict:
    groups = group_lines(lines, threshold=20)
    texts = [row_text(g) for g in groups]

    result = {'sportsbook': 'Fanatics', 'bet_type': '', 'total_odds': '',
              'wager': '', 'payout': '', 'selections': []}

    # Fanatics formats:
    # 1) Long Ball: player+line+odds / market / matchup / wager+payout
    # 2) SGP Stack: "4 Leg SGP Stack +8291" / repeating player / market / matchup
    # 3) Parlay grid: "6 Leg Parlay +1175" / grid rows "1+ 8+" / "Player-Stat Player-Stat"
    #    / matchup lines

    i = 0
    while i < len(texts):
        text = texts[i]

        # ── Skip branding/promo ──
        if re.match(r'^(LONG BALL|DAILY|FANCASH|JACKPOT|F\s+Fanatics|RG\s+MUST)', text, re.I):
            i += 1; continue
        if re.match(r'^(Share|Reuse|Cash out)', text, re.I):
            i += 1; continue

        # ── Bet type + total odds: "4 Leg SGP Stack +8291" or "6 Leg Parlay +1175"
        bt_m = re.match(r'^(\d+)\s+Leg\s+(.+?)\s+([+-]\d+)', text, re.I)
        if bt_m and not result['bet_type']:
            result['bet_type'] = f"{bt_m.group(1)} Leg {bt_m.group(2).strip()}"
            result['total_odds'] = bt_m.group(3)
            i += 1; continue

        # ── Market name that doubles as bet type ──
        if re.search(r'(Long\s*Ball|Same Game)', text, re.I) and not result['bet_type']:
            result['bet_type'] = text.strip()
            i += 1; continue

        # ── Wager / payout ──
        if re.match(r'^Wager\s+(Payout|FCash)', text, re.I):
            if i + 1 < len(texts):
                amounts = re.findall(r'\$?([\d,\.]+)', texts[i + 1])
                if len(amounts) >= 2:
                    result['wager'] = amounts[0].replace(',', '')
                    result['payout'] = amounts[1].replace(',', '')
            i += 1; continue
        if re.match(r'^FCash$', text, re.I):
            # "FCash" on its own followed by "$1.95"
            if i + 1 < len(texts):
                amt = re.search(r'\$?([\d,\.]+)', texts[i + 1])
                # Don't overwrite wager/payout
            i += 1; continue

        wager_m = re.search(r'(?:Wager|Stake)[:\s]+\$?([\d,\.]+)', text, re.I)
        pay_m = re.search(r'(?:Payout|Paid)[:\s]+\$?([\d,\.]+)', text, re.I)
        if wager_m: result['wager'] = wager_m.group(1).replace(',', '')
        if pay_m: result['payout'] = pay_m.group(1).replace(',', '')

        # ── Format 1: Player + odds on one line ──
        # "Kyle Schwarber1+ +185" or "P Kyle Schwarber 1 1+ +185"
        odds_m = re.search(r'([+-]\d{3,})', text)
        player_m = re.search(r'(?:^[A-Z]?\s+)?([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)+)', text)
        if odds_m and player_m and not re.search(r'(Fanatics|Sportsbook|Leg\s)', text, re.I):
            sel = {'player': player_m.group(1), 'odds': odds_m.group(1)}
            line_m = re.search(r'(\d+)\+', text)
            if line_m: sel['line'] = line_m.group(1) + '+'
            result['selections'].append(sel)
            if not result['total_odds']: result['total_odds'] = odds_m.group(1)
            i += 1; continue

        # ── Format 0: Standalone pick word / market — "Neither" / "Race To 6 Goals" / matchup
        if re.match(r'^(Neither|Yes|No|Over|Under)$', text.strip(), re.I):
            if i + 1 < len(texts) and any(w.lower() in texts[i+1].lower() for w in FAN_MARKET_WORDS):
                sel = {'pick': text.strip(), 'market': texts[i+1].strip()}
                consumed = 2
                if i + 2 < len(texts) and re.search(r'\s(at|vs\.?|@)\s', texts[i+2], re.I):
                    sel['event'] = texts[i+2].strip()
                    consumed = 3
                result['selections'].append(sel)
                i += consumed; continue

        # ── Format 2: SGP Stack — standalone player name ──
        # "Tage Thompson" or "gor Chernyshov" (truncated OCR) followed by market
        if re.match(r'^[A-Za-z][a-z]+(\s[A-Z][a-zA-Z\'\-]+)+$', text.strip()):
            if i + 1 < len(texts) and any(w.lower() in texts[i + 1].lower() for w in FAN_MARKET_WORDS):
                sel = {'player': text.strip(), 'market': texts[i + 1].strip()}
                if i + 2 < len(texts) and re.search(r'\s(at|vs\.?|@)\s', texts[i + 2], re.I):
                    sel['event'] = texts[i + 2].strip()
                    i += 3
                else:
                    i += 2
                result['selections'].append(sel)
                continue

        # ── Format 3a: Individual line+stat: "3+ -180" / "Player-Stat" / event
        line_odds_m = re.match(r'^(\d+)\+\s*([+-]\d+)?$', text.strip())
        if line_odds_m and i + 1 < len(texts):
            next_text = texts[i + 1]
            # Check for "Player-Stat" on next line
            ps_m = re.match(r'^(.+?)\s*[-–]\s*(.+)$', next_text)
            if ps_m:
                sel = {'player': ps_m.group(1).strip(), 'market': ps_m.group(2).strip(),
                       'line': line_odds_m.group(1) + '+'}
                if line_odds_m.group(2): sel['odds'] = line_odds_m.group(2)
                # Look for event on following lines
                consumed = 2
                if i + 2 < len(texts) and re.search(r'(NHL|NBA|MLB|NFL|at|vs\.?|@)', texts[i+2], re.I):
                    event_parts = [texts[i+2].strip()]
                    consumed = 3
                    if i + 3 < len(texts) and re.match(r'^\d+:\d+', texts[i+3]):
                        event_parts.append(texts[i+3].strip())
                        consumed = 4
                    sel['event'] = ' '.join(event_parts)
                result['selections'].append(sel)
                i += consumed; continue

        # ── Format 3b: Paired player grid — "Brandon Lowe1+ Mike Busch 1+" / "ALT Hits ALT Hits"
        paired_m = re.findall(r'([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)*)\s*(\d+)\+', text)
        if len(paired_m) >= 2 and i + 1 < len(texts):
            market_line = texts[i + 1]
            # Split market line: "ALT Hits ALT Hits" → ["ALT Hits", "ALT Hits"]
            market_parts = re.split(r'\s{2,}', market_line.strip())
            if len(market_parts) < len(paired_m):
                market_parts = [market_line.strip()] * len(paired_m)
            # Look for matchup lines below
            event_parts = []
            consumed = 2
            for j in range(i + 2, min(i + 4, len(texts))):
                if re.search(r'\s(at|vs\.?|@)\s', texts[j], re.I):
                    event_parts.append(texts[j].strip())
                    consumed += 1
                else:
                    break
            for idx, (player, line) in enumerate(paired_m):
                sel = {'player': player.strip(), 'line': line + '+'}
                if idx < len(market_parts):
                    sel['market'] = market_parts[idx].strip()
                if event_parts:
                    sel['event'] = event_parts[min(idx, len(event_parts)-1)]
                result['selections'].append(sel)
            i += consumed; continue

        # ── Format 3c: Soccer grid — "Over 1.5 Arsenal" / "Total Goals Money Line 3-Way"
        soccer_grid_m = re.match(r'^(Over|Under|Yes|No)\s+([\d\.]+)?\s*(.+)$', text.strip(), re.I)
        if soccer_grid_m and i + 1 < len(texts) and any(w.lower() in texts[i+1].lower() for w in ['Total', 'Both Teams', 'Money Line', 'Moneyline']):
            pick1 = soccer_grid_m.group(1)
            line1 = soccer_grid_m.group(2) or ''
            team_or_extra = soccer_grid_m.group(3).strip()
            market_line = texts[i + 1].strip()
            # May have two selections merged: "Total Goals Money Line 3-Way"
            # Followed by matchup: "Sassuolo vs Atalanta Arsenal vs Chelsea"
            sel1 = {'pick': pick1, 'line': line1, 'market': market_line.split('  ')[0] if '  ' in market_line else market_line}
            result['selections'].append(sel1)
            if team_or_extra and not re.match(r'^[\d\.]+$', team_or_extra):
                # Second selection (team/pick): "Arsenal" / "Money Line 3-Way"
                market2_parts = market_line.split('  ')
                sel2 = {'team': team_or_extra, 'market': market2_parts[-1].strip() if len(market2_parts) > 1 else 'Moneyline'}
                result['selections'].append(sel2)
            i += 2; continue

        # ── Format 3: Parlay grid — "1+ 8+" line / "Player-Stat Player-Stat" ──
        # Lines like "1+ 8+" or "8+ 1+" are pick line pairs
        if re.match(r'^[\d\+\s]+$', text.strip()) and '+' in text:
            grid_lines = re.findall(r'(\d+)\+', text)
            # Next line should be player-stat pairs: "Timo Meier-Points Dominick Barlow-Points"
            if i + 1 < len(texts):
                stat_line = texts[i + 1]
                # Split by finding player-stat patterns
                pairs = re.findall(r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s*[-–]\s*(\w+)', stat_line)
                matchup_parts = []
                if i + 2 < len(texts):
                    # Matchup may span 2 lines
                    for j in range(i + 2, min(i + 4, len(texts))):
                        if re.search(r'\s(at|vs\.?|@)\s', texts[j], re.I):
                            matchup_parts.append(texts[j].strip())
                        else:
                            break
                event = ' '.join(matchup_parts)
                for idx, (player, stat) in enumerate(pairs):
                    sel = {'player': player, 'market': stat}
                    if idx < len(grid_lines):
                        sel['line'] = grid_lines[idx] + '+'
                    if event:
                        sel['event'] = event
                    result['selections'].append(sel)
                i += 2 + len(matchup_parts)
                continue

        # ── Market description (attach to last selection) ──
        if any(w.lower() in text.lower() for w in FAN_MARKET_WORDS):
            if result['selections'] and 'market' not in result['selections'][-1]:
                result['selections'][-1]['market'] = text.strip()
            i += 1; continue

        # ── Matchup (attach to last selection) ──
        if re.search(r'\s(at|vs\.?|@)\s', text, re.I):
            if result['selections'] and 'event' not in result['selections'][-1]:
                result['selections'][-1]['event'] = text.strip()
            i += 1; continue

        # ── BetID ──
        if re.match(r'^Bet\s*ID', text, re.I):
            i += 1; continue

        i += 1

    return result

# ── Onyx Odds Parser ─────────────────────────────────────────────────────────
def parse_onyx(lines: list[dict]) -> dict:
    groups = group_lines(lines, threshold=20)
    texts = [row_text(g) for g in groups]

    result = {'sportsbook': 'Onyx', 'bet_type': '', 'total_odds': '',
              'wager': '', 'payout': '', 'status': '', 'selections': []}

    # Onyx has 3 formats:
    # 1) Pre-game picks: seed+team+spread+odds / SPREAD / matchup / time
    # 2) Settled single: "Pick Placed +800" / "Market: Yes" / matchup / wager+payout
    # 3) Settled with result: "Mitchell Robinson Over 0.5 +510 WON" / market / wager+paid

    i = 0
    current_sel = None
    while i < len(texts):
        text = texts[i]

        # Skip UI elements
        if re.match(r'^(Hide|Show|selections?|Submit|Place\s+Bet|ODDS\s+ONYX)', text, re.I):
            i += 1; continue

        # ── Status ──
        if re.match(r'^(Won|Lost|Open|Settled|Void)$', text.strip(), re.I):
            result['status'] = text.strip()
            i += 1; continue

        # ── Wager/payout — multiple formats ──
        # "Wager: 25.00 Payout: 225.00" or "Wager: Paid:" + "2.50 15.25"
        wager_m = re.search(r'(?:Wager|Stake)[:\s]+\$?([\d,\.]+)', text, re.I)
        pay_m = re.search(r'(?:Payout|Paid|To Win)[:\s]+\$?([\d,\.]+)', text, re.I)
        if wager_m: result['wager'] = wager_m.group(1).replace(',', '')
        if pay_m: result['payout'] = pay_m.group(1).replace(',', '')
        if re.match(r'^(Wager|Stake)[:\s]+(Paid|Payout)', text, re.I):
            # Labels only — amounts on next line
            if i + 1 < len(texts):
                amounts = re.findall(r'([\d,\.]+)', texts[i + 1])
                if len(amounts) >= 2:
                    result['wager'] = amounts[0].replace(',', '')
                    result['payout'] = amounts[1].replace(',', '')
                    i += 2; continue
        if wager_m or pay_m:
            i += 1; continue

        # ── Format 2: "Pick Placed +800" ──
        pp_m = re.match(r'^Pick\s+Placed\s+([+-]\d+)', text, re.I)
        if pp_m:
            result['total_odds'] = pp_m.group(1)
            result['bet_type'] = 'Single'
            i += 1; continue

        # ── Format 3: "Mitchell Robinson Over 0.5 +510 WON" or "Jamal Shead Over17.5 -111 WON" ──
        prop_m = re.match(r'^(.+?)\s+(Over|Under)\s*([\d\.]+)\s+([+-]\d+)\s*(WON|LOST|PUSH)?', text, re.I)
        if prop_m:
            current_sel = {
                'player': prop_m.group(1).strip(),
                'pick': prop_m.group(2),
                'line': prop_m.group(3),
                'odds': prop_m.group(4),
            }
            if prop_m.group(5):
                result['status'] = prop_m.group(5)
            result['selections'].append(current_sel)
            result['total_odds'] = prop_m.group(4)
            i += 1; continue

        # ── Format 4 (Pick Receipt): "Player Points: Dillon Brooks Over 13.5"
        # or "Player Triple Double: LeBron James Over 0.5"
        # or "Point Spread: Los Angeles Lakers -2.5"
        receipt_m = re.match(r'^(Player\s+\w[\w\s]*?|Point\s+Spread):\s*(.+?)(?:\s+(Over|Under)\s+([\d\.]+))?(?:\s+([+-][\d\.]+))?$', text, re.I)
        if receipt_m:
            market = receipt_m.group(1).strip()
            name = receipt_m.group(2).strip()
            pick = receipt_m.group(3) or ''
            line = receipt_m.group(4) or ''
            spread_line = receipt_m.group(5) or ''
            current_sel = {'market': market}
            if 'Spread' in market:
                current_sel['team'] = name
                current_sel['line'] = spread_line or line
            else:
                current_sel['player'] = name
                if pick: current_sel['pick'] = pick
                if line: current_sel['line'] = line
            result['selections'].append(current_sel)
            i += 1; continue

        # ── Format 2: Market line — "Will There Be Overtime: Yes" ──
        market_m = re.match(r'^(.+?):\s*(Yes|No|Over|Under)$', text, re.I)
        if market_m:
            current_sel = {
                'market': market_m.group(1).strip(),
                'pick': market_m.group(2),
            }
            result['selections'].append(current_sel)
            i += 1; continue

        # ── Format 3: Market description ──
        if re.match(r'^(Player|Team)', text, re.I) or re.search(r'(Double|Triple|Goals|Assists)', text, re.I):
            if current_sel:
                current_sel['market'] = text.strip()
            i += 1; continue

        # ── Total odds standalone ──
        if re.match(r'^[+-]\d{3,}$', text.strip()):
            if not result['total_odds']:
                result['total_odds'] = text.strip()
            i += 1; continue

        # ── Format 1: Selection with spread+odds ──
        sel_m = re.match(
            r'^[A-Za-z]?\s*(\d+)\s+([A-Za-z][A-Za-z\s]+?)\s+([+-][\d\.]+)\s+([+-]\d+)$', text.strip()
        )
        if sel_m:
            current_sel = {
                'team': sel_m.group(2).strip(),
                'line': sel_m.group(3),
                'odds': sel_m.group(4),
            }
            result['selections'].append(current_sel)
            if not result['bet_type']: result['bet_type'] = 'Parlay'
            i += 1; continue

        # ── Format 1: Selection moneyline ──
        ml_m = re.match(
            r'^[A-Za-z]?\s*(\d+)\s+([A-Za-z][A-Za-z\s]+?)\s+([+-]\d+)$', text.strip()
        )
        if ml_m:
            current_sel = {
                'team': ml_m.group(2).strip(),
                'odds': ml_m.group(3),
            }
            result['selections'].append(current_sel)
            if not result['bet_type']: result['bet_type'] = 'Parlay'
            i += 1; continue

        # ── Market type ──
        if re.match(r'^(SPREAD|TO\s*WIN|MONEYLINE|TOTAL)', text.strip(), re.I):
            if current_sel:
                current_sel['market'] = text.strip()
            i += 1; continue

        # ── Matchup ──
        if re.search(r'\s(@|vs\.?\s)', text, re.I):
            if current_sel:
                current_sel['event'] = text.strip()
            elif result['selections']:
                result['selections'][-1]['event'] = text.strip()
            i += 1; continue

        # ── Score line: "CHI Bulls NYK Knicks" + "96 VS 136" ──
        if re.search(r'\d+\s+VS\s+\d+', text, re.I):
            if current_sel:
                current_sel['score'] = text.strip()
            i += 1; continue

        # ── Date/time ──
        if re.search(r'(Today|Tomorrow|Final|\d+:\d+\s*(am|pm|AM|PM)|\w+\s+\d+,\s*\d{4})', text, re.I):
            if current_sel:
                current_sel['date'] = re.sub(r'^[a-z]\d?\s+', '', text).strip()
            i += 1; continue

        i += 1

    return result

# ── Hard Rock Bet Parser ─────────────────────────────────────────────────────
HR_MARKET_WORDS = ['HOME RUNS', 'ANYTIME GOALSCORER', 'SHOTS ON TARGET',
                   'TOTAL GOALS', '1ST HALF TOTAL GOALS', 'TO WIN',
                   'STRIKEOUTS', 'HITS', 'MONEYLINE', 'SPREAD']

def parse_hardrock(lines: list[dict]) -> dict:
    groups = group_lines(lines, threshold=20)
    texts = [row_text(g) for g in groups]

    result = {'sportsbook': 'Hard Rock', 'bet_type': '', 'total_odds': '',
              'wager': '', 'payout': '', 'status': '', 'selections': []}

    i = 0
    current_event = ''
    while i < len(texts):
        text = texts[i]

        # Skip branding
        if re.match(r'^(Hard Rock|BET|SPORTSBOOK|My Bets|Upcoming|Live|Finished|Home|Discover|Rewards|Account)', text, re.I):
            i += 1; continue
        if re.match(r'^(Share|Hide selections|Early cash out|ID:)', text, re.I):
            i += 1; continue
        if re.search(r'Profit Boost applied', text, re.I):
            i += 1; continue

        # ── Bet type + odds: "SGPMAX 4-Bet Parlay +1197" or "PARLAY 6-Bet Parlay +451"
        bt_m = re.match(r'^(?:SGPMAX|PARLAY)\s+(\d+)-?\s*Bet\s+Parlay\s*([+-]\d+)?', text, re.I)
        if bt_m:
            result['bet_type'] = f"{bt_m.group(1)}-Bet Parlay"
            if bt_m.group(2):
                result['total_odds'] = bt_m.group(2)
            i += 1; continue

        # Standalone odds
        if re.match(r'^[+-]\d{3,}$', text.strip()) and not result['total_odds']:
            result['total_odds'] = text.strip()
            i += 1; continue

        # ── Wager / Payout ──
        wager_m = re.search(r'(?:Wager|Stake)[:\s]+\$?([\d,\.]+)', text, re.I)
        pay_m = re.search(r'(?:Payout|To Win|To Pay)[:\s]+\$?([\d,\.]+)', text, re.I)
        if wager_m: result['wager'] = wager_m.group(1).replace(',', '')
        if pay_m: result['payout'] = pay_m.group(1).replace(',', '')
        if wager_m or pay_m:
            i += 1; continue

        # "$X ... $Y" wager/payout on same line
        if re.match(r'^\$[\d,\.]+\s+\$[\d,\.]+', text):
            amounts = re.findall(r'\$([\d,\.]+)', text)
            if len(amounts) >= 2:
                result['wager'] = amounts[0].replace(',', '')
                result['payout'] = amounts[1].replace(',', '')
            i += 1; continue

        # ── SGP sub-header: "SGP Girona vs FC Barcelona +750"
        sgp_m = re.match(r'^SGP\s+(.+?)\s+([+-]\d+)', text, re.I)
        if sgp_m:
            current_event = sgp_m.group(1).strip()
            i += 1; continue

        # ── Selection: "Over 0.5" + "PLAYER - MARKET" or just market text
        ou_m = re.match(r'^(Over|Under)\s+([\d\.]+)\s*([+-]\d+)?$', text.strip(), re.I)
        if ou_m:
            pick = ou_m.group(1)
            line = ou_m.group(2)
            odds = ou_m.group(3) or ''
            market = ''
            if i + 1 < len(texts):
                market = texts[i + 1].strip()
                i += 1
            # Extract player from market: "RAPHINHA - SHOTS ON TARGET"
            player = ''
            pm = re.match(r'^(.+?)\s*[-–]\s*(.+)$', market)
            if pm:
                player = pm.group(1).strip()
                market = pm.group(2).strip()
            sel = {'pick': pick, 'line': line, 'market': market, 'event': current_event}
            if odds: sel['odds'] = odds
            if player: sel['player'] = player
            result['selections'].append(sel)
            i += 1; continue

        # ── Selection: "O1 Texas [W] -525" / "TOWIN" or "O Blue Jays -275" / "TOWIN"
        # or "Ferran Torres" / "FERRAN TORRES ANYTIME GOALSCORER"
        team_odds_m = re.match(r'^O?\s*\d*\s*(.+?)\s+([+-]\d+)$', text.strip())
        if team_odds_m and i + 1 < len(texts) and re.match(r'^TO\s*WIN', texts[i+1], re.I):
            team = re.sub(r'^\d+\s*', '', team_odds_m.group(1)).strip()
            odds = team_odds_m.group(2)
            event = ''
            if i + 2 < len(texts) and re.search(r'(@|vs\.?)', texts[i+2], re.I):
                event = texts[i + 2].strip()
            sel = {'team': team, 'odds': odds, 'market': 'Moneyline', 'event': event}
            result['selections'].append(sel)
            i += 3 if event else i + 2; continue

        # ── Selection: standalone team name / "TOWIN"
        if (i + 1 < len(texts) and re.match(r'^TO\s*WIN', texts[i+1], re.I)
                and re.match(r'^O?\s*(.+)$', text.strip())):
            team = re.sub(r'^O\s*', '', text).strip()
            event = ''
            if i + 2 < len(texts) and re.search(r'(@|vs\.?)', texts[i+2], re.I):
                event = texts[i + 2].strip()
            sel = {'team': team, 'market': 'Moneyline', 'event': event}
            result['selections'].append(sel)
            i += 3 if event else i + 2; continue

        # ── Selection: player name + market below
        if (re.match(r'^[A-Z0-9]', text) and i + 1 < len(texts)
                and any(w in texts[i+1].upper() for w in HR_MARKET_WORDS)):
            player_or_team = text.strip()
            market_text = texts[i + 1].strip()
            odds = ''
            # Check for odds on the market line or player line
            odds_m = re.search(r'([+-]\d+)', text)
            if odds_m: odds = odds_m.group(1)
            # Extract player from market: "MICHAEL BUSCH - HOME RUNS"
            pm = re.match(r'^(.+?)\s*[-–]\s*(.+)$', market_text)
            sel = {'event': current_event}
            if pm:
                sel['player'] = pm.group(1).strip()
                sel['market'] = pm.group(2).strip()
            else:
                sel['team'] = player_or_team
                sel['market'] = market_text
            if odds: sel['odds'] = odds
            result['selections'].append(sel)
            i += 2; continue

        # ── Matchup: "Cubs @ Guardians" or "Padres @ Red Sox"
        if re.search(r'\s(@|vs\.?)\s', text, re.I):
            current_event = text.strip()
            if result['selections'] and not result['selections'][-1].get('event'):
                result['selections'][-1]['event'] = current_event
            i += 1; continue

        # ── Date/time ──
        if re.search(r'(Today|Tomorrow),?\s+\d+:\d+', text, re.I):
            i += 1; continue

        i += 1

    # Detect sport
    all_text = ' '.join(l['text'] for l in lines)
    result['sport'] = detect_sport_from_text(all_text)

    return result

# ── Main parse function ───────────────────────────────────────────────────────
def parse_slip(img_path: str, sportsbook: str = None) -> dict:
    lines = extract_lines(img_path)
    if not lines:
        return {'error': 'No text detected'}

    # Use provided sportsbook or fall back to auto-detection
    book = sportsbook or detect_sportsbook(lines)

    parsers = {
        'DraftKings': parse_draftkings,
        'FanDuel': parse_fanduel,
        'PrizePicks': parse_prizepicks,
        'Underdog': parse_underdog,
        'BetMGM': parse_betmgm,
        'Fanatics': parse_fanatics,
        'Onyx': parse_onyx,
        'Hard Rock': parse_hardrock,
    }

    parser = parsers.get(book)
    if parser:
        result = parser(lines)
    else:
        result = {'sportsbook': book, 'raw_lines': [l['text'] for l in lines]}

    # Clean up event fields across all selections (strip time stamps, ellipsis)
    for sel in result.get('selections', []) + result.get('picks', []):
        if sel.get('event'):
            sel['event'] = clean_event(sel['event'])

    return result
