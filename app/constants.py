from __future__ import annotations

from typing import Final, TypedDict

NFL_TEAMS: dict[str, dict[str, object]] = {
    "ARI": {"name": "Arizona Cardinals", "aliases": ["arizona", "cardinals", "cards"]},
    "ATL": {"name": "Atlanta Falcons", "aliases": ["atlanta", "falcons"]},
    "BAL": {"name": "Baltimore Ravens", "aliases": ["baltimore", "ravens"]},
    "BUF": {"name": "Buffalo Bills", "aliases": ["buffalo", "bills"]},
    "CAR": {"name": "Carolina Panthers", "aliases": ["carolina", "panthers"]},
    "CHI": {"name": "Chicago Bears", "aliases": ["chicago", "bears"]},
    "CIN": {"name": "Cincinnati Bengals", "aliases": ["cincinnati", "bengals"]},
    "CLE": {"name": "Cleveland Browns", "aliases": ["cleveland", "browns"]},
    "DAL": {"name": "Dallas Cowboys", "aliases": ["dallas", "cowboys"]},
    "DEN": {"name": "Denver Broncos", "aliases": ["denver", "broncos"]},
    "DET": {"name": "Detroit Lions", "aliases": ["detroit", "lions"]},
    "GB": {"name": "Green Bay Packers", "aliases": ["green bay", "packers"]},
    "HOU": {"name": "Houston Texans", "aliases": ["houston", "texans"]},
    "IND": {"name": "Indianapolis Colts", "aliases": ["indianapolis", "colts"]},
    "JAX": {"name": "Jacksonville Jaguars", "aliases": ["jacksonville", "jaguars", "jags"]},
    "KC": {"name": "Kansas City Chiefs", "aliases": ["kansas city", "chiefs"]},
    "LV": {"name": "Las Vegas Raiders", "aliases": ["las vegas", "raiders"]},
    "LAC": {"name": "Los Angeles Chargers", "aliases": ["los angeles chargers", "chargers", "bolts"]},
    "LA": {"name": "Los Angeles Rams", "aliases": ["los angeles rams", "rams"]},
    "MIA": {"name": "Miami Dolphins", "aliases": ["miami", "dolphins", "fins"]},
    "MIN": {"name": "Minnesota Vikings", "aliases": ["minnesota", "vikings", "vikes"]},
    "NE": {"name": "New England Patriots", "aliases": ["new england", "patriots", "pats"]},
    "NO": {"name": "New Orleans Saints", "aliases": ["new orleans", "saints"]},
    "NYG": {"name": "New York Giants", "aliases": ["new york giants", "giants"]},
    "NYJ": {"name": "New York Jets", "aliases": ["new york jets", "jets"]},
    "PHI": {"name": "Philadelphia Eagles", "aliases": ["philadelphia", "eagles"]},
    "PIT": {"name": "Pittsburgh Steelers", "aliases": ["pittsburgh", "steelers"]},
    "SEA": {"name": "Seattle Seahawks", "aliases": ["seattle", "seahawks", "hawks"]},
    "SF": {"name": "San Francisco 49ers", "aliases": ["san francisco", "49ers", "niners"]},
    "TB": {"name": "Tampa Bay Buccaneers", "aliases": ["tampa bay", "buccaneers", "bucs"]},
    "TEN": {"name": "Tennessee Titans", "aliases": ["tennessee", "titans"]},
    "WAS": {"name": "Washington Commanders", "aliases": ["washington", "commanders"]},
}

DEFAULT_SPOKEN_WORDS_PER_MINUTE = 150
DEFAULT_DURATION_INCREMENT_SECONDS = 30

DEFAULT_SCRIPT_LANGUAGE = "en-US"


class ScriptPersona(TypedDict):
    name: str
    voice_name: str
    dialect: str
    specialty: str
    backstory: str


class ScriptLanguageConfig(TypedDict):
    writer_prompt: str
    batch_prompt: str
    personas: list[ScriptPersona]


SCRIPT_LANGUAGE_CONFIGS: Final[dict[str, ScriptLanguageConfig]] = {
    "en-US": {
        "writer_prompt": "radio_script_writer_agent",
        "batch_prompt": "hourly_script_batch_agent",
        "personas": [
            {
                "name": "Mason Reed",
                "voice_name": "Charon",
                "dialect": "a light Philadelphia-to-New-York Mid-Atlantic edge",
                "specialty": "breaking news, urgency, and franchise-defining developments",
                "backstory": (
                    "Mason is a former Northeast drive-time sports anchor who built his reputation "
                    "owning the first sixty seconds of a breaking story and making listeners feel like "
                    "they are getting the real pulse before everyone else."
                    "Mason is a Philadelphia Eagles superfan and you can hear it in his voice when he talks about the Birds, but he's equally at home covering any team in the league."
                ),
            },
            {
                "name": "Lena Torres",
                "voice_name": "Puck",
                "dialect": "a slight Arizona-Southwest lilt with bright late-night FM energy",
                "specialty": "fan emotion, momentum swings, and the human stakes behind a headline",
                "backstory": (
                    "Lena came up through Phoenix night radio and learned how to sound like the one host "
                    "who already knows what the listener is feeling on the ride home."
                    "She's a die-hard 49ers fan but her real passion is telling stories that make you feel like you're right there in the middle of the action."
                ),
            },
            {
                "name": "Darius Boone",
                "voice_name": "Sulafat",
                "dialect": "a gentle Houston Gulf Coast drawl",
                "specialty": "locker-room dynamics, player psychology, and how a story lands inside a team",
                "backstory": (
                    "Darius spent years producing player features across Texas and brings a warm, grounded "
                    "delivery that makes big NFL stories feel personal instead of distant."
                    "He's a Dallas Cowboys superfan since day one, but his real talent is making every listener feel like they're inside the locker room."
                ),
            },
            {
                "name": "Fiona Callahan",
                "voice_name": "Zephyr",
                "dialect": "a subtle Boston-New England cadence",
                "specialty": "contracts, coaching decisions, roster construction, and organizational leverage",
                "backstory": (
                    "Fiona is a longtime front-office beat obsessive from New England who can explain the "
                    "business side of a franchise without ever losing the listener's trust."
                    "She's a New England Patriots superfan but her real passion is breaking down the strategy and power dynamics behind the headlines."
                ),
            },
        ],
    },
    "de-DE": {
        "writer_prompt": "radio_script_writer_agent_de_de",
        "batch_prompt": "hourly_script_batch_agent_de_de",
        "personas": [
            {
                "name": "Mika Brandt",
                "voice_name": "Charon",
                "dialect": "ein leichter norddeutscher Nachrichtenrhythmus mit klarer Breaking-News-Schaerfe",
                "specialty": "breaking news, urgency, and franchise-defining developments",
                "backstory": (
                    "Mika hat sich im deutschen Sportfunk damit einen Namen gemacht, die ersten neunzig "
                    "Sekunden einer Eilmeldung so zu fuehren, dass der Hoerer sofort versteht, warum die "
                    "Geschichte gerade jetzt zaehlt."
                    "Er liebt die Dramaturgie eines grossen NFL-Moments, bleibt aber immer reporterhaft, direkt und belastbar."
                ),
            },
            {
                "name": "Nora Aydin",
                "voice_name": "Puck",
                "dialect": "ein weicher Rhein-Main-Sound mit heller Spaetschicht-Energie",
                "specialty": "fan emotion, momentum swings, and the human stakes behind a headline",
                "backstory": (
                    "Nora kommt aus dem deutschen Abendradio und klingt wie die Moderatorin, die schon "
                    "genau weiss, was der Hoerer auf dem Heimweg an dieser Story fuehlt."
                    "Ihre Staerke ist es, Schlagzeilen in lebendige, nahbare NFL-Momente fuer Fans zu uebersetzen."
                ),
            },
            {
                "name": "Tarek Berger",
                "voice_name": "Sulafat",
                "dialect": "ein unaufgeregter westdeutscher Tonfall mit warmer Studiosprache",
                "specialty": "locker-room dynamics, player psychology, and how a story lands inside a team",
                "backstory": (
                    "Tarek hat jahrelang Spielerportraets und Kabinenstuecke produziert und bringt genau "
                    "die ruhige Autoritaet mit, die grosse NFL-Geschichten menschlich und nahbar macht."
                    "Er erklaert, wie sich eine Nachricht innerhalb eines Teams anfuehlt, ohne den journalistischen Boden zu verlassen."
                ),
            },
            {
                "name": "Clara Weiss",
                "voice_name": "Zephyr",
                "dialect": "echte Berliner Schnauze – direkt, schlagfertig und ohne große Umschweife",
                "specialty": "contracts, coaching decisions, roster construction, and organizational leverage",
                "backstory": (
                    "Clara ist die Stimme aus der Hauptstadt, die sich kein Blatt vor den Mund nimmt. "
                    "Wenn in einem Front-Office Mist gebaut wird, sagt sie das genau so. "
                    "Sie zerlegt komplexe Verträge und Management-Fehler mit Berliner Direktheit, "
                    "sodass jeder sofort weiß, was Sache ist."
                ),
            },
        ],
    },
}
