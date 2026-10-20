"""Official club domains used for targeted personnel research."""

from urllib.parse import urlparse

CLUB_DOMAINS = {
    "ARI": "azcardinals.com",
    "ATL": "atlantafalcons.com",
    "BAL": "baltimoreravens.com",
    "BUF": "buffalobills.com",
    "CAR": "panthers.com",
    "CHI": "chicagobears.com",
    "CIN": "bengals.com",
    "CLE": "clevelandbrowns.com",
    "DAL": "dallascowboys.com",
    "DEN": "denverbroncos.com",
    "DET": "detroitlions.com",
    "GB": "packers.com",
    "HOU": "houstontexans.com",
    "IND": "colts.com",
    "JAX": "jaguars.com",
    "KC": "chiefs.com",
    "LA": "therams.com",
    "LAC": "chargers.com",
    "LV": "raiders.com",
    "MIA": "miamidolphins.com",
    "MIN": "vikings.com",
    "NE": "patriots.com",
    "NO": "neworleanssaints.com",
    "NYG": "giants.com",
    "NYJ": "newyorkjets.com",
    "PHI": "philadelphiaeagles.com",
    "PIT": "steelers.com",
    "SEA": "seahawks.com",
    "SF": "49ers.com",
    "TB": "buccaneers.com",
    "TEN": "tennesseetitans.com",
    "WAS": "commanders.com",
}


def official_publisher(url):
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or parsed.username or parsed.password:
        return None
    for domain in ("nfl.com", *CLUB_DOMAINS.values()):
        if host == domain or host.endswith("." + domain):
            return domain
    return None
