from src.utils.team_names import TEAM_NAME_MAPPING

def normalize_team_name(name: str) -> str:
    return TEAM_NAME_MAPPING.get(name, name)