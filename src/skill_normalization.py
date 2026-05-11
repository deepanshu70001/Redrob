from utils.constants import SKILL_ALIASES
from utils.exceptions import NormalizationError


class SkillNormalization:
    def __init__(self, resume_data):
        self.resume_data = resume_data

    def process(self):
        try:
            normalized_skills = []
            for skill in self.resume_data["skills"]:
                skill = skill.strip().lower()
                if skill in SKILL_ALIASES:
                    normalized_skills.append(SKILL_ALIASES[skill])
            return normalized_skills
        except Exception as e:
            raise NormalizationError(f"Error normalizing skills: {str(e)}")
