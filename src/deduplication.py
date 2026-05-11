from utils.exceptions import DeduplicationError


class Deduplication:
    def __init__(self, normalized_skills):
        self.normalized_skills = normalized_skills

    def process(self):
        try:
            deduplicated_skills = []
            seen = set()
            for skill in self.normalized_skills:
                if skill not in seen:
                    deduplicated_skills.append(skill)
                    seen.add(skill)
            return deduplicated_skills
        except Exception as e:
            raise DeduplicationError(f"Error deduplicating skills: {str(e)}")
