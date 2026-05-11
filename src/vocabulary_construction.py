from utils.exceptions import VocabularyConstructionError


class VocabularyConstruction:
    def __init__(self, deduplicated_skills):
        self.deduplicated_skills = deduplicated_skills

    def process(self):
        try:
            vocabulary = set()
            for skills in self.deduplicated_skills:
                vocabulary.update(skills)
            return sorted(vocabulary)
        except Exception as e:
            raise VocabularyConstructionError(f"Error constructing vocabulary: {str(e)}")
