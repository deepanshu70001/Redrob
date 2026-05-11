import math
from utils.exceptions import CosineSimilarityError


class CosineSimilarity:
    def __init__(self, tfidfvectors, jobdescriptionvectors):
        self.tfidfvectors = tfidfvectors
        self.jobdescriptionvectors = jobdescriptionvectors

    def calculatecosinesimilarity(self, vector1, vector2):
        dotproduct = sum(x * y for x, y in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(x ** 2 for x in vector1))
        magnitude2 = math.sqrt(sum(x ** 2 for x in vector2))
        denominator = magnitude1 * magnitude2
        if denominator == 0:
            return 0
        return dotproduct / denominator

    def process(self):
        try:
            similarity_scores = []
            for tfidfvector in self.tfidfvectors:
                resume_scores = []
                for jobdescriptionvector in self.jobdescriptionvectors:
                    resume_scores.append(
                        self.calculatecosinesimilarity(tfidfvector, jobdescriptionvector)
                    )
                similarity_scores.append(resume_scores)
            return similarity_scores
        except Exception as e:
            raise CosineSimilarityError(f"Error calculating cosine similarity: {str(e)}")
