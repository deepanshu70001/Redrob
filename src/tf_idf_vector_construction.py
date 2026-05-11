import math
from utils.exceptions import TFIDFVectorConstructionError


class TFIDFVectorConstruction:
    def __init__(self, vocabulary, resume_data):
        self.vocabulary = vocabulary
        self.resume_data = resume_data

    def calculate_tf(self, skill, resume):
        if not resume:
            return 0
        return resume.count(skill) / len(resume)

    def calculate_idf(self, skill, resumes):
        document_frequency = sum(1 for resume in resumes if skill in resume)
        if document_frequency == 0:
            return 0
        return math.log(len(resumes) / document_frequency)

    def process(self):
        try:
            tfidfvectors = []
            resumes = [resume["skills"] for resume in self.resume_data]
            for resume in self.resume_data:
                tfidfvector = []
                for skill in self.vocabulary:
                    tf = self.calculate_tf(skill, resume["skills"])
                    idf = self.calculate_idf(skill, resumes)
                    tfidfvector.append(tf * idf)
                tfidfvectors.append(tfidfvector)
            return tfidfvectors
        except Exception as e:
            raise TFIDFVectorConstructionError(f"Error constructing TF-IDF vectors: {str(e)}")
