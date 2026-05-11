from utils.exceptions import RankingError


class Ranking:
    def __init__(self, similarity_scores, resume_data=None):
        self.similarity_scores = similarity_scores
        self.resume_data = resume_data

    def process(self):
        try:
            ranked_resumes = []
            job_count = len(self.similarity_scores[0]) if self.similarity_scores else 0

            for job_index in range(job_count):
                scored_resumes = []
                for resume_index, scores in enumerate(self.similarity_scores):
                    name = (
                        self.resume_data[resume_index]["name"]
                        if self.resume_data
                        else resume_index
                    )
                    scored_resumes.append((name, scores[job_index]))

                scored_resumes.sort(key=lambda candidate: (-candidate[1], candidate[0]))
                ranked_resumes.append(scored_resumes[:3])

            return ranked_resumes
        except Exception as e:
            raise RankingError(f"Error ranking resumes: {str(e)}")
