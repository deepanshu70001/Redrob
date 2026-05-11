import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skill_normalization import SkillNormalization
from src.deduplication import Deduplication
from src.vocabulary_construction import VocabularyConstruction
from src.tf_idf_vector_construction import TFIDFVectorConstruction
from src.cosine_similarity import CosineSimilarity
from src.ranking import Ranking
from utils.constants import (
    JOB_DESCRIPTION_DATASET_PATH,
    RESULTS_PATH,
    RESUME_DATASET_PATH,
)


def splitskills(skill_text):
    return [skill.strip() for skill in skill_text.split(",") if skill.strip()]


def loadresumedata():
    resumedata = []
    with open(RESUME_DATASET_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            resumedata.append({
                "name": row["Candidate"],
                "skills": splitskills(row["Raw Skills"]),
            })
    return resumedata


def loadjobdescriptiondata():
    jobdescriptiondata = []
    with open(JOB_DESCRIPTION_DATASET_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            jobdescriptiondata.append({
                "job_description": f'{row["JD"]} - {row["Company"]} ({row["Role"]})',
                "skills": splitskills(row["Required Skills"]) + splitskills(row["Preferred Skills"]),
            })
    return jobdescriptiondata


def normalizeanddeduplicate(data):
    processeddata = []
    for item in data:
        skillnormalization = SkillNormalization(item)
        normalizedskills = skillnormalization.process()

        deduplication = Deduplication(normalizedskills)
        deduplicatedskills = deduplication.process()

        processeditem = item.copy()
        processeditem["skills"] = deduplicatedskills
        processeddata.append(processeditem)
    return processeddata


def buildjobdescriptionvectors(jobdescriptiondata, vocabulary):
    jobdescriptionvectors = []
    for jobdescription in jobdescriptiondata:
        skills = set(jobdescription["skills"])
        jobdescriptionvectors.append([1 if skill in skills else 0 for skill in vocabulary])
    return jobdescriptionvectors


def saveresults(jobdescriptiondata, rankedcandidates):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for jobdescription, candidates in zip(jobdescriptiondata, rankedcandidates):
        formattedcandidates = ", ".join(
            f"{name}({score:.2f})"
            for name, score in candidates
        )
        lines.append(f'{jobdescription["job_description"]}: {formattedcandidates}')
    RESULTS_PATH.write_text("\n".join(lines) + "\n")
    return lines


def main():
    # Load data
    resumedata = loadresumedata()
    jobdescriptiondata = loadjobdescriptiondata()

    # Normalize and deduplicate skills
    resumedata = normalizeanddeduplicate(resumedata)
    jobdescriptiondata = normalizeanddeduplicate(jobdescriptiondata)

    # Construct vocabulary
    deduplicatedskills = [resume["skills"] for resume in resumedata]
    vocabularyconstruction = VocabularyConstruction(deduplicatedskills)
    vocabulary = vocabularyconstruction.process()

    # Construct TF-IDF vectors
    tfidfvectorconstruction = TFIDFVectorConstruction(vocabulary, resumedata)
    tfidfvectors = tfidfvectorconstruction.process()

    # Build JD vectors
    jobdescriptionvectors = buildjobdescriptionvectors(jobdescriptiondata, vocabulary)

    # Calculate cosine similarity
    cosinesimilarity = CosineSimilarity(tfidfvectors, jobdescriptionvectors)
    similarityscores = cosinesimilarity.process()

    # Rank candidates
    ranking = Ranking(similarityscores, resumedata)
    rankedcandidates = ranking.process()

    # Save and print results
    for line in saveresults(jobdescriptiondata, rankedcandidates):
        print(line)


if __name__ == "__main__":
    main()
