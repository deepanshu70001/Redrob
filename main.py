import csv
import json
import math


# Load the resume dataset
def loadresumedataset(file_path):
    resume_dataset = []
    with open(file_path, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            resume_dataset.append({
                "name": row["Candidate"],
                "skills": row["Raw Skills"].split(","),
            })
    return resume_dataset


# Load the job description dataset
def loadjobdescriptiondataset(file_path):
    jobdescriptiondataset = []
    with open(file_path, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            jobdescriptiondataset.append({
                "job_description": f'{row["JD"]} - {row["Company"]} ({row["Role"]})',
                "required_skills": row["Required Skills"].split(","),
                "preferred_skills": row["Preferred Skills"].split(","),
            })
    return jobdescriptiondataset


# Load the skill aliases
def loadskillaliases(file_path):
    with open(file_path, "r") as file:
        skill_aliases = json.load(file)
    return skill_aliases


# Normalize the skills
def normalizeskills(skillstring, skill_aliases):
    normalized_skills = []
    for skill in skillstring.split(","):
        skill = skill.strip().lower()
        if skill in skill_aliases:
            normalized_skills.append(skill_aliases[skill])
    return list(set(normalized_skills))


# Compute the TF-IDF vectors
def computetfidf(resumedataset, skillaliases):
    normalizedresumeskills = []
    vocabulary = set()

    for resume in resumedataset:
        skills = []
        for skill in resume["skills"]:
            skills.extend(normalizeskills(skill, skillaliases))
        skills = list(set(skills))
        normalizedresumeskills.append(skills)
        vocabulary.update(skills)

    vocabulary = sorted(vocabulary)
    tfidfvectors = []

    for skills in normalizedresumeskills:
        tfidfvector = []
        for vocabskill in vocabulary:
            if vocabskill in skills and len(skills) > 0:
                tf = 1 / len(skills)
                df = sum(1 for resume_skills in normalizedresumeskills if vocabskill in resume_skills)
                idf = math.log(len(resumedataset) / df)
                tfidfvector.append(tf * idf)
            else:
                tfidfvector.append(0)
        tfidfvectors.append(tfidfvector)

    return tfidfvectors, vocabulary


# Compute the job description vectors
def computejobdescriptionvectors(jobdescriptiondataset, skillaliases, vocabulary):
    jobdescriptionvectors = []
    for jobdescription in jobdescriptiondataset:
        skills = []
        allskills = jobdescription["required_skills"] + jobdescription["preferred_skills"]
        for skill in allskills:
            skills.extend(normalizeskills(skill, skillaliases))
        skills = set(skills)
        jobdescriptionvector = [1 if skill in skills else 0 for skill in vocabulary]
        jobdescriptionvectors.append(jobdescriptionvector)
    return jobdescriptionvectors


# Compute the cosine similarities
def computecosinesimilarity(tfidfvectors, jobdescriptionvectors):
    cosine_similarities = []
    for tfidfvector in tfidfvectors:
        similarities = []
        for jobdescriptionvector in jobdescriptionvectors:
            dotproduct = sum(x * y for x, y in zip(tfidfvector, jobdescriptionvector))
            magnitudetfidf = math.sqrt(sum(x ** 2 for x in tfidfvector))
            magnitudejobdescription = math.sqrt(sum(x ** 2 for x in jobdescriptionvector))
            denominator = magnitudetfidf * magnitudejobdescription
            similarity = dotproduct / denominator if denominator != 0 else 0
            similarities.append(similarity)
        cosine_similarities.append(similarities)
    return cosine_similarities


# Rank the candidates
def rankcandidates(cosinesimilarities, resumedataset, jobdescriptiondataset):
    ranked_candidates = []
    for i, jobdescription in enumerate(jobdescriptiondataset):
        similarities = [x[i] for x in cosinesimilarities]
        indices = sorted(
            range(len(similarities)),
            key=lambda x: (-similarities[x], resumedataset[x]["name"]),
        )
        top3candidates = []
        for index in indices[:3]:
            top3candidates.append((resumedataset[index]["name"], round(similarities[index], 2)))
        ranked_candidates.append(top3candidates)
    return ranked_candidates


# Main function
def main():
    resumedataset = loadresumedataset("data/resume_dataset.csv")
    jobdescriptiondataset = loadjobdescriptiondataset("data/job_description_dataset.csv")
    skillaliases = loadskillaliases("data/skill_aliases.json")
    tfidfvectors, vocabulary = computetfidf(resumedataset, skillaliases)
    jobdescriptionvectors = computejobdescriptionvectors(jobdescriptiondataset, skillaliases, vocabulary)
    cosinesimilarities = computecosinesimilarity(tfidfvectors, jobdescriptionvectors)
    rankedcandidates = rankcandidates(cosinesimilarities, resumedataset, jobdescriptiondataset)
    for i, jobdescription in enumerate(jobdescriptiondataset):
        print(f'{jobdescription["job_description"]}:')
        for candidate in rankedcandidates[i]:
            print(f"{candidate[0]} ({candidate[1]:.2f})")
        print()


if __name__ == "__main__":
    main()
