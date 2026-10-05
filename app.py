from flask import Flask, render_template, request
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)


# --------------------------------
# Skills our analyzer can detect
# --------------------------------

skill_patterns = {
    "python": r"\bpython\b",
    "java": r"\bjava\b",
    "sql": r"\bsql\b",
    "flask": r"\bflask\b",
    "django": r"\bdjango\b",
    "html": r"\bhtml\b",
    "css": r"\bcss\b",
    "javascript": r"\bjavascript\b",

    "machine learning": r"\bmachine learning\b",
    "deep learning": r"\bdeep learning\b",
    "generative ai": r"\bgenerative ai\b",
    "llm": r"\bllms?\b",
    "rag": r"\brag\b",
    "nlp": r"\bnlp\b",

    "git": r"\bgit\b",
    "github": r"\bgithub\b",

    "aws": r"\baws\b",
    "azure": r"\bazure\b",
    "docker": r"\bdocker\b",
    "mysql": r"\bmysql\b",

    "pandas": r"\bpandas\b",
    "numpy": r"\bnumpy\b",
    "scikit-learn": r"\bscikit[- ]learn\b",

    "rest api": r"\brestful?\s+apis?\b",

    "linux": r"\blinux\b",
    "tensorflow": r"\btensorflow\b",
    "pytorch": r"\bpytorch\b"
}


# --------------------------------
# Learning recommendations
# --------------------------------

recommendation_map = {
    "python":
        "Learn Python basics, OOP, functions, modules and error handling.",

    "sql":
        "Learn SQL queries, joins, subqueries and database basics.",

    "flask":
        "Learn Flask routing, templates, forms and REST APIs.",

    "django":
        "Learn Django models, views, URLs, templates and REST APIs.",

    "html":
        "Learn HTML structure, forms, tables and semantic elements.",

    "css":
        "Learn CSS selectors, Flexbox, Grid and responsive design.",

    "javascript":
        "Learn JavaScript basics, DOM, events and API integration.",

    "machine learning":
        "Learn ML algorithms, training, testing and model evaluation.",

    "deep learning":
        "Learn neural networks, CNNs, RNNs and deep learning frameworks.",

    "generative ai":
        "Learn LLMs, prompt engineering, embeddings and RAG.",

    "llm":
        "Learn LLM fundamentals, prompting, embeddings and LLM applications.",

    "rag":
        "Learn chunking, embeddings, vector databases and retrieval.",

    "nlp":
        "Learn text preprocessing, tokenization, embeddings and NLP models.",

    "git":
        "Learn Git commands, commits, branches and version control.",

    "github":
        "Learn GitHub repositories, branches, pull requests and collaboration.",

    "aws":
        "Learn AWS basics, EC2, S3, IAM and cloud deployment.",

    "azure":
        "Learn Azure basics, virtual machines and cloud services.",

    "docker":
        "Learn Docker images, containers, Dockerfile and Docker Compose.",

    "linux":
        "Learn Linux commands, files, permissions and processes.",

    "mysql":
        "Learn MySQL databases, tables, queries and joins."
}


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Get uploaded resume
        resume = request.files.get("resume")

        # Get job description
        job_description = request.form.get("job_description", "")

        if not resume:
            return "Please upload a resume PDF."

        # --------------------------------
        # Read PDF
        # --------------------------------

        reader = PdfReader(resume)

        resume_text = ""

        for page in reader.pages:
            resume_text += page.extract_text() or ""

        # Convert to lowercase
        resume_text = resume_text.lower()
        job_description = job_description.lower()

        # --------------------------------
        # Detect resume skills
        # --------------------------------

        resume_skills = []

        for skill, pattern in skill_patterns.items():

            if re.search(pattern, resume_text):
                resume_skills.append(skill)

        # --------------------------------
        # Detect job skills
        # --------------------------------

        job_skills = []

        for skill, pattern in skill_patterns.items():

            if re.search(pattern, job_description):
                job_skills.append(skill)

        # --------------------------------
        # Matched skills
        # --------------------------------

        matched_skills = []

        for skill in job_skills:

            if skill in resume_skills:
                matched_skills.append(skill)

        # --------------------------------
        # Missing skills
        # --------------------------------

        missing_skills = []

        for skill in job_skills:

            if skill not in resume_skills:
                missing_skills.append(skill)

        # --------------------------------
        # Keyword score
        # --------------------------------

        if job_skills:

            keyword_score = (
                len(matched_skills) /
                len(job_skills)
            ) * 100

        else:

            keyword_score = 0

        # --------------------------------
        # NLP similarity
        # --------------------------------

        if resume_text.strip() and job_description.strip():

            documents = [
                resume_text,
                job_description
            ]

            vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            tfidf_matrix = vectorizer.fit_transform(documents)

            similarity = cosine_similarity(
                tfidf_matrix[0:1],
                tfidf_matrix[1:2]
            )[0][0]

            similarity_score = similarity * 100

        else:

            similarity_score = 0

        # --------------------------------
        # Final score
        # --------------------------------

        if job_skills:

            final_score = (
                keyword_score * 0.7
                + similarity_score * 0.3
            )

        else:

            final_score = similarity_score

        final_score = min(
            max(final_score, 0),
            100
        )

        # --------------------------------
        # Recommendations
        # --------------------------------

        recommendations = []

        for skill in missing_skills:

            if skill in recommendation_map:

                recommendations.append(
                    recommendation_map[skill]
                )

        # --------------------------------
        # Show result
        # --------------------------------

        return render_template(
            "result.html",
            match_percentage=round(final_score, 1),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            recommendations=recommendations,
            similarity_score=round(similarity_score, 1)
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)