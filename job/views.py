import uuid
from datetime import datetime
from django.http import HttpResponse
from rest_framework import status, viewsets
from rest_framework.response import Response
import requests
from .serializers import CvSerializer
from jobfinder_backend.utils import extract_text_from_pdf, extract_section
from jobfinder_backend.settings import api_key
from .models import JobPosition, Cv
from transformers import BertTokenizer, BertModel
import torch
from openai import OpenAI
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from functools import lru_cache
import logging
import json
from rest_framework.decorators import action

logger = logging.getLogger(__name__)

load_dotenv()

# Lazy loading of BERT model
_bert_tokenizer = None
_bert_model = None


def get_bert_model():
    """Lazy load BERT model and tokenizer"""
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        _bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if _bert_model is None:
        _bert_model = BertModel.from_pretrained("bert-base-uncased")
    return _bert_tokenizer, _bert_model


@lru_cache(maxsize=128)
def get_bert_embedding(text):
    """
    Generate BERT embeddings for a given text
    Uses caching to avoid recomputing embeddings for the same text
    """
    tokenizer, model = get_bert_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


class CvUpload(viewsets.ViewSet):
    """
    Viewset for handling CV uploads
    """

    @action(detail=False, methods=["post"], url_path="processApplicants")
    def process_first_export(self, request):
        """
        Process a JSON file provided in the request body, extract the CV, feed it to an OpenAI prompt,
        and save the response to the applicantbg folder.
        """
        try:
            # Ensure the applicantbg folder exists
            applicantbg_folder = "applicantbg"
            if not os.path.exists(applicantbg_folder):
                os.makedirs(applicantbg_folder)

            # Check if a file is included in the request
            if "data" not in request.FILES:
                return Response({"error": "No file uploaded."}, status=400)

            # Get the uploaded file
            uploaded_file = request.FILES["data"]

            # Read and parse the JSON file
            try:
                data = json.loads(uploaded_file.read().decode("utf-8"))
            except json.JSONDecodeError:
                return Response({"error": "Invalid JSON file."}, status=400)

            # Debug: Print the incoming JSON data
            print("Incoming JSON data:", data)

            # Extract CV data
            cv_data = data.get("cv", {})
            if not cv_data:
                return Response({"error": "No CV data found in the JSON file."}, status=404)

            # Debug: Print the extracted CV data
            print("Extracted CV data:", cv_data)

            # Prepare the OpenAI prompt
            openai_prompt = f"""
            Extract the following details from the given resume. Return the output in JSON format.
            ### Resume:
            {cv_data.get("skills", "")} {cv_data.get("experience", "")} {cv_data.get("education", "")}

            ### Extract the following details:
            1. Role: Identify the candidate’s current or most relevant job role ("software engineering", "web development", "data science",
                        "machine learning", "artificial intelligence", "cloud computing",
                        "cybersecurity", "mobile development", "devops", "backend development",
                        "frontend development", "full stack development", "Project Management" ).
            2. Education Level: Extract the highest level of one education achieved [
                                                                                      "HSD",
                                                                                      "BSc",
                                                                                      "MSc",
                                                                                      "PhD",
                                                                                      "Certifications",
                                                                                      "Other"
                                                                                    ].
            3. Experience Years: Estimate the total number of years of professional work experience if you do not know say unknown.

            ### Output Format (JSON):
            {{
              "Role": "<Extracted Role> (only choose from the following above 1.Roles if it does not fit any choose the closest one to it)",
              "Education Level": "<Extracted Education> (only choose from the following above Education Levels if it does not fit any choose closest one to it)",
              "Experience Years": <Extracted Years>
            }}
            """

            # Debug: Print the OpenAI prompt
            print("OpenAI Prompt:", openai_prompt)

            # Send the prompt to OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not os.getenv("OPENAI_API_KEY"):
                return Response({"error": "OpenAI API key not configured"}, status=500)

            response = client.chat.completions.create(
                model="gpt-4",  # Use a model that supports JSON output
                messages=[{"role": "user", "content": openai_prompt}],
            )

            # Debug the OpenAI response
            print("OpenAI Response Content:", response.choices[0].message.content)

            # Parse the AI response
            try:
                ai_response = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                ai_response = {
                    "error": "The AI response could not be parsed as JSON.",
                    "raw_response": response.choices[0].message.content
                }

            # Save the AI response to the applicantbg folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            response_filename = f"response_{timestamp}.json"
            response_file_path = os.path.join(applicantbg_folder, response_filename)

            with open(response_file_path, "w") as f:
                json.dump({
                    "cv_data": cv_data,
                    "ai_response": ai_response
                }, f, indent=4)

            return Response({
                "cv_data": cv_data,
                "ai_response": ai_response,
                "saved_to": response_file_path
            }, status=200)

        except Exception as e:
            # Add comprehensive error handling
            import traceback
            return Response({
                "error": f"An error occurred while processing the first export: {str(e)}",
                "traceback": traceback.format_exc()
            }, status=500)

    @action(detail=False, methods=["post"], url_path="json-ranker")
    def json_ranker(self, request):
        """
        Rank jobs using data from a JSON file sent in the request body.
        Save the AI recommendations to the ai_recommendation_json folder.
        Replace job names in ai_recommendation with internal IDs.
        """
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not os.getenv("OPENAI_API_KEY"):
                return Response({"error": "OpenAI API key not configured"}, status=500)

            # Get JSON data from the request body
            data = request.data
            if not data:
                return Response({"error": "No JSON data provided in the request body."}, status=400)

            # Extract CV and job data from the JSON file
            cv_data = data.get("cv", {})
            top_cosine_jobs = data.get("top_cosine_jobs", [])
            top_bert_jobs = data.get("top_bert_jobs", [])

            # Modified validation
            if not isinstance(cv_data, dict) or not isinstance(top_cosine_jobs, list) or not isinstance(top_bert_jobs,
                                                                                                        list):
                return Response({"error": "Invalid JSON structure."}, status=400)

            if not top_cosine_jobs or not top_bert_jobs:
                return Response({"error": "Job lists cannot be empty."}, status=400)

            # Function to safely truncate text
            def safe_truncate(text, length=500):
                if text and isinstance(text, str):
                    return text[:length] + "..." if len(text) > length else text
                return "Not available"

            # Prepare the prompt for the AI judge
            openai_prompt = f"""
                    You are an impartial AI judge for NextMatch, a job recommender system. Your task is to evaluate how well a candidate's 
                    resume matches job vacancies. The system has already calculated similarity scores using two different methods.

                    Input:
                    Candidate's CV: 
                    {cv_data.get("skills", "")} {cv_data.get("experience", "")} {cv_data.get("education", "")}

                    Method A - Top 3 Job Matches (TF-IDF with Cosine Similarity):
                    1. {top_cosine_jobs[0].get("job_position", "Not Available")} at {top_cosine_jobs[0].get("company_name", "Not Available")}
                       Description: {safe_truncate(top_cosine_jobs[0].get("job_description", "No description available"))}

                    2. {top_cosine_jobs[1].get("job_position", "Not Available")} at {top_cosine_jobs[1].get("company_name", "Not Available")}
                       Description: {safe_truncate(top_cosine_jobs[1].get("job_description", "No description available"))}

                    3. {top_cosine_jobs[2].get("job_position", "Not Available")} at {top_cosine_jobs[2].get("company_name", "Not Available")}
                       Description: {safe_truncate(top_cosine_jobs[2].get("job_description", "No description available"))}

                    Method B - Top 3 Job Matches (BERT Semantic Embeddings):
                    1. {top_bert_jobs[0].get("job_position", "Not Available")} at {top_bert_jobs[0].get("company_name", "Not Available")}
                       Description: {safe_truncate(top_bert_jobs[0].get("job_description", "No description available"))}

                    2. {top_bert_jobs[1].get("job_position", "Not Available")} at {top_bert_jobs[1].get("company_name", "Not Available")}
                       Description: {safe_truncate(top_bert_jobs[1].get("job_description", "No description available"))}

                    3. {top_bert_jobs[2].get("job_position", "Not Available")} at {top_bert_jobs[2].get("company_name", "Not Available")}
                       Description: {safe_truncate(top_bert_jobs[2].get("job_description", "No description available"))}

                    Please provide an analysis in **valid JSON format** with the following structure:
                    Method A Scores:
                    X
                    X
                    X
                    Method B Scores:
                    X
                    X
                    X
                    """

            response = client.chat.completions.create(
                model="gpt-4",  # Use a model that supports JSON output
                messages=[{"role": "user", "content": openai_prompt}],
            )

            # Debug the OpenAI response
            print("OpenAI Response Content:", response.choices[0].message.content)

            # Parse the AI recommendation
            try:
                ai_recommendation = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                ai_recommendation = {
                    "error": "The AI recommendation could not be parsed as JSON.",
                    "raw_response": response.choices[0].message.content
                }

            # Assign internal IDs to jobs (1-500)
            def assign_internal_ids(jobs):
                return {i + 1: job for i, job in enumerate(jobs)}

            method_a_jobs_with_ids = assign_internal_ids(top_cosine_jobs)
            method_b_jobs_with_ids = assign_internal_ids(top_bert_jobs)

            # Replace job names in ai_recommendation with internal IDs
            def replace_job_names_with_ids(ai_recommendation, method_a_jobs_with_ids, method_b_jobs_with_ids):
                updated_recommendation = {"Method A Scores": {}, "Method B Scores": {}}

                # Replace Method A job names with IDs
                for job_name, score in ai_recommendation.get("Method A Scores", {}).items():
                    for job_id, job in method_a_jobs_with_ids.items():
                        if job_name == f"{job['job_position']} at {job['company_name']}":
                            updated_recommendation["Method A Scores"][job_id] = score
                            break

                # Replace Method B job names with IDs
                for job_name, score in ai_recommendation.get("Method B Scores", {}).items():
                    for job_id, job in method_b_jobs_with_ids.items():
                        if job_name == f"{job['job_position']} at {job['company_name']}":
                            updated_recommendation["Method B Scores"][job_id] = score
                            break

                return updated_recommendation

            ai_recommendation_with_ids = replace_job_names_with_ids(ai_recommendation, method_a_jobs_with_ids,
                                                                    method_b_jobs_with_ids)

            # Save the AI recommendation to the ai_recommendation_json folder
            ai_recommendation_folder = "ai_recommendation_json"
            if not os.path.exists(ai_recommendation_folder):
                os.makedirs(ai_recommendation_folder)

            # Generate a unique filename for the recommendation
            recommendation_filename = f"recommendation_{uuid.uuid4().hex}.json"
            recommendation_file_path = os.path.join(ai_recommendation_folder, recommendation_filename)

            # Save the AI recommendation with IDs
            with open(recommendation_file_path, "w") as f:
                json.dump({
                    "cv": cv_data,  # Include CV data at the top
                    "top_cosine_jobs": top_cosine_jobs,
                    "top_bert_jobs": top_bert_jobs,
                    "ai_recommendation": ai_recommendation_with_ids
                }, f, indent=4)

            return Response({
                "cv": cv_data,  # Include CV data at the top
                "top_cosine_jobs": top_cosine_jobs,
                "top_bert_jobs": top_bert_jobs,
                "ai_recommendation": ai_recommendation_with_ids
            }, status=200)

        except Exception as e:
            # Add comprehensive error handling
            import traceback
            return Response({
                "error": f"An error occurred during job ranking: {str(e)}",
                "traceback": traceback.format_exc()
            }, status=500)

    @action(detail=False, methods=["post"], url_path="check-and-rerun-incomplete-files")
    def check_and_rerun_incomplete_files(self, request):
        """
        Check JSON files in the ai_recommendation_json folder for missing Method A or Method B scores.
        Rerun the json-ranker process for incomplete files.
        """
        try:
            # Path to the ai_recommendation_json folder
            ai_recommendation_folder = "ai_recommendation_json"
            if not os.path.exists(ai_recommendation_folder):
                return Response({"error": f"The folder '{ai_recommendation_folder}' does not exist."}, status=404)

            # Get all JSON files in the folder
            json_files = [f for f in os.listdir(ai_recommendation_folder) if f.endswith(".json")]
            if not json_files:
                return Response({"error": f"No JSON files found in the '{ai_recommendation_folder}' folder."},
                                status=404)

            # List to store results
            results = []

            # Loop through each JSON file
            for json_file in json_files:
                json_file_path = os.path.join(ai_recommendation_folder, json_file)

                # Load the JSON data
                with open(json_file_path, "r") as f:
                    data = json.load(f)

                # Check if Method A Scores or Method B Scores are missing
                ai_recommendation = data.get("ai_recommendation", {})
                method_a_scores = ai_recommendation.get("Method A Scores", {})
                method_b_scores = ai_recommendation.get("Method B Scores", {})

                if not method_a_scores or not method_b_scores:
                    # Rerun the json-ranker process for this file
                    rerun_response = self.rerun_json_ranker(data)
                    results.append({
                        "file": json_file,
                        "status": "Rerun completed",
                        "response": rerun_response.data
                    })
                else:
                    results.append({
                        "file": json_file,
                        "status": "Complete (no rerun needed)"
                    })

            return Response({
                "results": results
            }, status=200)

        except Exception as e:
            # Add comprehensive error handling
            import traceback
            return Response({
                "error": f"An error occurred while checking and rerunning incomplete files: {str(e)}",
                "traceback": traceback.format_exc()
            }, status=500)

    def rerun_json_ranker(self, data):
        """
        Helper function to rerun the json-ranker process for a given JSON data.
        """
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not os.getenv("OPENAI_API_KEY"):
                return Response({"error": "OpenAI API key not configured"}, status=500)

            # Extract CV and job data from the JSON file
            cv_data = data.get("cv", {})
            top_cosine_jobs = data.get("top_cosine_jobs", [])
            top_bert_jobs = data.get("top_bert_jobs", [])

            # Modified validation
            if not isinstance(cv_data, dict) or not isinstance(top_cosine_jobs, list) or not isinstance(top_bert_jobs,
                                                                                                        list):
                return Response({"error": "Invalid JSON structure."}, status=400)

            if not top_cosine_jobs or not top_bert_jobs:
                return Response({"error": "Job lists cannot be empty."}, status=400)

            # Function to safely truncate text
            def safe_truncate(text, length=500):
                if text and isinstance(text, str):
                    return text[:length] + "..." if len(text) > length else text
                return "Not available"

            # Prepare the prompt for the AI judge
            openai_prompt = f"""
                    You are an impartial AI judge for NextMatch, a job recommender system. Your task is to evaluate how well a candidate's 
                    resume matches job vacancies. The system has already calculated similarity scores using two different methods.

                    Input:
                    Candidate's CV: 
                    {cv_data.get("skills", "")} {cv_data.get("experience", "")} {cv_data.get("education", "")}

                    Method A - Top 3 Job Matches (TF-IDF with Cosine Similarity):
                    1. {top_cosine_jobs[0].get("job_position", "Not Available")} at {top_cosine_jobs[0].get("company_name", "Not Available")}
                       Description: {safe_truncate(top_cosine_jobs[0].get("job_description", "No description available"))}

                    2. {top_cosine_jobs[1].get("job_position", "Not Available")} at {top_cosine_jobs[1].get("company_name", "Not Available")}
                       Description: {safe_truncate(top_cosine_jobs[1].get("job_description", "No description available"))}

                    3. {top_cosine_jobs[2].get("job_position", "Not Available")} at {top_cosine_jobs[2].get("company_name", "Not Available")}
                       Description: {safe_truncate(top_cosine_jobs[2].get("job_description", "No description available"))}

                    Method B - Top 3 Job Matches (BERT Semantic Embeddings):
                    1. {top_bert_jobs[0].get("job_position", "Not Available")} at {top_bert_jobs[0].get("company_name", "Not Available")}
                       Description: {safe_truncate(top_bert_jobs[0].get("job_description", "No description available"))}

                    2. {top_bert_jobs[1].get("job_position", "Not Available")} at {top_bert_jobs[1].get("company_name", "Not Available")}
                       Description: {safe_truncate(top_bert_jobs[1].get("job_description", "No description available"))}

                    3. {top_bert_jobs[2].get("job_position", "Not Available")} at {top_bert_jobs[2].get("company_name", "Not Available")}
                       Description: {safe_truncate(top_bert_jobs[2].get("job_description", "No description available"))}

                    Please provide an analysis in **valid JSON format** with the following structure:
                    Method A Scores:
                    X
                    X
                    X
                    Method B Scores:
                    X
                    X
                    X
                    """

            response = client.chat.completions.create(
                model="gpt-4",  # Use a model that supports JSON output
                messages=[{"role": "user", "content": openai_prompt}],
            )

            # Debug the OpenAI response
            print("OpenAI Response Content:", response.choices[0].message.content)

            # Parse the AI recommendation
            try:
                ai_recommendation = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                ai_recommendation = {
                    "error": "The AI recommendation could not be parsed as JSON.",
                    "raw_response": response.choices[0].message.content
                }

            # Assign internal IDs to jobs (1-500)
            def assign_internal_ids(jobs):
                return {i + 1: job for i, job in enumerate(jobs)}

            method_a_jobs_with_ids = assign_internal_ids(top_cosine_jobs)
            method_b_jobs_with_ids = assign_internal_ids(top_bert_jobs)

            # Replace job names in ai_recommendation with internal IDs
            def replace_job_names_with_ids(ai_recommendation, method_a_jobs_with_ids, method_b_jobs_with_ids):
                updated_recommendation = {"Method A Scores": {}, "Method B Scores": {}}

                # Replace Method A job names with IDs
                for job_name, score in ai_recommendation.get("Method A Scores", {}).items():
                    for job_id, job in method_a_jobs_with_ids.items():
                        if job_name == f"{job['job_position']} at {job['company_name']}":
                            updated_recommendation["Method A Scores"][job_id] = score
                            break

                # Replace Method B job names with IDs
                for job_name, score in ai_recommendation.get("Method B Scores", {}).items():
                    for job_id, job in method_b_jobs_with_ids.items():
                        if job_name == f"{job['job_position']} at {job['company_name']}":
                            updated_recommendation["Method B Scores"][job_id] = score
                            break

                return updated_recommendation

            ai_recommendation_with_ids = replace_job_names_with_ids(ai_recommendation, method_a_jobs_with_ids,
                                                                    method_b_jobs_with_ids)

            # Save the AI recommendation to the ai_recommendation_json folder
            ai_recommendation_folder = "ai_recommendation_json"
            if not os.path.exists(ai_recommendation_folder):
                os.makedirs(ai_recommendation_folder)

            # Generate a unique filename for the recommendation
            recommendation_filename = f"recommendation_{uuid.uuid4().hex}.json"
            recommendation_file_path = os.path.join(ai_recommendation_folder, recommendation_filename)

            # Save the AI recommendation with IDs
            with open(recommendation_file_path, "w") as f:
                json.dump({
                    "cv": cv_data,  # Include CV data at the top
                    "top_cosine_jobs": top_cosine_jobs,
                    "top_bert_jobs": top_bert_jobs,
                    "ai_recommendation": ai_recommendation_with_ids
                }, f, indent=4)

            return Response({
                "cv": cv_data,  # Include CV data at the top
                "top_cosine_jobs": top_cosine_jobs,
                "top_bert_jobs": top_bert_jobs,
                "ai_recommendation": ai_recommendation_with_ids
            }, status=200)

        except Exception as e:
            # Add comprehensive error handling
            import traceback
            return Response({
                "error": f"An error occurred during job ranking: {str(e)}",
                "traceback": traceback.format_exc()
            }, status=500)


def create(self, request):
    """
    Handle the file upload, parse the CV for information, and save it.
    """
    serializer = CvSerializer(data=request.data)

    if serializer.is_valid():
        cv_instance = serializer.save()

        cv_file_path = cv_instance.cv_file.path
        cv_text = extract_text_from_pdf(cv_file_path)

        skills_keywords = ["skills", "technologies", "expertise"]
        education_keywords = ["education", "university", "degree"]
        experience_keywords = ["experience", "worked at", "role", "job"]

        # Extract all sections at once
        cv_instance.skills = extract_section(cv_text, skills_keywords)
        cv_instance.education = extract_section(cv_text, education_keywords)
        cv_instance.experience = extract_section(cv_text, experience_keywords)
        cv_instance.save()

        return Response({
            "full_name": cv_instance.full_name,
            "email": cv_instance.email,
            "phone_number": cv_instance.phone_number,
            "location": cv_instance.location,
            "skills": cv_instance.skills,
            "education": cv_instance.education,
            "experience": cv_instance.experience,
            "cv_file": cv_instance.cv_file.url
        }, status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@action(detail=False, methods=["get"], url_path="export-top-jobs")
def export_top_jobs(self, request):
    """
    Export the top 3 jobs from both ranking methods (TF-IDF and BERT) along with the CV to a file.
    """
    try:
        # Fetch the latest CV
        user_cv = Cv.objects.order_by('-uploaded_at').first()
        if not user_cv:
            return Response({"error": "No CV found."}, status=404)

        # Get CV text for both methods
        cv_text = f"{user_cv.skills} {user_cv.experience} {user_cv.education}"

        # Get jobs with meaningful descriptions
        jobs = JobPosition.objects.exclude(job_description__isnull=True) \
            .exclude(job_description__exact="") \
            .exclude(job_description__icontains="No description available") \
            .exclude(job_description__icontains="Fetching description...")

        if not jobs.exists():
            return Response({"error": "No jobs with meaningful descriptions found."}, status=404)

        # Deduplicate jobs based on position, company, and description
        unique_jobs = {}
        for job in jobs:
            key = (job.job_position, job.company_name, job.job_description)
            if key not in unique_jobs:
                unique_jobs[key] = job
        jobs = list(unique_jobs.values())

        # Create text representations for all jobs
        job_texts = []
        job_objects = []

        for job in jobs:
            job_text = f"{job.job_description} {job.job_function or ''} {job.industries or ''}"
            job_texts.append(job_text)
            job_objects.append(job)

        # METHOD A: TF-IDF + Cosine Similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        all_texts = [cv_text] + job_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        cv_vector = tfidf_matrix[0:1]
        job_vectors = tfidf_matrix[1:]
        cosine_similarities = sklearn_cosine_similarity(cv_vector, job_vectors).flatten()
        cosine_scores = [(score, job_objects[i]) for i, score in enumerate(cosine_similarities)]
        cosine_scores.sort(key=lambda x: x[0], reverse=True)

        # Get top 3 TF-IDF jobs
        top_cosine_jobs = []
        top_cosine_job_ids = set()
        for score, job in cosine_scores:
            if job.id not in top_cosine_job_ids:
                top_cosine_job_ids.add(job.id)
                top_cosine_jobs.append((score, job))
                if len(top_cosine_jobs) >= 3:
                    break

        # METHOD B: BERT Embeddings
        cv_embedding = get_bert_embedding(cv_text)
        bert_scores = []
        for i, job_text in enumerate(job_texts):
            job_embedding = get_bert_embedding(job_text)
            similarity = torch.nn.functional.cosine_similarity(cv_embedding, job_embedding)
            bert_scores.append((similarity.item(), job_objects[i]))
        bert_scores.sort(key=lambda x: x[0], reverse=True)

        # Get top 3 BERT jobs
        top_bert_jobs = []
        top_bert_job_ids = set()
        for score, job in bert_scores:
            if job.id not in top_bert_job_ids:
                top_bert_job_ids.add(job.id)
                top_bert_jobs.append((score, job))
                if len(top_bert_jobs) >= 3:
                    break

        # Prepare data for export
        export_data = {
            "cv": {
                "full_name": user_cv.full_name,
                "skills": user_cv.skills,
                "experience": user_cv.experience,
                "education": user_cv.education,
            },
            "top_cosine_jobs": [
                {
                    "job_position": job[1].job_position,
                    "company_name": job[1].company_name,
                    "job_description": job[1].job_description,
                    "score": job[0],
                }
                for job in top_cosine_jobs
            ],
            "top_bert_jobs": [
                {
                    "job_position": job[1].job_position,
                    "company_name": job[1].company_name,
                    "job_description": job[1].job_description,
                    "score": job[0],
                }
                for job in top_bert_jobs
            ],
        }

        # Save to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"top_jobs_export_{timestamp}.json"
        filepath = os.path.join("exports", filename)

        # Ensure the exports directory exists
        os.makedirs("exports", exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=4)

        # Return the file as a downloadable response
        with open(filepath, "rb") as f:
            response = HttpResponse(f.read(), content_type="application/json")
            response["Content-Disposition"] = f"attachment; filename={filename}"
            return response

    except Exception as e:
        import traceback
        return Response({
            "error": f"An error occurred during export: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


@action(detail=False, methods=["post"], url_path="fetch-job")
def fetch_job(self, request):
    """
    Fetch job data using the scrapingDog API for multiple fields and locations.
    """
    # Broaden the search criteria
    fields = [
        "software engineering", "web development", "data science",
        "machine learning", "artificial intelligence", "cloud computing",
        "cybersecurity", "mobile development", "devops", "backend development",
        "frontend development", "full stack development"
    ]
    locations = ["103420483"]  # Example location IDs
    page = request.data.get("page", 1)  # Allow dynamic pagination

    url = "https://api.scrapingdog.com/linkedinjobs"
    all_jobs = []  # Store all jobs fetched from the API

    # Fetch jobs for all combinations of fields and locations
    for field in fields:
        for location in locations:
            params = {
                "api_key": api_key,
                "field": field,
                "geoid": location,
                "page": page,
                "date_posted": "past_week",  # Broaden date range if needed
                "experience_level": "entry_level"  # Include other experience levels if needed
            }

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list):
                    all_jobs.extend(data)  # Add jobs to the master list
                else:
                    logger.error(f"Unexpected response format for field={field}, location={location}: {data}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request to ScrapingDog API failed for field={field}, location={location}: {e}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON for field={field}, location={location}: {response.text}")

    # If no jobs were fetched, return an error
    if not all_jobs:
        return Response({
            "error": "No jobs fetched from the ScrapingDog API for the given criteria."
        }, status=status.HTTP_404_NOT_FOUND)

    # Process and store job details
    processed_job_ids = set()
    detailed_jobs = []

    for job in all_jobs:
        job_id = job.get("job_id")
        if not job_id or job_id in processed_job_ids:
            continue  # Skip duplicates

        processed_job_ids.add(job_id)

        # Fetch detailed job information
        job_details_url = "https://api.scrapingdog.com/linkedinjobs"
        job_details_params = {
            "api_key": api_key,
            "job_id": job_id
        }

        try:
            job_details_response = requests.get(job_details_url, params=job_details_params)
            job_details_response.raise_for_status()
            job_details = job_details_response.json()

            if isinstance(job_details, list) and job_details:
                job_details = job_details[0]  # Use the first item if it's a list

            # Create job data dictionary
            job_data = {
                "job_position": job_details.get("job_position", "Not Available"),
                "job_link": job_details.get("job_link", "Not Available"),
                "company_name": job_details.get("company_name", "Not Available"),
                "job_location": job_details.get("job_location", "Not Available"),
                "job_posting_date": job_details.get("job_posting_time", "Not Available"),
                "job_description": job_details.get("job_description", "No description available"),
                "seniority_level": job_details.get("seniority_level", "Not Available"),
                "employment_type": job_details.get("employment_type", "Not Available"),
                "job_function": job_details.get("job_function", "Not Available"),
                "industries": job_details.get("industries", "Not Available"),
                "job_apply_link": job_details.get("job_apply_link", "Not Available"),
            }

            # Save job to the database
            job_instance = JobPosition.objects.create(**job_data)
            detailed_jobs.append(job_instance)

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logger.error(f"Error fetching details for job {job_id}: {e}")
            continue

    return Response(
        {"jobs": f"Successfully processed {len(detailed_jobs)} jobs."},
        status=status.HTTP_200_OK
    )


@action(detail=False, methods=["get"], url_path="rank-jobs")
def rank_jobs(self, request):
    """
    Rank jobs using two distinct methods:
    Method A: Pure cosine similarity on TF-IDF vectors
    Method B: BERT embeddings with semantic similarity
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not os.getenv("OPENAI_API_KEY"):
            return Response({"error": "OpenAI API key not configured"}, status=500)

        user_cv = Cv.objects.order_by('-uploaded_at').first()
        if not user_cv:
            return Response({"error": "No CV found."}, status=404)

        # Get CV text for both methods
        cv_text = f"{user_cv.skills} {user_cv.experience} {user_cv.education}"

        # Get jobs with meaningful descriptions
        jobs = JobPosition.objects.exclude(job_description__isnull=True) \
            .exclude(job_description__exact="") \
            .exclude(job_description__icontains="No description available") \
            .exclude(job_description__icontains="Fetching description...")

        if not jobs.exists():
            return Response({"error": "No jobs with meaningful descriptions found."}, status=404)

        # Deduplicate jobs based on position, company, and description
        unique_jobs = {}
        for job in jobs:
            key = (job.job_position, job.company_name, job.job_description)
            if key not in unique_jobs:
                unique_jobs[key] = job
        jobs = list(unique_jobs.values())

        # Create text representations for all jobs
        job_texts = []
        job_objects = []

        for job in jobs:
            job_text = f"{job.job_description} {job.job_function or ''} {job.industries or ''}"
            job_texts.append(job_text)
            job_objects.append(job)

        # METHOD A: TF-IDF + Cosine Similarity (traditional NLP approach)
        vectorizer = TfidfVectorizer(stop_words='english')

        # Create document-term matrix
        all_texts = [cv_text] + job_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Get CV vector (first row) and job vectors (remaining rows)
        cv_vector = tfidf_matrix[0:1]
        job_vectors = tfidf_matrix[1:]

        # Calculate cosine similarity between CV and all jobs
        cosine_similarities = sklearn_cosine_similarity(cv_vector, job_vectors).flatten()

        # Create sorted list of (similarity, job) pairs
        cosine_scores = [(score, job_objects[i]) for i, score in enumerate(cosine_similarities)]
        cosine_scores.sort(key=lambda x: x[0], reverse=True)

        # Prevent duplicate recommendations
        top_cosine_job_ids = set()
        unique_cosine_scores = []

        for score, job in cosine_scores:
            if job.id not in top_cosine_job_ids:
                top_cosine_job_ids.add(job.id)
                unique_cosine_scores.append((score, job))
                if len(unique_cosine_scores) >= 3:
                    break

        top_cosine_jobs = unique_cosine_scores

        # Check if we have enough unique jobs
        if len(top_cosine_jobs) < 3:
            return Response({"error": "Not enough unique jobs to rank."}, status=404)

        # METHOD B: BERT Embeddings with Semantic Similarity
        # Get BERT embedding for CV
        cv_embedding = get_bert_embedding(cv_text)

        bert_scores = []
        for i, job_text in enumerate(job_texts):
            # Get BERT embedding for job
            job_embedding = get_bert_embedding(job_text)

            # Calculate similarity using BERT embeddings
            similarity = torch.nn.functional.cosine_similarity(cv_embedding, job_embedding)
            bert_scores.append((similarity.item(), job_objects[i]))

        # Sort by the similarity score
        bert_scores.sort(key=lambda x: x[0], reverse=True)

        # Prevent duplicate recommendations
        top_bert_job_ids = set()
        unique_bert_scores = []

        for score, job in bert_scores:
            if job.id not in top_bert_job_ids:
                top_bert_job_ids.add(job.id)
                unique_bert_scores.append((score, job))
                if len(unique_bert_scores) >= 3:
                    break

        top_bert_jobs = unique_bert_scores

        # Function to safely truncate text
        def safe_truncate(text, length=500):
            if text and isinstance(text, str):
                return text[:length] + "..." if len(text) > length else text
            return "Not available"

        # Normalize scores for better comparison
        def normalize_scores(scores, min_val=0, max_val=100):
            if not scores:
                return []
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:  # Avoid division by zero
                return [max_val] * len(scores)
            return [min_val + (score - min_score) * (max_val - min_val) / (max_score - min_score) for score in
                    scores]

        # Extract just the scores for normalization
        cosine_scores_values = [score for score, _ in top_cosine_jobs]
        bert_scores_values = [score for score, _ in top_bert_jobs]

        # Normalize scores
        normalized_cosine = normalize_scores(cosine_scores_values)
        normalized_bert = normalize_scores(bert_scores_values)

        # Prepare the prompt for the AI judge with normalized scores
        openai_prompt = f"""
            You are an impartial AI judge for NextMatch, a job recommender system. Your task is to evaluate how well a candidate's 
            resume matches job vacancies. The system has already calculated similarity scores (0-100) using two different methods.

            Input:
            Candidate's CV: 
            {cv_text}

            Method A - Top 3 Job Matches (TF-IDF with Cosine Similarity):
            1. {top_cosine_jobs[0][1].job_position} at {top_cosine_jobs[0][1].company_name} (Score: {normalized_cosine[0]:.2f})
               Description: {safe_truncate(top_cosine_jobs[0][1].job_description)}

            2. {top_cosine_jobs[1][1].job_position} at {top_cosine_jobs[1][1].company_name} (Score: {normalized_cosine[1]:.2f})
               Description: {safe_truncate(top_cosine_jobs[1][1].job_description)}

            3. {top_cosine_jobs[2][1].job_position} at {top_cosine_jobs[2][1].company_name} (Score: {normalized_cosine[2]:.2f})
               Description: {safe_truncate(top_cosine_jobs[2][1].job_description)}

            Method B - Top 3 Job Matches (BERT Semantic Embeddings):
            1. {top_bert_jobs[0][1].job_position} at {top_bert_jobs[0][1].company_name} (Score: {normalized_bert[0]:.2f})
               Description: {safe_truncate(top_bert_jobs[0][1].job_description)}

            2. {top_bert_jobs[1][1].job_position} at {top_bert_jobs[1][1].company_name} (Score: {normalized_bert[1]:.2f})
               Description: {safe_truncate(top_bert_jobs[1][1].job_description)}

            3. {top_bert_jobs[2][1].job_position} at {top_bert_jobs[2][1].company_name} (Score: {normalized_bert[2]:.2f})
               Description: {safe_truncate(top_bert_jobs[2][1].job_description)}

            Please provide an analysis in **valid JSON format** with the following structure:
            {{
              "methodAssessment": {{
                "betterMethod": "A or B",
                "justification": "Explanation of which method provided better matches"
              }},
              "adjustedMatchScores": {{
                "1": {{
                  "jobTitle": "Job Title",
                  "adjustedScore": 0-100
                }},
                "2": {{
                  "jobTitle": "Job Title",
                  "adjustedScore": 0-100
                }},
                "3": {{
                  "jobTitle": "Job Title",
                  "adjustedScore": 0-100
                }}
              }},
              "bestMatch": {{
                "jobTitle": "Job Title",
                "justification": {{
                  "skillsAlignment": "Explanation of skills alignment",
                  "experienceRelevance": "Explanation of experience relevance",
                  "careerGrowthPotential": "Explanation of career growth potential"
                }}
              }}
            }}
            """

        response = client.chat.completions.create(
            model="gpt-4",  # Use a model that supports JSON output
            messages=[{"role": "user", "content": openai_prompt}],
        )

        # Debug the OpenAI response
        import json
        print("OpenAI Response Content:", response.choices[0].message.content)

        # Parse the AI recommendation
        try:
            ai_recommendation = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            ai_recommendation = {
                "error": "The AI recommendation could not be parsed as JSON.",
                "raw_response": response.choices[0].message.content
            }

        # Format the results for the frontend
        method_a_jobs = [
            {
                "job_position": job[1].job_position,
                "company_name": job[1].company_name,
                "job_description": safe_truncate(job[1].job_description, 300),
                "raw_score": job[0],
                "score": round(normalized_cosine[i], 2),  # Use normalized scores
                "apply_link": job[1].job_apply_link,
                "job_id": job[1].id
            }
            for i, job in enumerate(top_cosine_jobs)
        ]

        method_b_jobs = [
            {
                "job_position": job[1].job_position,
                "company_name": job[1].company_name,
                "job_description": safe_truncate(job[1].job_description, 300),
                "raw_score": job[0],
                "score": round(normalized_bert[i], 2),  # Use normalized scores
                "apply_link": job[1].job_apply_link,
                "job_id": job[1].id
            }
            for i, job in enumerate(top_bert_jobs)
        ]

        return Response({
            "method_a_jobs": method_a_jobs,
            "method_b_jobs": method_b_jobs,
            "ai_recommendation": ai_recommendation,  # Return parsed JSON or error
        }, status=200)

    except Exception as e:
        # Add comprehensive error handling
        import traceback
        return Response({
            "error": f"An error occurred during job ranking: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)
