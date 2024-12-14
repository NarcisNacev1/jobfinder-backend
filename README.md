---

# **Job Finder Backend**

This repository contains the backend implementation of the **Job Finder Application**, which uses a Django REST framework to manage CV uploads, fetch job data from external APIs, and rank job positions based on the relevance to a user's CV.

---

## **Features**

1. **CV Upload:**
   - Users can upload their CV in PDF format.
   - Extracts skills, education, and experience from the CV using keyword-based text extraction.

2. **Job Fetching:**
   - Integrates with the **ScrapingDog API** to fetch job postings.
   - Stores job data in the database for efficient access.

3. **Job Ranking:**
   - Ranks job positions based on their relevance to the uploaded CV.
   - Uses **TF-IDF** and **cosine similarity** for relevance scoring.
   - Ensures unique positions for the same company are not duplicated.

---

## **Installation**

### Prerequisites
- Python 3.8 or higher
- Django
- Django REST Framework
- pdfplumber
- Requests

### Steps
1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   Create a `.env` file in the project root and add the following:
   ```
   API_KEY=<your_scrapingdog_api_key>
   ```

5. **Apply Database Migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Run the Development Server:**
   ```bash
   python manage.py runserver
   ```

---

## **Endpoints**

### **CV Endpoints**
1. **Upload CV**
   - **URL:** `/api/cv/`
   - **Method:** `POST`
   - **Description:** Uploads a CV and extracts relevant details.
   - **Payload:**
     ```json
     {
       "full_name": "John Doe",
       "email": "john.doe@example.com",
       "phone_number": "123456789",
       "location": "New York",
       "cv_file": "<PDF file>"
     }
     ```

2. **Fetch Job Data**
   - **URL:** `/api/cv/fetch-job/`
   - **Method:** `POST`
   - **Description:** Fetches job postings from the ScrapingDog API and stores them in the database.
   - **Payload:**
     ```json
     {
       "field": "programming",
       "geoid": "103420483",
       "page": 1
     }
     ```

3. **Rank Jobs**
   - **URL:** `/api/cv/rank-jobs/`
   - **Method:** `GET`
   - **Description:** Ranks job positions based on their relevance to the most recently uploaded CV.

---

## **Code Structure**

### **Core Files**

1. **`views.py`**
   - Contains the `CvUpload` viewset for managing CV uploads, job fetching, and job ranking.
   - Implements logic for parsing CVs, fetching external job data, and calculating relevance scores.

2. **`models.py`**
   - Defines the `Cv` and `JobPosition` models:
     - `Cv`: Stores details of the uploaded CV, including extracted skills, education, and experience.
     - `JobPosition`: Stores job posting details fetched from the ScrapingDog API.

3. **`serializers.py`**
   - Handles serialization for the `Cv` model to validate and transform API requests.

4. **`utils.py`**
   - Implements helper functions:
     - `extract_text_from_pdf`: Extracts raw text from a PDF file.
     - `extract_section`: Filters specific sections of the text based on keywords.

5. **`urls.py`**
   - Configures API endpoints for the backend application.

---

## **Key Algorithms**

1. **TF-IDF Calculation:**
   - Computes term frequency (TF) and inverse document frequency (IDF) for words in the CV and job descriptions.

2. **Cosine Similarity:**
   - Measures similarity between the CV and job postings based on vector space.

3. **Priority Queue:**
   - Ensures jobs are ranked by relevance scores in descending order.

4. **Duplicate Removal:**
   - Filters out duplicate job entries for the same position and company.

---

## **Sample API Workflow**

1. **Upload CV:**
   - The user uploads their CV using the `/api/cv/` endpoint.

2. **Fetch Jobs:**
   - The user triggers a job fetch using the `/api/cv/fetch-job/` endpoint.
   - Jobs are stored in the database.

3. **Rank Jobs:**
   - The user retrieves ranked jobs using the `/api/cv/rank-jobs/` endpoint.
   - The system responds with a list of job positions ranked by relevance.

---

## **Technologies Used**
- **Framework:** Django REST Framework
- **API Integration:** ScrapingDog
- **PDF Processing:** pdfplumber
- **Database:** SQLite (can be swapped with PostgreSQL)

---

## **Future Enhancements**
- Add user authentication for personalized CV management.
- Support advanced filtering for job ranking.
- Implement live updates for job postings.

---

## **Contributing**
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch.
3. Make your changes and test them.
4. Submit a pull request.

