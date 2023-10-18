# AlumnAI-Job-Recommandation-App


## Overview

The **AlumnAI Job Recommendation App** is a powerful platform designed to provide job recommendations and detailed profiling for users based on their CV in PDF format or Digital format (video) . This app goes beyond job recommendations; it offers insights into key performance indicators (KRPIs) around suggested companies, average salary ranges, and the skills required for each job. Additionally, it assists administrators in finding potential partners for universities or clients.

The AlumnAI Job Recommendation App is deployed on Streamlit, ensuring an intuitive and accessible user experience.

## Features

1. **User Authentication**: Users can create accounts, log in, and securely manage their profiles.

2. **Profile Creation and Analysis**: Users can input their skills, experience, education, and location preferences. The app generates profiles of individuals with similar qualifications for comparison.

3. **Job Search and Recommendations**: The app employs a recommendation algorithm to match user profiles with relevant job listings. Each job recommendation includes key information about the job role.

4. **Company Insights**: Users can access key performance indicators (KRPIs) related to suggested companies, helping them make informed decisions.

5. **Salary Ranges**: The app provides average salary ranges for different job roles based on available data.

6. **Required Skills**: Users can view the skills required for each suggested job, helping them understand the qualifications needed.

7. **Admin Interface**: Administrators can access a dedicated dashboard for managing partner recommendations for universities or clients.

8. **Partner Recommendations**: The app assists in finding potential partners based on specified criteria, enhancing collaboration opportunities.

## Installation

1. **Clone Repository**: Clone this repository to your local machine.

```bash
git clone https://github.com/yourusername/AlumnAI-Job-Recommandation-App.git
```
2. **Create Virtual Environment**: Navigate to the project directory and create a virtual environment.
```bash

cd AlumnAI-Job-Recommandation-App 
python -m venv env
```
3. **Activate Virtual Environment**: Activate the virtual environment.
On Windows:
```bash

.\env\Scripts\activate
```
On macOS and Linux:
```bash

source env/bin/activate
```
4. **Install Dependencies**: Install the required packages.
```bash

pip install -r requirements.txt
```
##Usage
 1.**Run the App**: Start the Streamlit app.
```bash
streamlit run app.py
```
1. **Access the App**: Open a web browser and navigate to your localhost ( displayed by streamlit ).

2. **Create an Account**: Sign up with a username and password.

3. **Complete Profile**: Provide details about your skills, experience, education, location preferences, and job industry interests by submitting a CV in pdf format or in digital format (video).

4. **Explore Job Recommendations**: Browse through the recommended job listings.

5. **Profile Analysis**: View profiles of individuals with similar qualifications for comparison.

6. **Company Insights**: Access key performance indicators (KRPIs) for suggested companies.

7. **Salary Ranges**: Explore average salary ranges for different job roles.

8. **Required Skills**: Understand the skills needed for each suggested job.

9. **Admin Dashboard (For Administrators)**: Access the admin interface to manage partner recommendations.
## Demonstration
[DEMO VIDEO](https://drive.google.com/file/d/1B4TiKbMjn7BYopfp-q1IQWrOA2KJcr8j/view?usp=sharing)
## Dependencies

- Streamlit
- Pandas
- Numpy
- Scikit-learn
- ...

## Contributing

If you'd like to contribute to this project, please follow the contribution guidelines.
