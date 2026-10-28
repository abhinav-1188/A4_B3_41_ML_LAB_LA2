import os
from fpdf import FPDF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, 'test_sample_resumes')
os.makedirs(TEST_DIR, exist_ok=True)


class ResumePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 60, 110)
        self.cell(0, 8, self.title_text, ln=True, align="C")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, self.contact_text, ln=True, align="C")
        self.ln(4)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 60, 110)
        self.cell(0, 6, title.upper(), ln=True)
        self.set_draw_color(40, 90, 160)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text)
        self.ln(3)


def create_sample_resume(filename, title, contact, summary, skills, experience, education):
    pdf = ResumePDF()
    pdf.title_text = title
    pdf.contact_text = contact
    pdf.add_page()

    pdf.section_title("Professional Summary")
    pdf.body_text(summary)

    pdf.section_title("Technical & Core Competencies")
    pdf.body_text(skills)

    pdf.section_title("Work Experience & Projects")
    pdf.body_text(experience)

    pdf.section_title("Education & Certifications")
    pdf.body_text(education)

    filepath = os.path.join(TEST_DIR, filename)
    pdf.output(filepath)
    print(f"Generated: {filepath}")


def main():
    # 1. Data Scientist Resume
    create_sample_resume(
        filename="Sample_Data_Scientist_Resume.pdf",
        title="ARJUN SHARMA - DATA SCIENTIST",
        contact="Email: arjun.sharma@example.com | Phone: +91 9876543210 | Bangalore, India | LinkedIn: linkedin.com/in/arjun-ds",
        summary="Results-driven Data Scientist with 4+ years of experience delivering machine learning solutions, predictive statistical modeling, and deep learning algorithms. Proficient in transforming massive structured and unstructured datasets into actionable business intelligence.",
        skills="Programming & Tools: Python, R, SQL, Pandas, NumPy, Scikit-Learn, TensorFlow, PyTorch, Keras, Matplotlib, Seaborn, Tableau.\nMachine Learning: Linear Regression, Random Forest, Support Vector Machines (SVM), K-Means Clustering, NLP, TF-IDF, Word2Vec, XGBoost.\nCloud & Big Data: AWS S3, EC2, Apache Spark, Docker, Git, CI/CD.",
        experience="Lead Data Scientist | Analytics Corp (2022 - Present)\n- Developed automated customer churn prediction pipelines using Random Forest and XGBoost achieving 92% ROC-AUC.\n- Engineered NLP resume screening and sentiment analysis engines with TF-IDF and BERT embeddings, reducing screening latency by 65%.\n- Deployed production ML inference APIs using FastAPI and Streamlit dashboards on AWS cloud.\n\nMachine Learning Engineer | Tech Innovators (2020 - 2022)\n- Designed exploratory data analysis (EDA) and automated outlier detection pipelines.\n- Built hyperparameter tuning pipelines using GridSearchCV and cross-validation for high-dimensional credit risk models.",
        education="Bachelor of Technology in Computer Science & Engineering | Top University (2016 - 2020)\nCertifications: DeepLearning.AI Machine Learning Specialization, AWS Certified Machine Learning Specialist."
    )

    # 2. Java Developer Resume
    create_sample_resume(
        filename="Sample_Java_Developer_Resume.pdf",
        title="ROHIT VERMA - SENIOR JAVA DEVELOPER",
        contact="Email: rohit.verma@example.com | Phone: +91 9811223344 | Pune, India | GitHub: github.com/rohit-java",
        summary="Senior Java Developer with 5 years of hands-on experience designing, developing, and deploying enterprise-grade distributed microservices using Spring Boot, Hibernate, and RESTful APIs. Expertise in scalable architectures and multithreaded backend systems.",
        skills="Languages & Frameworks: Core Java, J2EE, Spring Boot, Spring MVC, Spring Security, Hibernate ORM, JPA, Microservices.\nWeb & Database: REST APIs, SOAP, MySQL, PostgreSQL, Oracle DB, Redis, MongoDB.\nDevOps & Tools: Maven, Gradle, Docker, Kubernetes, Jenkins, Git, JIRA, JUnit, Mockito.",
        experience="Senior Backend Engineer | FinTech Global Solutions (2021 - Present)\n- Architected high-throughput payment processing microservices using Spring Boot and Apache Kafka handling 50,000 requests per minute.\n- Implemented database optimization, indexing, and connection pooling in PostgreSQL reducing query execution time by 40%.\n- Integrated OAuth2 and JWT authentication across distributed microservices.\n\nSoftware Engineer | Enterprise Soft (2019 - 2021)\n- Developed Core Java multithreaded order-matching engines and REST APIs.\n- Authored comprehensive unit and integration test suites using JUnit and Mockito achieving 90% code coverage.",
        education="B.E. in Information Technology | State Technical University (2015 - 2019)\nCertifications: Oracle Certified Professional (OCP) Java SE 11 Developer."
    )

    # 3. HR Manager Resume
    create_sample_resume(
        filename="Sample_HR_Manager_Resume.pdf",
        title="PRIYA NAIR - HUMAN RESOURCES MANAGER",
        contact="Email: priya.nair@example.com | Phone: +91 9765432109 | Mumbai, India",
        summary="Dynamic HR Professional with 6+ years of experience leading talent acquisition, employee relations, performance management, human capital strategy, and organizational development in fast-paced corporate environments.",
        skills="HR Core Competencies: End-to-End Talent Acquisition, Technical Recruitment, Employee Engagement, Onboarding, HR Compliance, Payroll Management.\nTools & Systems: Workday, SAP SuccessFactors, BambooHR, Taleo, LinkedIn Recruiter, MS Office, Advanced Excel.\nSoft Skills: Conflict Resolution, Leadership, Negotiation, Cross-Functional Team Building.",
        experience="Human Resources Manager | Apex Global Consulting (2021 - Present)\n- Spearheaded technical recruitment campaigns hiring 250+ software engineers, data scientists, and executives within strict deadlines.\n- Revamped the onboarding and induction experience, increasing new-hire retention by 28% year-over-year.\n- Administered bi-annual performance appraisal cycles and formulated corporate HR policies.\n\nAssistant HR Manager | PeopleFirst Solutions (2018 - 2021)\n- Screened hundreds of resumes, conducted initial HR behavioral interviews, and negotiated salary packages.\n- Implemented employee wellness initiatives and organized diversity, equity, and inclusion (DEI) workshops.",
        education="Master of Business Administration (MBA) in Human Resource Management (2016 - 2018)\nBachelor of Commerce | Mumbai University (2013 - 2016)"
    )

    # 4. DevOps Engineer Resume
    create_sample_resume(
        filename="Sample_DevOps_Engineer_Resume.pdf",
        title="VIKRAM PATEL - DEVOPS & CLOUD ARCHITECT",
        contact="Email: vikram.patel@example.com | Phone: +91 9988776655 | Hyderabad, India",
        summary="DevOps Engineer with 4+ years of expertise automating CI/CD pipelines, container orchestration, infrastructure as code (IaC), and cloud infrastructure monitoring across AWS and GCP environments.",
        skills="Cloud Platforms: AWS (EC2, S3, EKS, RDS, VPC, CloudWatch), Google Cloud Platform (GCP).\nCI/CD & Containers: Docker, Kubernetes, Helm, Jenkins, GitLab CI, GitHub Actions, ArgoCD.\nInfrastructure as Code: Terraform, Ansible, CloudFormation.\nScripting & Monitoring: Bash, Python scripting, Prometheus, Grafana, ELK Stack.",
        experience="Senior DevOps Engineer | CloudScale Networks (2022 - Present)\n- Architected production Kubernetes clusters on AWS EKS serving 10M+ daily active users with 99.99% uptime.\n- Automated multi-stage CI/CD pipelines using GitHub Actions reducing release cycle time from 2 days to 30 minutes.\n- Provisioned modular cloud infrastructure using Terraform, enforcing strict security compliance.\n\nCloud Operations Engineer | Infrastructure Hub (2020 - 2022)\n- Managed Docker containerization of legacy monolithic applications into microservices.\n- Configured real-time system health alerts using Prometheus and Grafana dashboards.",
        education="B.Tech in Computer Science | National Institute of Technology (2016 - 2020)\nCertifications: AWS Certified Solutions Architect - Associate, Certified Kubernetes Administrator (CKA)."
    )

    print("\nAll 4 test sample PDF resumes generated successfully in 'test_sample_resumes/' folder!")


if __name__ == '__main__':
    main()
