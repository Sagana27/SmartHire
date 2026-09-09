from fpdf import FPDF

candidates = [
    (
        "resume_data_scientist.pdf",
        "Elena Rostova - Data Scientist",
        "Data Scientist with experience in predictive modeling, deep learning, and statistical analysis.",
        "Python, SQL, R, Pandas, NumPy, Scikit-Learn, TensorFlow, PyTorch, Deep Learning, NLP, Tableau, Statistics",
        "Senior AI Specialist at DeepMetrics. Built regression and NLP pipelines using PyTorch and Scikit-Learn."
    ),
    (
        "resume_java_developer.pdf",
        "Marcus Vance - Java Backend Developer",
        "Backend Software Engineer specializing in distributed systems, Spring Boot microservices, and APIs.",
        "Java, Spring Boot, SQL, Docker, Kubernetes, Git, AWS, Spark",
        "Lead Backend Developer at FinServe Corp. Architected enterprise REST APIs in Java with Spring Boot."
    ),
    (
        "resume_devops_engineer.pdf",
        "Sarah Jenkins - DevOps Engineer",
        "DevOps Engineer focused on Kubernetes orchestration, AWS cloud infrastructure, and CI/CD automation.",
        "Docker, Kubernetes, AWS, Git, Python, Hadoop, Spark",
        "Infrastructure Engineer at CloudWorks. Configured multi-region Kubernetes clusters on AWS."
    )
]

for filename, name, summary, skills, experience in candidates:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, name, ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Email: candidate@example.com | Location: Remote", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Professional Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, summary)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Technical Skills", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, skills)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Experience", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, experience)
    pdf.output(filename)
    print(f"Generated {filename}")

