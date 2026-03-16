"""
generate_dataset.py
Creates a realistic synthetic fake_job_postings.csv (~3000 rows)
that mirrors the Kaggle 'Real or Fake Job Postings' dataset schema.
Run: python generate_dataset.py
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# ── Legitimate job templates ──────────────────────────────────────────────────
LEGIT_TITLES = [
    "Software Engineering Intern", "Data Science Intern", "Marketing Intern",
    "Business Development Intern", "UI/UX Design Intern", "Machine Learning Intern",
    "Content Writing Intern", "Finance Intern", "HR Intern", "Operations Intern",
    "Product Management Intern", "Cybersecurity Intern", "Cloud Engineer Intern",
    "Android Development Intern", "iOS Development Intern", "DevOps Intern",
    "Research Intern", "Sales Intern", "Customer Success Intern", "Graphic Design Intern",
    "Full Stack Developer Intern", "Backend Developer (Python) Intern",
    "Data Analyst Intern", "Digital Marketing Intern", "Supply Chain Intern",
]

LEGIT_COMPANIES = [
    "TechCorp Solutions Pvt. Ltd.", "InfoSys Technologies", "Wipro Limited",
    "Amazon Development Centre", "Microsoft India Pvt. Ltd.", "Google India",
    "Zomato Ltd.", "Swiggy", "Byju's", "Razorpay", "Freshworks", "Zoho Corporation",
    "Tata Consultancy Services", "HCL Technologies", "Mindtree Ltd.",
    "PayTM Payments Bank", "PhonePe Pvt. Ltd.", "Ola Cabs", "Flipkart India",
    "Meesho Inc.", "CRED", "Postman Inc.", "Browserstack", "Clevertap",
    "Nykaa E-Retail Ltd.", "Urban Company", "Dunzo", "Dream11", "Dailyhunt",
]

LEGIT_DESCRIPTIONS = [
    "We are looking for a motivated {title} to join our team. The intern will work on "
    "real-world projects alongside experienced engineers. You will participate in daily "
    "stand-ups, code reviews, and sprint planning. This is a paid internship with a "
    "duration of {duration}. Excellent performers may receive a pre-placement offer (PPO).",

    "Join our growing team as a {title}. You will be responsible for {responsibility}. "
    "You will collaborate closely with cross-functional teams including design, product, "
    "and business. We believe in a culture of learning and growth. Stipend: {stipend}/month.",

    "{company} is seeking a talented {title} for our {department} team. "
    "The selected candidate will gain hands-on experience in {domain}. We offer a "
    "structured internship program with regular mentoring sessions. Duration: {duration}.",

    "This internship offers an exciting opportunity to work on cutting-edge problems in "
    "{domain}. As a {title}, you will {responsibility}. Strong performers will be "
    "considered for full-time roles. Location: {location}. Stipend: {stipend}/month.",
]

LEGIT_REQUIREMENTS = [
    "Currently enrolled in B.Tech/B.E/MCA/BCA or equivalent. Strong knowledge of "
    "{skills}. Good communication skills. Ability to work in a team environment. "
    "Previous project experience preferred but not mandatory.",

    "Bachelor's/Master's degree in Computer Science or related field. Proficiency in "
    "{skills}. Problem-solving mindset. Eager to learn in a fast-paced environment.",

    "Strong foundation in {skills}. Experience with Git and version control. "
    "Understanding of software development lifecycle. Good analytical skills.",

    "Familiarity with {skills}. Ability to work independently. Strong attention to detail. "
    "Good written and verbal communication skills. Portfolio or GitHub profile preferred.",
]

LEGIT_PROFILES = [
    "We are a leading technology company with a mission to {mission}. "
    "Founded in {year}, we have grown to over {size} employees across {offices} offices. "
    "We are known for our innovative culture and employee-first approach.",

    "{company} is a fast-growing startup revolutionizing the {industry} industry. "
    "We are backed by top-tier VCs and have a presence in {offices}+ cities. "
    "Our team thrives on collaboration, creativity, and impact.",

    "Established in {year}, {company} has been at the forefront of digital transformation. "
    "We serve over {size} customers globally and are ranked among the Top 50 employers "
    "in India. We invest heavily in our employees' growth and development.",
]

# ── Fake job templates ────────────────────────────────────────────────────────
FAKE_TITLES = [
    "Work From Home Data Entry Operator", "Online Part Time Job",
    "Earn Money at Home — No Experience", "Freelance Form Filling Job",
    "Daily Payment Work From Home", "Easy Online Job — Immediate Joining",
    "Home Based Data Entry", "Online Typing Work — Fast Earnings",
    "Part Time Job for Students — Earn Daily", "Copy Paste Work From Home",
    "Guaranteed Income Work From Home", "Quick Earn Online Job",
    "Data Entry Executive — Work From Home", "Simple Online Job — No Investment (Paid)",
    "Virtual Assistant — Earn Per Hour Guaranteed",
]

FAKE_DESCRIPTIONS = [
    "URGENTLY HIRING! Work from home, no experience needed. Earn Rs.{salary} per month "
    "guaranteed. Just pay a small registration fee of Rs.{fee} to get your login credentials. "
    "Immediate joining. WhatsApp us now for more details. No interview required.",

    "Online part time job for students and housewives. You will need to fill simple forms "
    "online. Earn Rs.{salary}/week from home. Just pay Rs.{fee} one-time registration fee. "
    "Money will be refunded after first week. 100% genuine. Call us immediately.",

    "EARN MONEY DAILY WORKING FROM HOME! No qualification needed. Flexible hours. "
    "Salary: Rs.{salary}/month guaranteed. To confirm your seat, deposit Rs.{fee} registration "
    "charge. Slots filling fast! Do not wait. Click below or WhatsApp: +91-XXXXXXXXXX.",

    "Daily payment job! Work just {hours} hours per day from anywhere. "
    "Earn Rs.{salary} per week. First pay security deposit of Rs.{fee} which will be "
    "refunded with your first payout. 500 openings available. APPLY NOW.",

    "Genuine work from home opportunity. No target, no pressure. Free training provided. "
    "Salary: Rs.{salary}/month (guaranteed). Only requirement is to pay Rs.{fee} registration. "
    "Send your name, Aadhar card photo, and bank account details to start immediately.",
]

FAKE_REQUIREMENTS = [
    "No qualification required. Anyone can apply. Age 18-45. Must have a smartphone "
    "and internet connection. Must be willing to pay Rs.{fee} registration fee upfront.",

    "No experience needed. Housewives, students, retired persons welcome. "
    "Aadhar card and bank account mandatory. Pay small deposit to access work portal.",

    "Basic computer knowledge. Internet connection needed. "
    "Willing to pay security deposit of Rs.{fee}. Must share bank details for salary transfer.",

    "Only requirement: willingness to work hard. Pay Rs.{fee} to receive training material. "
    "No office visit required. Immediate joiners preferred.",
]

FAKE_PROFILES = [
    "",  # Fake jobs often have no company profile
    "XYZ Online Services — providing genuine work-from-home opportunities since {year}. "
    "Trusted by over {size} members. Government approved. 100% guaranteed payment.",
    "Our company offers legitimate online work to people across India. "
    "We are ISO certified and SEBI registered (false claims commonly used by scammers).",
    "",
]

# ── Helper data ────────────────────────────────────────────────────────────────
LOCATIONS = [
    "Mumbai, Maharashtra", "Bangalore, Karnataka", "Delhi, India",
    "Hyderabad, Telangana", "Chennai, Tamil Nadu", "Pune, Maharashtra",
    "Kolkata, West Bengal", "Ahmedabad, Gujarat", "Jaipur, Rajasthan",
    "Remote", "Work From Home", "Anywhere in India",
]

SALARY_RANGES = [
    "10000-15000", "15000-20000", "20000-25000", "25000-30000",
    "", "", "",  # Many legit jobs don't list salary range
]

TECH_SKILLS = [
    "Python, Machine Learning, TensorFlow",
    "Java, Spring Boot, MySQL",
    "JavaScript, React, Node.js",
    "Data Analysis, SQL, Tableau",
    "Android (Kotlin), REST APIs",
    "AWS, Docker, Kubernetes",
    "C++, Data Structures, Algorithms",
    "UI/UX Design, Figma, Adobe XD",
    "Digital Marketing, SEO, Google Analytics",
    "Content Writing, Research, Social Media",
]

RESPONSIBILITIES = [
    "develop and test software features",
    "analyse data and generate insights",
    "create marketing content and manage campaigns",
    "assist in product roadmap planning",
    "design and prototype user interfaces",
    "build and deploy machine learning models",
    "conduct market research and competitive analysis",
    "support the HR team in recruitment activities",
    "work on backend APIs and database optimization",
    "manage social media and create engaging posts",
]

DOMAINS = [
    "Artificial Intelligence", "Cloud Computing", "Data Engineering",
    "Mobile Development", "Cybersecurity", "FinTech", "EdTech",
    "E-Commerce", "Healthcare Tech", "SaaS Products",
]

DEPARTMENTS = [
    "Engineering", "Data Science", "Product", "Marketing",
    "Finance", "Operations", "Human Resources", "Design",
]


def make_legit_row(idx):
    title   = random.choice(LEGIT_TITLES)
    company = random.choice(LEGIT_COMPANIES)
    loc     = random.choice(LOCATIONS[:10])   # No "Work From Home" for legit
    sal     = random.choice(SALARY_RANGES)
    dur     = random.choice(["2 months", "3 months", "6 months", "1 year"])
    stip    = random.choice(["₹10,000", "₹15,000", "₹20,000", "₹25,000", "₹8,000"])
    skill   = random.choice(TECH_SKILLS)
    resp    = random.choice(RESPONSIBILITIES)
    domain  = random.choice(DOMAINS)
    dept    = random.choice(DEPARTMENTS)
    year    = random.randint(2000, 2020)
    size    = random.choice(["200", "500", "1000", "5000", "10,000"])
    offices = random.choice(["3", "5", "10", "15"])
    mission = random.choice(["build products that matter", "connect people with opportunities",
                              "transform digital commerce", "democratize education"])
    industry = random.choice(["fintech", "edtech", "healthtech", "logistics", "SaaS"])

    desc    = random.choice(LEGIT_DESCRIPTIONS).format(
        title=title, company=company, duration=dur, responsibility=resp,
        domain=domain, stipend=stip, department=dept, location=loc
    )
    req     = random.choice(LEGIT_REQUIREMENTS).format(skills=skill)
    profile = random.choice(LEGIT_PROFILES).format(
        company=company, mission=mission, year=year, size=size,
        offices=offices, industry=industry
    )

    return {
        "job_id":          idx,
        "title":           title,
        "location":        loc,
        "department":      dept,
        "salary_range":    sal,
        "company_profile": profile,
        "description":     desc,
        "requirements":    req,
        "benefits":        "Health insurance, flexible hours, learning budget",
        "telecommuting":   0,
        "has_company_logo": 1,
        "has_questions":   1,
        "employment_type": "Internship",
        "required_experience": "Internship",
        "required_education": "Unspecified",
        "industry":        industry,
        "function":        dept,
        "fraudulent":      0,
    }


def make_fake_row(idx):
    title  = random.choice(FAKE_TITLES)
    loc    = random.choice(["Work From Home", "Anywhere", "Remote", "Online"])
    fee    = random.choice([499, 999, 1999, 500, 2000, 299])
    salary = random.choice([50000, 30000, 40000, 25000, 60000, 15000])
    hours  = random.choice([2, 3, 4])
    year   = random.randint(2018, 2023)
    size   = random.choice(["5000", "10000", "50000", "1,00,000"])

    desc    = random.choice(FAKE_DESCRIPTIONS).format(
        salary=salary, fee=fee, hours=hours
    )
    req     = random.choice(FAKE_REQUIREMENTS).format(fee=fee)
    profile = random.choice(FAKE_PROFILES).format(year=year, size=size)

    return {
        "job_id":          idx,
        "title":           title,
        "location":        loc,
        "department":      "",
        "salary_range":    f"{salary-5000}-{salary}",
        "company_profile": profile,
        "description":     desc,
        "requirements":    req,
        "benefits":        "",
        "telecommuting":   1,
        "has_company_logo": 0,
        "has_questions":   0,
        "employment_type": "Part-time",
        "required_experience": "Not Applicable",
        "required_education": "Unspecified",
        "industry":        "",
        "function":        "",
        "fraudulent":      1,
    }


# ── Build dataset ──────────────────────────────────────────────────────────────
N_LEGIT = 2700   # ~90% legitimate
N_FAKE  = 300    # ~10% fake

rows = []
for i in range(N_LEGIT):
    rows.append(make_legit_row(i))
for i in range(N_FAKE):
    rows.append(make_fake_row(N_LEGIT + i))

df = pd.DataFrame(rows)

# Shuffle the rows so fake and legit are interleaved
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dataset", "fake_job_postings.csv")
df.to_csv(out_path, index=False)

print(f"✅ Dataset created: {out_path}")
print(f"   Total rows : {len(df)}")
print(f"   Legitimate : {(df['fraudulent']==0).sum()}")
print(f"   Fake       : {(df['fraudulent']==1).sum()}")
print(f"   Columns    : {list(df.columns)}")
