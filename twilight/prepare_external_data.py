"""
Download and prepare external open-source data for encoder + decoder training.
Sources:
  - jacob-hugging-face/job-descriptions  (real JDs)
  - ahmedheakl/resume-atlas              (2400+ real resumes by category)
  - rajpurkar/squad                      (87K QA pairs for decoder)

Run: /opt/llm-training/bin/python3 prepare_external_data.py
"""
import json, random, re
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

OUT = Path.home() / "twilight" / "external_data"
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. JOB DESCRIPTIONS  →  encoder pairs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/3] Downloading job descriptions...")
jd_ds = load_dataset("jacob-hugging-face/job-descriptions", split="train", streaming=False)
print(f"  {len(jd_ds)} job descriptions")

jd_records = []
for ex in tqdm(jd_ds, desc="  Processing JDs"):
    jd = ex.get("job_description", "").strip()
    title = ex.get("position_title", "").strip()
    if len(jd) < 100:
        continue
    jd_records.append({"title": title, "jd": jd[:2000]})

with open(OUT / "job_descriptions.jsonl", "w") as f:
    for r in jd_records:
        f.write(json.dumps(r) + "\n")
print(f"  Saved {len(jd_records)} JDs → external_data/job_descriptions.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# 2. RESUME ATLAS  →  encoder pairs + decoder QA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/3] Downloading resume-atlas...")
res_ds = load_dataset("ahmedheakl/resume-atlas", split="train", streaming=False)
print(f"  {len(res_ds)} resumes")

# Category → JD template mapping
CATEGORY_JD = {
    "Accountant":        "We are hiring an Accountant with financial reporting, Excel, and auditing skills.",
    "Engineer":          "Seeking a Software Engineer with programming and system design experience.",
    "Data Science":      "Looking for a Data Scientist with Python, ML, and statistical analysis skills.",
    "Web Designing":     "Hiring a Web Designer with HTML, CSS, JavaScript, and UI/UX skills.",
    "Java Developer":    "Java Developer needed with Spring Boot, REST APIs, and SQL experience.",
    "Python Developer":  "Python Developer with Django/Flask, REST APIs, and database skills required.",
    "DevOps Engineer":   "DevOps Engineer with Docker, Kubernetes, CI/CD, and cloud platform experience.",
    "HR":                "HR professional with recruitment, employee relations, and HRIS experience.",
    "Sales":             "Sales executive with CRM, lead generation, and client management skills.",
    "Marketing":         "Marketing specialist with digital marketing, SEO, and analytics experience.",
    "Teacher":           "Teacher with curriculum development, classroom management, and subject expertise.",
    "Advocate":          "Legal professional with litigation, contract drafting, and legal research skills.",
    "Arts":              "Creative professional with design, visual arts, and portfolio experience.",
    "Fitness":           "Fitness trainer with exercise science, nutrition, and client coaching skills.",
    "Nursing":           "Registered Nurse with patient care, clinical skills, and medical knowledge.",
    "Business Analyst":  "Business Analyst with requirements gathering, SQL, and process modeling skills.",
    "Consultant":        "Consultant with problem solving, stakeholder management, and domain expertise.",
    "Digital Media":     "Digital media specialist with content creation, social media, and video editing.",
    "Automobile":        "Automobile engineer with mechanical design, CAD, and vehicle systems knowledge.",
    "Chef":              "Chef with culinary arts, kitchen management, and food safety certification.",
    "Aviation":          "Aviation professional with flight operations, safety protocols, and regulations.",
    "Banking":           "Banking professional with financial products, compliance, and customer service.",
    "Construction":      "Construction manager with project planning, site management, and safety skills.",
    "Public Relations":  "PR specialist with media relations, press releases, and brand communication.",
}

resume_records = []
encoder_pairs  = []  # (jd, resume_text, category)

for ex in tqdm(res_ds, desc="  Processing resumes"):
    cat  = ex.get("Category", "Engineer").strip()
    text = ex.get("Text", "").strip()
    if len(text) < 150:
        continue
    resume_records.append({"category": cat, "text": text[:2000]})
    jd = CATEGORY_JD.get(cat, f"We are looking for a {cat} with relevant skills and experience.")
    encoder_pairs.append({"jd": jd, "resume": text[:1500], "category": cat})

with open(OUT / "resumes.jsonl", "w") as f:
    for r in resume_records:
        f.write(json.dumps(r) + "\n")

with open(OUT / "encoder_pairs_resumes.jsonl", "w") as f:
    for r in encoder_pairs:
        f.write(json.dumps(r) + "\n")

print(f"  Saved {len(resume_records)} resumes → external_data/resumes.jsonl")
print(f"  Saved {len(encoder_pairs)} encoder pairs → external_data/encoder_pairs_resumes.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SQUAD  →  decoder SFT data (reading comprehension QA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/3] Downloading SQuAD...")
squad_ds = load_dataset("rajpurkar/squad", split="train", streaming=False)
print(f"  {len(squad_ds)} QA examples")

squad_records = []
for ex in tqdm(squad_ds, desc="  Processing SQuAD"):
    ctx  = ex["context"].strip()
    q    = ex["question"].strip()
    ans_list = ex["answers"]["text"]
    if not ans_list:
        continue
    ans = ans_list[0].strip()
    if len(ctx) < 50 or len(q) < 5 or len(ans) < 1:
        continue
    squad_records.append({
        "context":  ctx[:1200],
        "question": q,
        "answer":   ans,
        "answerable": True,
        "source":   "squad",
    })

# Also add unanswerable examples (SQuAD v2 style — use squad_v2)
try:
    squad2_ds = load_dataset("rajpurkar/squad_v2", split="train", streaming=False)
    unanswerable = 0
    for ex in squad2_ds:
        if ex["answers"]["text"]:
            continue  # skip answerable
        ctx = ex["context"].strip()
        q   = ex["question"].strip()
        if len(ctx) < 50:
            continue
        squad_records.append({
            "context":    ctx[:1200],
            "question":   q,
            "answer":     "NOT_FOUND",
            "answerable": False,
            "source":     "squad_v2_unanswerable",
        })
        unanswerable += 1
        if unanswerable >= 10000:
            break
    print(f"  Added {unanswerable} unanswerable examples from SQuAD v2")
except Exception as e:
    print(f"  SQuAD v2 skipped: {e}")

random.shuffle(squad_records)
with open(OUT / "squad_sft.jsonl", "w") as f:
    for r in squad_records:
        f.write(json.dumps(r) + "\n")
print(f"  Saved {len(squad_records)} QA examples → external_data/squad_sft.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build JD-Resume cross pairs (JD from job_descriptions + resume from atlas)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Building cross JD-Resume pairs...")

# Match JDs to resumes by keyword overlap
def keyword_overlap(jd: str, resume: str) -> int:
    jd_words    = set(re.findall(r'\b\w{4,}\b', jd.lower()))
    res_words   = set(re.findall(r'\b\w{4,}\b', resume.lower()))
    return len(jd_words & res_words)

# Sample 3000 JDs and find best matching resume for each
random.shuffle(jd_records)
random.shuffle(resume_records)
sample_jds     = jd_records[:3000]
sample_resumes = resume_records[:500]  # compare against 500 resumes

cross_pairs = []
for jd_rec in tqdm(sample_jds[:2000], desc="  Matching JDs to resumes"):
    best_res, best_score = None, 0
    for res_rec in sample_resumes:
        score = keyword_overlap(jd_rec["jd"], res_rec["text"])
        if score > best_score:
            best_score = score
            best_res   = res_rec
    if best_res and best_score >= 3:
        cross_pairs.append({
            "jd":      jd_rec["jd"][:1500],
            "resume":  best_res["text"][:1500],
            "category": best_res["category"],
            "overlap_score": best_score,
        })

with open(OUT / "encoder_pairs_cross.jsonl", "w") as f:
    for r in cross_pairs:
        f.write(json.dumps(r) + "\n")
print(f"  Saved {len(cross_pairs)} cross pairs → external_data/encoder_pairs_cross.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"  JDs:              {len(jd_records):>6}")
print(f"  Resumes:          {len(resume_records):>6}")
print(f"  Encoder pairs:    {len(encoder_pairs) + len(cross_pairs):>6}")
print(f"  Decoder SFT QA:   {len(squad_records):>6}")
print(f"\nAll saved to: {OUT}")
