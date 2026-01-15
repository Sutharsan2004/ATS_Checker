import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Initializing Model.....")
model = SentenceTransformer("all-MiniLm-L6-v2")

def extract_text_from_pdf(pdf_path):
  try:
    with open(pdf_path, 'rb') as file:
      reader = PyPDF2.PdfReader(file)
      text = ""
      for page in reader.pages:
        text = text + page.extract_text() + " "
      return text.strip()
  except FileNotFoundError:
    print("File not found")

def ats_checker(pdf_path, jd):
  resume_text = extract_text_from_pdf(pdf_path)

  resume_vector = model.encode([resume_text])
  jd_vector = model.encode([jd])

  similarity_score =cosine_similarity(resume_vector, jd_vector)[0][0]

  percentage = similarity_score * 100

  print(f"PDF Path {pdf_path}")

  print(f"Resume Score: {percentage:.2f}%")

  if percentage > 75:
    print("Excellent match for JD")
  elif percentage > 50 and percentage < 75:
    print("Good match.. ")
  else:
    print("Improve resume...")


print("ATS Score Checker!!!")
jd=input("Enter Job Description:")
resume = input("Enter Resume Path : ")
ats_checker(resume, jd)
