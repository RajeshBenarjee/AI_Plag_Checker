from model.cross_student_detector import CrossStudentDetector

detector = CrossStudentDetector()

with open("data/dataset.txt") as f:
    text = f.read()

detector.add_submission(text)

print("Student index built")