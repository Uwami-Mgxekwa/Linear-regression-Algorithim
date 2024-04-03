from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Let's assume we have some data
exam_marks = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95]).reshape((-1, 1))
student_performance = np.array([3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9])

# Split the data into training/testing sets
exam_marks_train, exam_marks_test, student_performance_train, student_performance_test = train_test_split(exam_marks, student_performance, test_size=0.2, random_state=42)

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(exam_marks_train, student_performance_train)

# Make predictions using the testing set
student_performance_pred = regr.predict(exam_marks_test)

print('Predicted student performance: \n', student_performance_pred)