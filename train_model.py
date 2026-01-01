import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "score": [35, 40, 50, 55, 65, 70, 80, 90]
}
df = pd.DataFrame(data)
X = df[["study_hours"]]
y = df["score"]
model = LinearRegression()
model.fit(X, y)
os.makedirs("model", exist_ok=True)
with open("model/grade_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("Model trained and saved successfully!")
