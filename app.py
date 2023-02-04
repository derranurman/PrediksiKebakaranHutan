import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("Forest_fire.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

inputt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(inputt)]

b = log_reg.predict_proba(final)

pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

from flask import Flask, request, render_template

app = Flask(__name__)

model_file = open('model.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', insurance_cost=0)

@app.route('/predict', methods=['POST'])
def predict():
    temp, RH, wind = [x for x in request.form.values()]
    
    data = []
    data.append(int(temp))
    data.append(int(RH))
    data.append(int(wind))

    prediction = model.predict([data])
    output = round(prediction[0], 2)
    
    if output == 0:
        result = "Hutan tidak berbakar"
    else:
        result = "Hutan Terbakar"

    return render_template('index.html', insurance_cost=output, temp=temp, RH=RH, wind=wind, result=result)


if __name__ == '__main__':
    app.run(debug=True)
