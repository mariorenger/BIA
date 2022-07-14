from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from keras.models import load_model

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from ortools.linear_solver import pywraplp
import pickle
import xgboost as xg

app = Flask(__name__)

df = pd.read_csv('KAG_conversion_data.csv')

data = df[~(df.Impressions > 1000000)]
data = data[~(data.Clicks > 250)]
data = data[~(data.Spent > 350)]
data = data[~(data.Total_Conversion > 20)]
data = data[~(data.Approved_Conversion > 8)]
df = data

encoder = LabelEncoder()
encoder.fit(df["gender"])
df["gender"] = encoder.transform(df["gender"])

encoder.fit(df["xyz_campaign_id"])
df["xyz_campaign_id"] = encoder.transform(df["xyz_campaign_id"])

encoder.fit(df["ad_id"])
df["ad_id"] = encoder.transform(df["ad_id"])

encoder.fit(df["fb_campaign_id"])
df["fb_campaign_id"] = encoder.transform(df["fb_campaign_id"])

encoder.fit(df["age"])
df["age"] = encoder.transform(df["age"])
print(encoder.classes_, encoder.transform(encoder.classes_))

x = np.array(df.drop(labels=["ad_id", "xyz_campaign_id",
                             "fb_campaign_id", "Approved_Conversion", ], axis=1))
x2 = np.array(df.drop(labels=["ad_id", "xyz_campaign_id", "fb_campaign_id",
                              "Total_Conversion", "Approved_Conversion", ], axis=1))

y = np.array(df["Approved_Conversion"])

preX = x
preY = y

scaler = StandardScaler()
scaler.fit(x)

scaler2 = StandardScaler()
scaler2.fit(x2)
x2 = scaler2.transform(x2)
x = scaler.transform(x)

model1 = load_model('model_lr.h5')
with open('model_dt.pkl', 'rb') as f:
    model2 = pickle.load(f)
with open('model_rf.pkl', 'rb') as f:
    model3 = pickle.load(f)
with open('model_xg.pkl', 'rb') as f:
    model4 = pickle.load(f)


@app.route('/')
def index():
    return render_template('form.html')


age = {0: "30-34", 1: "35-39", 2: "40-44", 3: "45-49"}
gender = {0: "Nam", 1: "Nữ"}


@app.route('/predict', methods=['POST'])
def predict():
    print(request.form)
    Age = request.form['Age']
    Gender = request.form['Gender']
    Interest = request.form['Interest']
    Impressions = request.form['Impressions']
    Click = request.form['Click']
    Spent = request.form['Spent']

    inp = np.array([[int(Age), int(Gender), int(Interest),
                     int(Impressions), int(Click), int(Spent)]])
    inp = scaler2.transform(inp)
    out = model1.predict(inp)
    message = ''
    message += 'Liner Regression: '+str(model1.predict(inp))+', \n'
    message += 'Decision Tree: '+str(model2.predict(inp))+', \n'
    message += 'Random Forest: '+str(model3.predict(inp))+', \n'
    message += 'Xgboost : '+str(model4.predict(inp))+'\n'
    if Age and Gender and Interest and Spent and Click and Impressions:
        # return jsonify({'total': message})
        
        rs = dict()
        rs['Liner Regression'] = float(model1.predict(inp))
        rs['Decision Tree'] = float(model2.predict(inp))
        rs['Random Forest'] = float(model3.predict(inp))
        rs['Xgboost'] = float(model4.predict(inp))
        return jsonify({'status': 'success', 'data': rs}), 200
    else:
        return jsonify({'error': 'Missing Values'})


@app.route('/process', methods=['POST'])
def process():
    global age, gender

    AgeMin = request.form['AgeMin']
    AgeMax = request.form['AgeMax']
    InterestMin = request.form['InterestMin']
    InterestMax = request.form['InterestMax']
    ImpressionsMin = request.form['ImpressionsMin']
    ImpressionsMax = request.form['ImpressionsMax']
    ClickMin = request.form['ClickMin']
    ClickMax = request.form['ClickMax']
    SpentMin = request.form['SpentMin']
    SpentMax = request.form['SpentMax']

    constraint = {}
    constraint['AgeMin'] = AgeMin
    constraint['AgeMax'] = AgeMax
    constraint['InterestMin'] = InterestMin
    constraint['InterestMax'] = InterestMax
    constraint['ImpressionsMin'] = ImpressionsMin
    constraint['ImpressionsMax'] = ImpressionsMax
    constraint['ClickMin'] = ClickMin
    constraint['ClickMax'] = ClickMax
    constraint['SpentMin'] = SpentMin
    constraint['SpentMax'] = SpentMax

    if SpentMax and ClickMax and ImpressionsMax:
        obj, out = LinearProgramming(model, scaler, constraint)
        # if obj == -1:
        #     message = 'Không có lời giải tối ưu'
        # else:
        #     message = 'Tổng số người mua hàng: '+str(int(obj))+', \n'
        #     message += 'Độ tuổi: '+age[int(out[0][0]+0.1)]+', \n'
        #     message += 'Giới tính: '+gender[int(out[0][1]+0.1)]+', \n'
        #     message += 'Mã sở thích: '+str(int(out[0][2]+0.1))+', \n'
        #     message += 'Thời gian quảng cáo được hiển thị: ' + \
        #         str(int(out[0][3]+0.1))+'\n'
        if obj == -1:
            message = 'Không có lời giải tối ưu'
            return jsonify({'error': message})
        else:
            rs = dict()
            rs['Tổng số người mua hàng'] = int(obj)
            rs['Độ tuổi'] = age[int(out[0][0]+0.1)]
            rs['Giới tính'] = gender[int(out[0][1]+0.1)]
            rs['Mã sở thích'] = int(out[0][2]+0.1)
            rs['Thời gian quảng cáo được hiển thị'] = int(out[0][3]+0.1)
            return jsonify({'status': 'success', 'data': rs})

    return jsonify({'error': 'Missing data!'})


def LinearProgramming(model, scaler, constraint):
    global output
    output = [0 for i in range(
        len(model.layers[0].get_weights()[0].flatten()))]
    """Linear programming sample."""
    # Instantiate a Glop solver, naming it LinearExample.
    solver = pywraplp.Solver.CreateSolver('GLOP')

    xr = [solver.NumVar(np.min(x, axis=0)[i], np.max(x, axis=0)[i], 'x'+str(i))
          for i in range(len(model.layers[0].get_weights()[0].flatten()))]
    # xr = [ solver.NumVar(np.min(x, axis = 0)[i], np.max(x, axis = 0)[i], 'x'+str(i)) if i>2 else solver.IntVar(np.min(x, axis = 0)[i], np.max(x, axis = 0)[i], 'x'+str(i)) for i in range(len(model.layers[0].get_weights()[0].flatten()))]
    print('Number of variables =', solver.NumVariables())

    print('Number of constraints =', solver.NumConstraints())

    ct_min = [[np.min(preX, axis=0)[i] for i in range(
        len(model.layers[0].get_weights()[0].flatten()))]]
    ct_max = [[np.max(preX, axis=0)[i] for i in range(
        len(model.layers[0].get_weights()[0].flatten()))]]
    print(ct_max)
    print(ct_min)
    if constraint['SpentMax']:
        ct_max[0][-2] = constraint['SpentMax']
    if constraint['ClickMax']:
        print(min(3.4, 6), constraint['ClickMax'])
        ct_max[0][-3] = min(float(constraint['ClickMax']),
                            float(constraint['SpentMax'])*0.8)
    if constraint['ImpressionsMax']:
        ct_max[0][-4] = min(float(constraint['ImpressionsMax']),
                            float(constraint['SpentMax'])*490)
    print(ct_max)
    print(ct_min)
    # Add constraints
    for i in range(len(model.layers[0].get_weights()[0].flatten())):
        solver.Add(xr[i] <= scaler.transform(ct_max)[0][i])

    for i in range(len(model.layers[0].get_weights()[0].flatten())):
        solver.Add(xr[i] >= scaler.transform(ct_min)[0][i])

    sum = model.layers[0].get_weights()[1].flatten()[0]
    for i in range(len(model.layers[0].get_weights()[0].flatten())):
        sum += (xr[i])*model.layers[0].get_weights()[0].flatten()[i]

    # Object
    solver.Maximize(sum)

    # Solve the system.
    status = solver.Solve()

    obj = -1
    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        r = [0 for i in range(len(model.layers[0].get_weights()[0].flatten()))]
        for i in range(len(model.layers[0].get_weights()[0].flatten())):
          # print('x =', x[i].solution_value())
            r[i] = xr[i].solution_value()
        obj = int(solver.Objective().Value())
        output = scaler.inverse_transform([[i for i in r]])
        print(output)
    else:
        print('The problem does not have an optimal solution.')

    print('\nAdvanced usage:')

    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    return obj, output


if __name__ == '__main__':

        # df =pd.read_csv('F:\BIA\AJAX_Forms_jQuery_Flask\KAG_conversion_data.csv')

    model = load_model('model.h5')
    model.summary()
    # LinearProgramming(model, scaler)

    app.run(debug=True)
