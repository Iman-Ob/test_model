from prediction_models.services.pre_process_text import pre_process_txt
from rest_framework.response import Response
from sklearn.externals import joblib
import pickle
import glob

def control_request(request):
    my_path = 'models/'
    files = glob.glob(my_path + '*.pkl')
    print(files)
    rating = {'1': 'Very_Bad', '2': 'Bad', '3': 'Neutral', '4': 'Good', '5': 'Very_Good'}
    vectorizer = joblib.load("vect/vectorizer.pkl")
    fin_res={}
    for i in range(len(request['data'])):
        request['data'][i] = pre_process_txt(request['data'][i])
        fin_res[i]={}
        for f in files:
            loaded_model = joblib.load(f)
            x = vectorizer.transform([request['data'][i]])
            pred = loaded_model.predict(x)
            fin_res[i]['content']=request['data'][i]
            fin_res[i]['model']=str(type(loaded_model)).split(".")[-1][:-2]
            fin_res[i]['rating']=rating[pred[0]]
            print("edited")
    return Response(fin_res)