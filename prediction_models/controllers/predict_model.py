from prediction_models.services.pre_process_text import pre_process_txt
import fasttext
from rest_framework.response import Response
from sklearn.externals import joblib
import pickle
import glob
loaded_model_svm = joblib.load("models/finalized_svm_model.pkl")
loaded_model_mlp = joblib.load("models/finalized_mlp_model.pkl")
loaded_model_ft = fasttext.load_model("models/fast_text_model.bin")
import re
rating = {'1': 'Very_Bad', '2': 'Bad', '3': 'Neutral', '4': 'Good', '5': 'Very_Good'}

def cpred(text,model):
    if model=="svm":
        return rating[loaded_model_svm.predict(text)[0]], 0
    elif model=="mlp":
        return rating[loaded_model_mlp.predict(text)[0]], 0
    else:
        res=re.sub("__label__", '',loaded_model_ft.predict(text)[0][0])

        return  res , loaded_model_ft.predict(text)[1]
res = []
def control_request(request):
    my_path = 'models/'
    files = glob.glob(my_path + '*.*')
    #print(files)
    vectorizer = joblib.load("vect/vectorizer.pkl")
    sub_dict={}
    for i in range(len(request['data'])):
        if request['data'][i]['model'] == "svm" or request['data'][i]['model'] == "mlp":
            b = pre_process_txt(request['data'][i]['content'])
            x = vectorizer.transform([b])
            print(x)
            pred, score = cpred(x,request['data'][i]['model'])
        else:
            pred, score= cpred(request['data'][i]['content'],request['data'][i]['model'])
        sub_dict['content'] = request['data'][i]['content']
        sub_dict['model'] = [request['data'][i]['model']]#pred))#.split(".")[-1][:-2]
        sub_dict['rating'] = pred
        sub_dict['weight'] = score
        res.append(sub_dict)
    print(res)
    return Response(res)