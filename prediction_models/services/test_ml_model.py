from sklearn import metrics

def test_ml_model(model,X_train,y_train,X_test):
    model.fit(X_train,y_train)
    modelName = type(model).__name__
    pred=model.predict(X_test)
    print(modelName)
    #print(classification_report(y_test,model.predict(X_test)))
    #score=np.mean(cross_val_score(model, X, y, cv=5))

    return model,{"model":modelName,"score":pred}