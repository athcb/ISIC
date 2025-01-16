
def evaluate_model(model, train_dataset, val_dataset):

    test_loss, test_precision, test_recall, test_auc = model.evaluate(val_dataset)
    #train_loss, train_accuracy, train_auc = model.evaluate(train_dataset)
    print("test_Loss ", test_loss)

    test_y_pred = model.predict(test_dataset)
    #train_y_pred = model.predict(train_dataset)

    return test_loss, test_f1_score, test_precision, test_recall, test_auc,  test_y_pred
