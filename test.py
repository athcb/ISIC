print(26500/(2*467))


def randomized_search(train_dataset, param_grid, num_iter, cvfolds, custom_scoring, greaterisbetter_param):

    early_stop = EarlyStopping(monitor="val_loss", mode = "min", verbose = 1, patience =5)
    print("test 1")
    model = KerasWrapper(build_fn=design_model_conv)
    print("test 2")
    custom_scorer = make_scorer(custom_scoring, greater_is_better=greaterisbetter_param)
    precision_scorer = make_scorer(precision_score)
    recall_scorer = make_scorer(recall_score)
    print("test 3")
    grid = RandomizedSearchCV(estimator=model,
                              param_distributions=param_grid,
                              scoring=custom_scorer,
                              #scoring = {"log_loss": custom_scorer, "precision": precision_scorer, "recall": recall_scorer},
                              #refit = "log_loss",
                              n_iter=num_iter,
                              cv = cvfolds, # default 5,
                              verbose = 2,
                              random_state = 11)
    print("test 4")
    # convert (batched) tf.data.Dataset object to a Python iterator that yields batches of numpy arrays
    # each element is one batch of data
    #X_train, y_train = dataset_to_numpy(train_dataset)
    X_train, y_train = dataset_to_numpy_from_generator(train_dataset)
    print("test 5")
    fit_params = {"callbacks": [early_stop]}
    grid_result = grid.fit(X_train, y_train, **fit_params)
    #print("generator shape:", )
    #grid_result = grid.fit(data_generator(train_dataset), **fit_params)
    print("test 6")
    #best_model = grid_result.best_estimator_
    #y_pred = best_model.predict(train_dataset)
    #print("y_pred from .predict")
    #print(y_pred)
    #print("Log loss from best model on train set: ", log_loss(y_train, y_pred))

    return grid_result
