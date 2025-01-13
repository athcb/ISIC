from sklearn.model_selection import KFold, StratifiedKFold

from process_data2 import design_model_conv

## Custom Randomized Search Function

def custom_randomised_search(train_dataset, param_grid, num_iter, cvfolds, greaterisbetter_param):
    # define number of folds to split the training dataset
    kf = KFold(n_splits = cvfolds,, shuffle = False, random_state = 11)
    best_score = float("inf")
    best_model = None


    for _ in range(num_iter):
        params = {key: np.random.choice(values) for key, values in param_grid.items()}

        fold_scores = []

        for i, (train_index, val_index) in enumerate(kf.split(range(len(train_dataset)))):
            #fold = i
            #train_indices = train_index
            #val_indices = val_index
            print(train_indices)
            print(test_indices)

            train_dataset = train_dataset.enumerate().filter(lambda idx, _: tf.reduce_any(tf.equal(idx, list(train_index)))).map(lambda _, data: data)
            val_dataset = val_dataset.enumerate().filter(lambda idx, _: tf.reduce_any(tf.equal(idx, list(val_index)))).map(lambda _, data: data)

            early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)

            model = KerasWrapper(build_fn = design_model_conv, **params)
            callbacks_params = {"callbacks": [early_stop]}

            model.fit(train_dataset,
                      steps_per_epoch=train_dataset.cardinality().numpy(),
                      #steps_per_epoch = len(train_index) // train
                      epochs=num_epochs,
                      validation_data=val_dataset,
                      validation_steps=val_dataset.cardinality().numpy(),
                      class_weight=class_weight,
                      verbose=verbose)
            #fit_model(model, train_dataset, val_dataset, num_epochs, weight_positive, verbose)

            test_loss, test_f1_score, test_precision, test_recall, test_auc = model.evaluate(val_dataset, sample_weight = params["weight_positive"])

            fold_scores.append(test_loss)

        mean_score = np.mean(fold_scores)
        if (not greaterisbetter_param and mean_score < best_score) or (greaterisbetter_param and mean_score > best_score):
            best_score = mean_score
            best_model = model


    return best_model

