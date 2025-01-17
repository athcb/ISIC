
def fit_model(model, train_dataset, steps_per_epoch, validation_dataset, validation_steps, num_epochs, weight_positive, callbacks, verbose):

    class_weight = {0: 1.0, 1: weight_positive}

    history = model.fit(train_dataset,
                        #steps_per_epoch= steps_per_epoch,
                        epochs=num_epochs,
                        validation_data=validation_dataset,
                        #validation_steps=validation_steps,
                        class_weight=class_weight,
                        callbacks = callbacks,
                        verbose=verbose)

    return model, history

