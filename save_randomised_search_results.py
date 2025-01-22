import pandas as pd
import logging

logger = logging.getLogger("MainLogger")

def save_randomised_search_results(best_model,  best_params, mean_scores_best_model, val_scores_best_model,
                                   output_best_params, output_mean_scores, output_val_scores):

    best_params_df = pd.DataFrame([best_params])
    mean_scores_best_model_df = pd.DataFrame([mean_scores_best_model])
    val_scores_best_model_df = pd.DataFrame(val_scores_best_model)
    #train_scores_best_model_df = pd.DataFrame(train_scores_best_model)

    logger.info("Best params DF: ")
    logger.info(best_params_df)

    best_params_df.to_csv(output_best_params, index=False)
    mean_scores_best_model_df.to_csv(output_mean_scores, index=False)
    val_scores_best_model_df.to_csv(output_val_scores, index=False)
    #train_scores_best_model_df.to_csv(output_train_scores, index=False)
