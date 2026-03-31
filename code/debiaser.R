# -----------------------------------------------------------------------------
# Logistic regression classifier with SLR adversarial model, penalizing when a
# given variable is highly predictive from the model residuals.
# -----------------------------------------------------------------------------
# R package keras3 + Python Keras 3 / TensorFlow 2.16+ (see supplement tf_r_env_mac.yml).
library(tidyverse)
library(tidymodels)
library(tensorflow)
library(keras3)
# library(estimatr)
library(modelsummary)
# library(tfdatasets)
library(matrixcalc)

### Construct main adversarial debiasing model.
###             x: Full data set
###        y_name: Name of outcome column
###    treat_name: Name of treatment column
###             f: Indicates which value of x$fold to filter to for validation
###                data.
###     debias_wt: Weight to put on adversarial model loss
###       y_class: type of data for the y variable, either "categorical" or 
###                "continuous"=
###         model: the prediction model, either "single_layer" or "deep_nn"
### debias_method: the debiaser, either "slr" or "corr" or "debiaser_beta"
###       verbose: Option for printing during keras model training.
adb_model <- function(x, y_name, treat_name, f, debias_wt = 0,
                      debias_method = "slr", y_class = "categorical",
                      model = "single_layer", 
                      verbose = NA, predicttrain = FALSE, epochs = 50) {
  
  ### Split data set x into training and validation features and outcomes
  x_train <- as.matrix(x[x$fold != f, !names(x) %in% c("fold", y_name,
                                                       treat_name)])
  y_train <- as.matrix(x[x$fold != f, y_name])
  treat_train <- x[x$fold != f, treat_name]
  
  # normalize treatment 
  treat_train <- as.matrix(scale(treat_train))
  
  x_test <- as.matrix(x[x$fold == f, !names(x) %in% c("fold", y_name,
                                                      treat_name)])
  
  class_wt <- list("0" = 1,  # For balancing classes
                   "1" = round(sum(y_train == 0) / sum(y_train == 1)))  
  
  ### Define model inputs
  features <- layer_input(shape = shape(ncol(x_train)), name = "features")
  
  treatment <- layer_input(shape = shape(1), name = "treatment")
  
  truth <- layer_input(shape = shape(1), name = "truth") 
  
  normalizer <- layer_normalization(axis = -1L) %>% 
    adapt(as.matrix(x_train))
  
  ### Initializes the logistic regression model
  
  if(model == "single_layer") {
    logits <- features %>%
      normalizer() %>%
      layer_dense(units = 1, 
                  if(y_class == "categorical") {
                    activation = "sigmoid"
                  } else if (y_class == "continuous") {
                    activation = "linear"
                  } else {
                    stop("y_class must be categorical or continuous")
                  })
  } else if (model == "deep_nn") {
    logits <- features %>%
      normalizer() %>%
      layer_dense(units = 32) %>%
      layer_dense(units = 32) %>%
      layer_dense(units = 1, 
                  if(y_class == "categorical") {
                    activation = "sigmoid"
                  } else if (y_class == "continuous") {
                    activation = "linear"
                  } else {
                    stop("y_class must be categorical or continuous")
                  })
  } else {
    stop("model must be single_layer or deep_nn")
  }
  
  
  ### Create debiaser layer that minimizes the difference of the global MSE and 
  ### MSE of a SLR model predicting treatment using the model residuals
  debiaser_layer_slr <- new_layer_class(
    
    classname = "debiaser_slr",
    initialize = function(name = "debiaser_slr") {
      super()$`__init__`(name = name)
    },
    call = function(logits, treatment, truth) {
      
      x <- truth - logits  # Residuals
      ## Add intercept term
      ones <- tf$ones_like(x)
      x <- op_concatenate(list(ones, x), axis = 2L)
      ## normalize treatment
      t <- tf$math$divide(tf$math$subtract(treatment, tf$math$reduce_mean(treatment)),
                          tf$math$reduce_std(treatment))
      ## least squares and predictions
      betas <- tf$linalg$lstsq(x, treatment)
      preds <- tf$linalg$matmul(x, betas)  # OLS predictions of treatment
      
      # Calculate Adversary loss
      adv_res <- tf$math$subtract(treatment, preds)
      amse <- tf$math$reduce_mean(tf$math$square(adv_res))
      armse <- tf$math$reduce_mean(tf$math$sqrt(tf$math$square(adv_res)))
      
      self$add_loss(-1*debias_wt*amse)  # Want to *maximize* adversary's loss
      logits
    }
  )
  
  debiaser_layer_beta <- new_layer_class(
    
    classname = "debiaser_beta",
    initialize = function(name = "debiaser_beta") {
      super()$`__init__`(name = name)
    },
    call = function(logits, treatment, truth) {
      
      x <- truth - logits  # Residuals
      ## Add intercept term
      ones <- tf$ones_like(x)
      x <- op_concatenate(list(ones, x), axis = 2L)
      ## normalize treatment
      t <- tf$math$divide(tf$math$subtract(treatment, tf$math$reduce_mean(treatment)),
                          tf$math$reduce_std(treatment))
      ## least squares and predictions
      betas <- tf$linalg$lstsq(x, treatment)
      preds <- tf$linalg$matmul(x, betas)  # OLS predictions of treatment
      
      # Calculate Adversary loss
      betaloss <- tf$math$reduce_mean(tf$abs(betas))

      self$add_loss(debias_wt*betaloss)
      logits
    }
  )
  
  
  ### Create debiaser layer that minimizes the correlation between the treatment
  ### and primary model residuals.
  debiaser_layer_corr <- new_layer_class(
    
    classname = "debiaser_corr",
    initialize = function(name = "debiaser_corr") {
      super()$`__init__`(name = name)
    },
    call = function(logits, treatment, truth) {
      
      # Calculate Adversary loss (covariance between treatment and residuals)
      x <- truth - logits
      y <- tf$math$divide(tf$math$subtract(treatment, tf$math$reduce_mean(treatment)),
                          tf$math$reduce_std(treatment))
      cov_xy <- tf$math$reduce_sum(tf$math$multiply(x - tf$math$reduce_mean(x),
                                                    y - tf$math$reduce_mean(y)))
      corr_xy <- cov_xy / (tf$math$reduce_std(y) * tf$math$reduce_std(y))
      
      self$add_loss(debias_wt*tf$math$sqrt(tf$math$square(corr_xy)))
      logits
    }
  )
  
  ### Create debiaser layer that uses the modified correlation calculation
  debiaser_layer_mg <- new_layer_class(
    
    classname = "debiaser_mg",
    initialize = function(name = "debiaser_mg") {
      super()$`__init__`(name = name)
    },
    call = function(logits, treatment, truth) {
      
      # Calculate Adversary loss (covariance between treatment and residuals)
      x <- truth - logits
      y <- tf$math$divide(tf$math$subtract(treatment, tf$math$reduce_mean(treatment)),
                          tf$math$reduce_std(treatment))
      cov_xy <- tf$math$reduce_sum(tf$math$multiply(x - tf$math$reduce_mean(x),
                                                    y - tf$math$reduce_mean(y)))
      corr_xy <- tf$sqrt(cov_xy^2)  # Modified correlation calculation
      
      self$add_loss(debias_wt * corr_xy)
      logits
    }
  )
  
  ### Add controls (experimental - not implemented)
  # debiaser_layer_corr <- new_layer_class(
    
  #   classname = "debiaser_corr",
  #   initialize = function(name = "debiaser_corr") {
  #     super()$`__init__`(name = name)
  #   },
  #   call = function(inputs) {
  #     logits <- inputs[[1]]
  #     treatment <- inputs[[2]]
  #     truth <- inputs[[3]]
  #     controls <- inputs[[4]]
      
  #     # Calculate Adversary loss (covariance between treatment and residuals)
  #     x <- truth - logits
  #     y <- tf$math$divide(tf$math$subtract(treatment, tf$math$reduce_mean(treatment)),
  #                         tf$math$reduce_std(treatment))
      
  #     ctc <- tf$linalg$matmul(controls, controls, transpose_a = TRUE)
  #     ### add epsilon to diagonal (ridge regression)
  #     ctc <- ctc + tf$eye(tf$shape(ctc)[1]) * 1e-8
      
  #     ### residualize x and y
  #     x <- x - tf$linalg$matmul(controls, tf$linalg$solve(ctc,
  #                                                         tf$linalg$matmul(controls, x, transpose_a = TRUE)
  #     ))
      
  #     y <- y - tf$linalg$matmul(controls, tf$linalg$solve(ctc,
  #                                                         tf$linalg$matmul(controls, y, transpose_a = TRUE)
  #     ))
      
  #     cov_xy <- tf$math$reduce_sum(tf$math$multiply(x - tf$math$reduce_mean(x),
  #                                                   y - tf$math$reduce_mean(y))) 
  #     corr_xy <- tf$sqrt(cov_xy^2) #cov_xy / (tf$math$reduce_std(x) * tf$math$reduce_std(y))
      
  #     self$add_loss(debias_wt*(corr_xy))
  #     logits
  #   },
    
  #   compute_output_shape = function(input_shape) {
  #     return(input_shape[[1]])
  #   }
  # )
  
  ### Determine which loss function to use
  if (debias_method == "slr") {
    debiased_preds <- debiaser_layer_slr()(logits, treatment, truth)
  } else if (debias_method == "corr") {
    debiased_preds <- debiaser_layer_corr()(logits, treatment, truth)
  } else if (debias_method == "beta") {
    debiased_preds <- debiaser_layer_beta()(logits, treatment, truth)
  } else if (debias_method == "mg") {
    debiased_preds <- debiaser_layer_mg()(logits, treatment, truth)
  } else {
    stop("The parameter debias_method must be 'slr', 'corr', 'beta', or 'mg'.")
  }
  
  
  ### Initialize and compile full model
  model <- keras_model(inputs = list(features, treatment, truth), 
                       outputs = debiased_preds)
  
  model %>% compile(optimizer = "adam",
                     if(y_class == "categorical") {
                       loss = "binary_crossentropy"
                     } else if (y_class == "continuous") {
                       loss = "mean_squared_error"
                     } else {
                       stop("y_class must be categorical or continuous")
                     },  # For logistic regression or linear regression
                     metrics = "mean_absolute_error")  
  
  ### Define early stopping callback
  early_stopping_callback <- callback_early_stopping(
    monitor = "val_mean_absolute_error",    # monitor validation loss
    patience = 10,                          # number of epochs with no improvement
    verbose = 1,                            # print a message when training stops early
    mode = "auto"                           # stops when `val_loss` stops decreasing
  )
  
  ### Fit the model
  # If we want to change the `verbose` argument for model training, keep
  # specification given, else use defaults.
  verbose <- ifelse(!is.na(verbose), verbose,
                    getOption("keras.fit_verbose", default = "auto"))
  
  class_wt <- if (y_class == "categorical") class_wt else NULL
  model %>% fit(
    x = list(x_train, treat_train, y_train),
    y = y_train,
    epochs = epochs,
    validation_split = 0.2,
    class_weight = class_wt,
    verbose = verbose,
    batch_size = 128,
    callbacks = list(early_stopping_callback)  # add early stopping callback here
  )
  
  yhats <- model %>% predict(list(x_test, rep(0, nrow(x_test)), rep(0, nrow(x_test))))
  
  if (predicttrain) {
    yhatstrain <- model %>% predict(list(x_train, rep(0, nrow(x_train)), rep(0, nrow(x_train))))
    yhats <- list(yhats, yhatstrain)
  }
  
  return(yhats)
  
}


### For a batch's control variables, remove linearly dependent columns (ex. may
### happen with dummy variables where some levels are not represented in a
### batch) and return a set of linearly independent control variables.
drop_dependent_cols <- function(batch_controls) {
  
  # While loop removes linearly dependent columns until covariance matrix is non
  # singular.
  batch_controls <- as.array(batch_controls)
  flag <- is.singular.matrix(t(batch_controls) %*% batch_controls)
  while (flag) {
    
    # Helpful way to find linearly dependent columns from:
    # https://stats.stackexchange.com/questions/16327/testing-for-linear-dependence-among-the-columns-of-a-matrix
    # We know from flag that the matrix is not full rank. Then either:
    #    There are linearly independent columns - removing these would decrease
    #    the rank, while removing linearly dependent columns would not change
    #    the rank (find these)
    #                               OR
    #    All columns are linearly dependent, and removing any of them would not
    #    change the rank (we can remove any one of them)
    rank_if_removed <- sapply(1:ncol(batch_controls), function(i){
      qr(batch_controls[, -i])$rank})
    lin_dep_cols <- which(rank_if_removed == max(rank_if_removed))
    
    # Remove first linearly dependent column and check again
    batch_controls <- batch_controls[, -lin_dep_cols[1]]
    flag <- is.singular.matrix(t(batch_controls) %*% batch_controls)
  
  }
  
  return(tf$cast(batch_controls, dtype = tf$float32))
  
}


### Predicts primary model prediction errors using linear regression with
### controls and treatment variable.
### TODO: Option to return just covariance for controls case?
linear_adversary <- function(controls, treat, nus) {
  
  # If we have control variables, the adversary is a linear regression of
  # (part of prediction error  explained by the controls) ~ (part of treatment
  # not explained by controls). 
  if (!is.null(controls)) {
    
    # Remove any linearly dependent control variables in this batch
    controls <- as.array(drop_dependent_cols(controls))
    
    # Prediction error residuals from nu ~ controls (part of nu not explained
    # by controls)
    nu_reg_c <- controls %*% solve(t(controls) %*% controls) %*% t(controls) %*% as.array(nus)
    nu_reg_c_res <- as.array(nus) - as.numeric(nu_reg_c)
    
    # If this batch has constant treatment, predict the mean (intercept only).
    if (length(unique(as.array(treat))) == 1) {
      return(rep(tf$reduce_mean(nu_reg_c_res), length(nus)))
    }
    
    # Treatment residuals from treatment ~ controls (part of treatment not
    # explained by controls)
    treat_reg_c <- controls %*% solve(t(controls) %*% controls) %*% t(controls) %*% as.array(treat)
    treat_reg_c_res <- as.array(treat) - as.numeric(treat_reg_c)
    treat_reg_c_res <- cbind(1, treat_reg_c_res)
    
    # TODO: Better way to handle high imbalance of binary treatment groups in
    # some batches? Can have computationally singular treat_reg_c_res with high
    # imbalance (ex. 1 treated vs 63 control). Create the batches ourselves,
    # always enforcing some threshold of treated units in each batch? 
    test <- try(solve(t(treat_reg_c_res) %*% treat_reg_c_res), silent = TRUE)
    if (length(test) == 1) {
      if (class(test) == "try-error") {
       return(rep(tf$reduce_mean(nu_reg_c_res), length(nus)))
      }
    }
    
    k <- solve(t(treat_reg_c_res) %*% treat_reg_c_res) %*% t(treat_reg_c_res) %*% nu_reg_c_res
    preds <- treat_reg_c_res %*% k
    
  } else {  # Otherwise, just using treatment directly to predict errors.
    
    # If this batch has constant treatment, predict the mean (intercept only).
    if (length(unique(as.array(treat))) == 1) {
      return(rep(tf$reduce_mean(nus), length(nus)))
    }
    
    M <- cbind(rep(1, length(treat)), as.array(treat))  # Add intercept
    preds <- M %*% solve(t(M) %*% M) %*% t(M) %*% as.array(nus)
  }
  
  return(as.numeric(preds))
  
}


### Construct more flexible adversarial debiasing model.
###             x: Full data set
###        y_name: Name of outcome column
###    treat_name: Name of treatment column
###             f: Indicates which value of x$fold to filter to for validation
###                data.
###     debias_wt: Weight to put on adversarial model loss
###      controls: matrix of control variables
adb_model_flex <- function(x, y_name, treat_name, f, debias_wt = 0,
                           controls = NULL, epochs = 50, batchsize = 64,
                           adversary_type = "linear") {
  
  ### Split data set x into training and validation features and outcomes
  x_train <- as.matrix(x[x$fold != f, !names(x) %in% c("fold", y_name, treat_name)])
  y_train <- as.matrix(x[x$fold != f, y_name])
  treat_train <- as.matrix(x[x$fold != f, treat_name])
  
  if (!is.null(controls)) {
    control_train <- as.matrix(controls[x$fold != f, ])
  } else {
    control_train <- NULL
  }
  
  x_test <- as.matrix(x[x$fold == f, !names(x) %in% c("fold", y_name,
                                                      treat_name)])
  
  class_wt <- ifelse(y_train==1, sum(y_train == 0)/sum(y_train == 1),1) 
  
  ### Initializes the learner
  learner <- keras_model_sequential(name = "learner",
                                    input_shape = shape(ncol(x_train))) %>%
    layer_dense(units = 32) %>%
    layer_dense(units = 32) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  ## Adversarial loss function
  loss_wrap <- function(adv_preds, w){
    l_loss <- function(y_true, y_preds){
      logloss <- tf$losses$binary_crossentropy(y_true, y_preds)
      advloss <- tf$losses$mse(y_true - y_preds, adv_preds)
      wloss <- tf$reduce_sum(w*(logloss - debias_wt*advloss))
      return(wloss)
    }
    return(l_loss)
  }
  
  ## Initialize optimizers and data
  l_optimizer = optimizer_adam(learning_rate = 1e-3)
  
  tf_ds <- tensorflow::tf$data$Dataset
  if (!is.null(controls)) {
    batched_data <- tf_ds$from_tensor_slices(reticulate::tuple(
      x_train, y_train, treat_train, class_wt, control_train
    ))$shuffle(1024L)$batch(as.integer(batchsize))
  } else {
    batched_data <- tf_ds$from_tensor_slices(reticulate::tuple(
      x_train, y_train, treat_train, class_wt
    ))$shuffle(1024L)$batch(as.integer(batchsize))
  }
  
  
  ## Training loop
  for (epoch in seq_len(epochs)) {
    cat("Start of epoch ", epoch, "\n")
    
    tfautograph::autograph(for (batch in batched_data) {
      x0 = tf$cast(batch[[1]], dtype=tf$float32); y0 = tf$cast(batch[[2]], dtype=tf$float32); 
      z = tf$cast(batch[[3]], dtype=tf$float32); w = tf$cast(batch[[4]], dtype=tf$float32)
      
      if (!is.null(controls)) {
        c = tf$cast(batch[[5]], dtype=tf$float32)
        covs <- cbind(as.array(z), as.array(c))
      } else {
        c = NULL
      }
      
      with(tf$GradientTape() %as% tape, {
        predictions <- learner(x0)
        nus <- y0 - tf$squeeze(predictions)
        if (adversary_type=="linear") {
          adv_preds <- linear_adversary(c, z, nus)
        } else if (adversary_type=="lasso"){
          adversary <- cv.glmnet(covs, nus)
          adv_preds <- predict(adversary, covs, s = "lambda.min")
        } else if (adversary_type=="rf") {
          rfdata <- data_frame("nus" = nus)
          rfdata <- cbind(rfdata, covs)
          colnames(rfdata) <- c("nus", paste("X", seq(1:(ncol(covs))), sep = ""))
          adversary <- randomForest(nus ~ ., data = rfdata, ntree = 150)
          adv_preds <- predict(adversary, covs)
        } else {
          adversary <- keras_model_sequential(name = "adversary",
                                              input_shape = shape(ncol(covs))) %>%
            layer_dense(units = 1, activation = "linear")
          
          adversary %>% compile(optimizer = "adam", loss = "mse")
          adversary %>% fit(x = covs, y = nus, epochs = 5, batch_size = 64)
          adv_preds <- predict(adversary, covs)
        }
        adv_preds <- tf$cast(adv_preds, dtype=tf$float32)
        l_loss <- loss_wrap(adv_preds, w)
        llossval <- l_loss(y0, predictions)
      })
      
      grads <- tape$gradient(llossval, learner$trainable_weights)
      l_optimizer$apply_gradients(
        zip_lists(grads, learner$trainable_weights))
    })
  }
  
  yhats <- learner %>% predict(x_test)
  
  return(yhats)
  
}



### Train and cross-fit adversarial debiasing model.
###             x: Full data set
###        y_name: Name of outcome column
###    treat_name: Name of treatment column
###             f: Indicates which value of x$fold to filter to for validation
###                data.
###     debias_wt: Weight to put on adversarial model loss
###       verbose: Option for printing during keras model training.
train_all_folds <- function(x, y_name, treat_name, debias_wt = 0,
                            debias_method = "slr", y_class = "categorical",
                            model = "single_layer",
                            verbose = NA, epochs = 50) {
  
  preds <- numeric(nrow(x))
  for (f in unique(x$fold)) {
    
    p <- adb_model(x = x, y_name =  y_name, treat_name = treat_name,
                   f = f, debias_wt = debias_wt, debias_method = debias_method,
                   y_class = y_class, model = model,
                   verbose = verbose, epochs = epochs)
    
    preds[which(x$fold == f)] <- p
    
  }
  
  return(preds)
  
}

train_all_folds_flex <- function(x = x, y_name =  y_name, treat_name = treat_name,
                                 f = f, debias_wt = debias_wt,controls = NULL, epochs = 20,
                                 batchsize = 64,adversary_type = "linear") {
  
  preds <- numeric(nrow(x))
  for (f in unique(x$fold)) {
    
    p <- adb_model_flex(x = x, y_name =  y_name, treat_name = treat_name,
                   f = f, debias_wt = debias_wt, controls = controls, epochs = epochs,
                   batchsize = batchsize, adversary_type = adversary_type)
    
    preds[which(x$fold == f)] <- p
    
  }
  
  return(preds)
  
}
