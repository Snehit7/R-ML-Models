library(rpart)
library(rpart.plot)
library(caret)
library(pROC)
library(ggplot2)
library(reshape2)
library(dplyr)
library(caret)
library(randomForest)
library(class)
library(ranger)

setwd("C:/Users/Lenovo/OneDrive/Desktop/AML")
# Load the data
all_data <- read.csv("Bank Marketing Campaign.csv", stringsAsFactors = FALSE)

# View structure
str(all_data)

# Convert target variable to factor (binary classification)
all_data$y <- as.factor(all_data$y)

# Check and handle missing values
sum(is.na(all_data))  # Count total missing values

# For simplicity, remove rows with any NA (can also consider imputation)
all_data <- na.omit(all_data)

# Convert categorical variables to factors
categorical_cols <- c("job", "marital", "education", "default", 
                      "housing", "loan", "contact", "month", "poutcome")
all_data[categorical_cols] <- lapply(all_data[categorical_cols], as.factor)

# Scale numeric columns
num_cols <- sapply(all_data, is.numeric)
all_data[num_cols] <- scale(all_data[num_cols])

# Check the cleaned data
summary(all_data)
str(all_data)

set.seed(123)

# Create 70/30 split
trainIndex <- createDataPartition(all_data$y, p = 0.7, list = FALSE)
train <- all_data[trainIndex, ]
test  <- all_data[-trainIndex, ]

# Confirm sizes
cat("Training set size:", nrow(train), "\n")
cat("Testing set size:", nrow(test), "\n")

# Check class balance
prop.table(table(train$y))
prop.table(table(test$y))
# Set seed for reproducibility
set.seed(123)

# Create 70/30 split
trainIndex <- createDataPartition(all_data$y, p = 0.7, list = FALSE)
train <- all_data[trainIndex, ]
test  <- all_data[-trainIndex, ]

# Confirm sizes
cat("Training set size:", nrow(train), "\n")
cat("Testing set size:", nrow(test), "\n")

# Check class balance
prop.table(table(train$y))
prop.table(table(test$y))


# Logistic Regression Model
set.seed(123)
logit_model <- train(y ~ ., data = train, method = "glm", family = "binomial",
                     trControl = trainControl(method = "cv", number = 5))

# Summary of model
summary(logit_model)

# Predictions
logit_pred_train <- predict(logit_model, train)
logit_pred_test <- predict(logit_model, test)

# Confusion Matrix & Performance
conf_train <- confusionMatrix(logit_pred_train, train$y)
conf_test <- confusionMatrix(logit_pred_test, test$y)

# Print metrics
cat("Training Accuracy:", conf_train$overall["Accuracy"], "\n")
cat("Testing Accuracy :", conf_test$overall["Accuracy"], "\n")
cat("Kappa (Train):", conf_train$overall["Kappa"], "\n")
cat("Kappa (Test) :", conf_test$overall["Kappa"], "\n")

# Print confusion matrix
print(conf_test)
# Convert confusion matrix to dataframe for plotting
cm_logit <- as.data.frame(conf_test$table)
colnames(cm_logit) <- c("Predicted", "Actual", "Freq")

# Plot confusion matrix
ggplot(data = cm_logit, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "gray", high = "black") +
  labs(title = "Confusion Matrix - Logistic Regression (Before Tuning)") +
  theme_minimal()



# ROC and AUC
logit_prob_test <- predict(logit_model, test, type = "prob")[,2]
roc_logit <- roc(test$y, logit_prob_test, levels = c("no", "yes"))

# Plot ROC
plot(roc_logit, col = "blue", main = "ROC Curve - Logistic Regression")
auc_val <- auc(roc_logit)
cat("AUC:", auc_val, "\n")



##Tuned logistic regression
set.seed(123)
logit_tuned <- train(
  y ~ ., 
  data = train,
  method = "glm",
  family = "binomial",
  trControl = trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 3,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC"
)

# Print best ROC score during CV
print(logit_tuned)

# Predict probabilities on test set
logit_prob_test <- predict(logit_tuned, test, type = "prob")[, "yes"]

# Try custom threshold (e.g., 0.3 to increase sensitivity)
threshold <- 0.3
logit_pred_thresh <- ifelse(logit_prob_test > threshold, "yes", "no")
logit_pred_thresh <- factor(logit_pred_thresh, levels = c("no", "yes"))

# Confusion Matrix with tuned threshold
conf_thresh <- confusionMatrix(logit_pred_thresh, test$y)

# Print evaluation
cat("Accuracy (Threshold = 0.3):", conf_thresh$overall["Accuracy"], "\n")
cat("Kappa:", conf_thresh$overall["Kappa"], "\n")
print(conf_thresh)
# Convert confusion matrix to dataframe for plotting
cm_logit_tuned <- as.data.frame(conf_thresh$table)
colnames(cm_logit_tuned) <- c("Predicted", "Actual", "Freq")

# Plot confusion matrix
ggplot(data = cm_logit_tuned, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "skyblue", high = "maroon") +
  labs(title = "Confusion Matrix - Logistic Regression After Tuning") +
  theme_minimal()


# ROC Curve
roc_logit_tuned <- roc(test$y, logit_prob_test)
plot(roc_logit_tuned, col = "darkgreen", main = "ROC - Logistic Regression After Tuning")
cat("Tuned AUC:", auc(roc_logit_tuned), "\n")



# Train KNN using caret with cross-validation
set.seed(123)
knn_model <- train(
  y ~ ., 
  data = train, 
  method = "knn",
  preProcess = c("center", "scale"),  # add this!
  trControl = trainControl(
    method = "cv", 
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC",
  tuneLength = 10  # tests different k values automatically
)

# Print model details and best k
print(knn_model)

# Predictions on train and test
knn_pred_train <- predict(knn_model, train)
knn_pred_test <- predict(knn_model, test)

# Confusion matrices
conf_train_knn <- confusionMatrix(knn_pred_train, train$y)
conf_test_knn <- confusionMatrix(knn_pred_test, test$y)
print(conf_test_knn)

cm_knn_untuned <- conf_test_knn$table

# Convert to data frame for ggplot
cm_df_untuned <- as.data.frame(cm_knn_untuned)
colnames(cm_df_untuned) <- c("Predicted", "Actual", "Freq")

# Plot confusion matrix
ggplot(data = cm_df_untuned, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "gray", high = "black") +
  labs(title = "Confusion Matrix - KNN Before Tuning") +
  theme_minimal()

# Print metrics
cat("Training Accuracy:", conf_train_knn$overall["Accuracy"], "\n")
cat("Testing Accuracy :", conf_test_knn$overall["Accuracy"], "\n")
cat("Kappa (Train):", conf_train_knn$overall["Kappa"], "\n")
cat("Kappa (Test) :", conf_test_knn$overall["Kappa"], "\n")

# ROC and AUC
knn_prob_test <- predict(knn_model, test, type = "prob")[, "yes"]
roc_knn <- roc(test$y, knn_prob_test)

# Plot ROC
plot(roc_knn, col = "purple", main = "ROC Curve - KNN")
cat("AUC:", auc(roc_knn), "\n")








##Tuning Knn
set.seed(123)
ctrl <- trainControl(
  method = "cv", 
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Tune over a range of k (e.g., 1 to 30)
k_grid <- expand.grid(k = c(1, 3, 5, 7, 9))

# Train with tuning
knn_tuned <- train(
  y ~ ., 
  data = train,
  method = "knn",
  preProcess = c("center", "scale"),
  metric = "ROC",  # or "Accuracy"
  trControl = ctrl,
  tuneGrid = k_grid
)

# Print the best k
print(knn_tuned$bestTune)

# Plot Elbow Curve (AUC vs k)
library(ggplot2)
ggplot(knn_tuned, aes(x = k, y = ROC)) +
  geom_line(color = "blue") +
  geom_point() +
  labs(title = "Elbow Plot for KNN Tuning",
       x = "Number of Neighbors (k)",
       y = "ROC AUC") +
  theme_minimal()



# Predict probabilities from the tuned model
knn_prob_test <- predict(knn_tuned, test, type = "prob")[, "yes"]

# Generate ROC curve
roc_knn <- roc(test$y, knn_prob_test)

# Plot ROC
plot(roc_knn, col = "purple", main = "ROC Curve - Tuned KNN")
cat("AUC:", auc(roc_knn), "\n")



# Predict classes
knn_pred_test <- predict(knn_tuned, test)

# Confusion matrix
conf_matrix_knn <- confusionMatrix(knn_pred_test, test$y, positive = "yes")
print(conf_matrix_knn)

# Load libraries
library(caret)
library(ggplot2)
library(reshape2)

# Extract confusion matrix table
cm_table <- conf_matrix_knn$table

# Convert to data frame for ggplot
cm_df <- as.data.frame(cm_table)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")

# Plot
ggplot(data = cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "skyblue", high = "maroon") +
  labs(title = "Confusion Matrix - Tuned KNN") +
  theme_minimal()


#Naive Bayes
# Train Naive Bayes model using caret
set.seed(123)
nb_model <- train(
  y ~ ., 
  data = train,
  method = "naive_bayes",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC"
)

# Show model summary
print(nb_model)


# Predict on test set
nb_pred_test <- predict(nb_model, test)

# Confusion matrix
conf_matrix_nb <- confusionMatrix(nb_pred_test, test$y, positive = "yes")

# Print performance
cat("Testing Accuracy :", conf_matrix_nb$overall["Accuracy"], "\n")
cat("Kappa (Test)     :", conf_matrix_nb$overall["Kappa"], "\n")

# Full confusion matrix
print(conf_matrix_nb)

# 🔳 Confusion Matrix Plot
cm_nb <- as.data.frame(conf_matrix_nb$table)
colnames(cm_nb) <- c("Predicted", "Actual", "Freq")

ggplot(data = cm_nb, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "gray", high = "black") +
  labs(title = "Confusion Matrix - Naive Bayes") +
  theme_minimal()



# Predict probabilities
nb_prob_test <- predict(nb_model, test, type = "prob")[, "yes"]

# ROC and AUC
roc_nb <- roc(test$y, nb_prob_test)
plot(roc_nb, col = "darkblue", main = "ROC Curve - Naive Bayes")
cat("AUC:", auc(roc_nb), "\n")



#Tuning Naive Bayes
nb_ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Define tuning grid
nb_grid <- expand.grid(
  laplace = c(0, 1),
  usekernel = c(TRUE, FALSE),
  adjust = c(0.5, 1, 2)
)

# Train with tuning
set.seed(123)
nb_model_tuned <- train(
  y ~ ., 
  data = train,
  method = "naive_bayes",
  trControl = nb_ctrl,
  tuneGrid = nb_grid,
  metric = "ROC"
)

print(nb_model_tuned)


# Predict on test set
nb_pred_tuned <- predict(nb_model_tuned, test)

# Confusion matrix
conf_matrix_nb_tuned <- confusionMatrix(nb_pred_tuned, test$y, positive = "yes")

# Print performance
cat("Testing Accuracy (Tuned):", conf_matrix_nb_tuned$overall["Accuracy"], "\n")
cat("Kappa (Tuned):", conf_matrix_nb_tuned$overall["Kappa"], "\n")

# Full confusion matrix
print(conf_matrix_nb_tuned)

# 🔳 Plot Confusion Matrix
cm_nb_tuned <- as.data.frame(conf_matrix_nb_tuned$table)
colnames(cm_nb_tuned) <- c("Predicted", "Actual", "Freq")

ggplot(data = cm_nb_tuned, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "skyblue", high = "maroon") +
  labs(title = "Confusion Matrix - Naive Bayes After Tunig") +
  theme_minimal()


# Predict probabilities
nb_prob_tuned <- predict(nb_model_tuned, test, type = "prob")[, "yes"]

# ROC curve
roc_nb_tuned <- roc(test$y, nb_prob_tuned)
plot(roc_nb_tuned, col = "darkgreen", main = "ROC Curve - Tuned Naive Bayes")
cat("AUC (Tuned):", auc(roc_nb_tuned), "\n")



















#DECISION TREE
# Train a Decision Tree model with CV
set.seed(123)
dt_model <- train(
  y ~ ., 
  data = train,
  method = "rpart",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC"
)

# Print the model details
print(dt_model)

# Predict on test set
dt_pred_test <- predict(dt_model, test)

# Confusion matrix
conf_matrix_dt <- confusionMatrix(dt_pred_test, test$y, positive = "yes")

# Evaluation metrics
cat("Testing Accuracy :", conf_matrix_dt$overall["Accuracy"], "\n")
cat("Kappa (Test)     :", conf_matrix_dt$overall["Kappa"], "\n")

# Print full confusion matrix
print(conf_matrix_dt)

# 🔳 Confusion matrix plot
cm_dt <- as.data.frame(conf_matrix_dt$table)
colnames(cm_dt) <- c("Predicted", "Actual", "Freq")

ggplot(data = cm_dt, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "gray", high = "black") +
  labs(title = "Confusion Matrix - Decision Tree Before Tuning") +
  theme_minimal()


# Predict probabilities
dt_prob_test <- predict(dt_model, test, type = "prob")[, "yes"]

# ROC curve
roc_dt <- roc(test$y, dt_prob_test)
plot(roc_dt, col = "darkorange", main = "ROC Curve - Decision Tree Before Tuning")
cat("AUC:", auc(roc_dt), "\n")

# Plot the actual tree
rpart.plot(dt_model$finalModel, main = "Decision Tree Visualization")






##Decision #Tree Tuning
# Set tuning grid for cp (try small values for more complexity)
cp_grid <- expand.grid(cp = seq(0.001, 0.05, by = 0.005))

# Tune the DT model
set.seed(123)
dt_tuned <- train(
  y ~ ., 
  data = train,
  method = "rpart",
  metric = "ROC",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  tuneGrid = cp_grid
)

# Show best cp value and model details
print(dt_tuned)

# Predict with tuned DT
dt_pred_tuned <- predict(dt_tuned, test)

# Confusion Matrix
conf_matrix_dt_tuned <- confusionMatrix(dt_pred_tuned, test$y, positive = "yes")

# Print accuracy and kappa
cat("Testing Accuracy (Tuned):", conf_matrix_dt_tuned$overall["Accuracy"], "\n")
cat("Kappa (Tuned)            :", conf_matrix_dt_tuned$overall["Kappa"], "\n")

# Full confusion matrix
print(conf_matrix_dt_tuned)

# 🔳 Confusion matrix plot
cm_dt_tuned <- as.data.frame(conf_matrix_dt_tuned$table)
colnames(cm_dt_tuned) <- c("Predicted", "Actual", "Freq")

ggplot(data = cm_dt_tuned, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "skyblue", high = "maroon") +
  labs(title = "Confusion Matrix - Tuned Decision Tree") +
  theme_minimal()


# Predict probabilities
dt_prob_tuned <- predict(dt_tuned, test, type = "prob")[, "yes"]

# ROC and AUC
roc_dt_tuned <- roc(test$y, dt_prob_tuned)
plot(roc_dt_tuned, col = "darkgreen", main = "ROC Curve - Tuned Decision Tree")
cat("AUC (Tuned):", auc(roc_dt_tuned), "\n")


# Random Forest
# Train Random Forest using caret
set.seed(123)

rf_ctrl <- trainControl(
  method = "none",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

rf_model <- train(
  y ~ ., 
  data = train,
  method = "rf",
  trControl = rf_ctrl,
  metric = "ROC",
  tuneGrid = data.frame(mtry = 3),  # mtry can be tuned; here it's fixed
  ntree = 100  # Optional: control number of trees
)

#Print summary of the model
print(rf_model)


# Predict on test set
rf_pred_test <- predict(rf_model, test)

# Confusion Matrix
conf_matrix_rf <- confusionMatrix(rf_pred_test, test$y, positive = "yes")

# Print metrics
cat("Testing Accuracy :", conf_matrix_rf$overall["Accuracy"], "\n")
cat("Kappa (Test)     :", conf_matrix_rf$overall["Kappa"], "\n")

# Full confusion matrix
print(conf_matrix_rf)

# 🔳 Confusion matrix plot
cm_rf <- as.data.frame(conf_matrix_rf$table)
colnames(cm_rf) <- c("Predicted", "Actual", "Freq")

ggplot(data = cm_rf, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "skyblue", high = "navy") +
  labs(title = "Confusion Matrix - Random Forest") +
  theme_minimal()


# Predict probabilities
rf_prob_test <- predict(rf_model, test, type = "prob")[, "yes"]

# ROC curve
roc_rf <- roc(test$y, rf_prob_test)
plot(roc_rf, col = "forestgreen", main = "ROC Curve - Random Forest")
cat("AUC:", auc(roc_rf), "\n")




##Tuned Model
# Define training control with 5-fold CV
rf_ctrl_tuned <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Define grid of mtry values to try
mtry_grid <- expand.grid(mtry = c(2, 3, 4, 5, 6, 7, 8))

# Train RF model with tuning
set.seed(123)
rf_model_tuned <- train(
  y ~ ., 
  data = train,
  method = "ranger",  # MUCH faster
  trControl = rf_ctrl_tuned,
  tuneGrid = expand.grid(
    mtry = c(2, 3, 4, 5, 6), 
    splitrule = "gini",
    min.node.size = 5
  ),
  metric = "ROC",
  num.trees = 100,
  importance = "impurity"
)

# View best model
print(rf_model_tuned)

# Predictions
rf_pred_tuned <- predict(rf_model_tuned, test)

# Confusion Matrix
conf_matrix_rf_tuned <- confusionMatrix(rf_pred_tuned, test$y, positive = "yes")

# Print performance
cat("Testing Accuracy (Tuned):", conf_matrix_rf_tuned$overall["Accuracy"], "\n")
cat("Kappa (Tuned):", conf_matrix_rf_tuned$overall["Kappa"], "\n")

# Print full confusion matrix
print(conf_matrix_rf_tuned)

# 🔳 Confusion Matrix Plot
cm_rf_tuned <- as.data.frame(conf_matrix_rf_tuned$table)
colnames(cm_rf_tuned) <- c("Predicted", "Actual", "Freq")

ggplot(data = cm_rf_tuned, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "skyblue", high = "maroon") +
  labs(title = "Confusion Matrix - Random Forest After Tuning") +
  theme_minimal()


# Probabilities
rf_prob_tuned <- predict(rf_model_tuned, test, type = "prob")[, "yes"]

# ROC curve
roc_rf_tuned <- roc(test$y, rf_prob_tuned)
plot(roc_rf_tuned, col = "darkred", main = "ROC Curve - Random Forest After Tuning")
cat("AUC (Tuned):", auc(roc_rf_tuned), "\n")

# Plot all ROC curves before tuning
plot(roc_logit, col = "blue", lwd = 2, main = "ROC Curve - All Models (Before Tuning)")
lines(roc_knn, col = "purple", lwd = 2)
lines(roc_nb, col = "darkred", lwd = 2)
lines(roc_dt, col = "orange", lwd = 2)
lines(roc_rf, col = "forestgreen", lwd = 2)

legend("bottomright", legend = c("Logistic", "KNN", "Naive Bayes", "Decision Tree", "Random Forest"),
       col = c("blue", "purple", "darkred", "orange", "forestgreen"),
       lty = 1, lwd = 2)


# Plot all ROC curves after tuning
plot(roc_logit_tuned, col = "blue", lwd = 2, main = "ROC Curve - All Models (After Tuning)")
lines(roc_knn, col = "purple", lty = 2, lwd = 2)  # reusing knn_prob_test from tuned KNN
lines(roc_nb_tuned, col = "darkred", lwd = 2)
lines(roc_dt_tuned, col = "orange", lwd = 2)
lines(roc_rf_tuned, col = "forestgreen", lwd = 2)

legend("bottomright", legend = c("Logistic (Tuned)", "KNN (Tuned)", "Naive Bayes (Tuned)", "Decision Tree (Tuned)", "Random Forest (Tuned)"),
       col = c("blue", "purple", "darkred", "orange", "forestgreen"),
       lty = 1, lwd = 2)



