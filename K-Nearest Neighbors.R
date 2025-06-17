setwd("C:/Users/Lenovo/OneDrive/Desktop/AML")
# Load required libraries
library(dplyr)
library(caret)
library(ggplot2)
library(pROC)
library(class)

# Load the data
df <- read.csv("Bank Marketing Campaign.csv", stringsAsFactors = FALSE)

# View structure
str(df)

# Convert target variable to factor (binary classification)
df$y <- as.factor(df$y)

# Check and handle missing values
sum(is.na(df))  # Count total missing values

# For simplicity, remove rows with any NA (can also consider imputation)
df <- na.omit(df)

# Convert categorical variables to factors
categorical_cols <- c("job", "marital", "education", "default", 
                      "housing", "loan", "contact", "month", "poutcome")
df[categorical_cols] <- lapply(df[categorical_cols], as.factor)

# Scale numeric columns
num_cols <- sapply(df, is.numeric)
df[num_cols] <- scale(df[num_cols])

# Check the cleaned data
summary(df)
str(df)

set.seed(123)

# Create 70/30 split
trainIndex <- createDataPartition(df$y, p = 0.7, list = FALSE)
train <- df[trainIndex, ]
test  <- df[-trainIndex, ]

# Confirm sizes
cat("Training set size:", nrow(train), "\n")
cat("Testing set size:", nrow(test), "\n")

# Check class balance
prop.table(table(train$y))
prop.table(table(test$y))



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



