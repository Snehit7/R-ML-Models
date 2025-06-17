setwd("C:/Users/Lenovo/OneDrive/Desktop/AML")
# Load required libraries
library(dplyr)
library(caret)
library(ggplot2)
library(pROC)


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


# Load caret package (already loaded if you're following from earlier)
library(caret)

# Set seed for reproducibility
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



