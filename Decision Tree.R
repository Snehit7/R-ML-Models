library(rpart)
library(rpart.plot)
library(caret)
library(pROC)
library(ggplot2)
library(reshape2)

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
