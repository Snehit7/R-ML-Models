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



# Load required library
library(randomForest)

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




# Extract one tree from the tuned random forest model
# Use the same data and formula to grow a single tree using rpart for illustration
tree_from_rf <- rpart(
  y ~ ., 
  data = train,
  method = "class", 
  control = rpart.control(cp = 0.01)  # cp can be adjusted
)

# Plot the single tree
rpart.plot(tree_from_rf, main = "Example Tree - Random Forest Approximation")



