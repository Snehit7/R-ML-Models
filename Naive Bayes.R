library(rpart)
library(rpart.plot)
library(caret)
library(pROC)
library(ggplot2)
library(reshape2)
library(randomForest)
library(naivebayes)

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
