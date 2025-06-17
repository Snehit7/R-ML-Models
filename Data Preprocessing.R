setwd("C:/Users/snehi/OneDrive/Desktop/SnehitParajuli_Assessment_Part1")

all_bank<-read.csv("Bank Marketing Campaign.csv")

#Summary of the given dataset
summary(all_bank)

#Struture of the given dataset
str(all_bank)

#Dimensions of the dataset
dim(all_bank)

#Viewing first few rows
head(all_bank)


#Number of Customer who subscribed
table(all_bank$y)


#unique jobs
unique(all_bank$job)


#Most common contact method
table(all_bank$contact)

#average call duration for each subscription status

avg_duration <- all_bank %>%
  group_by(y) %>%
  summarise(Average_Duration = mean(duration, na.rm = TRUE)) %>%
  as.data.frame()

print(avg_duration)

#subscription rate per education level
education_subscription <- all_bank %>%
  group_by(education) %>%
  summarise(
    Total_Customers = n(),
    Subscribed = sum(y == "yes"),
    Subscription_Rate = round((Subscribed / Total_Customers) * 100, 2)
  ) %>%
  arrange(desc(Subscription_Rate)) %>%
  as.data.frame()

print(education_subscription)




#Descriptive statistics
numeric_vars <- all_bank %>% select_if(is.numeric)

# Compute descriptive statistics
descriptive_stats <- data.frame(
  Variable = colnames(numeric_vars),
  Mean = sapply(numeric_vars, mean, na.rm = TRUE),
  Median = sapply(numeric_vars, median, na.rm = TRUE),
  Std_Dev = sapply(numeric_vars, sd, na.rm = TRUE)
)

print(descriptive_stats)


#VISUALIZATION
#Box plot
ggplot(all_bank, aes(x = y, y = balance, fill = y)) +
  geom_boxplot() +
  labs(title = "Bank Balance by Subscription Status", x = "Subscribed?", y = "Balance (€)") +
  theme_minimal()

#Bar plot
ggplot(all_bank, aes(x = job, fill = y)) +
  geom_bar(position = "fill") +  # Fill to show proportion
  labs(title = "Subscription Rate by Job Type", x = "Job", y = "Proportion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


#Principal Component Analysis
# Selecting only numeric variables for PCA
numeric_data <- all_bank %>% select_if(is.numeric) %>% na.omit()


preprocess_params <- preProcess(numeric_data, method = c("center", "scale"))
scaled_data <- predict(preprocess_params, numeric_data)

# Run PCA
pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)

summary(pca_result)



# Check missing values in each column
colSums(is.na(all_bank))
#imputed data
library(mice)
imputed_data <- mice(all_bank, method = "pmm", m = 5)  # Predictive mean matching
all_bank <- complete(imputed_data)



# Boxplot to detect outliers for numerical variables
boxplot(all_bank$balance, main = "Balance Outliers")

# Identify outliers using IQR method
Q1 <- quantile(all_bank$balance, 0.25)
Q3 <- quantile(all_bank$balance, 0.75)
IQR_value <- Q3 - Q1

# Define outlier threshold
lower_bound <- Q1 - 1.5 * IQR_value
upper_bound <- Q3 + 1.5 * IQR_value

# Remove outliers
all_bank <- all_bank[all_bank$balance >= lower_bound & all_bank$balance <= upper_bound, ]



#Multicollinearity
cor_matrix <- cor(select_if(all_bank, is.numeric))  # Get correlation matrix
print(cor_matrix)

# Find highly correlated variables (above 0.8)
findCorrelation<- which(abs(cor_matrix)>0.8,arr.ind = TRUE)
print(findCorrelation)


# Investigate variables
# Checking variance of numeric variables
numeric_vars <- sapply(all_bank, is.numeric)
var_info <- sapply(all_bank[, numeric_vars, drop = FALSE], var)
low_var_vars <- names(var_info[var_info < 0.01])  # Identify low variance variables
print("Low variance variables:")
print(low_var_vars)

# Check for near-zero variance
nzv <- nearZeroVar(all_bank)
if (length(nzv) > 0) {
  print("Near-zero variance variables:")
  print(names(all_bank)[nzv])
} else {
  print("No near-zero variance variables detected.")
}

# Checking correlations to detect multicollinearity
cor_matrix <- cor(all_bank[, numeric_vars, drop = FALSE], use = "complete.obs")
high_cor_vars <- findCorrelation(cor_matrix, cutoff = 0.9)  # Remove highly correlated variables
print("Highly correlated variables to remove:")
print(names(all_bank)[high_cor_vars])

# Scaling numerical variables
all_bank_scaled <- all_bank
all_bank_scaled[, numeric_vars] <- lapply(all_bank[, numeric_vars, drop = FALSE], scale)

# Summary after scaling
summary(all_bank_scaled[, numeric_vars, drop = FALSE])
