##################################################################################
# Hiring R Analysis
# Sarah K. Stein & Samaksh (Avi) Goyal
#
# This file does many things.
# 1. Prep stage, it loads all applicants answers from Q1, Q2, Q3 and filters
# 2. Briefly attempts STM models for Q1
# 3. Builds Models (ie. building blocks to runs a regression for hired as DV and linguistic measure + covariates as IV)
# 4. TFIDF and Build Regressions
#
##################################################################################

library(tm)
library(SnowballC)
library(NLP)
library(qlcMatrix)
library(proxy)
library(ggplot2)
library(parallel)
library(lsa)
library(stargazer)
library(plyr)
library(lme4)
library(optimx)
library(stm)
library(quanteda)


load("SteinData/jobvite_1_2_merged_anonymized.RData")
data <- anon

# Remove duplicate applications from the same individual
hired_ids <- data[data$Hired==1, "Jobvite.ID"]
hired_ids_dup <- data[data$Jobvite.ID %in% hired_ids, c("Jobvite.ID", "Hired")]
hired_ids_dup <- hired_ids_dup[order(hired_ids_dup$Jobvite.ID, -hired_ids_dup$Hired),]
data <- data[order(data$Jobvite.ID, -data$Hired),]
data <- data[!duplicated(data$Jobvite.ID),]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q1 Prep: AnswerCorpus_Centroid.py, EmailCorpus_Centroid.py, AnswerCorpus_WMD.py, EmailCorpus_WMD.py
# We need to filter on "kept" responses after removing stop words when we made word embedding model
# This is set in "ones_to_keep" variable

summary(data$Q1!="")

# Create corpus
q1 <- VCorpus(VectorSource(data$Q1[data$Q1!=""]))
ones_to_keep <- scan("SteinData/np_q1_keep_Q1_index.txt") #indicies generated to keep
ones_to_keep <- ones_to_keep + 1 # R is 1 indexed
q1 <- q1[ones_to_keep]
writeLines(as.character(q1[[1]]))

# Transfromations
q1 <- tm_map(q1, removeNumbers)
q1 <- tm_map(q1, removePunctuation)
q1 <- tm_map(q1, content_transformer(tolower))
q1 <- tm_map(q1, removeWords, stopwords("english"))
q1 <- tm_map(q1, stemDocument)
q1 <- tm_map(q1, stripWhitespace)

# Unigram frequency
dtm1 <- DocumentTermMatrix(q1)
inspect(dtm1[1:10, 1:10])
q1_wc <- log(rowSums(as.matrix(dtm1))+1)
q1_wc <- q1_wc[ones_to_keep]

q1_gender <- as.character(data$Gender[data$Q1!=""])
q1_gender <- q1_gender[ones_to_keep]

q1_race <- data$Race[data$Q1!=""]
q1_race <- q1_race[ones_to_keep]

q1_source <- as.character(data$Source[data$Q1!=""])
q1_source <- q1_source[ones_to_keep]

q1_title <- as.character(data$RequisitionTitle[data$Q1!=""])
q1_title <- q1_title[ones_to_keep]

q1_hirelist <- as.numeric(data$Hired[data$Q1!=""])
q1_hirelist <- q1_hirelist[ones_to_keep]

q1_id <- data$Jobvite.ID[data$Q1!=""]
q1_id <- q1_id[ones_to_keep]

q1_dflag <- data$DateFlag[data$Q1!=""]
q1_dflag <- q1_dflag[ones_to_keep]

q1_oflag <- data$TopOrgFlag[data$Q1!=""]
q1_oflag <- q1_oflag[ones_to_keep]

q1_uflag <- data$TopUniFlag[data$Q1!=""]
q1_uflag <- q1_uflag[ones_to_keep]

q1_intid <- data[grep("Int", names(data), value=TRUE)][data$Q1!="",][ones_to_keep,]
#q1_intid <- q1_intid[ones_to_keep]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q1 Prep: Sent_Bert.py, USE_Google.py
# Use all applicant responses so all are "kept"

summary(data$Q1!="")

# Create corpus
q1 <- VCorpus(VectorSource(data$Q1[data$Q1!=""]))
writeLines(as.character(q1[[1]]))

# Transfromations
q1 <- tm_map(q1, removeNumbers)
q1 <- tm_map(q1, removePunctuation)
q1 <- tm_map(q1, content_transformer(tolower))
q1 <- tm_map(q1, removeWords, stopwords("english"))
q1 <- tm_map(q1, stemDocument)
q1 <- tm_map(q1, stripWhitespace)

# Unigram frequency
dtm1 <- DocumentTermMatrix(q1)
inspect(dtm1[1:10, 1:10])
q1_wc <- log(rowSums(as.matrix(dtm1))+1)
q1_gender <- as.character(data$Gender[data$Q1!=""])
q1_race <- data$Race[data$Q1!=""]
q1_source <- as.character(data$Source[data$Q1!=""])
q1_title <- as.character(data$RequisitionTitle[data$Q1!=""])
q1_hirelist <- as.numeric(data$Hired[data$Q1!=""])
q1_id <- data$Jobvite.ID[data$Q1!=""]
q1_dflag <- data$DateFlag[data$Q1!=""]
q1_oflag <- data$TopOrgFlag[data$Q1!=""]
q1_uflag <- data$TopUniFlag[data$Q1!=""]
q1_intid <- data[grep("Int", names(data), value=TRUE)][data$Q1!="",]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q2 Prep: AnswerCorpus_Centroid.py, EmailCorpus_Centroid.py, AnswerCorpus_WMD.py, EmailCorpus_WMD.py
# We need to filter on "kept" responses after removing stop words when we made word embedding model
# This is set in "ones_to_keep" variable

summary(data$Q2!="")

# Create corpus
q2 <- VCorpus(VectorSource(data$Q2[data$Q2!=""]))
ones_to_keep <- scan("SteinData/np_q2_keep_Q2_index.txt")
ones_to_keep <- ones_to_keep + 1
q2 <- q2[ones_to_keep]
writeLines(as.character(q2[[1]]))

# Transfromations
q2 <- tm_map(q2, removeNumbers)
q2 <- tm_map(q2, removePunctuation)
q2 <- tm_map(q2, content_transformer(tolower))
q2 <- tm_map(q2, removeWords, stopwords("english"))
q2 <- tm_map(q2, stemDocument)
q2 <- tm_map(q2, stripWhitespace)

# Unigram frequency
dtm2 <- DocumentTermMatrix(q2)
inspect(dtm2[1:10, 1:10])
q2_wc <- log(rowSums(as.matrix(dtm2))+1)
q2_wc <- q2_wc[ones_to_keep]

q2_gender <- as.character(data$Gender[data$Q2!=""])
q2_gender <- q2_gender[ones_to_keep]

q2_race <- data$Race[data$Q2!=""]
q2_race <- q2_race[ones_to_keep]

q2_source <- as.character(data$Source[data$Q2!=""])
q2_source <- q2_source[ones_to_keep]

q2_title <- as.character(data$RequisitionTitle[data$Q2!=""])
q2_title <- q2_title[ones_to_keep]

q2_hirelist <- as.numeric(data$Hired[data$Q2!=""])
q2_hirelist <- q2_hirelist[ones_to_keep]

q2_id <- data$Jobvite.ID[data$Q2!=""]
q2_id <- q2_id[ones_to_keep]

q2_dflag <- data$DateFlag[data$Q2!=""]
q2_dflag <- q2_dflag[ones_to_keep]

q2_oflag <- data$TopOrgFlag[data$Q2!=""]
q2_oflag <- q2_oflag[ones_to_keep]

q2_uflag <- data$TopUniFlag[data$Q2!=""]
q2_uflag <- q2_uflag[ones_to_keep]

q2_intid <- data[grep("Int", names(data), value=TRUE)][data$Q2!="",][ones_to_keep,]
#q2_intid <- q2_intid[ones_to_keep]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q2 Prep: Sent_Bert.py, USE_Google.py
# Use all applicant responses so all are "kept"

summary(data$Q2!="")

# Create corpus
q2 <- VCorpus(VectorSource(data$Q2[data$Q2!=""]))
writeLines(as.character(q2[[1]]))

# Transfromations
q2 <- tm_map(q2, removeNumbers)
q2 <- tm_map(q2, removePunctuation)
q2 <- tm_map(q2, content_transformer(tolower))
q2 <- tm_map(q2, removeWords, stopwords("english"))
q2 <- tm_map(q2, stemDocument)
q2 <- tm_map(q2, stripWhitespace)

# Unigram frequency
dtm2 <- DocumentTermMatrix(q2)
inspect(dtm2[1:10, 1:10])
q2_wc <- log(rowSums(as.matrix(dtm2))+1)
q2_gender <- as.character(data$Gender[data$Q2!=""])
q2_race <- data$Race[data$Q2!=""]
q2_source <- as.character(data$Source[data$Q2!=""])
q2_title <- as.character(data$RequisitionTitle[data$Q2!=""])
q2_hirelist <- as.numeric(data$Hired[data$Q2!=""])
q2_id <- data$Jobvite.ID[data$Q2!=""]
q2_dflag <- data$DateFlag[data$Q2!=""]
q2_oflag <- data$TopOrgFlag[data$Q2!=""]
q2_uflag <- data$TopUniFlag[data$Q2!=""]
q2_intid <- data[grep("Int", names(data), value=TRUE)][data$Q2!="",]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q3 Prep: AnswerCorpus_Centroid.py, EmailCorpus_Centroid.py, AnswerCorpus_WMD.py, EmailCorpus_WMD.py
# We need to filter on "kept" responses after removing stop words when we made word embedding model
# This is set in "ones_to_keep" variable

summary(data$Q3!="")

# Create corpus
q3 <- VCorpus(VectorSource(data$Q3[data$Q3!=""]))
ones_to_keep <- scan("SteinData/np_q3_keep_Q3_index.txt")
ones_to_keep <- ones_to_keep + 1
q3 <- q3[ones_to_keep]
writeLines(as.character(q3[[1]]))

# Transfromations
q3 <- tm_map(q3, removeNumbers)
q3 <- tm_map(q3, removePunctuation)
q3 <- tm_map(q3, content_transformer(tolower))
q3 <- tm_map(q3, removeWords, stopwords("english"))
q3 <- tm_map(q3, stemDocument)
q3 <- tm_map(q3, stripWhitespace)

# Unigram frequency
dtm3 <- DocumentTermMatrix(q3)
inspect(dtm3[1:10, 1:10])
q3_wc <- log(rowSums(as.matrix(dtm3))+1)
q3_wc <- q3_wc[ones_to_keep]

q3_gender <- as.character(data$Gender[data$Q3!=""])
q3_gender <- q3_gender[ones_to_keep]

q3_race <- data$Race[data$Q3!=""]
q3_race <- q3_race[ones_to_keep]

q3_source <- as.character(data$Source[data$Q3!=""])
q3_source <- q3_source[ones_to_keep]

q3_title <- as.character(data$RequisitionTitle[data$Q3!=""])
q3_title <- q3_title[ones_to_keep]

q3_hirelist <- as.numeric(data$Hired[data$Q3!=""])
q3_hirelist <- q3_hirelist[ones_to_keep]

q3_id <- data$Jobvite.ID[data$Q3!=""]
q3_id <- q3_id[ones_to_keep]

q3_dflag <- data$DateFlag[data$Q3!=""]
q3_dflag <- q3_dflag[ones_to_keep]

q3_oflag <- data$TopOrgFlag[data$Q3!=""]
q3_oflag <- q3_oflag[ones_to_keep]

q3_uflag <- data$TopUniFlag[data$Q3!=""]
q3_uflag <- q3_uflag[ones_to_keep]

q3_intid <- data[grep("Int", names(data), value=TRUE)][data$Q3!="",][ones_to_keep,]
#q3_intid <- q3_intid[ones_to_keep]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q3 Prep: Sent_Bert.py, USE_Google.py
# Use all applicant responses so all are "kept"

summary(data$Q3!="")

# Create corpus
q3 <- VCorpus(VectorSource(data$Q3[data$Q3!=""]))
writeLines(as.character(q3[[1]]))

# Transfromations
q3 <- tm_map(q3, removeNumbers)
q3 <- tm_map(q3, removePunctuation)
q3 <- tm_map(q3, content_transformer(tolower))
q3 <- tm_map(q3, removeWords, stopwords("english"))
q3 <- tm_map(q3, stemDocument)
q3 <- tm_map(q3, stripWhitespace)

# Unigram frequency
dtm3 <- DocumentTermMatrix(q3)
inspect(dtm3[1:10, 1:10])
q3_wc <- log(rowSums(as.matrix(dtm3))+1)
q3_gender <- as.character(data$Gender[data$Q3!=""])
q3_race <- data$Race[data$Q3!=""]
q3_source <- as.character(data$Source[data$Q3!=""])
q3_title <- as.character(data$RequisitionTitle[data$Q3!=""])
q3_hirelist <- as.numeric(data$Hired[data$Q3!=""])
q3_id <- data$Jobvite.ID[data$Q3!=""]
q3_dflag <- data$DateFlag[data$Q3!=""]
q3_oflag <- data$TopOrgFlag[data$Q3!=""]
q3_uflag <- data$TopUniFlag[data$Q3!=""]
q3_intid <- data[grep("Int", names(data), value=TRUE)][data$Q3!="",]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ALL QUESTIONS CORPUS
summary(data$Q1!="" | data$Q2!="" | data$Q3!="")

# Create corpus
data$Q <- paste(data$Q1, data$Q2, data$Q3, sep=" ")
q <- VCorpus(VectorSource(data$Q[data$Q!=""]))
writeLines(as.character(q[[1]]))

# Transfromations
q <- tm_map(q, removeNumbers)
q <- tm_map(q, removePunctuation)
q <- tm_map(q, content_transformer(tolower))
q <- tm_map(q, removeWords, stopwords("english"))
q <- tm_map(q, stemDocument)
q <- tm_map(q, stripWhitespace)

# Unigram frequency
dtm <- DocumentTermMatrix(q)
inspect(dtm[1:10, 1:10])
q_wc <- log(rowSums(as.matrix(dtm))+1)
q_gender <- as.character(data$Gender[data$Q!=""])
q_race <- data$Race[data$Q!=""]
q_source <- as.character(data$Source[data$Q!=""])
q_title <- as.character(data$RequisitionTitle[data$Q!=""])
q_hirelist <- as.numeric(data$Hired[data$Q!=""])
q_id <- data$Jobvite.ID[data$Q!=""]
q_dflag <- data$DateFlag[data$Q!=""]
q_oflag <- data$TopOrgFlag[data$Q!=""]
q_uflag <- data$TopUniFlag[data$Q!=""]
q_intid <- data[grep("Int", names(data), value=TRUE)][data$Q!="",]

####################################################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STM Trial: After making dtm1 from Q1 follow:
# https://cran.r-project.org/web/packages/stm/vignettes/stmVignette.pdf page 16 to try running STM estimate Effects

dfm1 <- as.dfm(dtm1)
fit0 <- stm(dfm1, K = 10, max.em.its = 75)
fit5 <- stm(dfm1, K = 5, max.em.its = 75)
fit15 <- stm(dfm1, K = 15, max.em.its = 75)

dfm1 <- as.dfm(dtm1)
fit_hirelist <- stm(dfm1, K = 10, prevalence =~ q1_hirelist, max.em.its = 75, init.type="Spectral")
prep <- estimateEffect(1:10 ~  q1_hirelist, fit_hirelist, uncertainty = "Global")
summary(prep, topic=c(1,2,3,4,5,6,7,8,9,10))
labelTopics(fit_hirelist, c(8,9))

####################################################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Q1 Models:
sim_score <- scan("SteinData/np_q1_SENT_BERT_similarity_score.txt") #fill in appropriate similarity metric

mdf1 <- data.frame(q1_hirelist, sim_score, q1_wc, q1_gender, q1_race, q1_source, q1_title, q1_dflag, q1_oflag, q1_uflag, q1_intid)
colnames(mdf1) <- c("hired", "sim_score", "log_wc", "gender", "race", "source", "title", "dflag", "oflag", "uflag", paste("intid", 1:161, sep=""))
mdf1$gender <- factor(mdf1$gender, levels=c("Male", "Female"))
mdf1$race <- factor(mdf1$race, levels=c("White", "Black or African American", "Asian", "Hispanic or Latino", "Two or more races"))
mdf1$i.age <- 2017 - (mdf1$dflag - 22)
mdf1$source <- tolower(as.character(mdf1$source))
mdf1$source <- ifelse(grepl("employee", mdf1$source), yes="Employee", no=mdf1$source)
mdf1$source <- ifelse(grepl("internal", mdf1$source), yes="Internal", no=mdf1$source)
mdf1$source <- ifelse(grepl("recruiter", mdf1$source), yes="Recruiter", no=mdf1$source)
mdf1$source <- ifelse(grepl("university", mdf1$source), yes="University", no=mdf1$source)
mdf1$source <- ifelse(grepl("sungevity career site", mdf1$source), yes="Career Site", no=mdf1$source)
mdf1$source <- ifelse(grepl("craigslist", mdf1$source), yes="Craigslist", no=mdf1$source)
mdf1$source <- ifelse(grepl("indeed", mdf1$source), yes="Indeed", no=mdf1$source)
mdf1$source <- ifelse(grepl("snagajob", mdf1$source), yes="SnagaJob", no=mdf1$source)
mdf1$source <- ifelse(grepl("glassdoor", mdf1$source), yes="Glassdoor", no=mdf1$source)
mdf1$source <- ifelse(grepl("linkedin", mdf1$source), yes="LinkedIn", no=mdf1$source)
mdf1$source <- ifelse(grepl("hotjobs", mdf1$source), yes="HotJobs", no=mdf1$source)
mdf1$source <- ifelse(grepl("dice", mdf1$source), yes="Dice", no=mdf1$source)
mdf1$source <- ifelse(grepl("monster", mdf1$source), yes="Monster", no=mdf1$source)
mdf1$source <- ifelse(grepl("simplyhired", mdf1$source), yes="SimplyHired", no=mdf1$source)
mdf1$source <- ifelse(grepl("Employee", mdf1$source) | grepl("Internal", mdf1$source) | grepl("Recruiter", mdf1$source)
                      | grepl("University", mdf1$source) | grepl("Career Site", mdf1$source) | grepl("Craigslist", mdf1$source)
                      | grepl("Indeed", mdf1$source)| grepl("SnagaJob", mdf1$source) | grepl("Glassdoor", mdf1$source)
                      | grepl("LinkedIn", mdf1$source) | grepl("HotJobs", mdf1$source) | grepl("Dice", mdf1$source)
                      | grepl("Monster", mdf1$source) | grepl("SimplyHired", mdf1$source), yes=mdf1$source, no="Other")
mdf1$source <- ifelse(mdf1$source=="Craigslist" | mdf1$source=="Indeed" | mdf1$source=="SnagaJob" | mdf1$source=="Glassdoor" |
                        mdf1$source=="LinkedIn" | mdf1$source=="HotJobs" | mdf1$source=="Dice" | mdf1$source=="Monster" |
                        mdf1$source=="SimplyHired", yes="Job Board", no=mdf1$source)
mdf1$source <- factor(mdf1$source, levels=c("Career Site", "Employee", "Internal", "Recruiter", "University",
                                            "Job Board", "Other"))
mdf1$recruiter <- ifelse(mdf1$source=="Recruiter", 1, 0)
mdf1$referral <- ifelse(mdf1$source=="Employee", 1, 0)
mdf1$title <- as.character(mdf1$title)
mdf1$manager <- ifelse(grepl("Manager", mdf1$title), yes=1, no=0)
mdf1$manager <- ifelse(grepl("Director", mdf1$title), yes=1, no=mdf1$manager)
mdf1$manager <- ifelse(grepl("VP", mdf1$title), yes=1, no=mdf1$manager)
mdf1$manager <- ifelse(grepl("President", mdf1$title), yes=1, no=mdf1$manager)

#rename and save model
save(mdf1, file="hiring_q1_simscore_SentBert.RData")

# Regression dataframe
#Q2 Models:
sim_score <- scan("SteinData/np_q2_similarity_score_Email_lenwmd.txt") #fill in appropriate similarity metric

mdf2 <- data.frame(q2_hirelist, sim_score, q2_wc, q2_gender, q2_race, q2_source, q2_title, q2_dflag, q2_oflag, q2_uflag, q2_intid)
colnames(mdf2) <- c("hired", "sim_score", "log_wc", "gender", "race", "source", "title", "dflag", "oflag", "uflag", paste("intid", 1:161, sep=""))
mdf2$gender <- factor(mdf2$gender, levels=c("Male", "Female"))
mdf2$race <- factor(mdf2$race, levels=c("White", "Black or African American", "Asian", "Hispanic or Latino", "Two or more races"))
mdf2$i.age <- 2017 - (mdf2$dflag - 22)
mdf2$source <- tolower(as.character(mdf2$source))
mdf2$source <- ifelse(grepl("employee", mdf2$source), yes="Employee", no=mdf2$source)
mdf2$source <- ifelse(grepl("internal", mdf2$source), yes="Internal", no=mdf2$source)
mdf2$source <- ifelse(grepl("recruiter", mdf2$source), yes="Recruiter", no=mdf2$source)
mdf2$source <- ifelse(grepl("university", mdf2$source), yes="University", no=mdf2$source)
mdf2$source <- ifelse(grepl("sungevity career site", mdf2$source), yes="Career Site", no=mdf2$source)
mdf2$source <- ifelse(grepl("craigslist", mdf2$source), yes="Craigslist", no=mdf2$source)
mdf2$source <- ifelse(grepl("indeed", mdf2$source), yes="Indeed", no=mdf2$source)
mdf2$source <- ifelse(grepl("snagajob", mdf2$source), yes="SnagaJob", no=mdf2$source)
mdf2$source <- ifelse(grepl("glassdoor", mdf2$source), yes="Glassdoor", no=mdf2$source)
mdf2$source <- ifelse(grepl("linkedin", mdf2$source), yes="LinkedIn", no=mdf2$source)
mdf2$source <- ifelse(grepl("hotjobs", mdf2$source), yes="HotJobs", no=mdf2$source)
mdf2$source <- ifelse(grepl("dice", mdf2$source), yes="Dice", no=mdf2$source)
mdf2$source <- ifelse(grepl("monster", mdf2$source), yes="Monster", no=mdf2$source)
mdf2$source <- ifelse(grepl("simplyhired", mdf2$source), yes="SimplyHired", no=mdf2$source)
mdf2$source <- ifelse(grepl("Employee", mdf2$source) | grepl("Internal", mdf2$source) | grepl("Recruiter", mdf2$source)
                      | grepl("University", mdf2$source) | grepl("Career Site", mdf2$source) | grepl("Craigslist", mdf2$source)
                      | grepl("Indeed", mdf2$source)| grepl("SnagaJob", mdf2$source) | grepl("Glassdoor", mdf2$source)
                      | grepl("LinkedIn", mdf2$source) | grepl("HotJobs", mdf2$source) | grepl("Dice", mdf2$source)
                      | grepl("Monster", mdf2$source) | grepl("SimplyHired", mdf2$source), yes=mdf2$source, no="Other")
mdf2$source <- ifelse(mdf2$source=="Craigslist" | mdf2$source=="Indeed" | mdf2$source=="SnagaJob" | mdf2$source=="Glassdoor" |
                        mdf2$source=="LinkedIn" | mdf2$source=="HotJobs" | mdf2$source=="Dice" | mdf2$source=="Monster" |
                        mdf2$source=="SimplyHired", yes="Job Board", no=mdf2$source)
mdf2$source <- factor(mdf2$source, levels=c("Career Site", "Employee", "Internal", "Recruiter", "University",
                                            "Job Board", "Other"))
mdf2$recruiter <- ifelse(mdf2$source=="Recruiter", 1, 0)
mdf2$referral <- ifelse(mdf2$source=="Employee", 1, 0)
mdf2$title <- as.character(mdf2$title)
mdf2$manager <- ifelse(grepl("Manager", mdf2$title), yes=1, no=0)
mdf2$manager <- ifelse(grepl("Director", mdf2$title), yes=1, no=mdf2$manager)
mdf2$manager <- ifelse(grepl("VP", mdf2$title), yes=1, no=mdf2$manager)
mdf2$manager <- ifelse(grepl("President", mdf2$title), yes=1, no=mdf2$manager)
#rename and save model
save(mdf2, file="hiring_q2_simscore_EmailWMD.RData")


# Q3 Models:
sim_score <- scan("SteinData/np_q3_similarity_score_Email_lenwmd.txt") #fill in appropriate similarity metric

mdf3 <- data.frame(q3_hirelist, sim_score, q3_wc, q3_gender, q3_race, q3_source, q3_title, q3_dflag, q3_oflag, q3_uflag, q3_intid)
colnames(mdf3) <- c("hired", "sim_score", "log_wc", "gender", "race", "source", "title", "dflag", "oflag", "uflag", paste("intid", 1:161, sep=""))
mdf3$gender <- factor(mdf3$gender, levels=c("Male", "Female"))
mdf3$race <- factor(mdf3$race, levels=c("White", "Black or African American", "Asian", "Hispanic or Latino", "Two or more races"))
mdf3$i.age <- 2017 - (mdf3$dflag - 22)
mdf3$source <- tolower(as.character(mdf3$source))
mdf3$source <- ifelse(grepl("employee", mdf3$source), yes="Employee", no=mdf3$source)
mdf3$source <- ifelse(grepl("internal", mdf3$source), yes="Internal", no=mdf3$source)
mdf3$source <- ifelse(grepl("recruiter", mdf3$source), yes="Recruiter", no=mdf3$source)
mdf3$source <- ifelse(grepl("university", mdf3$source), yes="University", no=mdf3$source)
mdf3$source <- ifelse(grepl("sungevity career site", mdf3$source), yes="Career Site", no=mdf3$source)
mdf3$source <- ifelse(grepl("craigslist", mdf3$source), yes="Craigslist", no=mdf3$source)
mdf3$source <- ifelse(grepl("indeed", mdf3$source), yes="Indeed", no=mdf3$source)
mdf3$source <- ifelse(grepl("snagajob", mdf3$source), yes="SnagaJob", no=mdf3$source)
mdf3$source <- ifelse(grepl("glassdoor", mdf3$source), yes="Glassdoor", no=mdf3$source)
mdf3$source <- ifelse(grepl("linkedin", mdf3$source), yes="LinkedIn", no=mdf3$source)
mdf3$source <- ifelse(grepl("hotjobs", mdf3$source), yes="HotJobs", no=mdf3$source)
mdf3$source <- ifelse(grepl("dice", mdf3$source), yes="Dice", no=mdf3$source)
mdf3$source <- ifelse(grepl("monster", mdf3$source), yes="Monster", no=mdf3$source)
mdf3$source <- ifelse(grepl("simplyhired", mdf3$source), yes="SimplyHired", no=mdf3$source)
mdf3$source <- ifelse(grepl("Employee", mdf3$source) | grepl("Internal", mdf3$source) | grepl("Recruiter", mdf3$source)
                      | grepl("University", mdf3$source) | grepl("Career Site", mdf3$source) | grepl("Craigslist", mdf3$source)
                      | grepl("Indeed", mdf3$source)| grepl("SnagaJob", mdf3$source) | grepl("Glassdoor", mdf3$source)
                      | grepl("LinkedIn", mdf3$source) | grepl("HotJobs", mdf3$source) | grepl("Dice", mdf3$source)
                      | grepl("Monster", mdf3$source) | grepl("SimplyHired", mdf3$source), yes=mdf3$source, no="Other")
mdf3$source <- ifelse(mdf3$source=="Craigslist" | mdf3$source=="Indeed" | mdf3$source=="SnagaJob" | mdf3$source=="Glassdoor" |
                        mdf3$source=="LinkedIn" | mdf3$source=="HotJobs" | mdf3$source=="Dice" | mdf3$source=="Monster" |
                        mdf3$source=="SimplyHired", yes="Job Board", no=mdf3$source)
mdf3$source <- factor(mdf3$source, levels=c("Career Site", "Employee", "Internal", "Recruiter", "University",
                                            "Job Board", "Other"))
mdf3$recruiter <- ifelse(mdf3$source=="Recruiter", 1, 0)
mdf3$referral <- ifelse(mdf3$source=="Employee", 1, 0)
mdf3$title <- as.character(mdf3$title)
mdf3$manager <- ifelse(grepl("Manager", mdf3$title), yes=1, no=0)
mdf3$manager <- ifelse(grepl("Director", mdf3$title), yes=1, no=mdf3$manager)
mdf3$manager <- ifelse(grepl("VP", mdf3$title), yes=1, no=mdf3$manager)
mdf3$manager <- ifelse(grepl("President", mdf3$title), yes=1, no=mdf3$manager)
#rename and save model
save(mdf3, file="hiring_q3_simscore_EmailWMD.RData")

####################################################################################################

# TF-IDF

# Q1
d <- as.matrix(dtm1)
d_binary <- ifelse(d >= 1, 1, 0)
tf1 <- d
idf1 <- log(nrow(d)/colSums(d_binary))
tfidf1 <- d
for (word in names(idf1)){
  tfidf1[,word] <- tf1[,word] * idf1[word]
}
tfidf1_means <- colMeans(tfidf1)
tfidf1_means <- rev(tfidf1_means[order(tfidf1_means)])
tfidf1_means[1:10]

# Q1 similarity weighted by hiring status
tfidf1 <- as.matrix(t(tfidf1))
tfidf1[1:10, 1:10]
s1 <- cosine(tfidf1)
diag(s1) <- 0
s1[is.na(s1)] <- 0
v1 <- q1_hirelist
wt_cossim <- scale(t(v1 %*% s1))
cossim <- scale(t(rep(1/length(v1), length(v1)) %*% s1))
save(s1, file="s1.RData")
save(q1_id, file="q1_id.RData")
save(q1_hirelist, file="q1_hirelist.RData")

# Regression dataframe
mdf1 <- data.frame(q1_hirelist, wt_cossim, cossim, q1_wc, q1_gender, q1_race, q1_source, q1_title, q1_dflag, q1_oflag, q1_uflag, q1_intid)
colnames(mdf1) <- c("hired", "wt_cossim", "cossim", "log_wc", "gender", "race", "source", "title", "dflag", "oflag", "uflag", paste("intid", 1:161, sep=""))
mdf1$gender <- factor(mdf1$gender, levels=c("Male", "Female"))
mdf1$race <- factor(mdf1$race, levels=c("White", "Black or African American", "Asian", "Hispanic or Latino", "Two or more races"))
mdf1$i.age <- 2017 - (mdf1$dflag - 22)
mdf1$source <- tolower(as.character(mdf1$source))
mdf1$source <- ifelse(grepl("employee", mdf1$source), yes="Employee", no=mdf1$source)
mdf1$source <- ifelse(grepl("internal", mdf1$source), yes="Internal", no=mdf1$source)
mdf1$source <- ifelse(grepl("recruiter", mdf1$source), yes="Recruiter", no=mdf1$source)
mdf1$source <- ifelse(grepl("university", mdf1$source), yes="University", no=mdf1$source)
mdf1$source <- ifelse(grepl("sungevity career site", mdf1$source), yes="Career Site", no=mdf1$source)
mdf1$source <- ifelse(grepl("craigslist", mdf1$source), yes="Craigslist", no=mdf1$source)
mdf1$source <- ifelse(grepl("indeed", mdf1$source), yes="Indeed", no=mdf1$source)
mdf1$source <- ifelse(grepl("snagajob", mdf1$source), yes="SnagaJob", no=mdf1$source)
mdf1$source <- ifelse(grepl("glassdoor", mdf1$source), yes="Glassdoor", no=mdf1$source)
mdf1$source <- ifelse(grepl("linkedin", mdf1$source), yes="LinkedIn", no=mdf1$source)
mdf1$source <- ifelse(grepl("hotjobs", mdf1$source), yes="HotJobs", no=mdf1$source)
mdf1$source <- ifelse(grepl("dice", mdf1$source), yes="Dice", no=mdf1$source)
mdf1$source <- ifelse(grepl("monster", mdf1$source), yes="Monster", no=mdf1$source)
mdf1$source <- ifelse(grepl("simplyhired", mdf1$source), yes="SimplyHired", no=mdf1$source)
mdf1$source <- ifelse(grepl("Employee", mdf1$source) | grepl("Internal", mdf1$source) | grepl("Recruiter", mdf1$source)
                     | grepl("University", mdf1$source) | grepl("Career Site", mdf1$source) | grepl("Craigslist", mdf1$source)
                     | grepl("Indeed", mdf1$source)| grepl("SnagaJob", mdf1$source) | grepl("Glassdoor", mdf1$source)
                     | grepl("LinkedIn", mdf1$source) | grepl("HotJobs", mdf1$source) | grepl("Dice", mdf1$source)
                     | grepl("Monster", mdf1$source) | grepl("SimplyHired", mdf1$source), yes=mdf1$source, no="Other")
mdf1$source <- ifelse(mdf1$source=="Craigslist" | mdf1$source=="Indeed" | mdf1$source=="SnagaJob" | mdf1$source=="Glassdoor" |
                       mdf1$source=="LinkedIn" | mdf1$source=="HotJobs" | mdf1$source=="Dice" | mdf1$source=="Monster" |
                       mdf1$source=="SimplyHired", yes="Job Board", no=mdf1$source)
mdf1$source <- factor(mdf1$source, levels=c("Career Site", "Employee", "Internal", "Recruiter", "University",
                                          "Job Board", "Other"))
mdf1$recruiter <- ifelse(mdf1$source=="Recruiter", 1, 0)
mdf1$referral <- ifelse(mdf1$source=="Employee", 1, 0)
mdf1$title <- as.character(mdf1$title)
mdf1$manager <- ifelse(grepl("Manager", mdf1$title), yes=1, no=0)
mdf1$manager <- ifelse(grepl("Director", mdf1$title), yes=1, no=mdf1$manager)
mdf1$manager <- ifelse(grepl("VP", mdf1$title), yes=1, no=mdf1$manager)
mdf1$manager <- ifelse(grepl("President", mdf1$title), yes=1, no=mdf1$manager)
save(mdf1, file="cossim_dataframes/hiring_q1_cossim.RData")

# Q2
d <- as.matrix(dtm2)
d_binary <- ifelse(d >= 1, 1, 0)
tf2 <- d
idf2 <- log(nrow(d)/colSums(d_binary))
tfidf2 <- d
for (word in names(idf2)){
  tfidf2[,word] <- tf2[,word] * idf2[word]
}
tfidf2_means <- colMeans(tfidf2)
tfidf2_means <- rev(tfidf2_means[order(tfidf2_means)])
tfidf2_means[1:10]

# Q2 similarity weighted by hiring status
tfidf2 <- as.matrix(t(tfidf2))
tfidf2[1:10, 1:10]
s2 <- cosine(tfidf2)
diag(s2) <- 0
s2[is.na(s2)] <- 0
v2 <- q2_hirelist
wt_cossim <- scale(t(v2 %*% s2))
cossim <- scale(t(rep(1/length(v2), length(v2)) %*% s2))
save(s2, file="s2.RData")
save(q2_id, file="q2_id.RData")
save(q2_hirelist, file="q2_hirelist.RData")

# Regression dataframe
mdf2 <- data.frame(q2_hirelist, wt_cossim, cossim, q2_wc, q2_gender, q2_race, q2_source, q2_title, q2_dflag, q2_oflag, q2_uflag, q2_intid)
colnames(mdf2) <- c("hired", "wt_cossim", "cossim", "log_wc", "gender", "race", "source", "title", "dflag", "oflag", "uflag", paste("intid", 1:161, sep=""))
mdf2$gender <- factor(mdf2$gender, levels=c("Male", "Female"))
mdf2$race <- factor(mdf2$race, levels=c("White", "Black or African American", "Asian", "Hispanic or Latino", "Two or more races"))
mdf2$i.age <- 2017 - (mdf2$dflag - 22)
mdf2$source <- tolower(as.character(mdf2$source))
mdf2$source <- ifelse(grepl("employee", mdf2$source), yes="Employee", no=mdf2$source)
mdf2$source <- ifelse(grepl("internal", mdf2$source), yes="Internal", no=mdf2$source)
mdf2$source <- ifelse(grepl("recruiter", mdf2$source), yes="Recruiter", no=mdf2$source)
mdf2$source <- ifelse(grepl("university", mdf2$source), yes="University", no=mdf2$source)
mdf2$source <- ifelse(grepl("sungevity career site", mdf2$source), yes="Career Site", no=mdf2$source)
mdf2$source <- ifelse(grepl("craigslist", mdf2$source), yes="Craigslist", no=mdf2$source)
mdf2$source <- ifelse(grepl("indeed", mdf2$source), yes="Indeed", no=mdf2$source)
mdf2$source <- ifelse(grepl("snagajob", mdf2$source), yes="SnagaJob", no=mdf2$source)
mdf2$source <- ifelse(grepl("glassdoor", mdf2$source), yes="Glassdoor", no=mdf2$source)
mdf2$source <- ifelse(grepl("linkedin", mdf2$source), yes="LinkedIn", no=mdf2$source)
mdf2$source <- ifelse(grepl("hotjobs", mdf2$source), yes="HotJobs", no=mdf2$source)
mdf2$source <- ifelse(grepl("dice", mdf2$source), yes="Dice", no=mdf2$source)
mdf2$source <- ifelse(grepl("monster", mdf2$source), yes="Monster", no=mdf2$source)
mdf2$source <- ifelse(grepl("simplyhired", mdf2$source), yes="SimplyHired", no=mdf2$source)
mdf2$source <- ifelse(grepl("Employee", mdf2$source) | grepl("Internal", mdf2$source) | grepl("Recruiter", mdf2$source)
                      | grepl("University", mdf2$source) | grepl("Career Site", mdf2$source) | grepl("Craigslist", mdf2$source)
                      | grepl("Indeed", mdf2$source)| grepl("SnagaJob", mdf2$source) | grepl("Glassdoor", mdf2$source)
                      | grepl("LinkedIn", mdf2$source) | grepl("HotJobs", mdf2$source) | grepl("Dice", mdf2$source)
                      | grepl("Monster", mdf2$source) | grepl("SimplyHired", mdf2$source), yes=mdf2$source, no="Other")
mdf2$source <- ifelse(mdf2$source=="Craigslist" | mdf2$source=="Indeed" | mdf2$source=="SnagaJob" | mdf2$source=="Glassdoor" |
                        mdf2$source=="LinkedIn" | mdf2$source=="HotJobs" | mdf2$source=="Dice" | mdf2$source=="Monster" |
                        mdf2$source=="SimplyHired", yes="Job Board", no=mdf2$source)
mdf2$source <- factor(mdf2$source, levels=c("Career Site", "Employee", "Internal", "Recruiter", "University",
                                            "Job Board", "Other"))
mdf2$recruiter <- ifelse(mdf2$source=="Recruiter", 1, 0)
mdf2$referral <- ifelse(mdf2$source=="Employee", 1, 0)
mdf2$title <- as.character(mdf2$title)
mdf2$manager <- ifelse(grepl("Manager", mdf2$title), yes=1, no=0)
mdf2$manager <- ifelse(grepl("Director", mdf2$title), yes=1, no=mdf2$manager)
mdf2$manager <- ifelse(grepl("VP", mdf2$title), yes=1, no=mdf2$manager)
mdf2$manager <- ifelse(grepl("President", mdf2$title), yes=1, no=mdf2$manager)
save(mdf2, file="cossim_dataframes/hiring_q2_cossim.RData")

# Q3
d <- as.matrix(dtm3)
d_binary <- ifelse(d >= 1, 1, 0)
tf3 <- d
idf3 <- log(nrow(d)/colSums(d_binary))
tfidf3 <- d
for (word in names(idf3)){
  tfidf3[,word] <- tf3[,word] * idf3[word]
}
tfidf3_means <- colMeans(tfidf3)
tfidf3_means <- rev(tfidf3_means[order(tfidf3_means)])
tfidf3_means[1:10]

# Q3 similarity weighted by hiring status
tfidf3 <- as.matrix(t(tfidf3))
tfidf3[1:10, 1:10]
s3 <- cosine(tfidf3)
diag(s3) <- 0
s3[is.na(s3)] <- 0
v3 <- q3_hirelist
wt_cossim <- scale(t(v3 %*% s3))
cossim <- scale(t(rep(1/length(v3), length(v3)) %*% s3))
save(s3, file="s3.RData")
save(q3_id, file="q3_id.RData")
save(q3_hirelist, file="q3_hirelist.RData")

# Regression dataframe
mdf3 <- data.frame(q3_hirelist, wt_cossim, cossim, q3_wc, q3_gender, q3_race, q3_source, q3_title, q3_dflag, q3_oflag, q3_uflag, q3_intid)
colnames(mdf3) <- c("hired", "wt_cossim", "cossim", "log_wc", "gender", "race", "source", "title", "dflag", "oflag", "uflag", paste("intid", 1:161, sep=""))
mdf3$gender <- factor(mdf3$gender, levels=c("Male", "Female"))
mdf3$race <- factor(mdf3$race, levels=c("White", "Black or African American", "Asian", "Hispanic or Latino", "Two or more races"))
mdf3$i.age <- 2017 - (mdf3$dflag - 22)
mdf3$source <- tolower(as.character(mdf3$source))
mdf3$source <- ifelse(grepl("employee", mdf3$source), yes="Employee", no=mdf3$source)
mdf3$source <- ifelse(grepl("internal", mdf3$source), yes="Internal", no=mdf3$source)
mdf3$source <- ifelse(grepl("recruiter", mdf3$source), yes="Recruiter", no=mdf3$source)
mdf3$source <- ifelse(grepl("university", mdf3$source), yes="University", no=mdf3$source)
mdf3$source <- ifelse(grepl("sungevity career site", mdf3$source), yes="Career Site", no=mdf3$source)
mdf3$source <- ifelse(grepl("craigslist", mdf3$source), yes="Craigslist", no=mdf3$source)
mdf3$source <- ifelse(grepl("indeed", mdf3$source), yes="Indeed", no=mdf3$source)
mdf3$source <- ifelse(grepl("snagajob", mdf3$source), yes="SnagaJob", no=mdf3$source)
mdf3$source <- ifelse(grepl("glassdoor", mdf3$source), yes="Glassdoor", no=mdf3$source)
mdf3$source <- ifelse(grepl("linkedin", mdf3$source), yes="LinkedIn", no=mdf3$source)
mdf3$source <- ifelse(grepl("hotjobs", mdf3$source), yes="HotJobs", no=mdf3$source)
mdf3$source <- ifelse(grepl("dice", mdf3$source), yes="Dice", no=mdf3$source)
mdf3$source <- ifelse(grepl("monster", mdf3$source), yes="Monster", no=mdf3$source)
mdf3$source <- ifelse(grepl("simplyhired", mdf3$source), yes="SimplyHired", no=mdf3$source)
mdf3$source <- ifelse(grepl("Employee", mdf3$source) | grepl("Internal", mdf3$source) | grepl("Recruiter", mdf3$source)
                      | grepl("University", mdf3$source) | grepl("Career Site", mdf3$source) | grepl("Craigslist", mdf3$source)
                      | grepl("Indeed", mdf3$source)| grepl("SnagaJob", mdf3$source) | grepl("Glassdoor", mdf3$source)
                      | grepl("LinkedIn", mdf3$source) | grepl("HotJobs", mdf3$source) | grepl("Dice", mdf3$source)
                      | grepl("Monster", mdf3$source) | grepl("SimplyHired", mdf3$source), yes=mdf3$source, no="Other")
mdf3$source <- ifelse(mdf3$source=="Craigslist" | mdf3$source=="Indeed" | mdf3$source=="SnagaJob" | mdf3$source=="Glassdoor" |
                        mdf3$source=="LinkedIn" | mdf3$source=="HotJobs" | mdf3$source=="Dice" | mdf3$source=="Monster" |
                        mdf3$source=="SimplyHired", yes="Job Board", no=mdf3$source)
mdf3$source <- factor(mdf3$source, levels=c("Career Site", "Employee", "Internal", "Recruiter", "University",
                                            "Job Board", "Other"))
mdf3$recruiter <- ifelse(mdf3$source=="Recruiter", 1, 0)
mdf3$referral <- ifelse(mdf3$source=="Employee", 1, 0)
mdf3$title <- as.character(mdf3$title)
mdf3$manager <- ifelse(grepl("Manager", mdf3$title), yes=1, no=0)
mdf3$manager <- ifelse(grepl("Director", mdf3$title), yes=1, no=mdf3$manager)
mdf3$manager <- ifelse(grepl("VP", mdf3$title), yes=1, no=mdf3$manager)
mdf3$manager <- ifelse(grepl("President", mdf3$title), yes=1, no=mdf3$manager)
save(mdf3, file="cossim_dataframes/hiring_q3_cossim.RData")

# ALL QUESTIONS
d <- as.matrix(dtm)
d_binary <- ifelse(d >= 1, 1, 0)
tf <- d
idf <- log(nrow(d)/colSums(d_binary))
tfidf <- d
for (word in names(idf)){
  tfidf[,word] <- tf[,word] * idf[word]
}
tfidf_means <- colMeans(tfidf)
tfidf_means <- rev(tfidf_means[order(tfidf_means)])
tfidf_means[1:10]

# Q similarity weighted by hiring status
tfidf <- as.matrix(t(tfidf))
tfidf[1:10, 1:10]
s <- cosine(tfidf)
diag(s) <- 0
s[is.na(s)] <- 0
v <- q_hirelist
wt_cossim <- scale(t(v %*% s))
cossim <- scale(t(rep(1/length(v), length(v)) %*% s))
save(s, file="s.RData")
save(q_id, file="q_id.RData")
save(q_hirelist, file="q_hirelist.RData")

# Regression dataframe
mdf <- data.frame(q_hirelist, wt_cossim, cossim, q_wc, q_gender, q_race, q_source, q_title, q_dflag, q_oflag, q_uflag, q_intid)
colnames(mdf) <- c("hired", "wt_cossim", "cossim", "log_wc", "gender", "race", "source", "title", "dflag", "oflag", "uflag", paste("intid", 1:161, sep=""))
mdf$gender <- factor(mdf$gender, levels=c("Male", "Female"))
mdf$race <- factor(mdf$race, levels=c("White", "Black or African American", "Asian", "Hispanic or Latino", "Two or more races"))
mdf$i.age <- 2017 - (mdf$dflag - 22)
mdf$source <- tolower(as.character(mdf$source))
mdf$source <- ifelse(grepl("employee", mdf$source), yes="Employee", no=mdf$source)
mdf$source <- ifelse(grepl("internal", mdf$source), yes="Internal", no=mdf$source)
mdf$source <- ifelse(grepl("recruiter", mdf$source), yes="Recruiter", no=mdf$source)
mdf$source <- ifelse(grepl("university", mdf$source), yes="University", no=mdf$source)
mdf$source <- ifelse(grepl("sungevity career site", mdf$source), yes="Career Site", no=mdf$source)
mdf$source <- ifelse(grepl("craigslist", mdf$source), yes="Craigslist", no=mdf$source)
mdf$source <- ifelse(grepl("indeed", mdf$source), yes="Indeed", no=mdf$source)
mdf$source <- ifelse(grepl("snagajob", mdf$source), yes="SnagaJob", no=mdf$source)
mdf$source <- ifelse(grepl("glassdoor", mdf$source), yes="Glassdoor", no=mdf$source)
mdf$source <- ifelse(grepl("linkedin", mdf$source), yes="LinkedIn", no=mdf$source)
mdf$source <- ifelse(grepl("hotjobs", mdf$source), yes="HotJobs", no=mdf$source)
mdf$source <- ifelse(grepl("dice", mdf$source), yes="Dice", no=mdf$source)
mdf$source <- ifelse(grepl("monster", mdf$source), yes="Monster", no=mdf$source)
mdf$source <- ifelse(grepl("simplyhired", mdf$source), yes="SimplyHired", no=mdf$source)
mdf$source <- ifelse(grepl("Employee", mdf$source) | grepl("Internal", mdf$source) | grepl("Recruiter", mdf$source)
                      | grepl("University", mdf$source) | grepl("Career Site", mdf$source) | grepl("Craigslist", mdf$source)
                      | grepl("Indeed", mdf$source)| grepl("SnagaJob", mdf$source) | grepl("Glassdoor", mdf$source)
                      | grepl("LinkedIn", mdf$source) | grepl("HotJobs", mdf$source) | grepl("Dice", mdf$source)
                      | grepl("Monster", mdf$source) | grepl("SimplyHired", mdf$source), yes=mdf$source, no="Other")
mdf$source <- ifelse(mdf$source=="Craigslist" | mdf$source=="Indeed" | mdf$source=="SnagaJob" | mdf$source=="Glassdoor" |
                        mdf$source=="LinkedIn" | mdf$source=="HotJobs" | mdf$source=="Dice" | mdf$source=="Monster" |
                        mdf$source=="SimplyHired", yes="Job Board", no=mdf$source)
mdf$source <- factor(mdf$source, levels=c("Career Site", "Employee", "Internal", "Recruiter", "University",
                                            "Job Board", "Other"))
mdf$recruiter <- ifelse(mdf$source=="Recruiter", 1, 0)
mdf$referral <- ifelse(mdf$source=="Employee", 1, 0)
mdf$title <- as.character(mdf$title)
mdf$manager <- ifelse(grepl("Manager", mdf$title), yes=1, no=0)
mdf$manager <- ifelse(grepl("Director", mdf$title), yes=1, no=mdf$manager)
mdf$manager <- ifelse(grepl("VP", mdf$title), yes=1, no=mdf$manager)
mdf$manager <- ifelse(grepl("President", mdf$title), yes=1, no=mdf$manager)
save(mdf, file="cossim_dataframes/hiring_q_cossim.RData")



