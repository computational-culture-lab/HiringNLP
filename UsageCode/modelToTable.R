##################################################################################
# TABLE CREATION
# Sarah K. Stein & Samaksh (Avi Goyal)
#
# This file creates output tables (ie. put all regressions in nice Overleaf format).
#
# 1. Load mdf's
# 2. Construct GLM's
# 3. Create Tables
##################################################################################

library(stargazer)
library(margins)
library(ggplot2)

stargazer2 <- function(model, odd.ratio=FALSE, ...) {
  if(!("list" %in% class(model))) model <- list(model)
  if (odd.ratio) {
    coefOR2 <- lapply(model, function(x) exp(coef(x)))
    seOR2 <- lapply(model, function(x) exp(coef(x)) * summary(x)$coef[, 2])
    t2 <- lapply(model, function(x) summary(x)$coef[, 3])
    p2 <- lapply(model, function(x) summary(x)$coef[, 4])
    stargazer(model, coef=coefOR2, se=seOR2, t=t2, p=p2, ...)
  } else {
    stargazer(model, ...)
  }
}


# Load saved mdf's
load("hiring_q1_simscore_USE.RData")
load("hiring_q2_simscore_USE.RData")
load("hiring_q3_simscore_USE.RData")


# Remove internal hires
mdf <- mdf[mdf$source!="Internal",]
mdf1 <- mdf1[mdf1$source!="Internal",]
mdf2 <- mdf2[mdf2$source!="Internal",]
mdf3 <- mdf3[mdf3$source!="Internal",]

# Q1 Table
#mdf1.5 <- mdf1[, c("hired", "wt_cossim", "manager", "log_wc", "gender", "i.age", "oflag", "uflag", "recruiter", "referral", paste("intid", 1:161, sep=""))]
mdf1.5 <- mdf1[, c("hired", "sim_score", "manager", "log_wc", "gender", "i.age", "oflag", "uflag", "recruiter", "referral", paste("intid", 1:161, sep=""))]
mdf1.5 <- mdf1.5[, !(names(mdf1.5) %in% c("intid44", "intid45", "intid70", "intid148", "intid156", "intid158"))]

q1.baseline <- glm(hired ~ sim_score + manager + log_wc, data=mdf1, family="binomial")
q1.demographics <- glm(hired ~ sim_score + manager + log_wc + gender + i.age, data=mdf1, family="binomial")
q1.humancap <- glm(hired ~ sim_score + manager + log_wc + gender + i.age + oflag + uflag + recruiter, data=mdf1, family="binomial")
q1.socialcap <- glm(hired ~ sim_score + manager + log_wc + gender + i.age + oflag + uflag + recruiter + referral, data=mdf1, family="binomial")

models.q1 <- list(q1.baseline, q1.demographics, q1.humancap, q1.socialcap)
stargazer2(models.q1, odd.ratio=TRUE, star.cutoffs=c(0.05, 0.01, 0.001), report="vct*")

# Q2 TFIDF
q1.baseline <- glm(hired ~ wt_cossim + manager + log_wc, data=mdf1, family="binomial")
q1.demographics <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age, data=mdf1, family="binomial")
q1.humancap <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter, data=mdf1, family="binomial")
q1.socialcap <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter + referral, data=mdf1, family="binomial")
q1.interviewerfe <- glm(hired ~ ., data=mdf1.5, family="binomial")

models.q1 <- list(q1.baseline, q1.demographics, q1.humancap, q1.socialcap, q1.interviewerfe)
stargazer2(models.q1, odd.ratio=TRUE, star.cutoffs=c(0.05, 0.01, 0.001), report="vct*")

# Q2 Table
#mdf2.5 <- mdf1[, c("hired", "wt_cossim", "manager", "log_wc", "gender", "i.age", "oflag", "uflag", "recruiter", "referral", paste("intid", 1:161, sep=""))]
mdf2.5 <- mdf2[, c("hired", "sim_score", "manager", "log_wc", "gender", "i.age", "oflag", "uflag", "recruiter", "referral", paste("intid", 1:161, sep=""))]
mdf2.5 <- mdf2.5[, !(names(mdf2.5) %in% c("intid44", "intid45", "intid70", "intid148", "intid156", "intid158"))]

q2.baseline <- glm(hired ~ sim_score + manager + log_wc, data=mdf2, family="binomial")
q2.demographics <- glm(hired ~ sim_score + manager + log_wc + gender + i.age, data=mdf2, family="binomial")
q2.humancap <- glm(hired ~ sim_score + manager + log_wc + gender + i.age + oflag + uflag + recruiter, data=mdf2, family="binomial")
q2.socialcap <- glm(hired ~ sim_score + manager + log_wc + gender + i.age + oflag + uflag + recruiter + referral, data=mdf2, family="binomial")

models.q2 <- list(q2.baseline, q2.demographics, q2.humancap, q2.socialcap, q2.interviewerfe)
stargazer2(models.q2, odd.ratio=TRUE, star.cutoffs=c(0.05, 0.01, 0.001), report="vct*")

# Q2 TFIDF
q2.baseline <- glm(hired ~ wt_cossim + manager + log_wc, data=mdf1, family="binomial")
q2.demographics <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age, data=mdf1, family="binomial")
q2.humancap <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter, data=mdf1, family="binomial")
q2.socialcap <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter + referral, data=mdf1, family="binomial")
q2.interviewerfe <- glm(hired ~ ., data=mdf1.5, family="binomial")

models.q2 <- list(q2.baseline, q2.demographics, q2.humancap, q2.socialcap, q2.interviewerfe)
stargazer2(models.q2, odd.ratio=TRUE, star.cutoffs=c(0.05, 0.01, 0.001), report="vct*")

# Q3 Table
#mdf3.5 <- mdf1[, c("hired", "wt_cossim", "manager", "log_wc", "gender", "i.age", "oflag", "uflag", "recruiter", "referral", paste("intid", 1:161, sep=""))]
mdf3.5 <- mdf3[, c("hired", "sim_score", "manager", "log_wc", "gender", "i.age", "oflag", "uflag", "recruiter", "referral", paste("intid", 1:161, sep=""))]
mdf3.5 <- mdf3.5[, !(names(mdf3.5) %in% c("intid44", "intid45", "intid70", "intid148", "intid156", "intid158"))]

q3.baseline <- glm(hired ~ sim_score + manager + log_wc, data=mdf3, family="binomial")
q3.demographics <- glm(hired ~ sim_score + manager + log_wc + gender + i.age, data=mdf3, family="binomial")
q3.humancap <- glm(hired ~ sim_score + manager + log_wc + gender + i.age + oflag + uflag + recruiter, data=mdf3, family="binomial")
q3.socialcap <- glm(hired ~ sim_score + manager + log_wc + gender + i.age + oflag + uflag + recruiter + referral, data=mdf3, family="binomial")

models.q3 <- list(q3.baseline, q3.demographics, q3.humancap, q3.socialcap, q3.interviewerfe)
stargazer2(models.q3, odd.ratio=TRUE, star.cutoffs=c(0.05, 0.01, 0.001), report="vct*")

# Q3 TFIDF
q3.baseline <- glm(hired ~ wt_cossim + manager + log_wc, data=mdf1, family="binomial")
q3.demographics <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age, data=mdf1, family="binomial")
q3.humancap <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter, data=mdf1, family="binomial")
q3.socialcap <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter + referral, data=mdf1, family="binomial")
q3.interviewerfe <- glm(hired ~ ., data=mdf1.5, family="binomial")

models.q3 <- list(q3.baseline, q3.demographics, q3.humancap, q3.socialcap, q3.interviewerfe)
stargazer2(models.q3, odd.ratio=TRUE, star.cutoffs=c(0.05, 0.01, 0.001), report="vct*")

# All Q Table
mdf.5 <- mdf[, c("hired", "wt_cossim", "manager", "log_wc", "gender", "i.age", "oflag", "uflag", "recruiter", "referral", paste("intid", 1:161, sep=""))]
mdf.5 <- mdf.5[, !(names(mdf.5) %in% c("intid44", "intid45", "intid70", "intid148", "intid156", "intid158"))]

q.baseline <- glm(hired ~ wt_cossim + manager + log_wc, data=mdf, family="binomial")
q.demographics <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age, data=mdf, family="binomial")
q.humancap <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter, data=mdf, family="binomial")
q.socialcap <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter + referral, data=mdf, family="binomial")
q.interviewerfe <- glm(hired ~ ., data=mdf.5, family="binomial")

models.q <- list(q.baseline, q.demographics, q.humancap, q.socialcap, q.interviewerfe)
stargazer2(models.q, odd.ratio=TRUE, star.cutoffs=c(0.05, 0.01, 0.001), report="vct*")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##################################################################################
# (Avi) I did not edit this bottom region, so my adjustment may not apply here

# Table 1
t1.q <- glm(hired ~ wt_cossim + manager + log_wc, data=mdf, family="binomial")
t1.q1 <- glm(hired ~ wt_cossim + manager + log_wc, data=mdf1, family="binomial")
t1.q2 <- glm(hired ~ wt_cossim + manager + log_wc, data=mdf2, family="binomial")
t1.q3 <- glm(hired ~ wt_cossim + manager + log_wc, data=mdf3, family="binomial")
models1 <- list(t1.q, t1.q1, t1.q2, t1.q3)
stargazer2(models1, odd.ratio=TRUE, star.cutoffs=c(0.05, 0.01, 0.001), report="vc*")

# Table 2
t2.q <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age, data=mdf, family="binomial")
t2.q1 <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age, data=mdf1, family="binomial")
t2.q2 <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age, data=mdf2, family="binomial")
t2.q3 <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age, data=mdf3, family="binomial")
models2 <- list(t2.q, t2.q1, t2.q2, t2.q3)
stargazer2(models2, odd.ratio=TRUE, star.cutoffs=c(0.05, 0.01, 0.001), report="vc*")

# Table 3
t3.q <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter, data=mdf, family="binomial")
t3.q1 <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter, data=mdf1, family="binomial")
t3.q2 <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter, data=mdf2, family="binomial")
t3.q3 <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter, data=mdf3, family="binomial")
models3 <- list(t3.q, t3.q1, t3.q2, t3.q3)
stargazer2(models3, odd.ratio=TRUE, star.cutoffs=c(0.05, 0.01, 0.001), report="vc*")

# Table 4
t4.q <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter + referral, data=mdf, family="binomial")
t4.q1 <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter + referral, data=mdf1, family="binomial")
t4.q2 <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter + referral, data=mdf2, family="binomial")
t4.q3 <- glm(hired ~ wt_cossim + manager + log_wc + gender + i.age + oflag + uflag + recruiter + referral, data=mdf3, family="binomial")
models4 <- list(t4.q, t4.q1, t4.q2, t4.q3)
stargazer2(models4, odd.ratio=TRUE, star.cutoffs=c(0.05, 0.01, 0.001), report="vc*")

# Table 5
mdf.5 <- mdf[, c("hired", "wt_cossim", "manager", "log_wc", "gender", "i.age", "oflag", "uflag", "recruiter", "referral", paste("intid", 1:161, sep=""))]
mdf.5 <- mdf.5[, !(names(mdf.5) %in% c("intid44", "intid45", "intid70", "intid148", "intid156", "intid158"))]
mdf1.5 <- mdf1[, c("hired", "wt_cossim", "manager", "log_wc", "gender", "i.age", "oflag", "uflag", "recruiter", "referral", paste("intid", 1:161, sep=""))]
mdf1.5 <- mdf1.5[, !(names(mdf1.5) %in% c("intid44", "intid45", "intid70", "intid148", "intid156", "intid158"))]
mdf2.5 <- mdf2[, c("hired", "wt_cossim", "manager", "log_wc", "gender", "i.age", "oflag", "uflag", "recruiter", "referral", paste("intid", 1:161, sep=""))]
mdf2.5 <- mdf2.5[, !(names(mdf2.5) %in% c("intid44", "intid45", "intid70", "intid148", "intid156", "intid158"))]
mdf3.5 <- mdf3[, c("hired", "wt_cossim", "manager", "log_wc", "gender", "i.age", "oflag", "uflag", "recruiter", "referral", paste("intid", 1:161, sep=""))]
mdf3.5 <- mdf3.5[, !(names(mdf3.5) %in% c("intid44", "intid45", "intid70", "intid148", "intid156", "intid158"))]
t5.q <- glm(hired ~ ., data=mdf.5, family="binomial")
t5.q1 <- glm(hired ~ ., data=mdf1.5, family="binomial")
t5.q2 <- glm(hired ~ ., data=mdf2.5, family="binomial")
t5.q3 <- glm(hired ~ ., data=mdf3.5, family="binomial")
models5 <- list(t5.q, t5.q1, t5.q2, t5.q3)
stargazer2(models5, odd.ratio=TRUE, star.cutoffs=c(0.05, 0.01, 0.001), report="vc*", omit="intid", omit.labels="Interviewer FE")

# Standardize cosine similarity
mdf$wt_cossim.std <- scale(mdf$wt_cossim)
mdf1$wt_cossim.std <- scale(mdf1$wt_cossim)
mdf2$wt_cossim.std <- scale(mdf2$wt_cossim)
mdf3$wt_cossim.std <- scale(mdf3$wt_cossim)

# Marginal Effects: Table 4
par(mfrow=c(2,2))
cplot(t4.q, x="wt_cossim", what="effect", main="All Questions", xlab="Cosine Similarity (All Questions)",
      ylab="Marginal Effect", se.type="shade", se.fill="lightgray", ylim=c(-0.01,0.05))
abline(h=0, lty=2, col="red")
cplot(t4.q1, x="wt_cossim", what="effect", main="Question 1", xlab="Cosine Similarity (Question 1)",
      ylab="Marginal Effect", se.type="shade", se.fill="lightgray", ylim=c(-0.01,0.05))
abline(h=0, lty=2, col="red")
cplot(t4.q2, x="wt_cossim", what="effect", main="Question 2", xlab="Cosine Similarity (Question 2)",
      ylab="Marginal Effect", se.type="shade", se.fill="lightgray", ylim=c(-0.01,0.05))
abline(h=0, lty=2, col="red")
cplot(t4.q3, x="wt_cossim", what="effect", main="Question 3", xlab="Cosine Similarity (Question 3)",
      ylab="Marginal Effect", se.type="shade", se.fill="lightgray", ylim=c(-0.01,0.05))
abline(h=0, lty=2, col="red") 

# Predicted Values: Table 4
par(mfrow=c(2,2))
cplot(t4.q, x="wt_cossim", what="prediction", main="All Questions", xlab="Cosine Similarity (All Questions)",
      ylab="Predicted Value (Hiring)", se.type="shade", se.fill="lightgray", ylim=c(0,0.1))
cplot(t4.q1, x="wt_cossim", what="prediction", main="Question 1", xlab="Cosine Similarity (Question 1)",
      ylab="Predicted Value (Hiring)", se.type="shade", se.fill="lightgray", ylim=c(0,0.1))
cplot(t4.q2, x="wt_cossim", what="prediction", main="Question 2", xlab="Cosine Similarity (Question 2)",
      ylab="Predicted Value (Hiring)", se.type="shade", se.fill="lightgray", ylim=c(0,0.1))
cplot(t4.q3, x="wt_cossim", what="prediction", main="Question 3", xlab="Cosine Similarity (Question 3)",
      ylab="Predicted Value (Hiring)", se.type="shade", se.fill="lightgray", ylim=c(0,0.1))


