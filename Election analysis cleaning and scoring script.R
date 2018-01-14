library(dplyr)
library(tidyr)
library(Hmisc)
library(readr)
library(MASS)
library(caret)
library(lubridate)
library(ggplot2)
library(C50)
library(pROC)
library(mda)


setwd("C:/Users/Louis/Desktop/MSTI Application/US Primary Election Analysis")

load("USA_county_data.RData")

results_2016_pres <- read_csv("results_G2016.csv")
county_facts <- read_csv("county_facts.csv")
fips_codes <- read_csv(file = "2010 FIPS Codes.csv")



fips_codes$State_FP_2 <- as.integer(fips_codes$State_FP)
fips_codes$FPCODE <- as.integer(paste(fips_codes$State_FP_2, fips_codes$County_FP, sep = ""))

county_facts_fips <- right_join(county_facts_1, fips_codes, by = c("fips" = "FPCODE")) 

results_2016_pres <- USA_county_data %>% dplyr::select(fips, name_16, votes, votes16_trumpd, votes16_clintonh)

county_results_facts <- right_join(results_2016_pres, county_facts_fips, by = "fips")
county_results_facts$TrumpCounty <- as.factor(ifelse(county_results_facts$votes16_clintonh > county_results_facts$votes16_trumpd, "NotTrumpCounty", "TrumpCounty"))

describe(county_results_facts$TrumpCounty)

county_results <- county_results_facts[!is.na(county_results_facts$TrumpCounty),]

save(county_results, file = "MergedClean.RData")






######################################
### Data Partition and Processing
######################################

#Register Cluster - issue with doParallel
#cl <- makeCluster(detectCores()-1)
#registerDoParallel(cl)

load("MergedClean.RData")


set.seed(8765)

#Model dataset
modMergedARV <- county_results %>% dplyr::select(PST045214:POP060210, TrumpCounty)
#save(modMergedARV, file = "modDataRaw.RData")

#Models created with 50% sample
inTrain <- createDataPartition(y=modMergedARV$TrumpCounty, p=0.3, list=FALSE)

training <- modMergedARV[inTrain,]
testing <- modMergedARV[-inTrain,]

#Start the clock! - 4.5 mins to do the whole process.
ptm <- proc.time()

#Doing the transform with medianImpute and nzv takes _____________ mins with 270 vars and 270,000 rows
preProcValues <- preProcess(training, method = c("medianImpute", "center", "scale", "nzv"), freqCut = 99/1, uniqueCut = 10)

trainTransformed <- predict(preProcValues, training)
testTransformed <- predict(preProcValues, testing)
ARVTransformed <- predict(preProcValues, modMergedARV)
#save(VANARVTransformed, file = "modDataTrans.RData")

#Stop the clock
proc.time() - ptm



##################
# Model Training #
##################

#GLM model
fit.glm <- train(TrumpCounty ~ .,
                 method = "glm",
                 trControl = trainControl(method = "none"),
                 data=trainTransformed)
summary(fit.glm)
#save(fit.glm, file = "fit.glm_3_9_17.RData")

trainpred.glm <- predict(fit.glm, trainTransformed, type="prob")
summary(trainpred.glm)
trainpredsplit.glm <- ntile(trainpred.glm[,1], 5)
table(trainTransformed$TrumpCounty,trainpredsplit.glm)
prop.table(table(trainTransformed$TrumpCounty,trainpredsplit.glm),2)
prop.table(table(trainTransformed$TrumpCounty,cut((trainpred.glm*10)[,1], 0:10)),2)
prop.table(table(trainTransformed$TrumpCounty,cut(trainpred.glm[,1], c(0,.2,.4,.6,.8,1))),2)

testpred.glm <- predict(fit.glm, testTransformed, type="prob")
summary(testpred.glm)
testpredsplit.glm <- ntile(testpred.glm[,1], 5)
table(testTransformed$TrumpCounty,testpredsplit.glm)
prop.table(table(testTransformed$TrumpCounty,testpredsplit.glm),2)
prop.table(table(testTransformed$TrumpCounty,cut((testpred.glm*10)[,1], 0:10)),2)
prop.table(table(testTransformed$TrumpCounty,cut(testpred.glm[,1], c(0,.2,.4,.6,.8,1))),2)

qplot(testpred.glm[,1]*100, geom="histogram", binwidth = 2) 
ROC.glm <- roc(testTransformed$TrumpCounty,testpred.glm[,1])
ROC.glm
plot(ROC.glm, legacy.axes = TRUE)
coords(ROC.glm, x="best", best.method = "closest.topleft")



# Start the clock!
ptm <- proc.time()

#gbm
fit.gbm <- train(TrumpCounty ~ .,
                 data = trainTransformed,
                 method = "gbm",
                 tuneLength = 10,
                 trControl = trainControl(method = "repeatedcv", number = 5, repeats = 1, classProbs = TRUE, summaryFunction = twoClassSummary),
                 metric = "ROC"
)
# Stop the clock
(proc.time() - ptm)/60

fit.gbm
plot(fit.gbm)
summary(fit.gbm)
#save(fit.gbm, file = "fit.gbm_3_9_17.RData")

trainpred.gbm <- predict(fit.gbm, trainTransformed, type="prob")
summary(trainpred.gbm)
trainpredsplit.gbm <- ntile(trainpred.gbm[,2], 10)
table(trainTransformed$TrumpCounty,trainpredsplit.gbm)
prop.table(table(trainTransformed$TrumpCounty,trainpredsplit.gbm),2)

testpred.gbm <- predict(fit.gbm, testTransformed, type="prob")
summary(testpred.gbm)
testpredsplit.gbm <- ntile(testpred.gbm[,2], 5)
table(testTransformed$TrumpCounty,testpredsplit.gbm)
prop.table(table(testTransformed$TrumpCounty,testpredsplit.gbm),2)
prop.table(table(testTransformed$TrumpCounty,cut((testpred.gbm)[,2], 0:100)),2)
prop.table(table(testTransformed$TrumpCounty,cut(testpred.gbm[,2], c(0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1))),2)

qplot(testpred.gbm[,2]*100, geom="histogram", binwidth = 2) 
ROC.gbm <- roc(testTransformed$TrumpCounty,testpred.gbm[,2])
ROC.gbm
plot(ROC.gbm, legacy.axes = TRUE)
coords(ROC.gbm, x="best", best.method = "closest.topleft")

hist(trainpred.gbm$TrumpCounty, breaks = 100, main = "Figure 1: Distribution of Trump County Scores")

df1 <- t(data.frame(seq(1,6,by=1),seq(6,1,by=-1)))    
colnames(df1) <- c("A","B","C","D","E","F")    
rownames(df1) <- c("a","b")    
df2 <- data.frame(rep(colnames(df1),2),rep(rownames(df1),6))    
colnames(df2) <- c("Vector1","Vector2")

library(reshape2)
imp <- varImp(fit.gbm, scale = F)
imp$importance$column_name <- row.names(df_imp)
imp$importance <- merge(facts_dict, imp$importance, by = "column_name")
row.names(imp$importance) <- imp$importance$description
imp$importance <- select(imp$importance, -column_name, -description)
plot(imp, top = 20, main = "Figure 3: gbm Model Variable Importance ")

imp


######################
# Save model objects #
######################

save(preProcValues, inTrain, fit.glm, fit.gbm, file = "TrumpModel_1_10_18.RData")
save(ARVTransformed, file = "ARVTransformed.RData")




########
# Data #
########

#Load clean data
load("MergedARVClean.RData")

#Load model objects and pre-processing values
load("TrumpModel_1_10_18.RData")



###########
# Scoring #
###########

#Select data for the model
modMergedARV <- county_results %>% dplyr::select(PST045214:POP060210, TrumpCounty)

#Transform data with old pre-processing values
ARVTransformed <- predict(preProcValues, modMergedARV)
#save(ARVTransformed, file = "ARVTransformed.RData")

#Score data
TrumpScore <- predict(fit.gbm, ARVTransformed, type="prob")
county_results$TrumpCountiesScore <- TrumpScore[,2]*100

table(county_results$TrumpCounty, cut(county_results$TrumpCountiesScore, breaks = 6))

View(county_results[,c("votes16_trumpd", "votes16_clintonh", "TrumpCounty", "TrumpCountiesScore")])
qplot(TrumpScore[,2]*100, geom="histogram", binwidth = 1) 



#############
# Save data #
#############


county_results <- county_results %>% mutate(TrumpShare = votes16_trumpd/votes,
                                            ClintonShare = votes16_clintonh/votes) %>%
  select(-ClassFP, - state_abbreviation, -area_name, -State_FP_2)

save(county_results, file = "TrumpCountyScore.RData")


