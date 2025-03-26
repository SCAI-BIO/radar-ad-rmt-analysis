# Univeriate analyses, for multimodal paper Manuel
# Marijn Muurling, 13-06-2023
# Adapted by Manuel Lentzen

### LOAD PACKAGES ####
library(car)
library(multcomp)
library(readxl)
library(tableone)
library(reshape)
library(reshape2)
library(ggplot2)
library(dplyr)
library(multcomp)
library(pROC)
library(lubridate)
library(gridExtra)
library(rstatix)
library(effectsize)

### LOAD DATA #####
rm(list = ls())

# Load merged data set
alldata <- read_excel(...) # TODO: add data path

### LABELLING ####

# Sex
alldata$sex <- factor(alldata$sex, levels = c(2, 1), labels = c("male", "female"))

# Smartphone type
alldata$rmt_smartphone_participant <- factor(alldata$rmt_smartphone_participant, levels = c(1, 2, 3, 4), labels = c("iOS", "Android", "Windows", "Other"))

# CDR as factor instead of continuous variable
alldata$inclusion_criteria_cdr <- factor(alldata$inclusion_criteria_cdr, levels = c(0.0, 0.5, 1.0, 2.0))

# Amyloid positivity
alldata$amyloid_positivity_csf <- factor(alldata$amyloid_positivity, levels = c(1, 2), labels = c("Positive", "Negative"))
alldata$amyloid_positivity_pet <- factor(alldata$amyloid_positivity_pet, levels = c(1, 2), labels = c("Positive", "Negative"))

# APOE genotype
alldata$apoe_genotype <- factor(alldata$apoe_genotype, levels = c(1, 2, 3, 4, 5, 6), labels = c("E2E2", "E2E3", "E2E4", "E3E3", "E3E4", "E4E4"))

# Study group
alldata$group <- factor(alldata$group, levels = c("MildAD", "ProAD", "PreAD", "HC"), labels = c("MildAD", "ProAD", "PreAD", "HC"))

### DEMOGRAPHICS TABLES ####
# Table 1 for all participants
tab1 <- CreateTableOne(
    data = alldata,
    vars = c("age", "sex", "education_years", "mmse_total", "inclusion_criteria_cdr", "adcs_total_score", "AIADL_SV_tscore"),
    strata = "group"
)
tab1Mat <- print(tab1, exact = "stage", quote = FALSE, noSpaces = TRUE, printToggle = FALSE)

# Fitbit
alldata$dailyMeanHeartRate <- as.numeric(alldata$dailyMeanHeartRate)
alldata$dailyMinHeartRate <- as.numeric(alldata$dailyMinHeartRate)
alldata$dailyMaxHeartRate <- as.numeric(alldata$dailyMaxHeartRate)
alldata$dailyMeanSteps <- as.numeric(alldata$dailyMeanSteps)
alldata$dailyMeanAsleepHours <- as.numeric(alldata$dailyMeanAsleepHours)
alldata$dailyMeanAwakeHours <- as.numeric(alldata$dailyMeanAwakeHours)
alldata$dailyMeanRemHours <- as.numeric(alldata$dailyMeanRemHours)
alldata$dailyMeanBedtimeHours <- as.numeric(alldata$dailyMeanBedtimeHours)
alldata$dailyMeanSleepEfficiency <- as.numeric(alldata$dailyMeanSleepEfficiency)
alldata$total_sleep_time <- as.numeric(alldata$total_sleep_time)
alldata$time_in_bed <- as.numeric(alldata$time_in_bed)
alldata$light_pct <- as.numeric(alldata$light_pct)
alldata$deep_pct <- as.numeric(alldata$deep_pct)
alldata$REM_pct <- as.numeric(alldata$REM_pct)
alldata$awake_pct <- as.numeric(alldata$awake_pct)
alldata$NREM_pct <- as.numeric(alldata$NREM_pct)
alldata$sleep_onset <- as.numeric(alldata$sleep_onset)
alldata$sleep_offset <- as.numeric(alldata$sleep_offset)
alldata$rem_latency <- as.numeric(alldata$rem_latency)
alldata$sleep_efficiency <- as.numeric(alldata$sleep_efficiency)
alldata$awakenings <- as.numeric(alldata$awakenings)
alldata$insomnia <- as.numeric(alldata$insomnia)
alldata$hypersomnia <- as.numeric(alldata$hypersomnia)
alldata$season <- as.factor(alldata$season)

# Altoida
alldata[which(alldata$group == "MildAD"), "DNS"] <- NA

### FEATURES ####
variables_banking <- c(
    "Total_Attempts_BankApp_count",
    "SUM_Duration_Attempts_BankApp_msec",
    "AVG_Duration_Attempts_BankApp_msec",
    "Correct_Attempts_BankApp_count",
    "Wrong_Attempts_BankApp_count",
    "SUM_Duration_Correct_Attempts_BankApp_msec",
    "AVG_Duration_Correct_Attempts_BankApp_msec",
    "Total_Attempts_PIN_count",
    "SUM_PIN_Duration_msec",
    "AVG_PIN_Duration_msec",
    "Correct_PIN_Attempts_count",
    "SUM_Correct_PIN_Duration_msec",
    "AVG_Correct_PIN_Duration_msec",
    "Wrong_PIN_Attempts_count",
    "Total_Attempts_Amount_count",
    "SUM_Amount_Duration_msec",
    "AVG_Amount_Duration_msec",
    "Correct_Amount_Attempts_count",
    "SUM_Correct_Amount_Duration_msec",
    "AVG_Correct_Amount_Duration_msec",
    "Wrong_Amount_Attempts_count",
    "Total_Confirm_attempts_count",
    "Total_Cancel_attempts_count",
    "Total_Correct_Confirm_attempts_count",
    "Total_Wrong_Confirm_attempts_count",
    "SUM_Duration_Correct_Confirm_msec",
    "AVG_Duration_Correct_Confirm_msec"
)

variables_fitbit <- c(
    "dailyMeanHeartRate",
    "dailyMaxHeartRate",
    "dailyMinHeartRate",
    "dailyMeanSteps",
    "dailyMeanAsleepHours",
    "dailyMeanAwakeHours",
    "dailyMeanBedtimeHours",
    "dailyMeanRemHours",
    "wearTimeMinutes",
    "wearTimePercentage",
    "total_sleep_time",
    "time_in_bed",
    "light_pct",
    "deep_pct",
    "REM_pct",
    "awake_pct",
    "NREM_pct",
    "sleep_onset",
    "sleep_offset",
    "rem_latency",
    "sleep_efficiency",
    "awakenings",
    "insomnia",
    "hypersomnia"
)

variables_physilog_dual <- c(
    "avg_peakswing",
    "cv_peakswing",
    "avg_speed",
    "cv_speed",
    "avg_LDr",
    "cv_LDr",
    "avg_DS",
    "cv_DS",
    "avg_swing",
    "cv_swing",
    "avg_PUr",
    "cv_PUr",
    "avg_stance",
    "cv_stance",
    "avg_HSP",
    "cv_HSP",
    "avg_FFr",
    "cv_FFr",
    "avg_PathLength",
    "cv_PathLength",
    "avg_cadence",
    "cv_cadence",
    "avg_TOP",
    "cv_TOP",
    "avg_slength",
    "cv_slength",
    "avg_swidth",
    "cv_swidth",
    "avg_gct",
    "cv_gct"
)

variables_physilog_tug <- c(
    "total_time",
    "sist_duration",
    "sist_angle_range",
    "turn_duration",
    "turnsi_duration",
    "NGaitCycles",
    "cadence",
    "gaitspeed"
)

variables_axivity <- c(
    "acc-overall-avg",
    "acc-overall-sd",
    "wearTime-overall(days)",
    "sedentary-overall-avg",
    "sedentary-overall-sd",
    "light-overall-avg",
    "light-overall-sd",
    "MVPA-overall-avg",
    "MVPA-overall-sd",
    "sleep-overall-avg",
    "sleep-overall-sd",
    "nonWearTime-overall(days)",
    "acc-Weekday-avg",
    "acc-Weekend-avg",
    "MVPA-Weekday-avg",
    "MVPA-Weekend-avg",
    "light-Weekday-avg",
    "light-Weekend-avg",
    "sedentary-Weekday-avg",
    "sedentary-Weekend-avg",
    "sleep-Weekday-avg",
    "sleep-Weekend-avg",
    "wear-Weekday-avg",
    "wear-Weekend-avg",
    "sedentary-overall-hour",
    "light-overall-hour",
    "MVPA-overall-hour",
    "sleep-overall-hour"
)

variables_altoida <- c(
    "DNS",
    "home_test1",
    "home_test2",
    "home_test3",
    "PerceptualMotorCoordination",
    "ComplexAttention",
    "CognitiveProcessingSpeed",
    "Inhibition",
    "Flexibility",
    "VisualPerception",
    "Planning",
    "ProspectiveMemory",
    "SpatialMemory",
    "FineMotorSkills",
    "Gait"
)

variables_mezurio_speech <- c(
    "audio_number_of_syllables",
    "audio_number_of_pauses",
    "audio_average_pause_duration",
    "audio_file_duration_in_second",
    "audio_total_speech_duration",
    "audio_speaking_rate",
    "audio_articulation_rate",
    "audio_average_syllable_duration",
    "audio_rms_energy",
    "audio_hesitation_ratio",
    "ost_F0semitoneFrom27_5Hz_sma3nz_amean",
    "ost_F0semitoneFrom27_5Hz_sma3nz_stddevNorm",
    "ost_F0semitoneFrom27_5Hz_sma3nz_percentile20_0",
    "ost_F0semitoneFrom27_5Hz_sma3nz_percentile50_0",
    "ost_F0semitoneFrom27_5Hz_sma3nz_percentile80_0",
    "ost_F0semitoneFrom27_5Hz_sma3nz_pctlrange0_2",
    "ost_F0semitoneFrom27_5Hz_sma3nz_meanRisingSlope",
    "ost_F0semitoneFrom27_5Hz_sma3nz_stddevRisingSlope",
    "ost_F0semitoneFrom27_5Hz_sma3nz_meanFallingSlope",
    "ost_F0semitoneFrom27_5Hz_sma3nz_stddevFallingSlope",
    "ost_loudness_sma3_amean",
    "ost_loudness_sma3_stddevNorm",
    "ost_loudness_sma3_percentile20_0",
    "ost_loudness_sma3_percentile50_0",
    "ost_loudness_sma3_percentile80_0",
    "ost_loudness_sma3_pctlrange0_2",
    "ost_loudness_sma3_meanRisingSlope",
    "ost_loudness_sma3_stddevRisingSlope",
    "ost_loudness_sma3_meanFallingSlope",
    "ost_loudness_sma3_stddevFallingSlope",
    "ost_spectralFlux_sma3_amean",
    "ost_spectralFlux_sma3_stddevNorm",
    "ost_mfcc1_sma3_amean",
    "ost_mfcc1_sma3_stddevNorm",
    "ost_mfcc2_sma3_amean",
    "ost_mfcc2_sma3_stddevNorm",
    "ost_mfcc3_sma3_amean",
    "ost_mfcc3_sma3_stddevNorm",
    "ost_mfcc4_sma3_amean",
    "ost_mfcc4_sma3_stddevNorm",
    "ost_jitterLocal_sma3nz_amean",
    "ost_jitterLocal_sma3nz_stddevNorm",
    "ost_shimmerLocaldB_sma3nz_amean",
    "ost_shimmerLocaldB_sma3nz_stddevNorm",
    "ost_HNRdBACF_sma3nz_amean",
    "ost_HNRdBACF_sma3nz_stddevNorm",
    "ost_logRelF0_H1_H2_sma3nz_amean",
    "ost_logRelF0_H1_H2_sma3nz_stddevNorm",
    "ost_logRelF0_H1_A3_sma3nz_amean",
    "ost_logRelF0_H1_A3_sma3nz_stddevNorm",
    "ost_F1frequency_sma3nz_amean",
    "ost_F1frequency_sma3nz_stddevNorm",
    "ost_F1bandwidth_sma3nz_amean",
    "ost_F1bandwidth_sma3nz_stddevNorm",
    "ost_F1amplitudeLogRelF0_sma3nz_amean",
    "ost_F1amplitudeLogRelF0_sma3nz_stddevNorm",
    "ost_F2frequency_sma3nz_amean",
    "ost_F2frequency_sma3nz_stddevNorm",
    "ost_F2bandwidth_sma3nz_amean",
    "ost_F2bandwidth_sma3nz_stddevNorm",
    "ost_F2amplitudeLogRelF0_sma3nz_amean",
    "ost_F2amplitudeLogRelF0_sma3nz_stddevNorm",
    "ost_F3frequency_sma3nz_amean",
    "ost_F3frequency_sma3nz_stddevNorm",
    "ost_F3bandwidth_sma3nz_amean",
    "ost_F3bandwidth_sma3nz_stddevNorm",
    "ost_F3amplitudeLogRelF0_sma3nz_amean",
    "ost_F3amplitudeLogRelF0_sma3nz_stddevNorm",
    "ost_alphaRatioV_sma3nz_amean",
    "ost_alphaRatioV_sma3nz_stddevNorm",
    "ost_hammarbergIndexV_sma3nz_amean",
    "ost_hammarbergIndexV_sma3nz_stddevNorm",
    "ost_slopeV0_500_sma3nz_amean",
    "ost_slopeV0_500_sma3nz_stddevNorm",
    "ost_slopeV500_1500_sma3nz_amean",
    "ost_slopeV500_1500_sma3nz_stddevNorm",
    "ost_spectralFluxV_sma3nz_amean",
    "ost_spectralFluxV_sma3nz_stddevNorm",
    "ost_mfcc1V_sma3nz_amean",
    "ost_mfcc1V_sma3nz_stddevNorm",
    "ost_mfcc2V_sma3nz_amean",
    "ost_mfcc2V_sma3nz_stddevNorm",
    "ost_mfcc3V_sma3nz_amean",
    "ost_mfcc3V_sma3nz_stddevNorm",
    "ost_mfcc4V_sma3nz_amean",
    "ost_mfcc4V_sma3nz_stddevNorm",
    "ost_alphaRatioUV_sma3nz_amean",
    "ost_hammarbergIndexUV_sma3nz_amean",
    "ost_slopeUV0_500_sma3nz_amean",
    "ost_slopeUV500_1500_sma3nz_amean",
    "ost_spectralFluxUV_sma3nz_amean",
    "ost_loudnessPeaksPerSec",
    "ost_VoicedSegmentsPerSec",
    "ost_MeanVoicedSegmentLengthSec",
    "ost_StddevVoicedSegmentLengthSec",
    "ost_MeanUnvoicedSegmentLength",
    "ost_StddevUnvoicedSegmentLength",
    "ost_equivalentSoundLevel_dBp"
)

### DESCRIPTIBE TABLES PER RMT ####
descr_table <- function(data, grouping_variable, RMT_variables, normality) {
    df <- dplyr::select(data, c(RMT_variables, grouping_variable))

    if (normality == "normal") {
        means <- df %>%
            group_by(group) %>%
            summarise_at(vars(RMT_variables), list(name = mean), na.rm = T)

        sds <- df %>%
            group_by(group) %>%
            summarise_at(vars(RMT_variables), list(name = sd), na.rm = T)

        descrtable <- as.data.frame(matrix(NA, ncol = 4, nrow = length(RMT_variables)))
        colnames(descrtable) <- levels(df$group)
        rownames(descrtable) <- RMT_variables
        for (i_var in 1:length(RMT_variables)) {
            for (i_group in 1:length(levels(df$group))) {
                descrtable[i_var, i_group] <- paste0(
                    round(dplyr::select(means, -group)[i_group, i_var], digits = 2), " (",
                    round(dplyr::select(sds, -group)[i_group, i_var], digits = 2), ")"
                )
            }
        }
    } else {
        medians <- df %>%
            group_by(group) %>%
            summarise_at(vars(RMT_variables), list(name = median), na.rm = T)

        quantile1 <- df %>%
            group_by(group) %>%
            summarise_at(vars(RMT_variables), list(name = quantile), 0, na.rm = T)
        quantile3 <- df %>%
            group_by(group) %>%
            summarise_at(vars(RMT_variables), list(name = quantile), 1, na.rm = T)

        descrtable <- as.data.frame(matrix(NA, ncol = 4, nrow = length(RMT_variables)))
        colnames(descrtable) <- levels(df$group)
        rownames(descrtable) <- RMT_variables
        for (i_var in 1:length(RMT_variables)) {
            for (i_group in 1:length(levels(df$group))) {
                descrtable[i_var, i_group] <- paste0(
                    round(dplyr::select(medians, -group)[i_group, i_var], digits = 2), " [",
                    round(as.numeric(dplyr::select(quantile1, -group)[i_group, i_var]), digits = 2), "-",
                    round(as.numeric(dplyr::select(quantile3, -group)[i_group, i_var]), digits = 2), "]"
                )
            }
        }
    }
    descrtable
}

variables_all <- c(variables_banking, variables_fitbit, variables_axivity, variables_altoida, variables_physilog_tug, variables_physilog_dual, variables_mezurio_speech)
table_all <- descr_table(alldata, "group", variables_all, "normal")
write.csv(table_all, file = paste0("descr_table_all_", Sys.Date(), ".csv"), append = F)

### CREATE ANCOVA TABLE ####
### Test normality
# If p<0.05, not normally distributed
test_normality <- function(variables) {
    pvalue <- as.data.frame(matrix(NA, nrow = 1, ncol = length(variables)))
    colnames(pvalue) <- variables
    for (i_var in 1:length(variables)) {
        pvalue[i_var] <- shapiro.test(as.matrix(dplyr::select(alldata, variables[i_var])))$p.value
    }
    pvalue
}

# Store the original p-values
anova_p_values <- data.frame()
tukey_p_values <- data.frame()

variables_all <- gsub("-", ".", variables_all)
colnames(alldata) <- gsub("-", ".", colnames(alldata))
variables_all <- gsub("\\(", ".", variables_all)
colnames(alldata) <- gsub("\\(", ".", colnames(alldata))
variables_all <- gsub("\\)", ".", variables_all)
colnames(alldata) <- gsub("\\)", ".", colnames(alldata))

alldata[alldata$group == "MildAD", colnames(alldata) %in% variables_altoida] <- NA

alldata <- alldata %>%
    mutate(
        site = as.factor(site),
        season = as.factor(season),
        sex = as.factor(sex),
        group = as.factor(group),
    )

cohens_d_calc <- function(fit, group_var = "group") {
    # Get group means
    group_means <- tapply(
        fitted(fit) + residuals(fit),
        alldata[[group_var]],
        mean
    )

    # Get pooled SD from residuals
    pooled_sd <- sqrt(sum(residuals(fit)^2) / df.residual(fit))

    # Calculate Cohen's d for each comparison
    d_values <- list()
    group_pairs <- combn(names(group_means), 2)

    for (i in 1:ncol(group_pairs)) {
        g1 <- group_pairs[1, i]
        g2 <- group_pairs[2, i]
        d <- (group_means[g1] - group_means[g2]) / pooled_sd
        d_values[[paste(g1, g2, sep = "-")]] <- d
    }

    return(d_values)
}

# Loop over outcome variables
for (outcome in variables_all) {
    if (!outcome %in% colnames(alldata)) {
        next
    }
    # Test for normality
    log <- FALSE
    if (shapiro.test(alldata[[outcome]])$p.value < 0.05) {
        # If p-value is less than 0.05, reject the null hypothesis of normality
        # Apply log transformation, adding a small constant to avoid undefined values
        alldata[[outcome]] <- log(alldata[[outcome]] + 1e-9)
        log <- TRUE
    }

    # Perform an ANOVA
    tryCatch(
        {
            if (outcome %in% variables_altoida) {
                fit <- aov(data = alldata, alldata[[outcome]] ~ group + site)
                # covariates <- "group + site"
            } else if (outcome %in% variables_axivity |
                outcome %in% variables_fitbit) {
                # covariates <- "group + age + sex + education_years + bmi + site"
                fit <- aov(data = alldata, alldata[[outcome]] ~ group + age + sex + education_years + bmi + site + season)
            } else if (outcome %in% variables_physilog_dual |
                outcome %in% variables_physilog_tug) {
                # covariates <- "group + age + sex + education_years + bmi + site"
                fit <- aov(data = alldata, alldata[[outcome]] ~ group + age + sex + education_years + bmi + site)
            } else {
                # covariates <- "group + age + sex + education_years + site"
                fit <- aov(data = alldata, alldata[[outcome]] ~ group + age + sex + education_years + site)
            }


            anova_results <- summary(fit)
            # Extract the p-value from the ANOVA results
            anova_p <- anova_results[[1]]["group", "Pr(>F)"]
            anova_f <- anova_results[[1]]["group", "F value"]
            anova_eta <- effectsize::eta_squared(fit)
            anova_eta_value <- anova_eta$Eta2[1] # Get the eta-squared value for the group effect
            anova_eta_ci <- sprintf("[%s, %s]", format(round(anova_eta$CI_low[1], digits = 2), nsmall = 2), format(round(anova_eta$CI_high[1], digits = 2), nsmall = 2))
            eta_string <- sprintf("%s %s", format(round(anova_eta_value, digits = 2), nsmall = 2), anova_eta_ci)

            tmp_outer <- data.frame(feature = outcome, ancova = anova_p, ancova_f = anova_f, log_trans = log, eta_string = eta_string)
            anova_p_values <- rbind(anova_p_values, tmp_outer)
            # If the ANOVA p-value is significant (assuming alpha = 0.05), perform Tukey's HSD test
            if (anova_p < 0.05) {
                tukey_results <- TukeyHSD(fit, which = "group")
                print(tukey_results)

                effect_sizes <- list()
                for (comparison in rownames(tukey_results$group)) {
                    groups <- strsplit(comparison, "-")[[1]]
                    g1 <- groups[1]
                    g2 <- groups[2]

                    # Subset data for the two groups being compared and remove NA values
                    group1_data <- na.omit(alldata[[outcome]][alldata$group == g1])
                    group2_data <- na.omit(alldata[[outcome]][alldata$group == g2])

                    if (length(group1_data) > 0 && length(group2_data) > 0) {
                        # Calculate Cohen's d with confidence intervals
                        tryCatch(
                            {
                                # Calculate Cohen's d with confidence intervals
                                d_result <- cohens_d(group1_data, group2_data,
                                    conf.level = 0.95, # 95% confidence interval
                                    paired = FALSE
                                )

                                # Store the results
                                effect_sizes[[paste0("d_", comparison)]] <- sprintf(
                                    "%s [%s, %s]",
                                    format(round(d_result$Cohens_d, digits = 2), nsmall = 2),
                                    format(round(d_result$CI_low, digits = 2), nsmall = 2),
                                    format(round(d_result$CI_high, digits = 2), nsmall = 2)
                                )
                            },
                            error = function(e) {
                                message(paste("Error in Cohen's d for", comparison, ":", e$message))
                            }
                        )
                    } else {
                        effect_sizes[[paste0("d_", comparison)]] <- NA
                        warning(paste("Sample size too small for Cohen's d calculation for", comparison))
                    }
                }

                # Extract p-values and add them to the list
                pvals <- unlist(tukey_results$group[, "p adj"])
                ts_list <- list()
                gnames <- names(tukey_results$group[, "diff"])
                for (gname in gnames) {
                    ts_string <- sprintf(
                        "%s [%s,%s]", format(round(tukey_results$group[gname, "diff"], digits = 2), nsmall = 2),
                        format(round(tukey_results$group[gname, "lwr"], digits = 2), nsmall = 2),
                        format(round(tukey_results$group[gname, "upr"], digits = 2), nsmall = 2)
                    )
                    # ts_string <- format(round(tukey_results$group[gname, "diff"], digits = 2), nsmall = 2)
                    ts_list[[paste0("TS_", gname)]] <- ts_string
                }
                tmp <- data.frame(as.list(pvals), feature = outcome, ts_list, effect_sizes)
                tmp

                tukey_p_values <- bind_rows(tukey_p_values, tmp)
            }
        },
        error = function(e) {
            # This code will run if there is an error
            message(paste("Error in ANOVA for", outcome, ":", e$message))
        }
    )
}
df <- merge(anova_p_values, tukey_p_values, all = TRUE)
for (i in seq_len(nrow(df))) {
    anc <- df$ancova[i]
    if (anc > 0.05) {
        df$HC.PreAD[i] <- NA
        df$HC.ProAD[i] <- NA
        df$HC.MildAD[i] <- NA
        df$PreAD.ProAD[i] <- NA
        df$PreAD.MildAD[i] <- NA
        df$ProAD.MildAD[i] <- NA
        df$TS_HC.PreAD[i] <- NA
        df$TS_HC.ProAD[i] <- NA
        df$TS_HC.MildAD[i] <- NA
        df$TS_PreAD.ProAD[i] <- NA
        df$TS_PreAD.MildAD[i] <- NA
        df$TS_ProAD.MildAD[i] <- NA
    }
}
df$HC.PreAD <- p.adjust(df$HC.PreAD, method = "holm")
df$HC.ProAD <- p.adjust(df$HC.ProAD, method = "holm")
df$HC.MildAD <- p.adjust(df$HC.MildAD, method = "holm")
df$PreAD.ProAD <- p.adjust(df$PreAD.ProAD, method = "holm")
df$PreAD.MildAD <- p.adjust(df$PreAD.MildAD, method = "holm")
df$ProAD.MildAD <- p.adjust(df$ProAD.MildAD, method = "holm")

write.csv(df, "ancova_results.csv")
warnings()
