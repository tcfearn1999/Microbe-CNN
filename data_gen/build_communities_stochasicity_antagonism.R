#!/bin/R

# setup ####
# install necessary packages, if not installed already
if(!requireNamespace("devtools",quietly = TRUE)){
  install.packages("devtools")
}
if(!requireNamespace("CommunityAssemblR",quietly = TRUE)){
  devtools::install_github("gzahn/CommunityAssemblR")
}
if(!requireNamespace("vegan",quietly = TRUE)){
  install.packages("vegan")
}
# load packages
library(tidyverse)
library(CommunityAssemblR)
library(vegan)
set.seed(666) # pseudo-random seed for reproducibility
options(scipen = 999) #makes it so the numbers aren't written in scientific notation

# 1:1000 with leading zeros
trial_number <- sprintf("%04d", 1:2000)

# start for-loop ####
# for-loop to do this 1000 times
for(i in trial_number){
  ## build communities ####
  
  ### build resident community ####
  
  # start with an even resident community
  resident <- 
    build_even_community() %>% 
    # add some more realistic relationships, keeping it fairly simple for now
    link_taxa_abundances(relationship = 'hub',
                         n.taxa = 30,
                         n.links = 3)
  
  # make relative abundance version for export later
  resident_ra <- vegan::decostand(resident,"total") %>% as("matrix")
  
  ### build donor community ####
  donor <- 
    build_donor_community_new(resident.comm = resident,
                          n.transplant.taxa = 50,
                          overlap = 0) # start off with no overlap in community membership for now
  
  ## simluate transplantation ####
  
  rand.ubiq <- runif(1,0,.5) %>% round(3)# random ubiquity
  rand.strength <- runif(1,0,1) %>% round(3) # random strength
  rand.stochast <- runif(1,0,.5) %>% round(3)
  
  final <- 
    transplant_w_antagonism(recipient = resident, 
                            donor = donor,
                            antag.ubiq = rand.ubiq, 
                            antag.strength = rand.strength,
                            antag.abundant = FALSE)
  
  final_null <- 
    transplant_w_stochasticity(recipient = resident,
                               donor = donor,
                               stochasticity = rand.stochast)
  
  ## get results ####
  # check success of transplanted taxa... before and after
  success <- 
    check_transplant_success(donor = donor,transplant = final,newtaxa.only = TRUE,tidy = FALSE,keep.all.taxa = FALSE) %>% 
    mutate(antag_ubiq = rand.ubiq, antag_strength = rand.strength)
  success_null <- 
    check_transplant_success(donor = donor,transplant = final_null,newtaxa.only = TRUE,tidy = FALSE,keep.all.taxa = FALSE) %>% 
    mutate(antag_ubiq = rand.ubiq, stoch.strength = rand.stochast)
  
  
  # pull out donor taxa from final community
  donor_before <- 
    success %>% 
    filter(timepoint == "Initial")
  #repair rownames
  row.names(donor_before) <- donor_before$sample
  
  # clean up and convert to matrix
  donor_before <- 
    donor_before %>% 
    select(-timepoint,-sample,-antag_ubiq,-antag_strength) %>% 
    as.matrix()
  
  # make sure columns are in same order in before and after matrices
  donor_before <- 
    donor_before[rownames(donor),colnames(donor)]
  
  
  # same, but for final values after transplantation
  donor_after <- 
    success %>% 
    filter(timepoint == "Final")
  #repair rownames
  row.names(donor_after) <- donor_after$sample
  
  # clean up and convert to matrix
  donor_after <- 
    donor_after %>% 
    select(-timepoint,-sample,-antag_ubiq,-antag_strength) %>% 
    as.matrix()
  
  # make sure columns are in same order in before and after matrices
  donor_after <- 
    donor_after[rownames(donor),colnames(donor)]
  
  # again, but with null transplant simulation (stochastic)
  # pull out donor taxa from final community
  donor_before <- 
    success %>% 
    filter(timepoint == "Initial")
  #repair rownames
  row.names(donor_before) <- donor_before$sample
  
  # clean up and convert to matrix
  donor_before <- 
    donor_before %>% 
    select(-timepoint,-sample,-antag_ubiq,-antag_strength) %>% 
    as.matrix()
  
  # make sure columns are in same order in before and after matrices
  donor_before <- 
    donor_before[rownames(donor),colnames(donor)]
  
  
  # same, but for final values after transplantation
  donor_after_null <- 
    success_null %>% 
    filter(timepoint == "Final")
  #repair rownames
  row.names(donor_after_null) <- donor_after_null$sample
  
  # clean up and convert to matrix
  donor_after_null <- 
    donor_after_null %>% 
    select(-timepoint,-sample,-antag_ubiq,-stoch.strength) %>% 
    as.matrix()
  
  # make sure columns are in same order in before and after matrices
  donor_after_null <- 
    donor_after_null[rownames(donor),colnames(donor)]
  
  
  ## build difference matrices ####
  antag_diff_matrix <- donor_after - donor_before
  stochast_diff_matrix <- donor_after_null - donor_before # negative values mean that taxon did poorly in new environment
  
  ## save files ####
  # compose file names
  resident_fn_01 <- file.path("/scratch/general/vast/u6049572/Micro_matrices/stochasticity",paste0("trial-",i,"_resident-community","_ubiq-",rand.ubiq,"_strength-",rand.strength,".csv"))
  resident_fn_02 <- file.path("/scratch/general/vast/u6049572/Micro_matrices/antagonism",paste0("trial-",i,"_resident-community","_ubiq-",rand.ubiq,"_strength-",rand.strength,".csv"))
  donor_diff_fn <- file.path("/scratch/general/vast/u6049572/Micro_matrices/antagonism",paste0("trial-",i,"_donor-difference_antag","_ubiq-",rand.ubiq,"_strength-",rand.strength,".csv"))
  donor_diff_null_fn <- file.path("/scratch/general/vast/u6049572/Micro_matrices/stochasticity",paste0("trial-",i,"_donor-difference_nullmodel","_noise_",rand.stochast,".csv"))
  
  # save csv files
  write_csv(as.data.frame(resident_ra),resident_fn_01)
  write_csv(as.data.frame(resident_ra),resident_fn_02)
  write_csv(as.data.frame(antag_diff_matrix),donor_diff_fn)
  write_csv(as.data.frame(stochast_diff_matrix),donor_diff_null_fn)
}

# end for-loop ####


# sanity check plot
check_transplant_success(donor = donor,transplant = final,newtaxa.only = TRUE,tidy = TRUE,keep.all.taxa = FALSE) %>% 
  mutate(antag_ubiq = rand.ubiq, antag_strength = rand.strength,
         timepoint = factor(timepoint,levels=c("Initial","Final"))) %>% 
  filter(sample == "sample_1") %>% 
  ggplot(aes(x=timepoint,y=relative_abundance,group=taxon)) +
  geom_path()
