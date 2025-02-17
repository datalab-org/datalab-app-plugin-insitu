#install NMR.Utils from https://github.com/jmstrat/NMR.Utils

#attach the necessary packages
library(NMR.Utils)
library(minpack.lm)
library(tidyverse)
library(pracma)
library(readr)

#because we run the NMR in many exp. folders we need to make a vector with each pathname, just altering the exp folder number.
#I do it by splicing Npath1 and Npath2 together in a for-loop below, using the vector "exp_folder" 
Npath1="E:/Cambridge/in situ NMR/2023-08-11_jana_insituLiLiTEGDME-02_galv"
Npath2="pdata/1/ascii-spec.txt"
nos_experiments = 600
#here I include 150 experiments and use each experiment. You can play around making this vector differently. Can also define it yourself if you ran experiment 1, 10 and 23, exp_folder = c(1,10,23)
exp_folder<- seq(1, nos_experiments, by = 1)
#if an experiment didn't work, you can exclude it from the exp_folder so you don't analyse it.
#exclude_exp <- seq(10, nos_experiments, by = 10)
#exp_folder <- exp_folder[!exp_folder %in% exclude_exp]

#This is the ID of the insitu experiment. I make a folder in my computer that has this pathname = subDir.
n = "LiLiTEGDMEinsitu_02"
subDir <- paste0("E:/Cambridge/in situ NMR/", n, "/")
#This creates the folder subDir in your computer if it doesn't exist already.

if (file.exists(subDir)){
  print("exists")
} else {
  dir.create(file.path(subDir))
  "created"
}

indx = exp_folder
mode(exp_folder) <- "character"

#the values I integrate between (not fitting, just total intensity). 
ppm2 = 310
ppm1 = 220
#if you have a lot of in situ experiment try to feed in as many information as possible so you can distinguish between different experiments when plotting later. 
#for example I could have a table that has electrolyte = LP30 and electrolyte = LiTFSI. When I plot, I can tell R, distinguish with colors between experiments with different electrolytes.
nucleus = "7Li"
compound = "Li metal"
compound2 = "metal"
cellsetup = "Li-Li"
electrolyte = "1M LiTFSI TEGDME"

input = c(n, Npath1, Npath2, ppm1, ppm2, compound, compound2, nucleus, cellsetup, electrolyte)

#we can either run this with the function read_insituCu (that I made myself) or just go through each step separately. Depends on what you like.
#input = c(n, Npath1, Npath2, ppm1, ppm2, compound, compound2, nucleus, cellsetup, Cutreatment, electrolyte)
#df_metal <- read_insituCu(input, exp_folder, indx)

#I directly save all tables as .rds files and then work on them in different folders when plotting etc. 
#That makes initial processing the same for each in-situ experiment and so you can copy&paste and it should always work.
#Later I read in the .rds file and analyse it further.
#saveRDS(df_metal, paste0(subDir, "/dfmetal", n, ".rds"))
###

  p2 = c()
  for (l in 1:length(exp_folder)) {
    p2[l] = paste(Npath1, exp_folder[l], Npath2, sep="/")
  }
  
  
  #combine strings into path to read the acqusition time and make a time vector
  p3 = c()
  d1 = c()
  t1 = c()
  for (l in 1:length(exp_folder)) {
    p3[l] = paste(Npath1, exp_folder[l], sep="/")
    r = read.acqu(p3[l])
    #store the time
    d1[l] = r$date
    #make time vector
    t1[l] = d1[l] - d1[1]
  }
  #this is the time when each NMR exp starts. This is something very hard to get from a pseudo-2D experiment.
  tNMR=t1
  
  #read first file to get ppm
  fdata2 <- read.table(p2[1], header=FALSE, sep=",", skip= 1)
  M2 = fdata2[,4]
  M.Li <- matrix(NA, length(M2), length(p2)+1)
  M.Li[,1] = M2
  M.Li[,2] = fdata2[,2]
  for (m in 2:length(p2)+1) {
    data2 <- read.table(p2[m-1], header=FALSE, sep=",", skip = 1)
    M.Li[,m] = data2[,2]
  }
  
  M.Li=as.data.frame(M.Li)
  #ppm1=210, ppm2=330
  M.Li <- filter(M.Li, V1>ppm1,V1 < ppm2)
  data.Li = M.Li
  #before renaming the column save the NMR for stackplot (call data.Li)
  saveRDS(data.Li, paste0(subDir, n, "_NMRforstackplot.rds"))
  #I throw out values below ppm2 and above ppm1 but not necessary. just to save space on computer.
  M.Li <- filter(M.Li, V1>ppm1,V1 < ppm2)
  names(M.Li) <- c('ppm',0:(ncol(M.Li)-2))
  saveRDS(M.Li, paste0(subDir, n, "_NMRdata.rds"))
  
  ### find the index where the ppm value in M.Li[,1] column is closest to ppm2 = 310 and ppm1 = 210
  ppm_indx2 <- which(abs(M.Li[,1]-ppm2)==min(abs(M.Li[,1]-ppm2)))
  ppm_indx1 <- which(abs(M.Li[,1]-ppm1)==min(abs(M.Li[,1]-ppm1)))
  
  ### Make vectors and integrate
  ppm = M.Li[ppm_indx1:ppm_indx2,1]
  #define env (stands for envelope) before the for-loop as an empty vector
  env = c()
  for (m in 1:length(M.Li)-1){
    #for each column m in M.Li (each column is one experiment we imported from topspin txt files)
    #we take out the values between ppm_indx1 and ppm_indx2 and integrate with trapezoidal method
    y = M.Li[ppm_indx1:ppm_indx2,m+1]
    env[m] = trapz(ppm,y)
  }
  #plot time vs. the integrated intensity
  plot(tNMR, env)
  
  #make dataframe
  insituLi.data <- data.frame(time = tNMR, #in seconds
                              intensity = env, #total intensity
                              norm_intensity = env/max(env), #I always look at normalised intensity
                              experiment = indx, #number of the exp folder in topspin
                              compound = compound,#additional ingo
                              nucleus = nucleus,
                              type = compound2,
                              insituLi = n,
                              cellsetup = cellsetup,
                              peak = "Total intensity", #define this "peak" as total intensity. Later I will add to this dataframe the intensity of the fitted peaks.
                              electrolyte = electrolyte
  )
  
  saveRDS(insituLi.data, paste0(subDir,"dfenv_",n, ".rds"))
  
  ggplot(insituLi.data, aes(time/3600, norm_intensity))+
    geom_point()+
    labs(x = "Time (hrs)",
         y = "Normalised intensity")+
    theme_bw()+
    theme(legend.position = c(0,1),
          legend.justification = c(0,1),
          legend.background = element_blank(),
          legend.box.background = element_rect(colour = "black"))+
    theme(plot.title = element_text(hjust = 0))
  
  printname = file.path(paste0(subDir, "normintensity.png"))
  ggsave(printname)
  
  write_tsv(M.Li, file="M.Li_02.txt")
  write_csv(M.Li, file="M.Li_02.cvs")
  write_tsv(insituLi.data, file="time-integral-metal02.txt")
  



