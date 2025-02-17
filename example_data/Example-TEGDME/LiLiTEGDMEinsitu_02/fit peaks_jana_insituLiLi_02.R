#"the fitting function". You don't need to run this as a function. 
#things to alter between 
fit_insituLi <-function(data_df, ppm1, ppm2) {
  
  fit=new_fit(integration_range=c(ppm1,ppm2))
  
  metal1 = new_model(pseudoVoigt, height=8.976e5, centre=248.0, hwhm=5,shape=0.3)
  metal2 = new_model(pseudoVoigt, height=0, centre=272.00, hwhm=5,shape=0.3)
  #metal3 = new_model(pseudoVoigt, height=0, centre=262.03, hwhm=5, shape=0.5)
  
  metal1=add_constraint_to_model(metal1, parameter='height',  constraint_type='range', constraint=c(1e5,6e6))
  metal1=add_constraint_to_model(metal1, parameter='centre', constraint_type='range', constraint=c(244.0,252.5))
  metal1=add_constraint_to_model(metal1,  parameter='hwhm', constraint_type='variable_range', constraint=c(0.5,0,6.5))
  metal1=add_constraint_to_model(metal1, parameter='shape', constraint_type='range', constraint=c(0.2,1))
  
  metal2=add_constraint_to_model(metal2, parameter='height',  constraint_type='range', constraint=c(0,5e7))
  metal2=add_constraint_to_model(metal2, parameter='centre', constraint_type='range', constraint=c(257,277.0))
  metal2=add_constraint_to_model(metal2, parameter='hwhm', constraint_type='variable_range', constraint=c(0.5,0,6.5))
  metal2=add_constraint_to_model(metal2,  parameter='shape', constraint_type='range', constraint=c(0.2,1))
  
  #metal3=add_constraint_to_model(metal3, parameter='height', constraint_type='range', constraint=c(0,1e6))
  #metal3=add_constraint_to_model(metal3, parameter='centre', constraint_type='range', constraint=c(261.0,265.0))
  #metal3=add_constraint_to_model(metal3, parameter='hwhm', constraint_type='variable_range', constraint=c(0.5,0,6.5))
  #metal3=add_constraint_to_model(metal3,  parameter='shape', constraint_type='range', constraint=c(0.2,1))
  
  fit=add_model_to_fit(fit,metal1)
  fit=add_model_to_fit(fit,metal2)
  #fit=add_model_to_fit(fit,metal3)
  
  fit=run_fit_for_data(fit,data_df)
  
}


#This is the ID of the insitu experiment. I make a folder in my computer that has this pathname = subDir.
n = "LiLiTEGDMEinsitu_02"
subDir <- paste0("E:/Cambridge/in situ NMR/", n, "/")

data_df <- readRDS(paste0(subDir, n, "_NMRdata.rds"))
#the ppm values I fit between
ppm2 = 310
ppm1 = 220
fit = fit_insituLi(data_df, ppm1, ppm2)

#access parameters via fit$results$name_of_parameter example:
plot(fit$result$hwhm_1)


plot_fit(fit,data_df,dummy.echem())
plot_examples(fit, data_df, 50, plot_offset = 10000000)
printname = file.path(paste0(subDir, "fit", ".png"))
dev.print(png, printname, units="in", width=12, height=8, pointsize=12,res=100)

printname =  paste0("E:/Cambridge/in situ NMR/fit", n, ".rds")
fit_df <- fit$result
saveRDS(fit,printname)

### bring dataframes together
df_env <- readRDS( paste0(subDir, "dfenv_", n, ".rds"))
tNMR <- df_env$time
env <- df_env$intensity

metal1 <- fit$result$integrated_area_1
metal2 <- fit$result$integrated_area_2


#if interest in fwhm etc, 

makesep_dfs <- function(fitresult, tNMR, env, input, peaktype){
  n = input[1]
  compound = input[6]
  compound2 = input[7]
  nucleus = input[8]
  cellsetup = input[9]
  data_fitted <- data.frame(time = tNMR, 
                            intensity = fitresult, 
                            norm_intensity = fitresult/max(env),
                            experiment = indx,
                            compound = compound,
                            nucleus = nucleus,
                            type = compound2,
                            insituCu = n,
                            cellsetup = cellsetup,
                            Cutreatment = input[10],
                            peak = peaktype
  )
}

df_peakfit1 <- makesep_dfs(metal1, tNMR, env, input, "Peak 1")
df_peakfit2 <- makesep_dfs(metal2, tNMR, env, input, "Peak 2")


write_tsv(df_peakfit1, file="peakfit1_Li_02.txt")
write_tsv(df_peakfit2, file="peakfit2_Li_02.txt")


## combine
df_all <- bind_rows(df_env, df_peakfit1, df_peakfit2)

ggplot(df_all, aes(time / 3600, norm_intensity, color = peak))+
  geom_line()
printname = file.path(paste0(subDir, "df_all", ".png"))
ggsave(printname)
saveRDS(df_all, paste0("E:/Cambridge/in situ NMR/", n, ".rds"))


